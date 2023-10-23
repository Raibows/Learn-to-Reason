import sys

import torch
import gc
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel, StoppingCriteria



def set_pad_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, pad_token='<|pad|>'):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=pad_token))

    current_size = model.get_input_embeddings().num_embeddings

    if len(tokenizer) > current_size:
        model.resize_token_embeddings(len(tokenizer))
        num_new_tokens = len(tokenizer) - current_size
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class StdOutFilter(object):
    def __init__(self, stream):
        self.stream = stream
        self.triggered = False
        self.filters = ['Invalidate trace cache', 'Positional args are being deprecated', 'Could not estimate the number', 'pytorch allocator cache flushes']

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        flag = False
        for fil in self.filters:
            if fil in data:
                flag = True
                break

        if data == '\n' and self.triggered:
            self.triggered = False
        else:
            if not flag:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()

def enable_logging_filter():
    sys.stdout = StdOutFilter(sys.stdout)
    sys.stderr = StdOutFilter(sys.stderr)

class MemCache:

    def __init__(self, rank, load, worldsize):
        self.cache = {}
        self.max_reserved = 0
        self.max_allocate = 0
        self.rank = rank

        self.freq = 0
        load = load.lower()
        if '6b' in load:
            self.freq = 1
        elif '2.7b' in load:
            self.freq = 2
        elif '1.3b' in load:
            self.freq = 4
        elif 'large' in load:
            self.freq = 4

        if worldsize < 4:
            self.freq = 1

        self.call_cnt = 0

    @staticmethod
    def torch_empty_cache():
        get_accelerator().empty_cache()

    @staticmethod
    def byte2mb(bt):
        return round(bt / (1024 ** 2), 3)



    def mclean(self):
        self.call_cnt += 1

        if self.call_cnt >= self.freq:
            self.call_cnt = 0

            r0 = torch.cuda.memory_reserved(self.rank)
            a0 = torch.cuda.memory_allocated(self.rank)
            f0 = r0 - a0

            for key in list(self.cache.keys()):
                del self.cache[key]
            gc.collect()
            get_accelerator().empty_cache()

            r1 = torch.cuda.memory_reserved(self.rank)
            a1 = torch.cuda.memory_allocated(self.rank)
            f1 = r1 - a1

        # print('Mem Free')
        # print(f'Reserved  \t {MemCache.byte2mb(r1 - r0)}MB')
        # print(f'Allocated \t {MemCache.byte2mb(a1 - a0)}MB')
        # print(f'Free      \t {MemCache.byte2mb(f1 - f0)}MB')

    def __setitem__(self, key, value):
        self.cache[key] = value
        self.max_reserved = max(self.max_reserved, torch.cuda.memory_reserved(self.rank))
        self.max_allocate = max(self.max_allocate, torch.cuda.memory_allocated(self.rank))

    def __getitem__(self, item):
        return self.cache[item]

    def __delitem__(self, *keys):
        r0 = torch.cuda.memory_reserved(self.rank)
        a0 = torch.cuda.memory_allocated(self.rank)
        f0 = r0 - a0

        for key in keys:
            del self.cache[key]

        r1 = torch.cuda.memory_reserved(self.rank)
        a1 = torch.cuda.memory_allocated(self.rank)
        f1 = r1 - a1

        # print('Cuda Free')
        # print(f'Reserved  \t {MemCache.byte2mb(r1 - r0)}MB')
        # print(f'Allocated \t {MemCache.byte2mb(a1 - a0)}MB')
        # print(f'Free      \t {MemCache.byte2mb(f1 - f0)}MB')

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(self.rank).total_memory
        r = torch.cuda.memory_reserved(self.rank)
        a = torch.cuda.memory_allocated(self.rank)
        f = r - a

        print('Cuda Info')
        print(f'Total     \t{MemCache.byte2mb(t)} MB')
        print(f'Reserved  \t{MemCache.byte2mb(r)} [{MemCache.byte2mb(self.max_reserved)}] MB')
        print(f'Allocated \t{MemCache.byte2mb(a)} [{MemCache.byte2mb(self.max_allocate)}] MB')
        print(f'Free      \t{MemCache.byte2mb(f)} MB')

class CosineSimilarityDistance(nn.Module):
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityDistance, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return -F.cosine_similarity(x1, x2, self.dim, self.eps)


def prepare_model(load, checkpoint=None, fp16=False, random_init=False):
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
    if isinstance(checkpoint, str):
        if checkpoint.lower() == 'none':
            checkpoint = None

    if random_init:
        assert checkpoint is None

    import torch
    load = load.lower()
    if 'xxl' in load or fp16:
        kwargs = {'torch_dtype': torch.float16}
    else:
        kwargs = {}

    seq2seq = ['flan', 't5']
    decoder = ['gpt', 'opt']

    for m in seq2seq:
        if m in load:
            if random_init:
                return AutoModelForSeq2SeqLM.from_config(AutoConfig.from_pretrained(load))
            else:
                return AutoModelForSeq2SeqLM.from_pretrained(checkpoint if checkpoint else load, **kwargs)

    for m in decoder:
        if m in load:
            if random_init:
                return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(load))
            else:
                return AutoModelForCausalLM.from_pretrained(checkpoint if checkpoint else load, **kwargs)

    raise NotImplementedError()


def get_generation_config(load: str, tokenizer, is_decode_eos_token: bool, max_new_tokens=None, dataset=None):
    from transformers import GenerationConfig
    from tools import tools_is_decoder_only, tools_get_logger
    if max_new_tokens is None:
        assert isinstance(dataset, str)
        max_new_tokens = {
            'gsm8k': 256,
            'svamp': 192,
            'multiarith': 192,
            'cqa': 96,
            'sqa': 128,
        }[dataset]


    if tools_is_decoder_only(load):
        if is_decode_eos_token:
            gen_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode('\n\n').append(tokenizer.eos_token_id),
                begin_suppress_tokens=tokenizer.all_special_ids,
            )
        else:
            gen_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.encode('\n\n')
            )
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
    else:
        try:
            gen_config = GenerationConfig.from_pretrained(
                load,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                # top_p=0.9,
                # repetition_penalty=0.6,
                # no_repeat_ngram_size=3,
            )
        except Exception as e:
            tools_get_logger('tools').error(f"the generation config not exists thus reset, error msg:\n{e}")

            gen_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )

    return gen_config


def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class StoppingCriteriaEos(StoppingCriteria):
    def __init__(self, eos_token_id, line_break_id, double_line_break_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.line_break_id = line_break_id
        self.double_line_break_id = double_line_break_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        should_stop = True
        gen_len = len(scores)
        for inputs in input_ids.flip(dims=[1]):
            flag = False
            for i in range(gen_len):
                if inputs[i] == self.eos_token_id: flag = True
                elif inputs[i] == self.double_line_break_id: flag = True
                elif i + 1 < gen_len and inputs[i] == self.line_break_id and inputs[i+1] == self.line_break_id: flag = True
            if not flag:
                should_stop = False
                break
        return should_stop


@torch.no_grad()
def infer_reasoning_dataset(task, test_loader, rank: int, local_rank: int, world_size: int, desc: str, model, gen_config, is_decoder_only: bool, infer_tokenizer, logger, group_gloo, verbose=False) -> dict:
    """
    :return: only world_rank == 0 will return a dict
    others will return None
    """
    from tools import cleanup, score


    correct = torch.tensor(0.0, device=rank)
    total = torch.tensor(0.0, device=rank)
    all_generated = []
    get_accelerator().empty_cache()

    stopping_criteria = [
        StoppingCriteriaEos(eos_token_id=infer_tokenizer.eos_token_id, line_break_id=infer_tokenizer.encode('\n')[0], double_line_break_id=infer_tokenizer.encode('\n\n')[0])
    ]
    # stopping_criteria = []


    with tqdm(total=len(test_loader), desc=desc, disable=local_rank > 0) as pbar:

        for batch in test_loader:

            inputs = batch['input']
            bsz = len(batch['answer'])
            total += bsz

            # direct generate the cot and answer, the Flan T5 default method
            outs = model.generate(**inputs.to(model.device), return_dict_in_generate=True, output_scores=True,generation_config=gen_config, stopping_criteria=stopping_criteria)['sequences']

            if is_decoder_only:
                outs = outs[:, inputs['input_ids'].shape[1]:]

            decode_outs = infer_tokenizer.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            raw_outs = infer_tokenizer.batch_decode(outs, skip_special_tokens=False, clean_up_tokenization_spaces=False)


            pred = []
            outs = [None for _ in decode_outs]
            for i, o in enumerate(decode_outs):
                outs[i], p = cleanup(o, task)
                pred.append(p)

            if local_rank <= 0 and (pbar.n < 1 or (verbose and pbar.n % 4 == 0)):
                logger.info(f"{'-'*160}\n"
                            f"-----------------input text--------------\n\n{batch['input_text'][0]}\n\n"
                            f"-----------------raw decode---------------\n\n{[raw_outs[0]]}\n\n"
                            f"-----------------clean special decode---------------\n\n{decode_outs[0]}\n\n"
                            f"-----------------cleaned generated---------------\n\n{outs[0]}\n\n"
                            f"-----------------prediction--------------\n\n{pred[0]}\n\n"
                            f"-----------------final answer------------\n\n{batch['final_ans'][0]}\n"
                            f"{'-'*160}\n")

            correct_num, flags = score(pred, batch['final_ans'])

            correct += correct_num

            for i in range(len(flags)):
                all_generated.append(
                    {
                        'question': batch['question'][i],
                        'answer': batch['answer'][i],
                        'final_ans': batch['final_ans'][i],
                        'generated': outs[i],
                        'pred': pred[i],
                        'correct': flags[i],
                    }
                )

            get_accelerator().empty_cache()

            pbar.set_postfix_str(f"Acc {correct.item() / total.item() * 100:.4f}, {correct.item()} / {total.item()}")
            pbar.update(1)

    # save results

    dist.monitored_barrier(group_gloo)

    gathers = [None for _ in range(world_size)] if rank <= 0 else None
    dist.gather_object(all_generated, gathers, group=group_gloo)

    if rank <= 0:
        res = []
        unique = set()
        temp_correct = 0

        for dist_gather in gathers:
            for item in dist_gather:
                if item['question'] not in unique:
                    unique.add(item['question'])
                    res.append(item)
                    if item['correct']: temp_correct += 1

        return {'generated': res, 'acc': temp_correct / len(res) * 100, 'correct': temp_correct, 'total': len(res)}

    else:

        return {}