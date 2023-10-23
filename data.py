import os.path

from datasets import load_dataset, Dataset, load_from_disk
from tools import tools_json_load
from transformers import PreTrainedTokenizer
from torch.utils.data import Sampler
from tqdm import tqdm
import torch
import math
import copy
import random




def process_cl_samples(eos_token, task, addition_data_path, limit_num, is_gpt, fewshot):
    data = {k: random.sample(v, k=max(min(limit_num, len(v)), 0)) for k, v in tools_json_load(addition_data_path).items()}
    temp = {}
    for item in get_dataset(eos_token, task, teacher_data=data, teacher_data_random=1.0, is_gpt=is_gpt, fewshot=fewshot, use_rationale=True)['train']:
        if item['question'] not in temp:
            temp[item['question']] = []
        reasoning = item['input'][len(item['prefix']):]
        temp[item['question']].append(reasoning)
    return temp

class ContrastiveDynamicCollator:
    def __init__(self, tokenizer, positive_samples, negative_samples, is_gpt, debug_with_local_rank_zero=False):
        # infer dataset should not include the reasoning path or answer, but just have the prompts
        assert is_gpt

        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

        self.verbose = debug_with_local_rank_zero


    def __call__(self, features):
        batch = {k: [item[k] for item in features] for k in features[0].keys()}

        pos = []
        neg = []
        mask = []

        anchor = [item[len(p):] for item, p in zip(batch['input'], batch['prefix'])]

        for que, anc in zip(batch['question'], anchor):

            if que in self.positive_samples and que in self.negative_samples and \
                len(self.positive_samples[que]) > 0 and len(self.negative_samples[que]) > 0:

                # select a different positive sample
                succ = False
                temp = self.positive_samples[que].copy()
                random.shuffle(temp)
                for p in temp:
                    if p != anc:
                        succ = True
                        pos.append(p)
                        break
            else:
                succ = False

            if succ:
                mask.append(1)
                neg.append(random.choice(self.negative_samples[que]))
            else:
                mask.append(0)
                pos.append(self.tokenizer.eos_token)
                neg.append(self.tokenizer.eos_token)

        #  adding eos token to input(prefix+reasoning), positive, negative
        if self.verbose:
            print('-'*60, 'input', '-'*60)
            print(batch['input'][0])
            print('-'*60, 'positive', '-'*60)
            print(pos[0])
            print('-'*60, 'negative', '-'*60)
            print(neg[0])
            print('-' * 60, 'anchor', '-' * 60)
            print(anchor[0])

        # reasoning part
        batch['sample'] = self.tokenizer(anchor, padding=True, return_tensors='pt', text_target=anchor, return_attention_mask=False, max_length=192, truncation=True)

        batch['positive'] = self.tokenizer(pos, padding=True, return_tensors='pt', return_attention_mask=False, max_length=192, truncation=True)

        batch['negative'] = self.tokenizer(neg, padding=True, return_tensors='pt', return_attention_mask=False, max_length=192, truncation=True)

        batch['mask'] = torch.tensor(mask, dtype=torch.float)

        batch['prefix'] = self.tokenizer(batch['prefix'], padding=True, return_tensors='pt')

        # prompt+reasoning
        batch['input'] = self.tokenizer(
            batch['input'],
            text_target=batch['target'],
            padding=True,
            return_tensors='pt',
        )

        return batch

class DynamicCollator:
    def __init__(self, tokenizer, prepare_target=True, only_return_hfinput=False):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.prepare_target = prepare_target
        self.only_return_hfinput = only_return_hfinput


    def __call__(self, features):
        batch = {k: [item[k] for item in features] for k in features[0].keys()}

        # copy the text for debug convention
        batch['input_text'] = copy.deepcopy(batch['input'])

        # sample is only the reasoning part
        temp = [item[len(p):] for item, p in zip(batch['input'], batch['prefix'])]
        batch['sample'] = self.tokenizer(temp, padding=True, return_tensors='pt',
                                         text_target=temp, return_attention_mask=False,
                                         max_length=192, truncation=True)

        # prefix usually eqauls to fewshot demo + question
        batch['prefix'] = self.tokenizer(batch['prefix'], padding=True, return_tensors='pt')

        # prompt + reasoning
        batch['input'] = self.tokenizer(
            batch['input'],
            text_target=batch['target'] if self.prepare_target else None,
            padding=True,
            return_tensors='pt',
        )

        if self.only_return_hfinput:
            return batch['input']

        return batch

class UniqueQuestionSampler(Sampler):
    def __init__(self, rank, world_size, batch_size, seed, dataset: Dataset):
        super(UniqueQuestionSampler, self).__init__(None)
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.seed = seed
        self.hf_dataset = dataset
        self.batches = []
        self.epoch_num = 0
        self.unqique = {}

        for idx, item in enumerate(self.hf_dataset):
            if item['question'] not in self.unqique: self.unqique[item['question']] = set()
            self.unqique[item['question']].add(idx)

        self.batches = self.generate_batches()
        self.per_gpu_batches = self.batches[self.rank:len(self.batches):self.world_size]

    def __len__(self):
        return len(self.per_gpu_batches)

    def set_epoch(self, epoch: int):
        self.epoch_num = epoch

    def __iter__(self):
        if self.epoch_num > 0:
            self.batches = self.generate_batches()
            self.per_gpu_batches = self.batches[self.rank:len(self.batches):self.world_size]

        for batch in self.per_gpu_batches:
            yield batch

        self.epoch_num += 1
        # debug
        # if self.epoch_num > 5:
        #     assert False

    def generate_batches(self) -> list:
        batches = []

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch_num)

        question_idx = list(self.unqique.keys())

        total = set()
        batch = []

        with tqdm(total=len(self.hf_dataset), desc='preparing unique batches', disable=self.rank>0) as pbar:
            while len(total) < len(self.hf_dataset):
                indices = torch.randperm(len(self.unqique), generator=g).tolist()

                for idx in indices:
                    que = question_idx[idx]
                    remain = list(self.unqique[que] - total)
                    if len(remain) > 0:
                        i = torch.randint(low=0, high=len(remain), size=[1], generator=g).item()
                        total.add(remain[i])
                        pbar.update(1)
                        batch.append(remain[i])

                        if len(batch) == self.batch_size:
                            batches.append(batch)
                            batch = []

            if len(batch) > 0:
                for _ in range(self.batch_size - len(batch)):
                    i = torch.randint(low=0, high=len(self.hf_dataset), size=[1], generator=g).item()
                    batch.append(i)

                batches.append(batch)
                batch = []


        per_gpu_batch_num = math.ceil(len(batches) / self.world_size)
        should_total_batch_num = per_gpu_batch_num * self.world_size


        # dummy batch
        for _ in range(should_total_batch_num - len(batches)):
            i = torch.randint(low=0, high=len(batches), size=[1], generator=g).item()
            batches.append(batches[i])

        # shuffle
        shuffled_batches = [batches[idx] for idx in torch.randperm(len(batches), generator=g).tolist()]


        return shuffled_batches



from args import ALLOWED_TASKS

def get_hf_datasets(task, cachedir='localdataset/hfdataset'):
    assert task in ALLOWED_TASKS

    if os.path.exists(f"{cachedir}/{task}"):
        dataset = load_from_disk(f"{cachedir}/{task}")
    else:
        raise RuntimeError()

    if task in {'gsm8k', 'svamp', 'multiarith', 'sqa'}:
        dataset['validation'] = copy.deepcopy(dataset['test'])
    elif task in {'cqa'}:
        dataset['test'] = copy.deepcopy(dataset['validation'])


    return dataset


from preprocessors import GSM8KProcessor, CQAProcessor, SVAMPProcessor, MultiArithProcessor, SQAProcessor

def get_dataset(eos_token, task, train_on='train', valid_on='validation', test_on='test', teacher_data=None, teacher_data_random=0.0, is_gpt=False, use_rationale=True, fewshot=False, **kwargs) -> dict:

    assert task in ALLOWED_TASKS

    res = {}

    processor = {
        'gsm8k': GSM8KProcessor,
        'cqa': CQAProcessor,
        'svamp': SVAMPProcessor,
        'multiarith': MultiArithProcessor,
        'sqa': SQAProcessor,
    }[task]

    dataset = get_hf_datasets(task)

    remove_columns = []
    if task == 'svamp':
        remove_columns = dataset['train'][0].keys()
    elif task == 'cqa':
        remove_columns = ['id', 'question_concept', 'answerKey']
    elif task == 'sqa':
        remove_columns = ['qid', 'term', 'description']


    res['train'] = dataset[train_on].map(
        processor(eos_token=eos_token, is_train_set=True, is_gpt=is_gpt, teacher_data_random=teacher_data_random,
                  teacher_data=teacher_data, use_rationale=use_rationale, fewshot=fewshot, **kwargs),
        load_from_cache_file=False, batched=True, remove_columns=remove_columns
    )
    res['validation'] = dataset[valid_on].map(
        processor(eos_token=eos_token, is_train_set=False, is_gpt=is_gpt, teacher_data_random=0.0,
                  teacher_data=None, use_rationale=use_rationale, fewshot=fewshot, **kwargs),
        load_from_cache_file=False, batched=True, remove_columns=remove_columns
    )
    res['test'] = dataset[test_on].map(
        processor(eos_token=eos_token, is_train_set=False, is_gpt=is_gpt, teacher_data_random=0.0,
                  teacher_data=None, use_rationale=use_rationale, fewshot=fewshot, **kwargs),
        load_from_cache_file=False, batched=True, remove_columns=remove_columns
    )

    return res


if __name__ == '__main__':
    dataset = get_dataset('<eostoken>', 'multiarith', is_gpt=True)['train']
    print(len(dataset), dataset[0].keys())
    print(dataset[0])


    exit(0)

