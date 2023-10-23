import logging
import time
from datetime import datetime
import pytz
import json
import re
import os
from typing import List, Tuple

ABSOLUTE_WRONG_FINAL_ANS = "will_never_be_the_right_answer"

ZONE = pytz.timezone("Asia/Chongqing")
def timetz(*args):
    return datetime.now(ZONE).timetuple()

logging.Formatter.converter = timetz

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s\n%(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %H:%M:%S",
)


def tools_prepare_trainargs_note(args) -> str:
    from args import TrainerArgs
    args: TrainerArgs

    if args.teacher_data:
        note_train_data = re.findall("round.*\.", args.teacher_data)
        if len(note_train_data) > 0:
            note_train_data = note_train_data[0][:-1]
        else:
            note_train_data = 'unknown'
    else:
        note_train_data = "original"


    flag = False
    if args.checkpoint is None:
        flag = True

    else:
        for end in ['gpt-j-6B', 'gpt-neo-2.7B', 'opt-imb-1.3b', 'gpt2-large']:
            if args.checkpoint.endswith(end):
                flag = True
                break
    if flag:
        note_init = "initHF"
    else:
        note_init = ""

    note_optim = f"lr{args.lr}wd{args.weight_decay}bs{args.train_bsz * args.world_size}"

    return f"{note_train_data}_{note_optim}_{note_init}"

def tools_prepare_inferargs_note(args) -> str:

    from args import InferArgs
    args: InferArgs

    if len(args.note) > 32:
        return args.note

    if '/' in args.load:
        model_name = args.load.split('/')[1]
    else:
        model_name = args.load

    if args.fewshot:
        fewshot = 'fs'
    else:
        fewshot = 'NOfs'

    if args.rationale:
        rationale = 'rat'
    else:
        rationale = 'NOrat'

    if args.decode_eos_token:
        eos = 'eos'
    else:
        eos = 'NOeos'

    return f"{model_name}_{args.task}_{args.test_on}set_{fewshot}_{rationale}_{eos}_{tools_get_time()}_{args.note}".strip('_')


def tools_is_decoder_only(model_name):
    model_name = model_name.lower()
    temp =  {'EleutherAI/gpt-j-6B', 'sshleifer/tiny-gpt2', "EleutherAI/gpt-neo-2.7B", 
                     'facebook/opt-iml-1.3b'}
    for n in {'gpt', 'opt'}:
        if n in model_name:
            return True
    
    return False

def tools_get_random_available_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    time.sleep(3)
    return port

def tools_json_load(path):
    with open(path, 'r') as file:
        return json.load(file)

def tools_json_dump(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file, indent=4)

def tools_get_model_name(load: str):
    if '/' in load:
        return load.split('/')[1]
    else:
        return load

def tools_set_device_env(device: str):
    if device != 'cpu':
        if os.system("rocm-smi > /dev/null 2>&1") == 0:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            # os.environ['ROCR_VISIBLE_DEVICES'] = device
            # os.environ["CUDA_VISIBLE_DEVICES"] = ""
            nvidia = False
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            nvidia = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ['ROCR_VISIBLE_DEVICES'] = ""
        nvidia = False
    return nvidia

def tools_is_nvidia() -> bool:
    import torch
    if os.system("rocm-smi > /dev/null 2>&1") == 0:
        return False
    elif torch.cuda.device_count() > 0:
        return True
    return False


def tools_get_time():
    return datetime.now(ZONE).strftime("%y-%m-%d-%H_%M_%S")

def tools_get_logger(name):
    return logging.getLogger(name)

def clean_repeat_generated(sample: str) -> str:
    for m in re.finditer(r'\(a\)|\(b\)|\(c\)|\(d\)|\(e\)', sample):
        sample = sample[:m.end()+1]
        break

    return sample.strip()

def cleanup(pred, dataset='gsm8k') -> Tuple[str, str]:
    """
    :param pred: generated text
    :param dataset: task
    :return: [cleaned_text, final_prediction]
    """
    if dataset in {'gsm8k', 'svamp', 'multiarith'}:
        pred = pred.strip().split('\n\n')[0]
        temp = pred

        struct_ans_flag = False
        for answer_prefix in ['\nAnswer', 'Therefore, the answer is']:
            if answer_prefix in pred:
                temp = pred.split(answer_prefix)[1].strip()
                struct_ans_flag = True
                break
        
        temp_ori = [item for item in re.findall(r'-?\d+\.?\$?,?\d*', temp)]
        temp = [item.strip('.') for item in re.findall(r'-?\d+\.?\d*', temp.replace(',', ''))]

        if len(temp) == 0:
            final_pred = ABSOLUTE_WRONG_FINAL_ANS
            if struct_ans_flag:
                answer_prefix_idx = pred.index(answer_prefix)
                next_word = pred[answer_prefix_idx+len(answer_prefix):].split()
                if next_word[0] == ':':
                    if len(next_word) == 1:
                        next_word = ' '
                    else:
                        next_word = ': ' + next_word[1]
                else:
                    next_word = ' ' + next_word[0]
                pred = pred[:answer_prefix_idx+len(answer_prefix)] + next_word

        elif struct_ans_flag:
            final_pred = temp[0]
            answer_prefix_idx = pred.index(answer_prefix)
            if final_pred in pred[answer_prefix_idx:]:
                temp_idx = pred[answer_prefix_idx:].index(final_pred)
                pred = pred[:answer_prefix_idx + temp_idx + len(final_pred)]
            else:
                next_word = pred[answer_prefix_idx+len(answer_prefix):].split()
                if next_word[0] == ':':
                    next_word = ': ' + next_word[1]
                else:
                    next_word = ' ' + next_word[0]
                pred = pred[:answer_prefix_idx + len(answer_prefix)] + next_word
                
        elif not struct_ans_flag:
            final_pred = temp[-1]
            if final_pred in pred:
                pred = pred[:pred.index(final_pred) + len(final_pred)]
            elif temp_ori[-1] in pred:
                pred = pred[:pred.index(temp_ori[-1]) + len(temp_ori[-1])]
            else:
                pass
        else:
            raise RuntimeError()

    elif dataset == 'sqa':
        pred = pred.strip().split('\n\n')[0]
        struct_ans_flag = False

        temp = pred
        for answer_prefix in ['\nAnswer']:
            if answer_prefix in pred:
                temp = pred.split(answer_prefix)[1].strip()
                struct_ans_flag = True
                break

        res = []
        for item in temp.split():
            item = item.lower().strip('.')
            if item in {'yes', 'no'}:
                res.append(item)

        if len(res) == 0:
            final_pred = ABSOLUTE_WRONG_FINAL_ANS
        elif struct_ans_flag:
            final_pred = res[0]
        elif not struct_ans_flag:
            final_pred = res[-1]
        else:
            raise NotImplementedError()

    elif dataset == 'cqa':
        pred = pred.strip().split('\n\n')[0].split('\n')[0].strip()
        pred = clean_repeat_generated(pred)

        final_pred = re.findall(r'\(a\)|\(b\)|\(c\)|\(d\)|\(e\)', pred)
        if len(final_pred) == 0 and pred in {'a', 'b', 'c', 'd', 'e'}:
            final_pred = pred
        elif len(final_pred) > 0:
            final_pred = final_pred[-1].replace('(', '').replace(')', '')
        else:
            final_pred = ABSOLUTE_WRONG_FINAL_ANS

    else:
        raise NotImplementedError()

    return pred, final_pred.lower()

def vote(after_cleanup_res_list):
    assert False
    return max(set(after_cleanup_res_list), key=after_cleanup_res_list.count)

def score(pred: List, final_ans: List):
    assert(len(pred) == len(final_ans))
    ans = [i.lower() == j.lower() for (i, j) in zip(pred, final_ans)]
    return sum(ans), ans

def template_split_get_only_generate(outs, template_id):
    assert template_id == 'v1'
    return [o.split("\nA: Let's think step by step.")[-1] for o in outs]


if __name__ == '__main__':
    print(tools_get_time())
    print(tools_get_time())