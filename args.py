from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments

ALLOWED_TASKS = ['gsm8k', 'cqa', 'svamp', 'multiarith', 'sqa']

def metahelp(desc="_", **kwargs):
    return {'help': desc, **kwargs}

class Seq2seqArgs(Seq2SeqTrainingArguments):
    pass

@dataclass()
class TrainerArgs:
    task: str = field(metadata={
        'required': True,
        'choices': ALLOWED_TASKS,
    })

    debug: bool = field(default=False)

    load: str = field(default="google/flan-t5-base", metadata={'required': True})
    checkpoint: str = field(default=None)

    device: str = field(default='cpu')

    train_on: str = field(default='train')
    valid_on: str = field(default='validation')
    test_on: str = field(default='test')

    teacher_data: str = field(default=None)
    teacher_data_random: float = field(default=1.0)
    fewshot: bool = field(default=False)

    save_best: int = field(default=0)

    note: str = field(default='test')

    # contrastive learning
    contrastive: bool = field(default=False)
    cl_pos_path: str = field(default=None)
    cl_neg_path: str = field(default=None)
    cl_pos_num: int = field(default=10)
    cl_neg_num: int = field(default=4)
    cl_p: float = field(default=1.0)
    cl_ratio: float = field(default=0.5)
    cl_repr: str = field(default='last', metadata={'choices': ['pool', 'last']})
    cl_clip: bool = field(default=False)
    merge_losses: bool = field(default=False, metadata={'help': "default=False, enable it will consume more VRAM"})

    # conditional mle
    mle: str = field(default='weight', metadata={'choices': ['hf', 'nograd', 'weight', 'no']})
    mle_w: float = field(default=0.1, metadata={'help': 'decide the weights of prefix tokens'})

    output_dir: str = field(default=None)

#     multi node
    master_addr: str = field(default='localhost')
    master_port: str = field(default=None)
    rank_offset: int = field(default=0)
    world_size: int = field(default=0)

#     hf training
    seed: int = field(default=42)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    train_bsz: int = field(default=2)
    eval_bsz: int = field(default=4)
    warmup_steps: int = field(default=100)
    weight_decay: float = field(default=0.1)
    lr: float = field(default=1e-6)
    deepspeed: str = field(default=None)

    epoch: int = field(default=10)
    max_steps: int = field(default=-1)

    resume: str = field(default=None)
    do_eval: bool = field(default=False)
    do_first_eval: bool = field(default=False)

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if isinstance(value, str) and value.lower() == 'none':
                self.__dict__[key] = None


@dataclass()
class InferArgs:
    task: str = field(metadata=metahelp(required=True, choices=ALLOWED_TASKS))

    load: str = field(metadata=metahelp(), default="EleutherAI/gpt-j-6B")

    checkpoint: str = field(default=None, metadata=metahelp())

    rationale: bool = field(default=True, metadata=metahelp('involve the prompt format'))
    fewshot: bool = field(default=False, metadata=metahelp())
    batch: int = field(default=4, metadata=metahelp())
    output_dir: str = field(default=None, metadata=metahelp())
    note: str = field(default="", metadata=metahelp())
    test_on: str = field(default='test', metadata=metahelp())
    seed: int = field(default=42)
    device: str = field(default='cpu', metadata=metahelp())
    decode_eos_token : bool = field(default=True, metadata=metahelp('use for gpt decoing, align the training prompt format'))

    fp16: bool = field(default=True, metadata=metahelp())

    # for star
    hint: bool = field(default=False, metadata=metahelp('star'))
    hint_filter_path: str = field(default=False, metadata=metahelp('star'))
    teacher_data: str = field(default=None, metadata=metahelp('star'))
    sample_num: int = field(default=None, metadata=metahelp())  # control the sample num
    # end


    def __post_init__(self):
        for key, value in self.__dict__.items():
            if isinstance(value, str) and value.lower() == 'none':
                self.__dict__[key] = None
    
    

