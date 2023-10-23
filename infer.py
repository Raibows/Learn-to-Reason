from args import InferArgs
from transformers import HfArgumentParser
from tools import tools_set_device_env
import os


os.environ['NCCL_DEBUG']='WARN'
os.environ['DATASETS_VERBOSITY']='error'
args,  = HfArgumentParser([InferArgs]).parse_args_into_dataclasses()
args: InferArgs
is_nvidia = tools_set_device_env(args.device)



import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from data import get_dataset, DynamicCollator
from tools import tools_json_dump, tools_prepare_inferargs_note, \
    tools_is_decoder_only, tools_get_random_available_port, tools_get_logger
from utils import infer_reasoning_dataset, seed_everything, prepare_model, get_generation_config, set_pad_token



def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    # initialize the process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else 'gloo', rank=rank, world_size=world_size)

@torch.no_grad()
def main(rank, is_nvidia, world_size, port, args: InferArgs):

    setup(rank, world_size, port)
    tools_set_device_env(args.device)

    logger = tools_get_logger('infer')
    if rank <= 0:
        logger.info(f'available devices {torch.cuda.device_count()}')

    seed_everything(args.seed)

    # prepare the dataset ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.load, padding_side='left' if tools_is_decoder_only(args.load) else 'right')
    assert tokenizer.eos_token is not None
    dataset = get_dataset(tokenizer.eos_token, args.task, test_on=args.test_on, is_gpt=tools_is_decoder_only(args.load), fewshot=args.fewshot, use_rationale=args.rationale)['test']


    # control the dataaset num ------------------------------------------------------------
    if isinstance(args.sample_num, int):
        args.sample_num = min(args.sample_num, len(dataset))
        if rank <= 0:
            logger.info(f"select sample num = {args.sample_num}")

        dataset = dataset.select([i for i in range(args.sample_num)])

    # prepare model -------------------------------------------------------------
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        if rank <= 0:
            logger.error(f"{args.checkpoint} do not exist, thus reset to None")
        args.checkpoint = None

    if rank <= 0:
        logger.info(f"loading {args.load}, checkpoint = {args.checkpoint}")

    model = prepare_model(args.load, args.checkpoint, fp16=args.fp16).to(rank)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # prepare test dataloader ------------------------------------------------------------
    set_pad_token(tokenizer, model)
    gen_config = get_generation_config(args.load, tokenizer, args.decode_eos_token, dataset=args.task)
    if rank <= 0:
        logger.info(f"gen_config={gen_config.__dict__}")

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        collate_fn=DynamicCollator(tokenizer, prepare_target=False),
        sampler=DistributedSampler(dataset, world_size, rank, shuffle=False, )
    )
    test_loader.sampler.set_epoch(0)

    group_gloo = dist.new_group(backend="gloo")
    # ready to evaluate -------------------------------------------------------------
    results = infer_reasoning_dataset(
        task=args.task,
        test_loader=test_loader,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        desc=f"infer={args.task} set={args.test_on} device={args.device}",
        model=model,
        gen_config=gen_config,
        is_decoder_only=tools_is_decoder_only(args.load),
        infer_tokenizer=tokenizer,
        logger=logger,
        group_gloo=group_gloo,
        verbose=True,
    )

    # save results -------------------------------------------------------------
    if rank <= 0:
        info = {
            'args': args.__dict__,
            'correct': results['correct'],
            'total': results['total'],
            'acc': results['acc'],
            'results': results['generated'],
            'gen_config': gen_config.__dict__,
        }
        tools_json_dump(info, f"{args.output_dir}/Acc{results['acc']:.4f}.json")
        logger.info(args)
        logger.info(f"correct {results['correct']} total {results['total']} acc {results['acc']:.4f}")


    dist.monitored_barrier(group_gloo)
    dist.destroy_process_group()

if __name__ == '__main__':
    gpu_num  = torch.cuda.device_count()
    print(f"device_count: {torch.cuda.device_count()}")

    if args.output_dir is None:
        args.output_dir = f"checkpoints/evaluate/{tools_prepare_inferargs_note(args)}"

    os.makedirs(args.output_dir, exist_ok=False)

    mp.spawn(main, args=(is_nvidia, gpu_num, tools_get_random_available_port(), args), nprocs=gpu_num, join=True)