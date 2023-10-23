import os
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, HfArgumentParser, GenerationConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler

from args import InferArgs
from data import get_dataset, DynamicCollator
from tools import tools_json_dump, tools_set_device_env, tools_get_time, \
    tools_is_decoder_only, tools_get_random_available_port, \
    tools_get_logger, cleanup
from utils import prepare_model, seed_everything, set_pad_token


args, = HfArgumentParser([InferArgs]).parse_args_into_dataclasses()
args: InferArgs

is_nvidia = tools_set_device_env(args.device)


def setup(rank, world_size, port, is_nvidia):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


@torch.no_grad()
def main(rank, is_nvidia, world_size, port, args: InferArgs):
    setup(rank, world_size, port, is_nvidia)
    tools_set_device_env(args.device)

    logger = tools_get_logger('infer')
    if rank <= 0:
        logger.info(f'available devices {torch.cuda.device_count()}')

    seed_everything(args.seed)

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

    # prepare the dataset ----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.load, padding_side='left' if tools_is_decoder_only(args.load) else 'right')

    set_pad_token(tokenizer, model)

    dataset = get_dataset(tokenizer.eos_token, args.task, test_on=args.test_on, is_gpt=tools_is_decoder_only(args.load), fewshot=args.fewshot,
                          use_rationale=args.rationale)['test']
    # tood debug
    # dataset = dataset.select([i for i in range(7)])

    # control the dataaset num ------------------------------------------------------------
    if isinstance(args.sample_num, int):
        args.sample_num = min(args.sample_num, len(dataset))
        if rank <= 0:
            logger.info(f"select sample num = {args.sample_num}")

        dataset = dataset.select([i for i in range(args.sample_num)])

    # print examples -------------------------------------------------------------
    if rank <= 0:
        logger.info(f"evaluate {args.load} from ckpt {args.checkpoint} on {args.test_on} of {args.task} dataset num = {len(dataset)}")

        logger.info(f"-----------------sample example---------------------------\n{dataset[0]}\n")
        logger.info(f"-----------------sample input---------------------------\n{dataset[0]['input']}\n")
        logger.info(f"-----------------sample final_ans---------------------------\n{dataset[0]['final_ans']}\n")



    # prepare test dataloader ------------------------------------------------------------
    gen_config = GenerationConfig(
                    do_sample=True,
                    max_new_tokens=128,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.encode('\n\n').append(tokenizer.eos_token_id),
                    num_return_sequences=4,
                )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        collate_fn=DynamicCollator(tokenizer, prepare_target=False),
        sampler=DistributedSampler(dataset, world_size, rank, shuffle=False, )
    )

    test_loader.sampler.set_epoch(0)

    # ready to evaluate -------------------------------------------------------------
    correct = 0
    total = 0
    all_generated = {}

    torch.cuda.empty_cache()

    with tqdm(total=len(test_loader), desc=f'infer {args.load} {args.task} set={args.test_on} cuda:{args.device}', disable=rank > 0) as pbar:
        for batch in test_loader:

            inputs = batch['input']
            bsz = len(batch['answer'])
            total += bsz

            # direct generete the cot and answer, the flant5 default method
            outs = model.generate(**inputs.to(rank), generation_config=gen_config)

            if tools_is_decoder_only(args.load):
                outs = outs[:, inputs['input_ids'].shape[1]:]

            outs = tokenizer.batch_decode(outs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            gen = None
            label = None

            for i, o in enumerate(outs):
                que = batch['question'][i // gen_config.num_return_sequences]
                label = batch['final_ans'][i // gen_config.num_return_sequences]
                gen, pred = cleanup(o, args.task)

                if pred.lower() != label.lower():
                    if que not in all_generated:
                        all_generated[que] = set()
                    all_generated[que].add(gen)
                else:
                    correct += 1

                total += 1

            if (pbar.n % 16 == 0 or pbar.n == 0) and rank <= 0:
                logger.info("------------------generated sample--------------------")
                logger.info(gen)
                logger.info("------------------label-------------------------------")
                logger.info(label)


            torch.cuda.empty_cache()

            pbar.set_postfix_str(f"#of Return Sequence {gen_config.num_return_sequences} Acc {correct / total*100:.4f}, {correct} / {total}")
            pbar.update(1)

    # save results -------------------------------------------------------------
    all_generated = {k: list(v) for k, v in all_generated.items()}

    correct = torch.tensor(correct, device=rank)
    total = torch.tensor(total, device=rank)
    group_gloo = dist.new_group(backend="gloo")
    dist.monitored_barrier(group_gloo)

    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    correct = correct.item()
    total = total.item()

    if rank <= 0:
        logger.info(f"#of Return Sequence {gen_config.num_return_sequences} Acc {correct / total:.5f}, {correct} / {total}")

    gathers = [None for _ in range(world_size)] if rank <= 0 else None
    dist.gather_object(all_generated, gathers, dst=0, group=group_gloo)

    if rank <= 0:
        res = {}
        for gens in gathers:
            for k, v in gens.items():
                res[k] = v

        tools_json_dump(res, f"{args.output_dir}/student_wrong.json")
        config = {
            'args': args.__dict__,
            'gen_config': gen_config.__dict__,
        }
        tools_json_dump(config, f"{args.output_dir}/config.json")

        logger.info(args.__dict__)

    group_gloo = dist.new_group(backend="gloo")
    dist.monitored_barrier(group_gloo)
    dist.destroy_process_group()


if __name__ == '__main__':
    assert os.path.exists('scripts/endrun.sh')

    gpu_num = torch.cuda.device_count()
    print(f"device_count: {torch.cuda.device_count()}")

    if '/' in args.load:
        model_name = args.load.split('/')[1]
    else:
        model_name = args.load

    if args.output_dir is None:
        args.output_dir = f"checkpoints/infer/{args.task}_{model_name}_{tools_get_time()}_{args.note}"

    os.makedirs(args.output_dir, exist_ok=False)

    mp.spawn(main, args=(is_nvidia, gpu_num, tools_get_random_available_port(), args), nprocs=gpu_num, join=True)

    os.system('bash scripts/endrun.sh')