from args import TrainerArgs
from tools import tools_set_device_env
from transformers import HfArgumentParser

args, = HfArgumentParser([TrainerArgs]).parse_args_into_dataclasses()
args: TrainerArgs
is_nvidia = tools_set_device_env(args.device)
local_world_size = 1 if args.device == 'cpu' else len(args.device.split(','))

from transformers import Seq2SeqTrainer, AutoTokenizer
from datetime import timedelta
import torch.distributed as dist
import torch.multiprocessing as mp
from tools import tools_is_decoder_only, tools_json_dump, tools_json_load, tools_prepare_trainargs_note, \
    tools_get_model_name, tools_get_time, tools_get_random_available_port, tools_get_logger
from utils import prepare_model, seed_everything, enable_logging_filter, set_pad_token
from data import get_dataset
from args import Seq2seqArgs
from trainer import CustomTrainer, TrainerHook
import glob
import wandb
import os


def setup(localrank, rank, world_size, addr, port):
    tools_set_device_env(args.device)
    enable_logging_filter()
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    os.environ['LOCAL_RANK'] = str(localrank)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("nccl" if args.device != 'cpu' else 'gloo', rank=rank, world_size=world_size)

def main_process(localrank, time_based, args: TrainerArgs):
    setup(localrank, localrank+args.rank_offset, args.world_size, args.master_addr, args.master_port)
    world_rank = localrank + args.rank_offset
    seed_everything(args.seed + world_rank)

    logger = tools_get_logger('trainer')

    if isinstance(args.max_steps, int) and args.max_steps > 0:
        args.epoch = 0
        save_strategy = 'no'
        args.do_eval = False
    else:
        save_strategy = 'epoch'

    hfargs = Seq2seqArgs(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=args.do_eval,
        do_predict=False,
        evaluation_strategy='no',
        save_strategy=save_strategy,
        report_to=[],
        remove_unused_columns=False,
        logging_steps=1 if args.debug else 16,
        ddp_find_unused_parameters=args.debug and args.mle == 'nograd',
        num_train_epochs=args.epoch,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        load_best_model_at_end=False,
        resume_from_checkpoint=args.resume,
        overwrite_output_dir=True,
        no_cuda=args.device == 'cpu',
        weight_decay=args.weight_decay,
    )


    tokenizer = AutoTokenizer.from_pretrained(args.load)

    if args.task == 'gsm8k' and args.valid_on == 'validation':
        args.valid_on = args.test_on
        if localrank <= 0:
            logger.warning(f"since gsm8k dataset does not have a validation set, so reset to {args.test_on}")

    if args.teacher_data:
        if localrank <= 0:
            logger.info(
                f"using teacher data from {args.teacher_data}, note that this will filter data that teacher answer wrong")
        teacher_data = tools_json_load(args.teacher_data)
    else:
        teacher_data = None

    dataset: dict = get_dataset(
        tokenizer.eos_token,
        args.task,
        train_on=args.train_on,
        valid_on=args.valid_on,
        test_on=args.test_on,
        is_gpt=tools_is_decoder_only(args.load),
        teacher_data=teacher_data,
        teacher_data_random=args.teacher_data_random,
        fewshot=args.fewshot,
        use_rationale=teacher_data is not None,
    )

    dataset['train'] = dataset['train'].shuffle(seed=args.seed)
    if args.debug:
        dataset['train'] = dataset['train'].select([i for i in range(97)])
        dataset['validation'] = dataset['validation'].select([i for i in range(33)])
        dataset['test'] = dataset['test'].select([i for i in range(33)])
        if localrank <= 0:
            logger.warning("you are using debug mode, the dataset size will be limited")

    if not args.do_eval:
        dataset['validation'] = dataset['validation'].select([])
        dataset['test'] = dataset['test'].select([])

    if len(dataset['train']) <= 0:
        if localrank <= 0:
            logger.error(f"fetal error, the num of train data = 0, check your teacher data {args.teacher_data}")
        dist.destroy_process_group()
        return False

    if localrank <= 0:
        for key in dataset.keys():
            logger.info(f"{key} data num={len(dataset[key])}")

        logger.info("example from train set: input------------------------------------------")
        logger.info(dataset['train'][0]['input'])
        logger.info("example from train set: target-----------------------------------------")
        logger.info(dataset['train'][0]['target'])

        logger.info(f"loading model from {args.load}\nckpt={args.checkpoint}")

    model = prepare_model(args.load, args.checkpoint, random_init=args.resume is not None)
    model = model.train()
    set_pad_token(tokenizer, model)


    group_gloo = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))

    if local_world_size < args.world_size:
    #     multiple servers
        args.output_dir = f"{args.output_dir}_world{args.world_size}_rank{args.rank_offset}-{args.rank_offset+local_world_size-1}"
    else:
        sync_obj = [args.output_dir]
        dist.monitored_barrier(group_gloo, timeout=timedelta(minutes=30))
        dist.broadcast_object_list(sync_obj, src=0, group=group_gloo)
        args.output_dir = sync_obj[0]


    if localrank <= 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/manual_ckpt", exist_ok=True)

        args_path = f"{args.output_dir}/args.json"
        hfargs_path = f"{args.output_dir}/hfargs.json"
        dsconfig_path = f"{args.output_dir}/dsconfig.json"

        if os.path.exists(args_path) or os.path.exists(hfargs_path) or os.path.exists(dsconfig_path):
            logger.warning(f"exists {args_path} or {hfargs_path}, reset to time based")
            args_path = f"{args.output_dir}/{time_based}.args.json"
            hfargs_path = f"{args.output_dir}/{time_based}.hfargs.json"
            dsconfig_path = f"{args.output_dir}/{time_based}.dsconfig.json"

        tools_json_dump(args.__dict__, args_path)
        tools_json_dump(hfargs.to_dict(), hfargs_path)
        if args.deepspeed:
            tools_json_dump(tools_json_load(args.deepspeed), dsconfig_path)

        logger.info(args.__dict__)
        logger.info(f"output dir is {args.output_dir}")


    if world_rank <= 0:
        if args.resume:
            wandb_save_dir = f"{args.output_dir}/wandblogs_{time_based}"
        else:
            wandb_save_dir = f"{args.output_dir}/wandblogs"
            os.makedirs(wandb_save_dir, exist_ok=True)

        wandbrun = wandb.init(dir=wandb_save_dir,
                              config={'args': args.__dict__, 'hf_args': hfargs.__dict__},
                              project='distil', name=args.output_dir, notes=args.note, mode='offline')
        wandbrun.define_metric('accuracy', 'epoch')

    else:
        wandbrun = None


    trainer: Seq2SeqTrainer = CustomTrainer(
        wandbrun,
        args,
        world_rank,
        logger,
        contrastive_loss=args.contrastive,
        model=model,
        args=hfargs,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[
            TrainerHook(
                myargs=args, hfargs=hfargs, logger=logger,
            ),
        ],
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    if localrank <= 0:
        logger.info(args.__dict__)

    # trainer.save_state()
    if args.max_steps > 0 and save_strategy == 'no':
        trainer.save_model(f"{args.output_dir}/last_model")

    if localrank <= 0 and False:
        pass
        # logger.info(f"try to move {args.output_dir} to {FINAL_OUTPUT_DIR}")
        # shutil.move(f"{args.output_dir}", f"{FINAL_OUTPUT_DIR}")

    if world_rank <= 0:
        wandbrun.finish()

    dist.monitored_barrier(group_gloo, timeout=timedelta(minutes=70))
    dist.destroy_process_group()


if __name__ == '__main__':
    logger = tools_get_logger('trainer')
    enable_logging_filter()
    time_based = tools_get_time()

    # env setup ------------------------------------------------------------------------------------------
    if args.world_size <= 0:
        args.world_size = local_world_size

    if args.master_port is None:
        assert args.world_size == local_world_size
        args.master_port = tools_get_random_available_port()

    if args.master_addr is None:
        args.master_addr = "localhost"

    if args.resume is not None:
        assert os.path.exists(args.resume)
        assert len(glob.glob(f"{args.resume}/global_step*")) > 0, "cannot resume without global step dir"
        args.checkpoint = None
        args.output_dir = '/'.join(args.resume.split('/')[:-1])
        logger.warning(f"resume enabled, reset checkpoint to None, reset output_dir to {args.output_dir}")

    if args.device == 'cpu':
        args.fp16 = False
        args.bf16 = False
        args.deepspeed = None
        logger.warning("in cpu mode, fp16/bf16/deepspeed are all disabled")
    else:
        assert sum([args.fp16, args.bf16]) <= 1

    if args.deepspeed:
        assert os.path.exists(args.deepspeed)

    if args.contrastive and args.cl_ratio < 1e-3:
        logger.warning(f"cl=True, but cl ratio={args.cl_ratio} < 1e-3 is too small, thus reset cl to False")
        args.contrastive = False

    if not args.contrastive:
        args.cl_neg_num = 0
        args.cl_pos_num = 0
        args.cl_neg_path = None
        args.cl_pos_path = None
        args.cl_p = 0
        args.cl_ratio = 0
        args.cl_repr = None
        assert args.mle not in {'no', 'nograd'}
    else:
        assert os.path.exists(args.cl_pos_path) and os.path.exists(args.cl_neg_path)

    if args.teacher_data:
        assert os.path.exists(args.teacher_data)

    if args.mle != 'weight':
        args.mle_w = 0

    if len(args.note) <= 48:
        args.note = tools_prepare_trainargs_note(args) + "_" + args.note

    if args.output_dir is None:
        args.output_dir = f"checkpoints/student_train/{args.task}/{tools_get_model_name(args.load)}/{time_based}_{args.note}"

    if args.debug:
        args.output_dir = f"checkpoints/debug/{tools_get_model_name(args.load)}_{time_based}_{args.note}"
        logger.warning(f"debug enabled, reset output_dir to {args.output_dir}")

    args.output_dir = args.output_dir.strip('_')
    args.note = args.note.strip('_')

    # env setup done ------------------------------------------------------------------------------------------
    mp.spawn(main_process, args=(time_based, args), nprocs=local_world_size, join=True)
