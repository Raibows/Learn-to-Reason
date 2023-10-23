import copy
import os.path

import torch
import glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import nn
from transformers import Seq2SeqTrainer, TrainerCallback, Seq2SeqTrainingArguments, TrainerState, TrainerControl, \
    AutoTokenizer
from typing import Dict, Optional, List, Union, Any
from datetime import timedelta
from deepspeed.runtime.engine import DeepSpeedEngine

from metric import Metric
from args import TrainerArgs
from tools import tools_is_decoder_only, tools_json_dump, tools_json_load

from data import DynamicCollator, ContrastiveDynamicCollator, UniqueQuestionSampler, process_cl_samples
from utils import MemCache, infer_reasoning_dataset, CosineSimilarityDistance, get_generation_config, set_pad_token


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, wandbrun, myargs: TrainerArgs, world_rank, logger, contrastive_loss=False, **kwargs):
        train_dataset = kwargs['train_dataset']
        self.wandbrun = wandbrun
        self.tokenizer = kwargs['tokenizer']
        self.contrastive_loss: bool = contrastive_loss
        assert isinstance(self.contrastive_loss, bool)

        train_batch_sampler = UniqueQuestionSampler(world_rank, myargs.world_size,
                                                    myargs.train_bsz, myargs.seed,
                                                    train_dataset)

        if self.contrastive_loss:
            pos_data = process_cl_samples(
                eos_token=self.tokenizer.eos_token,
                task=myargs.task,
                addition_data_path=myargs.cl_pos_path,
                limit_num=myargs.cl_pos_num,
                is_gpt=tools_is_decoder_only(myargs.load),
                fewshot=myargs.fewshot
            )
            neg_data = process_cl_samples(
                eos_token=self.tokenizer.eos_token,
                task=myargs.task,
                addition_data_path=myargs.cl_neg_path,
                limit_num=myargs.cl_neg_num,
                is_gpt=tools_is_decoder_only(myargs.load),
                fewshot=myargs.fewshot
            )

            train_collator = ContrastiveDynamicCollator(
                self.tokenizer,
                pos_data,
                neg_data,
                is_gpt=tools_is_decoder_only(myargs.load),
                debug_with_local_rank_zero=myargs.debug and kwargs['args'].local_rank <= 0
            )

            self.cl_criterion = nn.TripletMarginWithDistanceLoss(
                distance_function=CosineSimilarityDistance(),
                margin=myargs.cl_p,
                reduction='none'
            )

            self.cl_get_repr = {
                'pool': self.pooling_hidden,
                'last': self.last_valid_token
            }[myargs.cl_repr]


        else:
            train_collator = DynamicCollator(self.tokenizer, prepare_target=True,
                                             only_return_hfinput=False)

        self.train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                       collate_fn=train_collator)

        # for inference dataset
        kwargs['data_collator'] = DynamicCollator(self.tokenizer, prepare_target=True, only_return_hfinput=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')


        self.records = {}

        self.myargs = myargs

        self.world_rank = world_rank

        self.logger = logger

        self.infer_tokenizer = AutoTokenizer.from_pretrained(myargs.load, padding_side='left' if tools_is_decoder_only(myargs.load) else 'right')
        if tools_is_decoder_only(myargs.load):
            set_pad_token(self.infer_tokenizer, kwargs['model'])

        self.memcache = MemCache(world_rank, myargs.load, self.myargs.world_size)

        super(CustomTrainer, self).__init__(**kwargs)

    def compute_mle_weighted_loss(self, model, inputs):
        labels = inputs['input']['labels']
        del inputs['input']['labels']
        self.memcache['shift_logits'] = model(**inputs['input'], output_hidden_states=True, return_dict=True)['logits']

        device = self.memcache['shift_logits'].device

        self.memcache['shift_logits'] = self.memcache['shift_logits'][..., :-1, :].contiguous()
        self.memcache['shift_labels'] = labels[..., 1:].contiguous()

        # cannot use logical not, since you manually add the eos_token (=pad_token)
        self.memcache['valid_length'] = inputs['input']['attention_mask'].sum(dim=-1) - 1
        for i, vl in enumerate(self.memcache['valid_length']):
            self.memcache['shift_labels'][i][vl:] = -100

        self.memcache['prefix_length'] = inputs['prefix']['attention_mask'].sum(dim=-1)

        self.memcache['weights'] = torch.ones_like(self.memcache['shift_labels'], device=device, dtype=self.memcache['shift_logits'].dtype)
        for i, pl in enumerate(self.memcache['prefix_length']):
            self.memcache['weights'][i][:pl] = self.myargs.mle_w

        del self.memcache['prefix_length']

        losses = self.criterion(self.memcache['shift_logits'].view(-1, self.memcache['shift_logits'].shape[2]), self.memcache['shift_labels'].view(-1))

        return (losses * self.memcache['weights'].view(-1)).sum() / self.memcache['valid_length'].sum().float()

    def compute_hf_original_loss(self, model, inputs):
        return super().compute_loss(model, inputs['input'], False)

    def compute_cl_loss(self, model, inputs, require_mle_loss=False):
        assert 'prefix' in inputs
        assert 'sample' in inputs


        with torch.no_grad():
            self.memcache['prefix'] = model(**inputs['prefix'], use_cache=True, return_dict=True)['past_key_values']


        if require_mle_loss:
            self.memcache['mle_outputs'] = model(**inputs['sample'], past_key_values=self.memcache['prefix'], output_hidden_states=True, return_dict=True)
            loss = self.memcache['mle_outputs']['loss']
        else:
            del inputs['sample']['labels']
            self.memcache['mle_outputs'] = model(**inputs['sample'], past_key_values=self.memcache['prefix'], output_hidden_states=True, return_dict=True)

        self.memcache['repr'] = {}
        self.memcache['repr']['sample'] = self.cl_get_repr(inputs['sample']['input_ids'],
                                             self.tokenizer.pad_token_id,
                                             self.memcache['mle_outputs']['hidden_states'][-1])


        del self.memcache['mle_outputs']

        for k in ['positive', 'negative']:
            assert k in inputs
            self.memcache['temp'] = model(**inputs[k], past_key_values=self.memcache['prefix'], output_hidden_states=True,
                         return_dict=True)
            self.memcache['repr'][k] = self.cl_get_repr(inputs[k]['input_ids'],
                                          self.tokenizer.pad_token_id,
                                          self.memcache['temp']['hidden_states'][-1])

            del self.memcache['temp']

        cl_loss = self.compute_contrastive_loss(
            self.memcache['repr']['sample'],
            self.memcache['repr']['positive'],
            self.memcache['repr']['negative'],
            inputs['mask'],
            model.device,
        ) * self.myargs.cl_ratio

        if require_mle_loss:
            return loss, cl_loss

        return cl_loss

    @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
    def last_valid_token(self, input_ids, pad_token_id, hidden_state):
        self.memcache['mask'] = (input_ids != pad_token_id).long()
        self.memcache['select'] = torch.sum(self.memcache['mask'], dim=1) - 1
        hidden = hidden_state[torch.arange(hidden_state.shape[0]), self.memcache['select'], :]
        del self.memcache['mask'], self.memcache['select']
        return hidden

    @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
    def pooling_hidden(self, input_ids, pad_token_id, hidden_state):
        self.memcache['mask'] = (input_ids != pad_token_id).float()
        self.memcache['length'] = torch.sum(self.memcache['mask'], dim=1)
        pooling = (self.memcache['mask'].unsqueeze(1) @ hidden_state).squeeze() / self.memcache['length'].unsqueeze(1)
        del self.memcache['mask'], self.memcache['length']
        return pooling

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def logwandb(self, log: dict, step: int):
        for k, v in log.items():
            if k not in self.records:
                self.records[k] = []
            self.records[k].append(v)

        if self.state.is_world_process_zero:
            self.wandbrun.log(log, step=step, commit=step % self.args.logging_steps == 0)
        else:
            pass

    def log(self, logs: Dict[str, float]) -> None:

        logs["step"] = self.state.global_step

        for k, v in self.records.items():
            if len(v) > 0 and isinstance(v[0], float):
                logs[f'avg_{k}'] = sum(v[-100:]) / min(100, len(v))
            elif k == 'accuracy':
                temp = list(sorted(v, key=lambda x: (x['acc'], -x['step'])))
                logs['first_best'] = temp[-1]
            else:
                logs[k] = v[-1]

        return super().log(logs)

    # @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
    # def compute_contrastive_loss(self, p, sample_repr, positive_repr, negative_repr, mask, device):
    #     return torch.maximum(
    #         torch.tensor(0.0).to(device),
    #         torch.nan_to_num(
    #             p -
    #             torch.cosine_similarity(sample_repr, positive_repr, dim=1) +
    #             torch.cosine_similarity(sample_repr, negative_repr, dim=1)
    #         )
    #         * mask.to(device)
    #     ).sum() / torch.maximum(torch.sum(mask), torch.tensor(1e-6, device=device))

    @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
    def compute_contrastive_loss(self, sample_repr, positive_repr, negative_repr, mask, device):
        return (torch.nan_to_num(self.cl_criterion.forward(
            sample_repr, positive_repr, negative_repr
        )) * mask.to(device)).sum() / torch.maximum(torch.sum(mask), torch.tensor(1e-6, device=device))

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        from transformers.trainer import WEIGHTS_NAME

        if self.deepspeed and self.deepspeed.zero_optimization_stage() < 3:
            if self.args.should_save:
                self._save(output_dir)
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        os.remove(file)
                    for path in glob.glob(f"{output_dir}/pytorch_model*"):
                        if os.path.isfile(path): os.remove(path)

                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    self.logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)
        else:
            super(CustomTrainer, self).save_model(output_dir, _internal_call)

        if self.state.is_local_process_zero and isinstance(output_dir, str):
            tools_json_dump(self.myargs.__dict__, os.path.join(output_dir, 'args.json'))
            tools_json_dump(self.args.to_dict(), os.path.join(output_dir, 'hfargs.json'))
            if self.myargs.deepspeed:
                tools_json_dump(tools_json_load(self.myargs.deepspeed), os.path.join(output_dir, 'dsconfig.json'))


        #     try to remove previous ckpt_state, like global step
            if int(self.state.epoch) > 1:
                glob_checkpoints = [str(x) for x in glob.glob(f"{self.myargs.output_dir}/checkpoint-*") if os.path.isdir(x)]
                glob_checkpoints = [(x, int(x.split('checkpoint-')[1])) for x in glob_checkpoints]
                glob_checkpoints = list(sorted(glob_checkpoints, key=lambda x: x[1]))
                if len(glob_checkpoints) > 1:
                    remove_dir = f"{glob_checkpoints[-2][0]}/global_step{glob_checkpoints[-2][1]}"
                    self.logger.info(f"remove ckpt global step {remove_dir} background")
                    if os.path.isdir(remove_dir):
                        os.system(f"rm -rf {remove_dir} &")

        self.evaluate(ckpt_dir=output_dir)

    def process_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            raise NotImplementedError()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        if self.state.global_step == 0 and (self.deepspeed is None or self.deepspeed.zero_optimization_stage() < 3) and self.myargs.do_eval and self.myargs.do_first_eval:

            if self.state.is_local_process_zero:
                self.logger.info(f"first evaluating the init load={self.myargs.load}\nckpt={self.myargs.checkpoint}")

            self.evaluate()


        if self.state.global_step % 16 == 0:
            self.memcache.torch_empty_cache()


        model.train()
        inputs = self._prepare_inputs(inputs)


        # mle loss
        if self.myargs.mle == "weight":
            with self.compute_loss_context_manager():
                loss = self.compute_mle_weighted_loss(model, copy.deepcopy(inputs))
            self.memcache.mclean()

        elif self.myargs.mle == 'hf':
            with self.compute_loss_context_manager():
                loss = self.compute_hf_original_loss(model, copy.deepcopy(inputs))

        elif self.myargs.mle == 'nograd':
            with self.compute_loss_context_manager():
                loss, cl_loss = self.compute_cl_loss(model, copy.deepcopy(inputs), require_mle_loss=True)

        elif self.myargs.mle == 'no':
            assert not self.myargs.merge_losses
            loss = None
        else:
            raise NotImplementedError()


        # mle loss
        if not self.myargs.merge_losses:
            if loss is None:
                total_loss = torch.tensor(0.0, device=self.model.device)
            else:
                total_loss = self.process_loss(loss)

        # check cl loss
        if self.contrastive_loss:

            # nograd already calculated the cl_loss
            if self.myargs.mle != 'nograd':
                with self.compute_loss_context_manager():
                    cl_loss = self.compute_cl_loss(model, copy.deepcopy(inputs))

            if self.myargs.cl_clip and (loss > cl_loss or cl_loss < 1e-2):
                cl_loss *= 0.0

            self.memcache.mclean()

            if self.myargs.merge_losses:
                total_loss = self.process_loss(loss + cl_loss)
            else:
                total_loss += self.process_loss(cl_loss)


        # process logs
        logs = {}

        if self.contrastive_loss:
            logs['cl_loss'] = cl_loss.item()
            logs['loss'] = logs['cl_loss']
        else:
            logs['loss'] = 0.0

        if loss is not None:
            mle_loss = loss.item()
            logs['mle_loss'] = mle_loss
            logs['loss'] += mle_loss

        if self.deepspeed:
            self.deepspeed: DeepSpeedEngine
            cur_scale = self.deepspeed.optimizer.loss_scaler.cur_scale
            if self.state.is_local_process_zero:
                if 'deepspeed' in self.records and len(self.records['deepspeed']) > 0:
                    last_scale = self.records['deepspeed'][-1]['loss_scale']
                    if last_scale > cur_scale:
                        self.logger.warning(f"The loss scale reduce from {last_scale} to {cur_scale}")

            logs['deepspeed'] = {
                'skipped_steps': self.deepspeed.skipped_steps,
                'loss_scale': cur_scale,
            }
            if cur_scale < 4.0:
                self.deepspeed.optimizer.loss_scaler.cur_scale *= 2
                if self.state.is_local_process_zero:
                    self.logger.warning(f"To avoid exception, manually reset loss scale from {cur_scale} to {self.deepspeed.optimizer.loss_scaler.cur_scale}")


        self.logwandb(logs, self.state.global_step)

        return total_loss.detach()


    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        ckpt_dir = None,
        **gen_kwargs
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if ckpt_dir is None:
            ckpt_dir = self.myargs.output_dir

        epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        step = int(self.state.global_step) if self.state.global_step is not None else 0

        if len(glob.glob(f"{self.myargs.output_dir}/checkpoint-{step}/*Acc*")) > 0:
            # already evaluated...
            return {}

        self.deepspeed: DeepSpeedEngine

        if self.myargs.do_eval:

            group_gloo = dist.new_group(backend="gloo")

            self.model.eval()
            gen_config = get_generation_config(self.myargs.load, self.infer_tokenizer, True)
            test_loader = DataLoader(eval_dataset, batch_size=self.myargs.eval_bsz, collate_fn=DynamicCollator(self.infer_tokenizer, prepare_target=False), sampler=DistributedSampler(eval_dataset, self.args.world_size, self.world_rank, shuffle=False))
            test_loader.sampler.set_epoch(0)


            results = infer_reasoning_dataset(
                task=self.myargs.task,
                test_loader=test_loader,
                rank=self.world_rank,
                local_rank=self.args.local_rank,
                world_size=self.args.world_size,
                desc=f"infer={self.myargs.task} set={self.myargs.valid_on} epoch={epoch} step={step}",
                model=self.model,
                gen_config=gen_config,
                is_decoder_only=tools_is_decoder_only(self.myargs.load),
                infer_tokenizer=self.infer_tokenizer,
                logger=self.logger,
                group_gloo=group_gloo
            )
            if self.world_rank <= 0:
                dumppath = os.path.join(ckpt_dir, f"ep{epoch}step{step}.evaluate.{self.myargs.valid_on}.Acc{results['acc']:.4f}.json")
                tools_json_dump(results['generated'], dumppath)

                self.logger.info(f"{'-'*160}\n"
                                 f"evaluate done! saved to\n{dumppath}\n"
                                 f"Acc={results['acc']}\tCorrect={results['correct']}\tTotal={results['total']}\n"
                                 f"{'-'*160}")

                self.wandbrun.log({'accuracy': results['acc'], 'epoch': epoch}, step=step)
                info = {
                    'accuracy': {
                        'acc': results['acc'],
                        'epoch': epoch,
                        'step': step,
                        'ckpt': ckpt_dir,
                    }}
                self.logwandb(info, step=step)

            dist.monitored_barrier(group_gloo, timeout=timedelta(minutes=15))

            return {'eval_acc': results['acc'] if self.world_rank <= 0 else None}
        else:
            return {}




class TrainerHook(TrainerCallback):

    def __init__(self, myargs: TrainerArgs, hfargs: Seq2SeqTrainingArguments, logger) -> None:
        super().__init__()
        self.args = hfargs
        self.myargs = myargs
        self.metric = Metric(
            savedir=f"{myargs.output_dir}/manual_ckpt",
            best_save_num=myargs.save_best,
            save_metric_name='eval_loss',
            save_metric_lower_is_better=True,
            last_ep_save_num=3,
            total_ep=hfargs.num_train_epochs,
            logger=logger,
        )

    def on_epoch_end(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero:
            self.metric.logger.info(f"\n\n{'-' * 60} epoch {int(state.epoch)} done {'-' * 60}\n")
            self.metric.logger.info(self.myargs.__dict__)
            self.metric.logger.info(f"\n\n{'-' * 60} epoch {int(state.epoch)} done {'-' * 60}\n")
        return super().on_epoch_end(args, state, control, **kwargs)
    def on_evaluate(self, args: Seq2SeqTrainingArguments, state: TrainerState, control: TrainerControl,
                    **kwargs):

        if self.args.save_strategy == 'no' and args.deepspeed is None:
            if state.is_world_process_zero:
                if int(state.epoch) >= 0:
                    if 'eval_loss' not in kwargs['metrics']:
                        eval_loss = state.epoch
                    else:
                        eval_loss = kwargs['metrics']['eval_loss']

                    self.metric.add_record('eval_loss', int(state.epoch), eval_loss)

                    should_save, save_path = self.metric.check_save_model(int(state.epoch),
                                                                          kwargs['model'],
                                                                          state.global_step)

                    # control.should_save = should_save
                    # args.output_dir = save_path
                    if should_save:
                        kwargs['model'].save_pretrained(save_path)


            else:
                pass

            group_gloo = dist.new_group(backend="gloo")
            dist.monitored_barrier(group_gloo)

        return super().on_evaluate(args, state, control, **kwargs)