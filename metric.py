import os
import re
import shutil


class Metric():
    def __init__(self, savedir, best_save_num, save_metric_name, save_metric_lower_is_better, last_ep_save_num, total_ep, logger=None):
        os.makedirs(savedir, exist_ok=True)
        self.records = {}
        self.savedir = savedir
        self.best_save_num = best_save_num
        self.last_ep_save_num = last_ep_save_num
        self.save_metric_name = save_metric_name
        self.total_ep = total_ep
        self.save_metric_lower_is_better = save_metric_lower_is_better
        self.best_path = {}
        self.logger = logger


    def add_record(self, metric_name, epoch, value):
        # add record first
        assert epoch > 0
        if epoch not in self.records:
            self.records[epoch] = {}
        self.records[epoch][metric_name] = value

    def save_log(self):
        with open(f"{self.savedir}/log.json", 'w') as file:
            for ep, value in self.records.items():
                temp = f"epoch {ep:5d}"
                for k, v in value.items():
                    temp += f"  {k} {v:.5f}"
                file.write(temp)
                file.write('\n')

    def __repr__(self):
        return str(self.records)

    def find_best(self, metric_name='loss', lower_is_better=True, top=1):
        temp = sorted(self.records.items(), key=lambda item: item[1][metric_name], reverse=not lower_is_better)
        return dict(temp[:top])


    def check_save_model(self, ep, model, step='NA'):
        # try to save model second
        assert ep > 0
        assert ep in self.records
        assert self.save_metric_name in self.records[ep]
        assert hasattr(model, 'save_pretrained')

        # select the top n eps
        last_best_ep = list(self.find_best(self.save_metric_name, lower_is_better=self.save_metric_lower_is_better, top=self.best_save_num).keys())

        should_save = False
        save_path = None

        if ep in last_best_ep:
            # should save this epoch
            if len(self.best_path) == self.best_save_num:
                # should rm the worst one first to have a place for the new one
                shutil.rmtree(self.best_path[self.best_save_num - 1])
                del self.best_path[self.best_save_num - 1]
            
            # now we get one left place to save this new one
            # however, you have to reorder it since the new one may disrupt the order
            mapped = {}
            for old_idx, old_path in self.best_path.items():
                ep_t = int(re.findall("epoch\d+", old_path)[0].strip('epoch'))
                new_idx = last_best_ep.index(ep_t)
                new_path = old_path.replace(f'best{old_idx}', f'best{new_idx}')
                mapped[old_idx] = (new_idx, old_path, new_path)

            for old_idx, (new_idx, old_path, new_path) in mapped.items():
                os.rename(old_path, new_path)
                self.best_path[new_idx] = new_path

            # finally, you can save it now
            idx = last_best_ep.index(ep)
            save_path = f"{self.savedir}/best{idx}_epoch{ep}_step{step}.eval_loss{self.records[ep]['eval_loss']:.5f}"
            self.best_path[idx] = save_path
            should_save = True
            # model.save_pretrained(save_path)

            if self.logger:
                self.logger.info(f"saved best to {save_path}")
        else:
            # try to check if it is in 'last save' set
            if self.total_ep - self.last_ep_save_num < ep:
                idx = self.total_ep - ep
                save_path = f"{self.savedir}/last{idx}_epoch{ep}_step{step}.eval_loss{self.records[ep]['eval_loss']:.5f}"
                should_save = True
                # model.save_pretrained(save_path)

                if self.logger:
                    self.logger.info(f"saved last to {save_path}")
                    
        return should_save, save_path
