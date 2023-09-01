import warnings
import os
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


import time
from pathlib import Path
import gc
import hydra
from omegaconf import OmegaConf, open_dict
import yaml
import numpy as np
import torch
import wandb

import utils
from data import html
from logger import Logger
from data.custom_dataloader import create_dataloaders

torch.backends.cudnn.benchmark = True

def make_laco_model(cfg):
    return hydra.utils.instantiate(cfg.laco_model)

def make_mv_model(cfg):
    reload_dir = Path(cfg.restore_mv_snapshot_path).parent / '..'
    with open(reload_dir / '.hydra/config.yaml', 'r') as f:
        cfg2 = OmegaConf.create(yaml.safe_load(f))
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.reload_dir = str(reload_dir)
        for k, v in cfg2.mv_model.items():
            if k != "lr" and k != "device":
                cfg.mv_model[k] = v
        cfg.laco_model['obs_encoder_id'] = cfg.mv_model['name']
    return hydra.utils.instantiate(cfg.mv_model)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.plot_dir = self.work_dir / 'plot'
        self.plot_dir.mkdir(exist_ok=True)
        self.web_dir = self.work_dir / 'web'
        self.web_dir.mkdir(exist_ok=True)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        torch.cuda.set_device(cfg.device)
        self.shard = cfg.shard
        
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # data 
        self.sim2real = cfg.sim2real
        self.train_dataloader, self.eval_dataloader = create_dataloaders(cfg)
        
        # model
        if cfg.use_mv:
            mv_model = make_mv_model(cfg)
            model_ckpt = self.load_snapshot(self.cfg.restore_mv_snapshot_path)['model']
            print("creating multi-view model")
            if cfg.laco_model.mv_train_status != "from_scratch":
                mv_model.init_from(model_ckpt)
                print(f"initialized multi-view model from {self.cfg.restore_mv_snapshot_path}")
            else:
                mv_model.mae_decoder = None 
            model_ckpt.clear()
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
        self.model = make_laco_model(cfg)
        try:
            model_ckpt = self.load_snapshot(self.cfg.restore_snapshot_path)['model']
            self.model.init_from(model_ckpt)
            print(f"initialized agent from {self.cfg.restore_snapshot_path}")
            model_ckpt.clear()
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
        except Exception as e:
            print("failed to initialize model checkpoint", e)
        if cfg.use_mv:
            self.model.add_mv_model(mv_model)

        # misc
        self.step, self.epoch = 0, -1
        self.timer = utils.Timer()

    def eval(self):
        all_metrics, counters = {}, {}
        start_time = time.time()

        for i, data in enumerate(self.eval_dataloader): 
            with torch.no_grad():
                pred = self.model.forward(data)
            data['collisions'] = data['collisions'].to(self.cfg.device)
            data['all_collisions'] = data['all_collisions'].to(self.cfg.device)
            metrics = self.model.get_eval_metrics(pred, data)
            self.process_metrics(metrics, all_metrics, counters)

            if i >= self.cfg.eval_iters:
                break

        all_metrics['eval_iter_time'] = time.time() - start_time
        with self.logger.log_and_dump_ctx(self.step, ty='eval') as log:
            for m, v in all_metrics.items():
                if m in counters:
                    log(m, v / counters[m])
                else:
                    log(m, v)

    def process_metrics(self, metrics, all_metrics, counters):
        for m, v in metrics.items():
            if "accuracy" in m and v == -1:
                pass
            elif m[:4] == "num_":
                pass
            elif m == "collision_accuracy":
                all_metrics[m] = all_metrics.get(m, 0) + torch.round(v * metrics['num_collisions'])
                counters[m] = counters.get(m, 0) + metrics['num_collisions']
            elif m == "nocollision_accuracy":
                all_metrics[m] = all_metrics.get(m, 0) + torch.round(v * metrics['num_nocollision'])
                counters[m] = counters.get(m, 0) + metrics['num_nocollision']
            elif m == "collision_change_accuracy":
                all_metrics[m] = all_metrics.get(m, 0) + torch.round(v * metrics['num_collision_change'])
                counters[m] = counters.get(m, 0) + metrics['num_collision_change']
            elif m == "overall_accuracy":
                all_metrics[m] = all_metrics.get(m, 0) + torch.round(v * metrics['num_batch'])
                counters[m] = counters.get(m, 0) + metrics['num_batch']
            elif "overall_accuracy_" in m:
                # get the # masked objects of collision_accuracy_id
                n_mask = m.split("_")[-1]
                all_metrics[m] = all_metrics.get(m, 0) + torch.round(v * metrics[f'num_batch_{n_mask}'])
                counters[m] = counters.get(m, 0) + metrics[f'num_batch_{n_mask}']
            elif "accuracy_" in m:
                # get the # masked objects of collision_accuracy_id
                msplit = m.split("_")
                specificiation, n_mask = '_'.join(msplit[:-2]), msplit[-1]
                all_metrics[m] = all_metrics.get(m, 0) + torch.round(v * metrics[f'num_{specificiation}_{n_mask}'])
                counters[m] = counters.get(m, 0) + metrics[f'num_{specificiation}_{n_mask}']
            else:
                all_metrics[m] = all_metrics.get(m, 0) + v
                counters[m] = counters.get(m, 0) + 1

    def train(self):
        # predicates
        eval_every_step = utils.Every(self.cfg.eval_every_step)
        print_every_step = utils.Every(self.cfg.print_every_step)
        save_every_step = utils.Every(self.cfg.save_every_step)

        for epoch in range(self.cfg.n_epochs):
            self.epoch = epoch
            for i, data in enumerate(self.train_dataloader): 
                metrics = self.model.update(data, self.step) 
                self.logger.log_metrics(metrics, self.step, ty='train')

                if print_every_step(self.step):
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.step, ty='train') as log:
                        log('epoch_time', elapsed_time)
                        log('total_time', total_time)
                        log('step', self.step)
                        log('epoch', self.epoch)
                if eval_every_step(self.step):
                    self.eval()
                self.step += 1
                if save_every_step(self.step):
                    self.save_snapshot()

        
    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.step}.pt'
        payload = dict(model=self.model.get_model()) 
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, path):
        snapshot = Path(path)

        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=lambda storage, loc: storage.cuda(self.cfg.device))
        return payload


@hydra.main(config_path='.', config_name='train_laco')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(self.cfg.restore_snapshot_path)

    # create logger
    if cfg.use_wandb:
        import omegaconf
        wandb.init(entity=ENTITY,project=PROJECT,group=cfg.experiment_folder,name=cfg.experiment_name,tags=[cfg.experiment_folder], sync_tensorboard=True)
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

    workspace.train()


if __name__ == '__main__':
    main()
