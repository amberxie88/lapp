import warnings
import os

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


import time
from pathlib import Path
import gc
import hydra
import numpy as np
import torch
import seaborn as sns
from PIL import Image
import wandb

import utils
from logger import Logger
from data import html
from data.custom_dataloader import create_dataloaders

torch.backends.cudnn.benchmark = True

def make_mv_model(cfg):
    return hydra.utils.instantiate(cfg.mv_model)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.plot_dir = self.work_dir / 'plot'
        self.plot_dir.mkdir(exist_ok=True)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        torch.cuda.set_device(cfg.device)
        self.shard = cfg.shard
        
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        # data 
        self.train_dataloader, self.eval_dataloader = create_dataloaders(cfg)
       
        # model
        self.model = make_mv_model(cfg) 
        try:
            model_ckpt = self.load_snapshot()['model']
            self.model.init_from(model_ckpt)
            print(f"initialized model from {self.cfg.restore_snapshot_path}")
            model_ckpt.clear()
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()
        except:
            print("failed to initialize model checkpoint")

        # misc
        self.step, self.epoch = 0, -1
        self.timer = utils.Timer()

    def eval(self):
        all_metrics, counters = {}, {}
        n_iters = 0
        start_time = time.time()

        for i, data in enumerate(self.eval_dataloader): 
            with torch.no_grad():
                model_out = self.model.forward_verbose(data)
            metrics = self.model.get_eval_metrics(model_out, data)
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

    def train(self):
        # predicates
        eval_every_step = utils.Every(self.cfg.eval_every_step)
        plot_every_step = utils.Every(self.cfg.plot_every_step)
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
                if plot_every_step(self.step):
                    self.plot()
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

    def load_snapshot(self):
        snapshot = Path(self.cfg.restore_snapshot_path)

        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=lambda storage, loc: storage.cuda(self.cfg.device))
        return payload

    def process_metrics(self, metrics, all_metrics, counters):
        for m, v in metrics.items():
            if "accuracy" in m and v == -1:
                # don't count this!
                pass
            else:
                all_metrics[m] = all_metrics.get(m, 0) + v
                counters[m] = counters.get(m, 0) + 1

    def plot(self):
        for i, data in enumerate(self.eval_dataloader): 
            with torch.no_grad():
                featurized_seq, meta = self.model.forward_verbose(data)
            break
        if 'pixel_pred' not in meta.keys():
            return
            
        plot_folder = self.plot_dir / f'epoch{self.epoch}_step{self.step}'
        plot_folder.mkdir(exist_ok=True)
        webpage = html.HTML(plot_folder, f'Epoch {self.epoch} Step {self.step}')
        webpage.add_header(f'Epoch {self.epoch} Step {self.step}')

        mean = self.train_dataloader.dataset.image_processor.image_mean
        std = self.train_dataloader.dataset.image_processor.image_std
        mean = torch.tensor(mean).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        std = torch.tensor(std).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        pred_view1 = self.model.unpatchify(meta['pixel_pred'][:, :196, :]).permute(0,2,3,1).cpu()
        view1 = data['vit_obs'].permute(0,2,3,1).cpu()
        view1_mask = self.model.unpatchify(self.model.patchify(data['vit_obs']) * (1-meta['masks'][:,:196,:])).permute(0,2,3,1).cpu()
        pred_view1 = torch.clamp(pred_view1 * std + mean, 0, 1)
        view1 = view1 * std + mean
        view1_mask = view1_mask * std + mean
        if 'pixel_pred_w_mask' in meta.keys():
            pred_view1_mask = self.model.unpatchify(meta['pixel_pred_w_mask'][:, :196, :]).permute(0,2,3,1).cpu()
            pred_view1_mask = torch.clamp(pred_view1_mask * std + mean, 0, 1)
        view2 = None
        if 'extra_vit_obs' in data.keys() and data['extra_vit_obs'].shape[-1] > 0:
            view2 = data['extra_vit_obs'][:, 0, :, :, :].permute(0,2,3,1).cpu()
            view2 = view2 * std + mean
            view2_mask = self.model.unpatchify(self.model.patchify(data['extra_vit_obs'][:, 0, :, :, :]) * (1-meta['masks'][:,196:,:])).permute(0,2,3,1).cpu()
            view2_mask = view2_mask * std + mean


        for i in range(min(10, len(view1))):
            pred_view1_img = pred_view1[i].numpy()
            pred_view1_pil = Image.fromarray((pred_view1_img*255).astype(np.uint8))
            pred_view1_path = plot_folder / 'images' / f'pred_view1_{i}.png'
            pred_view1_pil.save(pred_view1_path)

            view1_img = view1[i].numpy()
            view1_pil = Image.fromarray((view1_img*255).astype(np.uint8))
            view1_path = plot_folder / 'images' / f'view1_{i}.png'
            view1_pil.save(view1_path)

            view1_mask_img = view1_mask[i].numpy()
            view1_mask_pil = Image.fromarray((view1_mask_img*255).astype(np.uint8))
            view1_mask_path = plot_folder / 'images' / f'view1_mask_{i}.png'
            view1_mask_pil.save(view1_mask_path)

            if 'pixel_pred_w_mask' in meta.keys():
                pred_view1_mask_img = pred_view1_mask[i].numpy()
                pred_view1_mask_pil = Image.fromarray((pred_view1_mask_img*255).astype(np.uint8))
                pred_view1_mask_path = plot_folder / 'images' / f'pred_view1_mask_{i}.png'
                pred_view1_mask_pil.save(pred_view1_mask_path)

            if view2 is not None:
                view2_img = view2[i].numpy()
                view2_pil = Image.fromarray((view2_img*255).astype(np.uint8))
                view2_path = plot_folder / 'images' / f'view2_{i}.png'
                view2_pil.save(view2_path)
                view2_mask_img = view2_mask[i].numpy()
                view2_mask_pil = Image.fromarray((view2_mask_img*255).astype(np.uint8))
                view2_mask_path = plot_folder / 'images' / f'view2_mask_{i}.png'
                view2_mask_pil.save(view2_mask_path)

            traj_files, traj_langs = [], []
            traj_files.append(pred_view1_path.name)
            traj_langs.append("predicted image")
            traj_files.append(view1_path.name)
            traj_langs.append("view1 image")
            traj_files.append(view1_mask_path.name)
            traj_langs.append("view1 image (unmasked patches)")
            if 'pixel_pred_w_mask' in meta.keys():
                traj_files.append(pred_view1_mask_path.name)
                traj_langs.append("predicted image (masked predictions)")
            if view2 is not None:
                traj_files.append(view2_path.name)
                traj_langs.append("view2 image")
                traj_files.append(view2_mask_path.name)
                traj_langs.append("view2 image (unmasked patches)")

            webpage.add_images(traj_files, traj_langs, traj_files, width=256) 
        webpage.save()


@hydra.main(config_path='.', config_name='train_mv')
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

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
