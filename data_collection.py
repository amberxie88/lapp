import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch

import pyrep_env
import utils
from data.collision_datasaver import CollisionDataSaver
import wandb

torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        torch.cuda.set_device(cfg.device)
        
        # create envs
        self.env = pyrep_env.make(cfg)

        # data 
        self.data_saver = CollisionDataSaver(cfg.save_folder, cfg.save_prefix, cfg.save_viz)

    def run(self):
        for _ in range(self.cfg.n_data):
            data = self.env.gen_data_step()
            self.data_saver.save(data)

        self.env.close()
        
    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = Path(self.cfg.restore_snapshot_dir)

        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload


@hydra.main(config_path='.', config_name='data_collection')
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
        wandb.init(entity="ENTITY",project="PROJECT",group=cfg.experiment_folder,name=cfg.experiment_name,tags=[cfg.experiment_folder], sync_tensorboard=True)
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )


    workspace.run()


if __name__ == '__main__':
    main()
