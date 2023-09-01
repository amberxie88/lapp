import torch
from pathlib import Path
from data.sim_dataset import SimDataset
from data.image_dataset import ImageDataset
from torch.utils.data.sampler import WeightedRandomSampler

class LacoDataloader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, cfg, eval_dl):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.cfg = cfg
        if cfg.name == "train_laco":
            self.dataset = SimDataset(cfg, eval_dl)
        elif cfg.name == "train_mv":
            self.dataset = ImageDataset(cfg, eval_dl)

        if (eval_dl and cfg.eval_sampling == "shuffle") or (not eval_dl and cfg.train_sampling == "shuffle"):
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                num_workers=int(cfg.num_threads),
                pin_memory=True)
        elif (eval_dl and cfg.eval_sampling in ["weighted"]) or (not eval_dl and cfg.train_sampling in ["weighted"]):
            sample_weights = self.dataset.sample_weights
            sampler = WeightedRandomSampler(sample_weights.double(), len(sample_weights))
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=int(cfg.num_threads),
                pin_memory=True)
        else:
            raise NotImplementedError

        self.dataset_size = len(self.dataset)

        assert cfg.batch_size <= len(self)
        print(f"Dataset size: {len(self)}")

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.dataset_size 

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

def create_dataloaders(cfg):
    train_dl = LacoDataloader(cfg, eval_dl=False)
    eval_dl = LacoDataloader(cfg, eval_dl=True)
    return train_dl, eval_dl