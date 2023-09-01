import os
from pathlib import Path
import numpy as np
import torch
import transformers

def make_image_dataset(simdata_folder, sample_weight, max_dataset_size):
    assert os.path.isdir(simdata_folder), '%s is not a valid directory' % simdata_folder

    idx_to_file, paths, sample_weights = [], [], []
    n_paths = 0
    for root, _, fnames in sorted(os.walk(simdata_folder)):
        for idx, fname in enumerate(fnames):
            path = Path(os.path.join(root, fname))
            paths.append(path)
            # 5 images per file
            idx_to_file.extend([(idx, i) for i in range(5)])
            n_paths += 5
            if n_paths >= max_dataset_size:
                return paths, idx_to_file, sample_weights

    return paths, idx_to_file, sample_weights

class ImageDataset:
    
    def __init__(self, cfg, eval_dl):
        """Initialize this dataset class.
        Parameters:
            cfg: hydra config
            eval_dl: is this the evaluation dataloader?
        """
        if eval_dl:
            self.main_folder = Path(cfg.eval_folder)
        else:
            self.main_folder = Path(cfg.folder)
        self.eval_dl = eval_dl
        self.simdata_folder = self.main_folder / 'simdata'
        self.max_dataset_size = cfg.max_dataset_size

        # get image paths
        self.paths, self.idx_to_file, self.sample_weights = make_image_dataset(self.simdata_folder, cfg.sample_weight, cfg.max_dataset_size)
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)
        self.MAIN_OBS_KEY = cfg.obs_key[0]
        self.obs_keys = cfg.obs_key

        self.preprocess = False
        if cfg.name == "train_laco":
            model_name = "laco_model"
        elif cfg.name == "train_mv":
            model_name = "mv_model" 
        else:
            raise NotImplementedError
        if "obs_encoder_id" in cfg[model_name].keys():
            self.preprocess = True
            if "clip16" in cfg[model_name].obs_encoder_id:
                self.image_processor = transformers.ViTImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            elif "clip14" in cfg[model_name].obs_encoder_id:
                self.image_processor = transformers.ViTImageProcessor.from_pretrained("openai/clip-vit-large-patch14") 
            elif "multimae_patch" in cfg[model_name].obs_encoder_id:
                self.image_processor = transformers.ViTImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        path_idx, img_idx = self.idx_to_file[index]
        path = Path(self.paths[index//5])
        with path.open('rb') as f:
            data = np.load(f)
            data = {k: data[k] for k in data.keys()}
            new_data = {}

            new_data['actual_obs'] = (((data[self.MAIN_OBS_KEY][img_idx].transpose((2, 0, 1))))*255).astype(np.uint8)
            extra_obs = np.array([(255*data[obs_key][img_idx]).astype(np.uint8) for obs_key in self.obs_keys[1:]])

            if self.preprocess:
                new_data['vit_obs'] = self.image_processor.preprocess(new_data['actual_obs'])['pixel_values'][0]
                new_data['extra_vit_obs'] = np.array([self.image_processor.preprocess(extra_obs[k])['pixel_values'][0] for k in range(extra_obs.shape[0])])
            return new_data
        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.idx_to_file)
