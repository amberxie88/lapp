import os
from pathlib import Path
import numpy as np
import torch
import random
import pickle
import transformers

def load_data(path, preprocess, image_processor, obs_keys, max_states_per_env, pickle):
    with path.open('rb') as f:
        data_tmp = np.load(f, allow_pickle=pickle) 
        data = dict()
        for key in data_tmp.keys():
            if key in ['obs', 'zoom', 'zoom_left', 'zoom_right']:
                # don't load unnecessary keys
                if not (key in obs_keys or key == obs_keys):
                    continue 
            data[key] = data_tmp[key]
        del data_tmp
        if preprocess:
            preprocess_obs(data, image_processor, obs_keys)
        n_states = min(data['states'].shape[0], max_states_per_env)
        data['states'] = data['states'][:n_states]
        data['collisions'] = data['collisions'][:n_states]
        data['relevant_collisions'] = data['relevant_collisions'][:n_states]
    return data

def make_lang_dataset(simdata_folder, max_dataset_size, preprocess, image_processor, obs_keys, max_states_per_env, weight, max_files_loaded, pickle=False):
    assert os.path.isdir(simdata_folder), '%s is not a valid directory' % simdata_folder

    idx_to_file, dataset, sample_weights = [], dict(), []
    n_paths = 0
    for root, _, fnames in sorted(os.walk(simdata_folder)):
        for idx, fname in enumerate(fnames):
            path = Path(os.path.join(root, fname))
            # get number of language conditions from path
            if (idx < max_files_loaded):
                data = load_data(path, preprocess, image_processor, obs_keys, max_states_per_env, pickle)
                collisions = (data['collisions'].sum(axis=1) > 0).astype(np.uint8) * (weight-1) + 1
                sample_weights.extend(collisions)
                dataset[path] = data
                n_states = data['states'].shape[0]
            else:
                with path.open('rb') as f:
                    data_tmp = np.load(f, allow_pickle=pickle) 
                    n_states = min(data_tmp['states'].shape[0], max_states_per_env)
                    del data_tmp
            idx_to_file.extend([(path, i) for i in range(n_states)])
            n_paths += n_states
            if n_paths >= max_dataset_size:
                return dataset, idx_to_file, sample_weights

    return dataset, idx_to_file, sample_weights

def preprocess_obs(data, image_processor, obs_keys):
    data['vit_obs'] = []
    for obs_key in obs_keys:
        for img in data[obs_key]:
            img = img.transpose((2, 0, 1)).astype(np.float32)
            vit_obs = image_processor.preprocess(img)['pixel_values'][0]
            data['vit_obs'].append(vit_obs)

class SimDataset:
    def __init__(self, cfg, eval_dl, folder=None):
        """Initialize this dataset class.
        Parameters:
            cfg: hydra config
            eval_dl: is this the evaluation dataloader?
        """
        if folder is not None:
            self.main_folder = folder 
        else:
            if eval_dl:
                self.main_folder = Path(cfg.eval_folder)
            else:
                self.main_folder = Path(cfg.folder)
        self.eval_dl = eval_dl
        self.max_dataset_size = cfg.max_dataset_size # refers to # envs
        self.max_states_per_env = cfg.max_states_per_env

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

        # get image paths
        self.dataset, self.idx_to_file, self.sample_weights = make_lang_dataset(self.main_folder / 'simdata', cfg.max_dataset_size, 
                                                                    self.preprocess, self.image_processor, cfg.obs_key, 
                                                                    self.max_states_per_env, cfg.sample_weight, cfg.max_files_loaded)
        self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)
        self.obs_keys = cfg.obs_key


    def __getitem__(self, index):
        path, state_idx = self.idx_to_file[index]
        if path in self.dataset:
            data = self.dataset[path]
        else:
            random_path = random.choice(list(self.dataset.keys()))
            to_delete = self.dataset.pop(random_path)
            del to_delete
            data = load_data(path, self.preprocess, self.image_processor, self.obs_keys, self.max_states_per_env, pickle=False)
            self.dataset[path] = data

        lang_idx = np.random.randint(0, data['language'].shape[0])
        obs_idx = np.random.randint(0, data[self.obs_keys[0]].shape[0])

        all_collision = np.float32(int(sum(data['collisions'][state_idx]) > 0))
        relevant_collision = data['relevant_collisions'][state_idx][lang_idx]
        lang = data['language'][lang_idx]

        obs = data[self.obs_keys[0]][obs_idx].transpose((2, 0, 1)).astype(np.float32)
        vit_obs = data['vit_obs'][obs_idx]
        state = data['states'][state_idx]
       
        data_item = dict(vit_obs=vit_obs, init_obs=obs, actual_obs=obs, states=state,
                    language=lang, collisions=relevant_collision, all_collisions=all_collision,)

        if len(self.obs_keys) > 1:
            extra_init_obs = []
            for key in self.obs_keys[1:]:
                extra_init_obs.append(data[key][obs_idx].transpose((2, 0, 1)).astype(np.float32))
            extra_init_obs = np.array(extra_init_obs)
            data_item['extra_vit_obs'] = np.array([self.image_processor.preprocess(extra_init_obs[k])['pixel_values'][0] for k in range(extra_init_obs.shape[0])])
            data_item['extra_init_obs'] = extra_init_obs

        return data_item

    def get_grid_data(self, n_plot_data):
        dataset, _, _ = make_lang_dataset(self.main_folder / 'griddata', np.infty, 
                                            self.preprocess, self.image_processor,
                                            self.obs_keys, np.infty, 0, pickle=True)
        return dataset[:n_plot_data]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.idx_to_file) # random number
