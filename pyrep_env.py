import os
import numpy as np
import math
import random

N_OBJ_TO_N_LANG = {
    2: (1, 2, 1),
    3: (1, 3, 2),
    4: (1, 4, 3),
    5: (1, 5, 10),
    6: (1, 6, 15)
}
LANG_ID_TO_COMBOS = {
    2: {1: [(0,), (1,)],
        2: [(0, 1)],
    },
    3: {1: [(0,), (1,), (2,)],
        2: [(0, 1), (0, 2), (1, 2)],
    },
    4: {1: [(0,), (1,), (2,), (3,)],
        2: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    },
    5: {1: [(0,), (1,), (2,), (3,), (4,)],
        2: [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    }, 
    6: {1: [(0,), (1,), (2,), (3,), (4,), (5,)],
        2: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), 
            (1, 2), (1, 3), (1, 4), (1, 5), 
            (2, 3), (2, 4), (2, 5), 
            (3, 4), (3, 5), 
            (4, 5)],
    },
}

class CollisionPyrepWrapper:
    def __init__(self, env, cfg):
        self.env = env 
        self.env_name = cfg.env
        self.reset_every = cfg.reset_every
        self.traj_count = 0
        self.radius = cfg.radius
        self.noise = cfg.noise
        self.num_obs_per_env = cfg.num_obs_per_env

    def close(self):
        self.env.shutdown()

    def gen_data_step(self):
        if self.env_name == "sim_shapenet":
            return self.gen_data_simdata()
        elif self.env_name == "real_ycb":
            return self.gen_data_realdata()
        else:
            raise NotImplementedError

    def get_lang_from_combo(self, names, combo):
        obj_names = [names[idx] for idx in combo]
        random.shuffle(obj_names)
        if len(obj_names) == 1:
            return obj_names[0]
        elif len(obj_names) == 2:
            return f"{obj_names[0]} and {obj_names[1]}"
        else:
            raise NotImplementedError

    def set_lang_combos(self, collision_data):
        n_objs = len(collision_data['names'])
        n_langs = N_OBJ_TO_N_LANG[n_objs]
        max_nlang = max(n_langs)

        langs = [''] * max_nlang
        all_combos = [()] * max_nlang
        indices_no_mask_lst = [[i for i in range(n_objs)]] * max_nlang
        collision_ids = [[0 for _ in range(n_objs)]] * max_nlang

        for lang_id, n_lang in enumerate(n_langs):
            if lang_id == 0: 
                # taken care of by empty string
                continue 
            combos = LANG_ID_TO_COMBOS[n_objs][lang_id]
            # pick n_lang combos by shuffling and slicing the list
            random.shuffle(combos)
            for idx in range(max_nlang):
                combo = combos[idx % len(combos)]
                all_combos.append(combo)
                collision_ids.append([1 if i in combo else 0 for i in range(n_objs)])
                indices_no_mask_lst.append([i for i in range(n_objs) if i not in combo])
                langs.append(self.get_lang_from_combo(collision_data['names'], combo))

        relevant_collisions_lst = []
        for indices_no_mask in indices_no_mask_lst:
            relevant_collisions = collision_data['collisions'][:, indices_no_mask]
            relevant_collisions = (relevant_collisions.sum(axis=-1) > 0).astype(np.float32)
            relevant_collisions_lst.append(relevant_collisions)
        collision_data['relevant_collisions'] = np.array(relevant_collisions_lst).transpose()
        collision_data['language'] = np.array(langs)
        collision_data['collision_ids'] = np.array(collision_ids)
        # expected shape: 
        # lang: (n_lang)
        # collision_ids: (n_lang, n_objs) (binary matrix)
        # relevant_collisions: (n_state, n_lang)
        return  

    def gen_data_simdata(self, use_reset=True):
        self.traj_count += 1 # even if traj isn't successful
        if use_reset and self.traj_count % self.reset_every == 0:
            self.env.reset()

        collision_data = self.env.get_traj_collision_data()
        if collision_data is None:
            self.traj_count -= 1 
            return self.gen_data_simdata(use_reset=False)
        collision_data['names'] = self.env.get_meta()['names'] 
        rand_idxs = np.random.randint(0, collision_data['collisions'].shape[0], size=5)
        collision_data.update(self.env.get_obs(collision_data['states'][rand_idxs]))
        # process language
        self.set_lang_combos(collision_data)
        return collision_data

    def gen_data_realdata(self, use_reset=True):
        self.traj_count += 1 # even if traj isn't successful
        if use_reset and self.traj_count % self.reset_every == 0:
            self.env.reset()

        collision_data = self.env.get_collision_data()
        collision_data['names'] = self.env.get_meta()['names'] 
        collision_data.update(self.env.get_obs(collision_data['states'][:self.num_obs_per_env]))
        # process language
        self.set_lang_combos(collision_data)
        return collision_data

def load_environment(cfg):
    name = cfg.env
    if name == "sim_shapenet":
        from envs.sim_env import SimEnv
        env = SimEnv(cfg) 
    elif name == "real_ycb":
        from envs.real_env import RealEnv
        env = RealEnv(cfg) 
    else:
        raise NotImplementedError
    return env

def make(cfg):
    env = load_environment(cfg)
    env = CollisionPyrepWrapper(env, cfg)
    return env