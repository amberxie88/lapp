import random
import re
import time
import math
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from typing import Any, NamedTuple
from PIL import Image
from pathlib import Path

from pyrep.objects.shape import Shape
from pyrep.backend import sim
import transforms3d

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    # if the item is a list, then update the data of the target net with the data of the net
    if isinstance(net, list):
        for n, param in enumerate(net):
            target_net[n].data.copy_(param.data)
    else:
        for (n, param), (tn, target_param) in zip(net.named_parameters(), target_net.named_parameters()):
            target_param.data.copy_(param.data)

def hard_update_params_data(param, target_param):
    target_param.data.copy_(param.data)

def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None or self._every == -1:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def str_mj_arr(arr):
    return ' '.join(['%0.3f' % arr[i] for i in range(len(arr))])


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

class ShapeNetObj:
    def __init__(self, path, name, parent_name):
        self.path = path
        self.name = name
        self.parent_name = parent_name

    def __repr__(self):
        return f"{self.name}: {self.path}"

class ShapeNetClass:
    def __init__(self, base_names, synset_id, num_instances, obj_split):
        self.base_names = base_names
        self.synset_id = synset_id
        self.num_instances = num_instances
        self.paths, self.names = get_shapenet_data(synset_id)
        self.obj_split = obj_split

    def __getitem__(self, idx):
        if idx >= self.num_instances:
            raise IndexError()
        if self.obj_split == "train" and idx >= self.num_instances - 5:
            raise Indexerror()
        elif self.obj_split == "eval" and idx < self.num_instances - 5:
            raise IndexError()
        path = f"/PATH/TO/shapenetcore_v2/{self.synset_id}/{self.paths[idx]}/models/model_normalized.obj"
        name = self.names[idx][np.random.randint(len(self.names[idx]))]
        return ShapeNetObj(path, name, self.base_names[0])

    def get_random_shape(self):
        if self.obj_split == "train":
            idx = np.random.randint(self.num_instances - 5)
        elif self.obj_split == "eval":
            idx = np.random.randint(self.num_instances - 5, self.num_instances)
        return self[idx]

def get_shapenet_data(synset_id):
    csv_base_dir = "/PATH/TO/shapenet/metadata"
    csv_file = f"{csv_base_dir}/{synset_id}.csv"
    paths, names = [], []
    with open(csv_file, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # first row of csv
            if row[0] == 'fullId':
                continue
            paths.append(row[0].split(".")[-1])
            names.append(row[2].split(','))
    return paths, names

def read_shapenet(shapenet_to_generate, obj_split):
    json_file = "/PATH/TO/shapenetcore_v2/taxonomy.json"

    with open(json_file) as json_data:
        data = json.load(json_data)

        all_objs = []
        all_children = []
        # keep track of children to process them last
        for obj in data:
            for child in obj['children']:
                all_children.append(child)

        # this object doesn't exist in the shapenet folder, not sure why
        all_children.append("02834778") # bicycle

        # process objects recursively
        for obj in data:
            if obj['synsetId'] not in all_children:
                name_lst = obj['name'].split(',')
                if len(shapenet_to_generate) > 0 and name_lst[0] not in shapenet_to_generate:
                    continue
                parent_obj = ShapeNetClass(name_lst, obj['synsetId'], obj['numInstances'], obj_split)
                all_objs.append(parent_obj)

                # assert path exists
                # path, _ = parent_obj[np.random.randint(parent_obj.num_instances)]
                # assert os.path.exists(path), path
        return all_objs 

def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

def prefix_sum(lst):
    ps = [0] * len(lst)
    for i in range(len(lst)):
        for j in range(i+1):
            ps[i] += lst[j]
    return ps 

YCB_PATH = Path("/PATH/TO/shapenet/final_ycb")
ITEM_TO_PATH = {
    'bleach': YCB_PATH / 'bleach' / 'textured.obj',
    'cheezit': YCB_PATH / 'cheezit' / 'textured.obj',
    'mustard': YCB_PATH / 'mustard' / 'textured.obj',
    'pringles': YCB_PATH / 'pringles' / 'tsdf' / 'textured.obj',
    'spam': YCB_PATH / 'spam' / 'textured.obj',
    'windex': YCB_PATH / 'windex' / 'textured.obj',
}

# add 150 to z axis because the coordinate system w/ end effector
OBJ_DIMENSIONS = {
    'bleach': (60, 100, 240+150),
    'cheezit': (62, 160, 215+150),
    'mustard': (80, 120, 200+150), 
    'pringles': (80, 80, 235+150),
    'spam': (60, 110, 85+150),
    'windex': (90, 110, 275+150),
}

def twin_pos_real_to_sim(x_real, y_real, z_real):
    x_sim = 0.001 * x_real + 0.49665668
    y_sim = 0.001 * y_real + 0.02898553
    z_sim = 0.001 * z_real + 0.58112336
    return np.array([x_sim, y_sim, z_sim])

def twin_pos_sim_to_real(x_sim, y_sim, z_sim):
    x_real = 999.99959574 * x_sim -496.65648302
    y_real = 999.99984797 * y_sim -28.98552269
    z_real = 999.99908161 * z_sim -581.12282562
    return np.array([x_real, y_real, z_real])

class RealWorldMeshes:
    """
    Class for loading in the ycb meshes that are used for real world data.
    This class is honestly just a static class, not that it matters.
    """
    def __init__(self):
        self.items = list(ITEM_TO_PATH.keys())

    def get(self, idx):
        if isinstance(idx, int):
            obj_name = self.items[idx]
        elif isinstance(idx, str):
            obj_name = idx
        obj = ShapeNetObj(str(ITEM_TO_PATH[obj_name]), obj_name, None)

        shape = Shape.import_shape(obj.path, scaling_factor=1)
        if obj_name == "bleach":
            self.process_bleach(shape)
        elif obj_name == 'cheezit':
            self.process_cheezit(shape)
        elif obj_name == 'mustard':
            self.process_mustard(shape)
        elif obj_name == 'pringles':
            self.process_pringles(shape)
        elif obj_name == 'spam':
            self.process_spam(shape)
        elif obj_name == 'windex':
            self.process_windex(shape)
        else:
            raise NotImplementedError

        # small random rotation
        rand_rot = np.random.rand() * math.pi / 3 - math.pi / 6
        euler = self.euler_world_to_shape(shape, [0, 0, rand_rot])
        shape.rotate(euler)

        return obj, shape

    def euler_world_to_shape(self, shape, euler):
        m = sim.simGetObjectMatrix(shape._handle, -1)
        x_axis = np.array([m[0], m[4], m[8]])
        y_axis = np.array([m[1], m[5], m[9]])
        z_axis = np.array([m[2], m[6], m[10]])
        R = transforms3d.euler.euler2mat(*euler, axes='rxyz')
        T = np.array([x_axis, y_axis, z_axis]).T
        new_R = np.linalg.inv(T)@R@T
        new_euler = transforms3d.euler.mat2euler(new_R, axes='rxyz')
        return new_euler

    def process_bleach(self, shape):
        shape.rotate([0, 0, -math.pi/2])
        x, y, z = shape.get_position()
        _, _, bnz, _, _, _ = shape.get_bounding_box()
        shape.set_position([x, y, 0.87])
        # cm to m conversion
        shape.real_dimensions = np.array([6, 10, 24]) * 10 
        pass 

    def process_cheezit(self, shape):
        x, y, z = shape.get_position()
        _, _, bnz, _, _, _ = shape.get_bounding_box()
        shape.set_position([x, y, 0.88])
        shape.real_dimensions = np.array([6.2, 16, 21.5]) * 10
        pass

    def process_mustard(self, shape):
        shape.rotate([0, 0, math.pi/2])
        x, y, z = shape.get_position()
        _, _, bnz, _, _, _ = shape.get_bounding_box()
        shape.set_position([x, y, 0.85])
        shape.real_dimensions = np.array([8, 12, 20]) * 10
        pass

    def process_pringles(self, shape):
        x, y, z = shape.get_position()
        shape.set_position([x, y, 0.88])
        shape.set_color([1.0, 1.0, 1.0])
        shape.real_dimensions = np.array([8, 8, 23.5]) * 10
        pass

    def process_spam(self, shape):
        shape.rotate([0, math.pi/2, 0])
        x, y, z = shape.get_position()
        _, _, bnz, _, _, _ = shape.get_bounding_box()
        shape.set_position([x, y, 0.8-0.01])  
        shape.real_dimensions = np.array([6, 10, 8.5]) * 10 
        pass

    def process_windex(self, shape):
        x, y, z = shape.get_position()
        _, _, bnz, _, _, _ = shape.get_bounding_box()
        shape.set_position([x, y, 0.9])
        shape.real_dimensions = np.array([9, 11, 27.5]) * 10
        pass
