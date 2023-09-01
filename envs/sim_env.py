"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.xarm7 import XArm7
from pyrep.objects.object import Object
from pyrep.objects.vision_sensor import VisionSensor

from pyrep.objects.shape import Shape
from pyrep.const RenderMode
from pyrep.errors import ConfigurationPathError
from pyrep.backend import sim

import numpy as np
import math
import random
import transforms3d
import matplotlib.pyplot as plt
from utils import read_shapenet, twin_pos_real_to_sim, twin_pos_sim_to_real
from pathlib import Path

SCENE_FILE = join(dirname(abspath(__file__)),
                  'assets/real_xarm.ttt')

class SimEnv(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=True)
        self.pr.start()
        arm = XArm7()
        self.agent = arm
        for arm_obj in arm.get_objects_in_tree():
            if 'visual' in arm_obj.get_name():
                arm_obj.set_renderable(True)
            else:
                arm_obj.set_renderable(False)
     
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = [0.000625, -0.591169, -0.001118, 0.176431, -0.003432, 0.767861, 0.002054]

        self.camera = VisionSensor('my_vision_sensor')
        if cfg.camera[0] == "zoom":
            self.camera.set_position([1.5, 0, 1.5])
            self.camera.set_orientation([-math.pi, -50 / 180 * math.pi, math.pi/2])
            self.camera.set_render_mode(RenderMode.OPENGL)
        elif cfg.camera[0] == "standard":
            self.camera.set_render_mode(RenderMode.OPENGL)
        else:
            raise NotImplementedError

        self.bounds = np.array([[0, -0.75, 0.6], [1, 0.75, 1.5]])
        self.add_offset = np.array([[-0.5, 0, -0.4]])

        self.extra_cameras = []
        for cam_name in cfg.camera[1:]:
            if cam_name == "zoom":
                new_camera = VisionSensor.create(self.camera.resolution,
                                                position=[1.5, 0, 1.5],
                                                orientation=[-math.pi, -50 / 180 * math.pi, math.pi/2])
                new_camera.set_render_mode(RenderMode.OPENGL)
                self.extra_cameras.append(new_camera)
            elif cam_name == "zoom_left":
                new_camera = VisionSensor.create(self.camera.resolution,
                                                position=[1.5, -0.5, 1.5],
                                                orientation=[-140 / 180 * math.pi, -40 / 180 * math.pi, 140 / 180 * math.pi])
                new_camera.set_render_mode(RenderMode.OPENGL) 
                self.extra_cameras.append(new_camera)
            elif cam_name == "zoom_right":
                new_camera = VisionSensor.create(self.camera.resolution, 
                                                position=[1.5, 0.5, 1.5], 
                                                orientation=[140 / 180 * math.pi, -40 / 180 * math.pi, 40 / 180 * math.pi])
                new_camera.set_render_mode(RenderMode.OPENGL) 
                self.extra_cameras.append(new_camera)
            else:
                raise NotImplementedError

        self.shapes = []
        self.shapenet_objs = read_shapenet(cfg.shapenet_to_generate, cfg.obj_split)
        self.table_coll_cuboid = Shape('TableCollCuboid')

        self.reset()
        self.render_resolution = self.camera.get_resolution()
        self.max_states_per_env = cfg.max_states_per_env
        self.grid_states = None

        self.noise = cfg.noise 
        self.radius = cfg.radius

    def get_obs(self, joint_states):
        obs, extra_obs = [], [[] for _ in range(len(self.extra_cameras))]
        for state in joint_states:
            self.agent.set_joint_positions(state, disable_dynamics=True)
            self.pr.step()
            obs.append(self.render())
            for cam_id in range(len(self.extra_cameras)):
                extra_obs[cam_id].append(self.render_extra(cam_id))
        out = dict(obs=np.array(obs))
        if len(self.extra_cameras) > 0:
            for cam_id in range(len(self.extra_cameras)):
                out[self.cfg.camera[cam_id+1]] = np.array(extra_obs[cam_id])
        return out

    def get_traj_collision_data(self):
        states = []
        collisions = []

        # get waypoints
        rand_idx = np.random.randint(len(self.shapes))
        cpos = self.shapes[rand_idx].get_position()
        radius = self.radius
        noise = np.random.random() * self.noise

        theta = np.random.random() * 2 * math.pi
        noise = np.random.random() * self.noise
       
        xrand = np.random.random(2) * 0.4
        yrand = np.random.random(2) * self.noise
        zrand = np.random.random(2) * 0.2 - 0.1
        pos1 = cpos + [np.cos(theta)*xrand[0], np.sin(theta)*yrand[0], zrand[0]]
        pos2 = cpos + [-np.cos(theta)*xrand[1], -np.sin(theta)*yrand[1], zrand[1]]
        
        pos_lst = [pos1, pos2]
        euler = [3.14, 0, 0]

        # run through paths
        for pos in pos_lst:
            try:
                path = self.agent.get_path(position=pos, euler=euler, ignore_collisions=True)
            except ConfigurationPathError as e:
                print("trying to get", pos)
                continue

            done_path = False
            while not done_path:
                done_path = path.step()
                self.pr.step()
                states.append(self._get_state())
                state_coll = []
                for i in range(len(self.shapes)):
                    state_coll.append(self.agent.check_arm_collision(self.shapes[i]))
                collisions.append(state_coll)

        if len(states) == 0:
            return None

        return dict(states=np.array(states), collisions=np.array(collisions))

    def get_collision_data(self):
        states = []
        collisions = [] 
        for _ in range(self.max_states_per_env):
            joint_state = self.sample_joint_state()
            states.append(joint_state)
            self.agent.set_joint_positions(joint_state, disable_dynamics=True)
            state_coll = []
            for i in range(len(self.shapes)):
                state_coll.append(self.agent.check_arm_collision(self.shapes[i]))
            # state_coll.append(self.agent.check_arm_collision(self.table_coll_cuboid))
            collisions.append(state_coll)
        return dict(states=np.array(states), collisions=np.array(collisions))

    def sample_joint_state(self):
        for _ in range(100):
            x = np.random.rand() * 400 + 230
            if x > 600:
                y = np.random.rand() * 100 - 50
            elif x > 500:
                y = np.random.rand() * 300 - 150
            else:
                y = np.random.rand() * 800 - 400
            sim_x, sim_y, _ = twin_pos_real_to_sim(x, y, 0)
            sim_z = np.random.rand() * 0.35 + 0.6
            ex, ey, ez = np.random.rand() * 0.6 - 0.3 + np.array([3.14, 0, 0])
            try: 
                joints = self.agent.solve_ik_via_sampling([sim_x, sim_y, sim_z], euler=[ex,ey,ez], ignore_collisions=True)
                return joints[0]
            except:
                pass
        raise Exception('Could not sample state') 
    def reset(self):
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        
        for shape in self.shapes:
            shape.remove()
        self.shapes = []

        position_min, position_max = [0.65, -0.4], [0.95, 0.4]
        x_len = position_max[0] - position_min[0]
        y_len = position_max[1] - position_min[1]
        pos_id = list(range(10))
        random.shuffle(pos_id)

        n_objects = np.random.randint(self.cfg.n_objects_min, self.cfg.n_objects_max + 1)
        for idx in range(n_objects):
            # sample random ShapeNetClass in self.shapenet_objs
            shapenet_class = random.choice(self.shapenet_objs)
            shapenet_obj = shapenet_class.get_random_shape()

            pid = pos_id[idx]
            pid_x = pid // 5 * x_len / 2 + position_min[0] 
            pid_y = pid % 5 * y_len / 5 + position_min[1]
            rx, ry = np.random.uniform(size=2) * np.array([0.01, 0.05]) 
            shape = Shape.import_shape(shapenet_obj.path, scaling_factor=0.35)
            shape.set_position([pid_x+rx, pid_y+ry, 0.9])
            shape.set_collidable(True)
            shape.set_dynamic(False)
            self.rotate_shape(shape)
            shape.name = shapenet_obj.name 
            shape.parent_name = shapenet_obj.parent_name
            self.shapes.append(shape)

        # post-processing: check for shapes in collision with arm 
        offset = len(self.shapes)
        removed = 0
        for idx in range(n_objects):
            while self.agent.check_arm_collision(self.shapes[idx-removed]):
                if offset >= len(pos_id):
                    print("Cannot find a valid position for all shapes")
                    shape_to_remove = self.shapes.pop(idx-removed)
                    shape_to_remove.remove()
                    removed += 1
                    break
                pid = pos_id[offset]
                pid_x = pid // 5 * x_len / 2 + position_min[0] 
                pid_y = pid % 5 * y_len / 5 + position_min[1]
                rx, ry = np.random.uniform(size=2) * np.array([0.01, 0.05]) 
                self.shapes[idx-removed].set_position([rx+pid_x, ry+pid_y, 0.8])
                offset += 1

        return self._get_state()

    def rotate_shape(self, shape):
        euler = self.euler_world_to_shape(shape, [math.pi/2, 0, 0])
        shape.rotate(euler)

    def euler_world_to_shape(self, shape, euler):
        m = sim.simGetObjectMatrix(shape._handle, -1)
        x_axis = np.array([m[0], m[4], m[8]])
        y_axis = np.array([m[1], m[5], m[9]])
        z_axis = np.array([m[2], m[6], m[10]])
        euler = np.array([math.pi/2, 0, 0])
        R = transforms3d.euler.euler2mat(*euler, axes='rxyz')
        T = np.array([x_axis, y_axis, z_axis]).T
        new_R = np.linalg.inv(T)@R@T
        new_euler = transforms3d.euler.mat2euler(new_R, axes='rxyz')
        return new_euler

    def _get_state(self):
        return np.concatenate([self.agent.get_joint_positions()])

    def _get_joint_positions(self):
        return self.agent.get_joint_positions()

    def get_path_collision_data(self, pos, euler=[0, math.radians(180), 0]):
        try:
            path = self.agent.get_path(position=pos, euler=euler, ignore_collisions=True)
        except ConfigurationPathError as e:
            return None

        states = []
        colls = []
        obs = []
        extra_obs = [[] for _ in range(len(self.extra_cameras))]

        done_path = False 

        while not done_path:
            done_path = path.step()
            self.pr.step()
            states.append(self._get_state())
            coll = []
            for shape in self.shapes:
                if self.agent.check_arm_collision(shape):
                    coll.append(1)
                else:
                    coll.append(0)
            colls.append(coll)
            obs.append(self.render())
            for cam_id in range(len(self.extra_cameras)):
                extra_obs[cam_id].append(self.render_extra(cam_id))
            
        return states, colls, obs, extra_obs


    def render(self):
        img_arr = self.camera.capture_rgb() 
        img = (img_arr * 255).astype(np.uint8)
        return img_arr

    def render_extra(self, cam_id):
        img_arr = self.extra_cameras[cam_id].capture_rgb() 
        img = (img_arr * 255).astype(np.uint8)
        return img_arr

    def get_meta(self):
        positions, names, parent_names = [], [], []
        for shape in self.shapes:
            positions.append(shape.get_position())
            names.append(shape.name)
            parent_names.append(shape.parent_name)
        positions = np.array(positions)
        names = np.array(names)
        meta = dict(positions=positions, names=names, parent_names=parent_names)
        return meta

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
