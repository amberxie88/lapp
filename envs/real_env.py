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
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.const import ObjectType
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.light import Light
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError
from pyrep.backend import sim

import numpy as np
import math
import random
import transforms3d
import csv
import yaml
import matplotlib.pyplot as plt
from utils import RealWorldMeshes, twin_pos_real_to_sim, twin_pos_sim_to_real
from PIL import Image

SCENE_FILE = join(dirname(abspath(__file__)),
                  'assets/real_xarm.ttt')

class RealEnv(object):
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

        self.set_up_randomization()
        self.randomize_every = cfg.randomize_every
        
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = [0.000625, -0.591169, -0.001118, 0.176431, -0.003432, 0.767861, 0.002054]

        self.camera = VisionSensor('my_vision_sensor')
        self.camera.set_position([1.45, 0, 1.4])
        self.camera.set_orientation([-math.pi, -65 / 180 * math.pi, math.pi/2 + math.pi * 2/180])
        self.camera.set_render_mode(RenderMode.OPENGL3) 

        self.cylinder = Shape('Cylinder')
        self.cylinder.remove()
        self.shapes = []
        self.real_world_meshes = RealWorldMeshes()

        self.reset()
        self.max_states_per_env = cfg.max_states_per_env
        self.table_coll_cuboid = Shape('TableCollCuboid')

        self.default_joint_positions = np.array([0.000625, -0.591169, -0.001118, 0.176431, -0.003432, 0.767861, 0.002054])
        self.grid_states = None

    def get_obs(self, joint_states):
        obs, extra_obs = [], []
        for i in range(len(joint_states)):
            joint_pos = self.default_joint_positions + 0.2 * np.random.randn(7)
            self.agent.set_joint_positions(joint_pos, disable_dynamics=True)
            obs.append(self.render())
        out = dict(obs=np.array(obs))
        return out

    def sample_joint_state(self):
        for i in range(40):
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
            if i % 10 == 0:
                print(f"still sampling at {i} for {x} {y} {sim_z*1000-581}")
        raise Exception('Could not sample state') 

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
            state_coll.append(self.agent.check_arm_collision(self.table_coll_cuboid))
            collisions.append(state_coll)
        return dict(states=np.array(states), collisions=np.array(collisions))

    def set_up_randomization(self):
        from rlbench import VisualRandomizationConfig
        # reinit walls
        wall = Shape('Wall1')
        x, y, z = wall.get_position()
        wall.set_position([x, -0.55, z])
        wall = Shape('Wall2')
        x, y, z = wall.get_position()
        wall.set_position([x, 0.55, z])
        wall = Shape('Wall3')
        x, y, z = wall.get_position()
        wall.set_position([0.4, y, z])

        # reinit lights
        light = Light('DefaultLightA')
        x, y, z = light.get_position()
        light.set_position([x, -0.35, z])
        light = Light('DefaultLightB')
        x, y, z = light.get_position()
        light.set_position([x, 0.35, z])
        light = Light('DefaultLightC')
        x, y, z = light.get_position()
        light.set_position([0.5, -0.35, z])

    def randomize(self):
        # randomize lighting
        for light_name in ['DefaultLightA', 'DefaultLightB', 'DefaultLightC', 'DefaultLightD']:
            light = Light(light_name)
            x, y, z = light.get_position()
            light.set_position([x+np.random.rand()*0.2-0.1, y+np.random.rand()*0.2-0.1, z+np.random.rand()*0.2+0.1])
            c = np.random.rand()*0.6+0.2
            light.set_diffuse([c+np.random.rand()/10-0.05, c+np.random.rand()/10-0.05, c+np.random.rand()/10-0.05])

        # randomize camera
        self.camera.set_position([1.45+np.random.rand()*0.1, np.random.rand()*0.2-0.1, 1.2 + np.random.rand()*0.2])
        self.camera.set_orientation([-math.pi + np.random.rand()*0.08, -65 / 180 * math.pi + np.random.rand()*5/180*math.pi, math.pi/2 + math.pi * 2/180])

        # wall background:
        for wall_name in ['Wall1', 'Wall2', 'Wall3', 'Wall4', 'Floor']:
            wall = Shape(wall_name)
            wall.remove_texture()
            wall_color = np.random.uniform(np.array([0, 61, 0]), np.array([50, 130, 50]), size=(3)) / 255
            wall.set_color(list(wall_color))

        # table domain randomization
        rand_shade = np.random.uniform(80/255, 220/255)
        noise_x, noise_y, noise_z = np.random.random() / 40, np.random.random() / 40, np.random.random() / 40
        rand_color = [rand_shade + noise_x, rand_shade + noise_y, rand_shade + noise_z]
        for table_name in ['diningTable_visible', 'diningTable_visible0']:
            table = Object.get_object(table_name)
            ungrouped  = table.ungroup()
            for o in ungrouped:
                o.remove_texture()
                o.set_color(rand_color)
            self.pr.group_objects(ungrouped)
            # text_ob.remove()
        self.pr.step()


    def reset(self):
        self.agent.set_joint_positions(self.initial_joint_positions, disable_dynamics=True)
        self.pr.step()

        if self.randomize_every > 0:
            self.randomize()

        for shape in self.shapes:
            shape.remove()
        self.shapes = []

        x_min, x_max = 300, 700
        y_min, y_max = -400, 400
        mins = twin_pos_real_to_sim(300, -400, 0)
        maxs = twin_pos_real_to_sim(700, 400, 0)
        position_min, position_max = [mins[0], mins[1]], [maxs[0], maxs[1]]

        x_len = position_max[0] - position_min[0]
        y_len = position_max[1] - position_min[1]
        pos_id = list(range(10))
        random.shuffle(pos_id)

        n_objects = np.random.randint(3, 6) # 3 to 5 objects inclusive
        obj_ids = list(range(6))
        random.shuffle(obj_ids)
        for idx in range(n_objects):
            # sample random ShapeNetClass in self.shapenet_objs
            shapenet_obj, shape = self.real_world_meshes.get(obj_ids[idx])

            pid = pos_id[idx]
            pid_x = pid // 5 * x_len / 2 + position_min[0] 
            pid_y = pid % 5 * y_len / 5 + position_min[1]
            rx, ry = np.random.uniform(size=2) * np.array([0.01, 0.05]) 
            shape.set_position([pid_x+rx, pid_y+ry, shape.get_position()[-1]])
            shape.set_collidable(True)
            shape.set_dynamic(False)
            shape.name = shapenet_obj.name 
            self.shapes.append(shape)

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
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions()])

    def _get_joint_positions(self):
        return self.agent.get_joint_positions()

    def generate_env_data(self):
        # get env_cfg
        shape_info_list = []
        for shape in self.shapes:
            real_pos = twin_pos_sim_to_real(*shape.get_position())
            real_dimensions = shape.real_dimensions

            name = shape.name
            xlim = [real_pos[0], real_pos[0] + real_dimensions[0]]
            ylim = [real_pos[1], real_pos[1] + real_dimensions[1]]
            xlim = [float(x) for x in xlim]
            ylim = [float(y) for y in ylim]
            shape_info = dict(name=name, xlim=xlim, ylim=ylim)
            shape_info_list.append(shape_info)
        cfg = dict(shapes=shape_info_list)

        img_lst = []
        for _ in range(25):
            self.randomize()
            img = self.camera.capture_rgb()
            img_lst.append(img)
        return cfg, img_lst

    def render(self, null=False):
        img_arr = self.camera.capture_rgb() 
        img = (img_arr * 255).astype(np.uint8)
        return img_arr

    def get_meta(self):
        positions, names, parent_names = [], [], []
        for shape in self.shapes:
            positions.append(shape.get_position())
            names.append(shape.name)
        positions = np.array(positions)
        names = np.array(names)
        meta = dict(positions=positions, names=names)
        return meta

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
