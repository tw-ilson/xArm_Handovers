from dis import dis
from typing import Tuple, List, Optional, Dict, Callable
import math
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import pybullet as pb
import pybullet_data
import gym
from gym.spaces.discrete import Discrete
import time
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import os
import nuro_arm
import nuro_arm.robot.robot_arm as robot

from curriculum import ObjectRoutine
# from augmentations import Preprocess

HOME_JPOS = [0, -1, 1.2, 1.4, 0]
TERMINAL_ERROR_MARGIN = 0.035

ROTATION_DELTA = 0.10
VERTICAL_DELTA = 0.02
DISTANCE_DELTA = 0.02
ROLL_DELTA = 0.04

# BACKGROUNDS_DIR =  "/Users/tom/Documents/tiny-imagenet-200/val/images"

class HandoverArm(robot.RobotArm):
    def __init__(self, controller_type='sim', headless=True, realtime=False, workspace=None, pb_client=None, serial_number=None):
        super().__init__(controller_type, headless,
                         realtime, workspace, pb_client, serial_number)

        self.end_effector_link_index = 11
        self.camera_link_index = 12
        self.open_gripper
        if controller_type == 'sim':
            self._id = self._sim.robot_id

        self.arm_ready_jpos = HOME_JPOS
        # self.base_rotation_radians = self.controller._to_radians(1, READY_JPOS[0])
#         self.object_id

    def ready(self, randomize=True):
        '''
        moves the arm to the 'ready' position (bent, pointing forward toward workspace)
        '''

        # self.open_gripper()
        self.mp._teleport_gripper(1)

        self.mp._teleport_arm(HOME_JPOS)

        if randomize:
            rot_off = np.random.randint(-4, 4)
            z_off = np.random.randint(-4, 4)
            self.execute_action(rot_off, z_off, 0, 0)

    def execute_action(self,
                       rot_act: int,
                       z_act: int,
                       dist_act: int,
                       roll_act: int):
        '''takes an action returned from policy neural network and moves joints accordingly.
        Params
        ------
        takes the four dimensions of the action space, all in {-1, 0, 1}
        rot_act: base rotation
        z_act: up/down on z axis
        dist_act: forward/backward from center of robot
        roll_act: gripper roll

        Returns
        -------
            True if collision-free action was predicted

        '''
        start_jpos = self.get_arm_jpos()
#         collide_list = self.mp.is_collision_free(start_jpos, False)[1]
#         for collision in collide_list:
#             print(collision.__str__())
#         obj_collide = np.any(map(lambda x: x.other_body == self.object_id, collide_list))
#         print('obj collide:', obj_collide)
        # TODO add to valid_Action return, change definition of collission to object collision
        (old_x, old_y, old_z), ee_quat = self.get_hand_pose()
        old_roll, _, old_yaw = R.from_quat(ee_quat).as_euler('zyz')
        # old_yaw = np.arctan2(old_y, old_x)
        # print(old_yaw, R.from_quat(ee_quat).as_euler('zyz'))
        old_radius = np.linalg.norm((old_x, old_y))

        x = old_x + (dist_act * DISTANCE_DELTA) * np.cos(old_yaw)
        y = old_y + (dist_act * DISTANCE_DELTA) * np.sin(old_yaw)
        z = old_z + (z_act * VERTICAL_DELTA)
        yaw = old_yaw + (rot_act * ROTATION_DELTA)
        roll = old_roll + (roll_act * ROLL_DELTA)

        new_pos = (x, y, z)
        new_quat = R.from_euler('zyz', (roll, np.pi/2, yaw)).as_quat()
        next_jpos, info = self.mp.calculate_ik(new_pos, new_quat)
        
        valid_action = info['ik_pos_error'] < 0.05 and info['ik_rot_error'] < 0.2

        valid_action = valid_action and self.mp.is_collision_free(next_jpos, False)[0]  

        # valid_action, collisions = self.mp.is_collision_free_trajectory(
            # start_jpos, next_jpos, ignore_gripper=False, n_substeps=4)
        if valid_action:
            self.mp._teleport_arm(next_jpos)

        return valid_action

        # self.mp._teleport_arm(joint_pos)
        # self.controller.power_off_servos()

    def try_grasp(self) -> bool:
        self.close_gripper()
        self.self.controller.read_gripper_state()


class WristCamera:
    def __init__(self, robot: HandoverArm, img_size: int) -> None:
        '''Camera that is mounted to view workspace from above
        Hint
        ----
        For this camera setup, it may be easiest if you use the functions
        `pybullet.computeViewMatrix` and `pybullet.computeProjectionMatrixFOV`.
        cameraUpVector should be (0,1,0)
        Parameters
        ----------
        workspace
            2d array describing extents of robot workspace that is to be viewed,
            in the format: ((min_x,min_y), (max_x, max_y))
        Attributes
        ----------
        img_size : int
            height, width of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_size = img_size
        self.robot = robot
        self.type = robot.controller_type

        self.computeView()

    def computeView(self):
        """
        Computes the view matrix and projection matrix based on the position and orientation of the robot's end effector
        """
        pos, quat = pb.getLinkState(
            self.robot._id, self.robot.camera_link_index)[:2]

        rotmat = R.from_quat(quat).as_matrix().T
        pos = - np.dot(rotmat, pos)

        self.view_mtx = np.eye(4)
        self.view_mtx[:3, :3] = rotmat
        self.view_mtx[:3, 3] = pos

        self.view_mtx = np.ravel(self.view_mtx, order='F')

        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=90,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=10)

    def get_image(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Takes rgb image
        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba, _, mask = tuple(
            pb.getCameraImage(width=self.img_size,
                              height=self.img_size,
                              viewMatrix=self.view_mtx,
                              projectionMatrix=self.proj_mtx,
                              renderer=pb.ER_TINY_RENDERER)[2:5])

        return rgba[..., :3], mask


class HandoverGraspingEnv(gym.Env):
    def __init__(self,
                 episode_length: int = 60,
                 sparse_reward: bool = True,
                 img_size: int = 64,
                 render: bool = False,
                 preprocess=False,
                 ) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        # add robot
        self.robot = HandoverArm(headless=not render)
        pb.setGravity(0, 0, 0)

        self.camera = WristCamera(self.robot, img_size)
        self.img_size = img_size

        self.prepro = preprocess
        if preprocess:
            self.prepro = Preprocess(augmentations=('brightness', 'blur'), bkrd_dir=BACKGROUNDS_DIR)

        self.object_routine = ObjectRoutine(moving_mode='noise', moving_dimensions=['horizontal', 'vertical', 'roll'], random_start=False)
        self.robot.object_id = self.object_routine._id

        # add object
        self.object_width = 0.02

        self.t_step = 0
        self.episode_length = episode_length
        self.sparse = sparse_reward

        self.observation_space = gym.spaces.Box(
            0, 255, shape=(img_size, img_size, 3), dtype=np.float32)

        # Four action spaces, each representing a positive, zero, or negative change
        self.base_rotation_actions = Discrete(n=3, start=-1)
        self.pitch_actions = Discrete(n=3, start=-1)
        self.forward_actions = Discrete(n=3, start=-1)
        self.wrist_rotation_actions = Discrete(n=3, start=-1)

        # NOTE: if sampling, subtract 1
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3])
        # self.action_space = gym.spaces.Discrete(81)

        self.reset()

    def reset(self) -> np.ndarray:
        '''Resets environment by randomly placing object
        '''
        # self.object_routine.reset()
        pos = np.random.uniform((0.15, -0.06, 0.1),(0.25, 0.06, 0.25))
        quat = pb.getQuaternionFromEuler((0,np.pi/2, 0))
        pb.resetBasePositionAndOrientation(self.object_routine._id, pos, quat)
        self.robot.ready()
        # self.reset_object_texture()
        self.t_step = 0

        return self.get_obs()

    def step(self, action: np.ndarray):
        '''
        Takes one step in the environment.

        Params
        ------
            action: 4-vector with discrete values in {-1, 0, 1}
        Returns
        ------
            obs, reward, done, info
        '''
        self.action_space.contains(action)

        # TODO maybe different reward schema?
        # tmp = action
        # action = []
        # for i in range(4):
            # action.append(tmp % 3 - 1)
            # tmp = tmp // 3


        if self.robot.execute_action(*action):
            collided = False
        else:
            collided = True

        self.t_step += 1

        obs = self.get_obs()
        reward, done = self.getReward(collided)

        done = done or self.t_step >= self.episode_length# or collided

        # diagnostic information, what should we put here?
        info = {'success': self.canGrasp()}

        # self.object_routine.step()

        return obs, reward, done, info

    def canGrasp(self) -> bool:
        '''Determines if the current position of the gripper's is such that the object is within a small error margin of grasp point.
        '''

        # grip_pos = pb.getLinkState(
        #     self.robot._id, self.robot.end_effector_link_index, computeForwardKinematics=True)[0]
        # obj_pos = pb.getBasePositionAndOrientation(self.object_id)[0]
        # return np.allclose(grip_pos, obj_pos, atol=TERMINAL_ERROR_MARGIN, rtol=0)
        return self.distToGrasp() < TERMINAL_ERROR_MARGIN

    def distToGrasp(self) -> float:
        ''' Euclidian distance to the grasping object '''

        grip_pos = pb.getLinkState(
            self.robot._id, self.robot.end_effector_link_index, computeForwardKinematics=True)[0]
        obj_pos = pb.getBasePositionAndOrientation(self.object_routine._id)[0]

        return float(np.linalg.norm(np.subtract(grip_pos, obj_pos)))

    def getReward(self, collided: bool) -> Tuple[float, bool]:
        ''' Defines the terminal states in the learning environment'''

        # TODO: Issue penalty if runs into an obstacle

        REWARD_SCALE = 1e-3
        done = self.canGrasp()

        if self.sparse:
#             if collided:
#                 return -1, True
#             else:
            return int(done), done
        else:
            return REWARD_SCALE/self.distToGrasp(), done

    def get_obs(self) -> dict:
        '''Takes picture using camera, returns rgb and segmentation mask of image
        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        self.camera.computeView()
        rgb, mask = self.camera.get_image()

        jpos = self.robot.get_arm_jpos()
        if self.prepro:
            rgb = self.prepro(rgb, mask)

        return {'rgb': rgb, 'joints':jpos}

    def plot_obs(self):
        plt.imshow(self.get_obs())
        plt.show()


if __name__ == "__main__":
    env = HandoverGraspingEnv()

    env.plot_obs()


    exit()
