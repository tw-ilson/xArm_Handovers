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
from augmentations import Preprocess

READY_JPOS = [0, -1, 1.2, 1.4, 0]
TERMINAL_ERROR_MARGIN = 0.004

# NOTE: Besides Forward, are these intended to be in radians or in joint units?
ROTATION_DELTA = 0.01
VERTICAL_DELTA = 0.01
DISTANCE_DELTA = 0.01
ROLL_DELTA = 0.01


class HandoverArm(robot.RobotArm):
    def __init__(self, controller_type='sim', headless=True, realtime=False, workspace=None, pb_client=None, serial_number=None):
        super().__init__(controller_type, headless,
                         realtime, workspace, pb_client, serial_number)

        self.end_effector_link_index = 11
        self.camera_link_index = 12
        self.open_gripper
        if controller_type == 'sim':
            self._id = self._sim.robot_id

        self.arm_ready_jpos = READY_JPOS
        # self.base_rotation_radians = self.controller._to_radians(1, READY_JPOS[0])


    def ready(self):
        '''
        moves the arm to the 'ready' position (bent, pointing forward toward workspace)
        '''
        # self.open_gripper()
        self.mp._teleport_gripper(1)
        # [pb.resetJointState(self.robot_id, i, jp, physicsClientId=self._client)
        #    for i,jp in zip(self.arm_joint_ids, arm_jpos)]
        self.mp._teleport_arm(READY_JPOS)

    def execute_action(self,
                       rot_act: int,
                       z_act: int,
                       dist_act: int,
                       roll_act: int) -> bool:
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
        (old_x, old_y, old_z), ee_quat = self.get_hand_pose()
        old_roll = R.from_quat(ee_quat).as_euler('zyz')[0]
        old_yaw = np.arctan2(old_y, old_x)
        old_radius = np.linalg.norm((old_x, old_y))

        x = old_x + (dist_act * DISTANCE_DELTA) * np.cos(old_yaw)
        y = old_y + (dist_act * DISTANCE_DELTA) * np.sin(old_yaw)
        z = old_z + (z_act * VERTICAL_DELTA)
        yaw = old_yaw + (rot_act * ROTATION_DELTA)
        roll = old_roll + (roll_act * ROLL_DELTA)

        new_pos = (x, y, z)
        new_quat = R.from_euler('zyz', (roll, np.pi/2, yaw)).as_quat()
        next_jpos = self.mp.calculate_ik(new_pos, new_quat)[0]


        valid_action, collisions = self.mp.is_collision_free_trajectory(start_jpos, next_jpos, ignore_gripper=False, n_substeps=4)
        if valid_action:
            self.mp._teleport_arm(next_jpos) 
        return valid_action


        # self.mp._teleport_arm(joint_pos)
        # self.controller.power_off_servos()


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

        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=CAMERA_FOV,
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
                 episode_length: int = 30,
                 sparse_reward: bool = True,
                 img_size: int = 128,
                 render: bool = False,
                 ) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        # add robot
        self.robot = HandoverArm(headless=not render)
        pb.setGravity(0, 0, 0)

        self.prepro = Preprocess(('blur', 'brightness'))

        self.camera = WristCamera(self.robot, img_size)
        self.img_size = img_size

        # add object
        self.object_id = pb.loadURDF(os.path.join(
            nuro_arm.constants.URDF_DIR, "object.urdf"))
        pb.changeDynamics(self.object_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
        self.object_width = 0.02

        # no options currently given
        self.object_routine = ObjectRoutine(
            self.object_id)#, moving_mode='noise', dimensions=['vertical', 'horizontal', 'depth'])

        pb.resetBasePositionAndOrientation(self.object_id, self.object_routine.getPos(
        ), pb.getQuaternionFromEuler(self.object_routine.getOrn()))

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
        self.action_space = gym.spaces.tuple.Tuple((
                self.base_rotation_actions, 
                self.pitch_actions, 
                self.forward_actions, 
                self.wrist_rotation_actions))
        self.action_space_shape = tuple([space.shape for space in self.action_space.spaces])

    def reset(self) -> np.ndarray:
        '''Resets environment by randomly placing object
        '''
        self.object_routine.reset()
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

        if self.robot.execute_action(*action):
            reward = self.getReward()
        else:
            reward = -100

        self.t_step += 1

        obs = self.get_obs()
        reward, done = self.getReward()

        done = done or self.t_step >= self.episode_length

        # diagnostic information, what should we put here?
        info = {'success': self.canGrasp()}

        self.object_routine.step()

        return obs, reward, done, info

    def getEEPosOrn(self):
        ee_ls = pb.getLinkState(
            self.robot._id, self.robot.end_effector_link_index, computeForwardKinematics=True)
        return tuple(ee_ls[:2])


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

        grip_pos, grip_orn = self.getEEPosOrn()
        obj_pos = pb.getBasePositionAndOrientation(self.object_id)[0]

        return float(np.linalg.norm(np.subtract(grip_pos, obj_pos)))

    def getReward(self) -> Tuple[float, bool]:
        ''' Defines the terminal states in the learning environment'''

        # TODO: Issue penalty if runs into an obstacle

        done = self.canGrasp()

        if self.sparse:
            return int(done), done
        else:
            return -self.distToGrasp(), done

    def get_obs(self) -> np.ndarray:
        '''Takes picture using camera, returns rgb and segmentation mask of image
        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        self.camera.computeView()
        rgb, mask = self.camera.get_image()
        rgb = self.prepro(rgb)
        return rgb

    def plot_obs(self):
        plt.imshow(self.get_obs())
        plt.show()


if __name__ == "__main__":
    env = HandoverGraspingEnv(render=False, img_size=86)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.resetDebugVisualizerCamera(cameraDistance=.4,
                                  cameraYaw=65.2,
                                  cameraPitch=-40.6,
                                  cameraTargetPosition=(.5, -0.36, 0.40))
    env.robot.ready()
    env.plot_obs()
    
    exit()
