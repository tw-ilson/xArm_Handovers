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
from scipy.spatial.transform.rotation import Rotation

import os

import nuro_arm
import nuro_arm.robot.robot_arm as robot

from curriculum import ObjectRoutine

READY_JPOS = [0, -1, 1.2, 1.4, 0]
TERMINAL_ERROR_MARGIN = 0.005

# NOTE: Besides Forward, are these intended to be in radians or in joint units?
ROTATION_DELTA = 0.02
VERTICAL_DELTA = 0.02
DISTANCE_DELTA = 0.05
ROLL_DELTA = 0.02


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
        self.controller.write_arm_jpos(READY_JPOS)

    def execute_action(self,
                       rot_act: int,
                       z_act: int,
                       dist_act: int,
                       roll_act: int):
        '''takes an action returned from policy neural network and moves joints accordingly.
        Params
        ------
        takes the four dimensions of the action space, all in {-1, 0, 1}

        '''
        xyz, rpy = self.get_hand_pose()
        x, y, z = xyz
        r, pt, yw = pb.getEulerFromQuaternion(rpy)

        if y > 0:
            rot_p = math.atan(x/y) + ROTATION_DELTA * rot_act
        else:
            rot_p = ROTATION_DELTA * rot_act
        z_p = z + VERTICAL_DELTA * z_act
        dist_p = math.sqrt(x**2 + y**2) + DISTANCE_DELTA*dist_act
        roll_p = r + ROLL_DELTA * roll_act

        x_p = dist_p * math.sin(rot_p)
        y_p = (y/abs(y)) * dist_p * math.cos(rot_p)

        # quat = pb.getQuaternionFromEuler((rot_p, 0, roll_p))
        quat = pb.getQuaternionFromEuler((rot_p, np.pi / 2, roll_p))

        joint_pos = self.mp.calculate_ik([x_p, y_p, z_p])[0]
        print('before:', joint_pos)

        self.controller.write_arm_jpos(joint_pos)
        # self.mp._teleport_arm(joint_pos)
        # self.controller.write_arm_jpos(joint_pos, speed=0.000001)


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

        rotmat = Rotation.from_quat(quat).as_matrix().T
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
                 episode_length: int = 3,
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

        self.camera = WristCamera(self.robot, img_size)

        # add object
        self.object_id = pb.loadURDF(os.path.join(
            nuro_arm.constants.URDF_DIR, "object.urdf"))
        pb.changeDynamics(self.object_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
        self.object_width = 0.02

        # no options currently given
        self.object_routine = ObjectRoutine(self.object_id)

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

    def reset(self) -> np.ndarray:
        '''Resets environment by randomly placing object
        '''
        self.object_routine.reset()
        # self.reset_object_texture()
        self.t_step = 0

        return self.get_obs()[0]

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

        assert self.base_rotation_actions.contains(action[0])
        assert self.pitch_actions.contains(action[2])
        assert self.forward_actions.contains(action[3])
        assert self.wrist_rotation_actions.contains(action[3])

        self.robot.execute_action(*action)

        self.t_step += 1

        obs = self.get_obs()
        reward, done = self.getReward()
        done = done or self.t_step >= self.episode_length

        # diagnostic information, what should we put here?
        info = {'s#uccess': 1}

        # self.object_routine.step()

        return obs, reward, done, info

    def canGrasp(self) -> bool:
        '''Determines if the current position of the gripper's is such that the object is within a small error margin of grasp point.
        '''

        grip_pos = pb.getLinkState(
            self.robot._id, self.robot.end_effector_link_index, computeForwardKinematics=True)[0]
        obj_pos = pb.getLinkState(self.object_id, 0)[0]

        return np.allclose(grip_pos, obj_pos, atol=TERMINAL_ERROR_MARGIN)

    def distToGrasp(self) -> float:
        ''' Euclidian distance to the grasping object '''

        grip_pos = pb.getLinkState(
            self.robot._id, self.robot.end_effector_link_index, computeForwardKinematics=True)[0]
        obj_pos = pb.getLinkState(self.object_id, 0)[0]

        return float(np.linalg.norm(grip_pos - obj_pos))

    def getReward(self) -> Tuple[float, bool]:
        ''' Defines the terminal states in the learning environment'''

        # TODO: Issue penalty if runs into an obstacle

        done = self.canGrasp()

        if self.sparse:
            return int(done), done
        else:
            return self.distToGrasp(), done

    def get_obs(self, background_mask: Optional[np.ndarray] = None) -> np.ndarray:
        '''Takes picture using camera, returns rgb and segmentation mask of image
        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        self.camera.computeView()
        rgb, mask = self.camera.get_image()

        # add virtual background for augmentation purposes (untested)
        if background_mask is not None:
            def map_fn(pix, bkrd_pix, mask_i,
                       ): return pix if mask_i != 0 else bkrd_pix
            rgb = np.vectorize(map_fn)(zip(rgb[:2], background_mask[:2], mask))

        return rgb

    def plot_obs(self):
        plt.imshow(self.get_obs())
        plt.show()


if __name__ == "__main__":
    env = HandoverGraspingEnv(render=True, img_size=200)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.resetDebugVisualizerCamera(cameraDistance=.4,
                                  cameraYaw=65.2,
                                  cameraPitch=-40.6,
                                  cameraTargetPosition=(.5, -0.36, 0.40))
    # env.plot_obs()
    env.robot.ready()

    while not np.allclose(env.robot.get_arm_jpos(), READY_JPOS, atol=.05):
        pb.stepSimulation()
        time.sleep(.001)

    # pose = env.robot.get_hand_pose()
    # print(pose[0])
    # print(pb.getEulerFromQuaternion(pose[1]))
    env.robot.execute_action(0, 0, 1, 0)

    for _ in range(100):
        [pb.stepSimulation() for _ in range(10)]
        time.sleep(.001)
    print('after', env.robot.get_arm_jpos())
    exit()
