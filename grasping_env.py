from typing import Tuple, List, Optional, Dict, Callable
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import pybullet as pb
import pybullet_data
import gym
import time
from tqdm import tqdm
import h5py

import os

import nuro_arm.camera 
from nuro_arm.robot import robot_arm



class HandoverArm(robot_arm.RobotArm):
    def __init__(self, controller_type='sim', headless=True, realtime=False, workspace=None, pb_client=None, serial_number=None):
        super().__init__(controller_type, headless, realtime, workspace, pb_client, serial_number)
        self.camera_link_index = 12;
        if controller_type == 'sim':
            self._id = self._sim.robot_id;

    

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

        self.computeView()

    def computeView(self):
        """
        Computes the view matrix and projection matrix based on the position and orientation of the robot's end effector
        """
        cam_linkstate = pb.getLinkState(self.robot._id, self.robot.camera_link_index)
        self.position = [*cam_linkstate[0]]
        self.orientation = [*pb.getMatrixFromQuaternion(cam_linkstate[1])] + [0, 0, 0]

        #pre-allocate view matrix
        self.view_mtx = np.zeros(shape=(4,4))
        for i in range(4):
            for j in range(3):
                self.view_mtx[i, j] = self.orientation[3*i +j]
        for i in range(3):
            self.view_mtx[i, 3] = self.position[i]
        self.view_mtx[3,3] = 1

        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=60,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=1)
        
        assert(self.view_mtx.shape == (4, 4))


    def get_rgb_image(self) -> np.ndarray:
        '''Takes rgb image
        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba = pb.getCameraImage(width=self.img_size,
                                 height=self.img_size,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]

        return rgba[...,:3]


class TopDownGraspingEnv(gym.Env):
    def __init__(self,
                 episode_length: int=3,
                 img_size: int=42,
                 render: bool=False,
                ) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        # add robot
        self.robot = HandoverArm(headless=False)

        self.camera = WristCamera(self.robot, 42);

        # add object
        self.object_id = pb.loadURDF("assets/urdf/object.urdf")
        pb.changeDynamics(self.object_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
        self.object_width = 0.02

        self.workspace = np.array(((0.10, -0.05), # ((min_x, min_y)
                                   (0.20, 0.05))) #  (max_x, max_y))


        self.t_step = 0
        self.episode_length = episode_length

        #self.observation_space = gym.spaces.Box(0, 255, shape=(img_size, img_size, 3), dtype=np.uint8)
        #self.action_space = gym.spaces.Box(0, img_size-1, shape=(2,), dtype=int)

    def reset(self) -> np.ndarray:
        '''Resets environment by randomly placing object
        '''
        self.reset_object_position()
        self.reset_object_texture()
        self.t_step = 0

        return self.get_obs()

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        x,y = self._convert_from_pixel(np.array(action))

        success = self.perform_grasp(x, y)
        self.t_step += 1

        obs = self.get_obs()
        reward = float(success)
        done = success or self.t_step >= self.episode_length
        info = {'success' : success}

        return obs, reward, done, info

    def _convert_from_pixel(self, pxy: np.ndarray) -> np.ndarray:
        xy_norm = pxy.astype(float) / self.img_size

        xy = xy_norm * np.subtract(*self.workspace[::-1]) + self.workspace[0]
        return xy

    def _convert_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy_norm = np.subtract(xy, self.workspace[0]) \
                    / np.subtract(*self.workspace[::-1])
        xy_norm = np.clip(xy_norm, 0, 1)

        #xy axis are flipped from world to image space
        pxy = self.img_size * xy_norm
        return pxy.astype(int)

    def perform_grasp(self, x: float, y: float, theta: float=0) -> bool:
        '''Perform top down grasp in the workspace.  All grasps will occur
        at a height of the center of mass of the object (i.e. object_width/2)
        Parameters
        ----------
        x
            x position of the grasp in world frame
        y
            y position of the grasp in world frame
        theta, default=0
            target rotation about z-axis of gripper during grasp
        Returns
        -------
        bool
            True if object was successfully grasped, False otherwise. It is up
            to you to decide how to determine success
        '''
        self.robot.teleport_arm(self.robot.home_arm_jpos)
        self.robot.set_gripper_state(self.robot.GRIPPER_OPENED)

        pos = np.array((x, y, self.object_width/2))

        self.robot.move_gripper_to(pos, theta, teleport=True)
        self.robot.set_gripper_state(self.robot.GRIPPER_CLOSED)

        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        # check if object is above plane
        min_object_height = 0.2
        obj_height = pb.getBasePositionAndOrientation(self.object_id)[0][2]
        success = obj_height > min_object_height

        return success

    def reset_object_position(self) -> None:
        '''Places object randomly in workspace.  The x,y position should be
        within the workspace, and the rotation performed only about z-axis.
        The height of the object should be set such that it sits on the plane
        '''
        ws_padding = 0.01
        x,y = np.random.uniform(self.workspace[0]+ws_padding,
                                self.workspace[1]-ws_padding)
        theta = np.random.uniform(-np.pi/2, np.pi/2)

        pos = np.array((x,y,self.object_width/2))
        quat = pb.getQuaternionFromEuler((np.random.randint(2)*np.pi,0, theta))
        pb.resetBasePositionAndOrientation(self.object_id, pos, quat)

    def reset_object_texture(self) -> None:
        '''Randomly assigns a texture to the object.  Available textures are
        located within the `assets/textures` folder.
        '''
        tex_id = self.tex_ids[np.random.randint(len(self.tex_ids))]
        r,g,b = np.random.uniform(0.5, 1, size=3)**0.8
        pb.changeVisualShape(self.object_id, -1, -1,
                             textureUniqueId=tex_id,
                             rgbaColor=(r,g,b,1))

    def get_obs(self) -> np.ndarray:
        '''Takes picture using camera
        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        return self.camera.get_rgb_image()



def watch_policy(env: TopDownGraspingEnv, policy: Optional[Callable]=None):
    if policy is None:
        policy = lambda s: env.action_space.sample()

    s = env.reset()
    while 1:
        a = policy(s)
        sp, r, d, info = env.step(a)
        time.sleep(0.5)

        s = sp.copy()
        if d:
            s = env.reset()
            time.sleep(1)


if __name__ == "__main__":
    env = TopDownGraspingEnv(render=True)
    while True:
        [pb.stepSimulation() for _ in range(10)]
        time.sleep(.01)
