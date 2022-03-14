from typing import Tuple, List, Optional, Dict, Callable
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

DELTA = 0.02


class HandoverArm(robot.RobotArm):
    def __init__(self, controller_type='sim', headless=True, realtime=False, workspace=None, pb_client=None, serial_number=None):
        super().__init__(controller_type, headless, realtime, workspace, pb_client, serial_number)
        
        self.end_effector_link_index = 11;
        self.camera_link_index = 12;
        self.open_gripper
        if controller_type == 'sim':
            self._id = self._sim.robot_id;

        self.arm_ready_jpos = READY_JPOS

    def ready(self):
        '''
        moves the arm to the 'ready' position (bent, pointing forward toward workspace)
        '''
        self.open_gripper()
        self.move_arm_jpos(self.arm_ready_jpos)

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
        pos, quat = pb.getLinkState(self.robot._id, self.robot.camera_link_index)[:2]

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


        return rgba[...,:3], mask

class HandoverGraspingEnv(gym.Env):
    def __init__(self,
                 episode_length: int=3,
                 img_size: int=128,
                 render: bool=False,
                ) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        # add robot
        self.robot = HandoverArm(headless=not render)

        self.camera = WristCamera(self.robot, img_size);

        # add object
        self.object_id = pb.loadURDF(os.path.join(nuro_arm.constants.URDF_DIR, "object.urdf"))
        pb.changeDynamics(self.object_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
        self.object_width = 0.02

        #no options currently given
        self.object_routine = ObjectRoutine()

        pb.resetBasePositionAndOrientation(self.object_id, self.object_routine.getPos(), pb.getQuaternionFromEuler(self.object_routine.getOrn))

        self.t_step = 0
        self.episode_length = episode_length

        self.observation_space = gym.spaces.Box(0, 255, shape=(img_size, img_size, 3), dtype=np.float32)

        self.action_delta = DELTA

        #Four action spaces, each representing a positive, zero, or negative change
        self.base_rotation_actions = Discrete( n=3, start=-1)
        self.pitch_actions = Discrete( n=3, start=-1)
        self.forward_actions = Discrete( n=3, start=-1)
        self.wrist_rotation_actions = Discrete( n=3, start=-1)

    def reset(self) -> np.ndarray:
        '''Resets environment by randomly placing object
        '''
        self.object_routine.reset()
        #self.reset_object_texture()
        self.t_step = 0

        return self.get_obs()[0]

    def step(self, action: np.ndarray):
        '''
        Takes one step in the environment.

        Params
        ------
            action: 4-vector with discre values in {-1, 0, 1}
        Returns
        ------
            obs, reward, done, info
        '''

        assert self.base_rotation_actions.contains(action[0])
        assert self.pitch_actions.contains(action[2])
        assert self.forward_actions.contains(action[3])
        assert self.wrist_rotation_actions.contains(action[3])
        
        current_effector_pos = pb.getLinkState(self.robot._id, self.robot.end_effector_link_index, calculateForwardKinematics=True)[0]

        next_pos = [x + self.action_delta * a for x, a in zip(current_effector_pos, action[:3])]
        
        self.robot.move_hand_to(next_pos)

        self.t_step += 1

        obs = self.get_obs()
        reward = float(success)
        done = success or self.t_step >= self.episode_length
        info = {'success' : success}

        self.object_routine.step()

        return obs, reward, done, info


    def reset_object_position(self) -> None:
        '''Places object randomly in workspace.  The x,y position should be
        within the workspace, and the rotation performed only about z-axis.
        The height of the object should be set such that it sits on the plane
        '''
        ws_padding = 0.01
        x,y,z = np.random.uniform(self.workspace[0]+ws_padding,
                                self.workspace[1]-ws_padding)
                                
        theta = np.random.uniform(-np.pi/2, np.pi/2)

        pos = np.array((x,y,z))
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

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Takes picture using camera, returns rgb and segmentation mask of image
        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        self.camera.computeView()
        rgb, mask = self.camera.get_image()

        # remove robot parts from correct segmentation labelling
        map_fn = lambda x: 0 if x == 1 else x
        mask = np.vectorize(map_fn)(mask)

        return rgb, mask
    
    def plot_obs(self):
        plt.imshow(self.get_obs()[0])
        plt.show()


if __name__ == "__main__":
    env = HandoverGraspingEnv(render=True, img_size=200)
    env.robot.ready()
    env.plot_obs()

    while True:
        [pb.stepSimulation() for _ in range(10)]
        time.sleep(.01)

