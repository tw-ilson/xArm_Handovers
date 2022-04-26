from random import seed
import numpy as np
from typing import Optional, Tuple, List
from os import path

from math import sin, cos
from time import time
from perlin_noise import PerlinNoise
from scipy.spatial.transform import Rotation as R

from pybullet import resetBasePositionAndOrientation, getQuaternionFromEuler, loadURDF, changeDynamics
from nuro_arm.constants import URDF_DIR

MOVE_OPTS = ['static', 'cyclic', 'noise']
DIMS = ['vertical', 'horizontal', 'depth', 'roll', 'pitch', 'yaw']

# NOTE: relevant?
WORKSPACE = np.array(((0.10, -0.05, 0.2),  # ((min_x, min_y, min_z)
                      (0.20, 0.05, 0.3)))  # (max_x, max_y, max_z))

NOISE_GRANULARITY = 0.025
NOISE_SCALING = 0.003
DIST_MAX = 0.2

DEFAULT_POS = (0.25, 0.0, 0.2)

POS_DELTA = 1
OR_DELTA = 20


class ObjectRoutine():
    """
    controls the routine which the object will follow throughout a training episode

    Params
    ------
        random_start: boolean, should the object be placed at a random position from the start
        position: position to start the object at beginning of episode, obfuscated by random_start=True
        moving_mode: The way that the object will move along each varying dimension. Either static, cyclic or noisy.
        dimensions: list containing the dimensions along which to vary each timestep
    """

    def __init__(self,
                 random_start: bool = False,
                 position: Optional[Tuple[float, float, float]] = DEFAULT_POS,
                 moving_mode: str = 'static',
                 moving_dimensions: List[str] = []) -> None:

        self._id = loadURDF(path.join(URDF_DIR, "object.urdf"))
        changeDynamics(self._id, -1,
                       lateralFriction=1,
                       spinningFriction=0.005,
                       rollingFriction=0.005)
        assert self._id

        self.workspace = WORKSPACE
        self.random_start = random_start
        self.home_position = [*position]
        self.position = self.home_position
        self.home_orientation = [0, np.pi/2, 0]
        self.orientation = self.home_orientation

        self.reset()

        if moving_mode in MOVE_OPTS:
            self.mode = moving_mode
            if self.mode == 'noise':
                self.noise = {dim: PerlinNoise() for dim in moving_dimensions}
        else:
            raise ValueError("invalid mode")

        self.routine = set()
        if len(moving_dimensions) == 0 or not all(arg in moving_dimensions for arg in DIMS):
            self.routine.update(moving_dimensions)
        else:
            raise ValueError("invalid dimensions specified")

    def reset(self):
        if self.random_start:
            ws_padding = 0.01
            # x, y, z = np.random.uniform(self.workspace[0, :]+ws_padding,
            #                             self.workspace[1, :]-ws_padding)

            # pick random start position based on radius around robot
            # theta = np.random.normal(-np.pi/4, np.pi/4)
            # r = np.random.uniform(.12, DIST_MAX)
            # self.home_position = list((r * cos(theta), r * sin(theta), np.random.uniform(0.1, 0.3)))
            self.home_position = np.random.uniform(
                (0.15, -0.06, 0.1), (0.25, 0.06, 0.25))
            self.position = self.home_position

            self.orientation = self.home_orientation
        else:
            self.orientation = self.home_orientation

        resetBasePositionAndOrientation(
            self._id, self.position, R.from_euler('zyz', angles=self.orientation).as_quat())

    def getPos(self):
        return self.position

    def getOrn(self, simulated_background: bool = False):
        return self.orientation

    def step(self):
        def issueUpdate(n, ax):
            if DIMS[0] == ax:
                self.position[0] = self.home_position[0] + POS_DELTA * n
            if DIMS[1] == ax:
                self.position[1] = self.home_position[1] + POS_DELTA * n
            if DIMS[2] == ax:
                self.position[2] = self.home_position[2] + POS_DELTA * n
            if DIMS[3] == ax:
                self.orientation[0] = self.home_orientation[0] + OR_DELTA * n
            if DIMS[4] == ax:
                self.orientation[1] = self.home_orientation[1] + OR_DELTA * n
            if DIMS[5] == ax:
                self.orientation[2] = self.home_orientation[2] + OR_DELTA * n

        if self.mode == 'static':
            pass

        elif self.mode == 'cyclic':
            for ax in self.routine:
                issueUpdate(
                    sin(time()),
                    ax)

        elif self.mode == 'noise':
            for ax in self.routine:
                issueUpdate(
                    NOISE_SCALING * self.noise[ax](NOISE_GRANULARITY * time()),
                    ax)

        resetBasePositionAndOrientation(
            self._id, self.position, R.from_euler('zyz', angles=self.orientation).as_quat())
