from random import seed
import numpy as np
from typing import Optional

from math import sin
from time import time
from perlin_noise import PerlinNoise

from pybullet import resetBasePositionAndOrientation, getQuaternionFromEuler

MOVE_OPTS = ['static', 'cyclic', 'noise']
DIMS = ['vertical', 'horizontal', 'depth', 'roll', 'pitch', 'yaw']

WORKSPACE = np.array(((0.10, -0.05, 0.2),  # ((min_x, min_y, min_z)
                      (0.20, 0.05, 0.3)))  # (max_x, max_y, max_z))

START_POS = [0.2, 0.0, 0.25]

DELTA = 1


class ObjectRoutine():
    """
    controls the routine which the object will follow throughout a training episode

    Params
    ------
        random_start: boolean, should the object be placed at a random position from the start
        moving_mode: The way that the object will move along each varying dimension. Either static, cyclic or noisy.
        dimensions: list containing the dimensions along which to vary each timestep
    """

    def __init__(self, _id: int, random_start: bool = False, moving_mode: str = 'static', dimensions=[]) -> None:

        assert(_id)

        self._id = _id

        self.workspace = WORKSPACE
        self.random_start = random_start

        self.reset()

        if moving_mode in MOVE_OPTS:
            self.mode = moving_mode
            if self.mode == 'noise':
                self.noise = {dim: PerlinNoise() for dim in dimensions}
        else:
            raise ValueError("invalid mode")

        self.routine = set()
        if len(dimensions) == 0 or not all(arg in dimensions for arg in DIMS):
            self.routine.update(dimensions)
        else:
            raise ValueError("invalid dimensions specified")

    def reset(self):
        if self.random_start:
            ws_padding = 0.01
            x, y, z = np.random.uniform(self.workspace[0, :]+ws_padding,
                                        self.workspace[1, :]-ws_padding)

            #theta = lambda: np.random.uniform(-np.pi/2, np.pi/2)

            self.position = np.array((x, y, z))
            #self.orientation = [theta(), 0, theta()]
            self.orientation = [0, 0, 0]
        else:
            self.position = START_POS
            self.orientation = [0, 0, 0]

        resetBasePositionAndOrientation(
            self._id, self.position, getQuaternionFromEuler(self.orientation))

    def getPos(self):
        return self.position

    def getOrn(self, simulated_background: bool = False):
        return self.orientation

    def step(self):
        def issueUpdate(n, ax):
            n = DELTA * n
            if DIMS[0] == ax:
                self.position[0] += n
            if DIMS[1] == ax:
                self.position[1] += n
            if DIMS[2] == ax:
                self.position[2] += n
            if DIMS[3] == ax:
                self.orientation[0] += n
            if DIMS[4] == ax:
                self.orientation[1] += n
            if DIMS[5] == ax:
                self.orientation[2] += n

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
                    0.03 * self.noise[ax](time()),
                    ax)

        resetBasePositionAndOrientation(
            self._id, self.position, getQuaternionFromEuler(self.orientation))
