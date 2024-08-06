#!/usr/bin/env python3

from __future__ import annotations

import gym

from .envs.blockedunlockpickup import BlockedUnlockPickup
from .envs.memory import MemoryEnv
from .window import Window
from .wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from gym import spaces
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from gym.core import ObservationWrapper, Wrapper
import numpy as np

class MemoryS11(MemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=11)
class MemoryS13(MemoryEnv):
    def __init__(self, seed=None):
        super().__init__(seed=seed, size=13)

class memoryar(MemoryS13):
    def __init__(self, tile_size=8, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        self.tile_size = tile_size
        super().__init__()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height * tile_size, self.width * tile_size, 3),
            dtype='uint8'
        )

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        obs, reward, done, info = super(memoryar, self).step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return rgb_img

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        seed = self.seeds[self.seed_idx]
        super().seed(seed)
        obs = super().reset(**kwargs)
        return self.observation(obs)

if __name__ == "__main__":
    env = memoryar()
    env.reset()
    print("okk")