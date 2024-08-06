#!/usr/bin/env python3

from __future__ import annotations

import gym

from .envs.blockedunlockpickup import BlockedUnlockPickup
from .envs.obstructedmaze import ObstructedMazeEnv,\
    ObstructedMaze_1Dlh,ObstructedMaze_2Dlhb,ObstructedMaze_Full,\
    ObstructedMaze_2Dlh,ObstructedMaze_1Q,ObstructedMaze_1Dl,\
    ObstructedMaze_1Dlhb,ObstructedMaze_2Q,ObstructedMaze_2Dl
from .window import Window
from .wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from gym import spaces
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from gym.core import ObservationWrapper, Wrapper
import numpy as np

class ObstructedMazeEnvAR(ObstructedMazeEnv):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_1DlhAR(ObstructedMaze_1Dlh):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_2DlhbAR(ObstructedMaze_2Dlhb):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_FullAR(ObstructedMaze_Full):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_2DlhAR(ObstructedMaze_2Dlh):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_1QAR(ObstructedMaze_1Q):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_1DlAR(ObstructedMaze_1Dl):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_1DlhbAR(ObstructedMaze_1Dlhb):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_2QAR(ObstructedMaze_2Q):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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

class ObstructedMaze_2DlAR(ObstructedMaze_2Dl):
    """
    fix a seed and return img observation
    """

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
        obs, reward, done, info = super().step(action)
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
    print("okk")