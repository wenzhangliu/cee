#!/usr/bin/env python3

from __future__ import annotations

from minigrid.envs.babyai.goto import GoToRedBallGrey
# from minigrid.wrappers import ImgObsWrapper,ReseedWrapper
from gym import spaces
import math

class GoToRedBallGreyPositionBonus(GoToRedBallGrey):
    def __init__(self, tile_size=8, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__()
        self.tile_size = tile_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height * tile_size, self.width * tile_size, 3),
            dtype='uint8'
        )
        action_size = self.action_space.n
        self.action_space = spaces.Discrete(action_size)
        self.counts = {}

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        obs, reward, terminated, truncated, info = super().step(action)
        done = truncated

        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)

        # note: reward is only action bonus
        reward = bonus
        info["counts"] = self.counts

        return self.observation(obs), reward, done, info

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)

        return rgb_img

    def render(self,mode = "human", **kwargs):
        self.render_mode = mode
        return super().render()

    def seed(self, seed):
        pass

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        obs,info = super().reset(seed=seed,options=None)
        return self.observation(obs)
if __name__ == "__main__":
    print("okk")