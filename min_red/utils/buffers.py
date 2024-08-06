from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch as th
from gym import spaces
from typing import Dict, Generator, Optional, Union, NamedTuple
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ISReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super(ISReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
        )
        self.pi = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, pi: int) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        # print(pi)
        if isinstance(pi, th.Tensor):
            pi = pi.cpu()
        self.pi[self.pos] = np.array(pi).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds],
            self._normalize_reward(self.rewards[batch_inds], env),
            self.pi[batch_inds],
        )
        return ISReplayBufferSamples(*tuple(map(self.to_torch, data)))


class ISReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    pi: th.Tensor
