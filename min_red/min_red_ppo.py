from typing import Any, Callable, Dict, Optional, Type, Union, Tuple
import types
import numpy as np
import torch as th
import gym
from gym import spaces
from torch.nn import functional as F
from stable_baselines3.ppo import PPO

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.vec_env import VecEnv
# from min_red.min_red_regularization import min_red_th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import explained_variance
import wandb
from stable_baselines3.common.utils import safe_mean

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.logger import Figure
from pureppo.heatmap import heatmap
from collections import deque
import sys
import time
import matplotlib.pyplot as plt

class MinRedPPO(PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        action_trainer=None,
        mf_trainer=None,
        method='none',
        absolute_threshold: bool = True,
        wandb: bool = True,
        min_red_ent_coef: float = 0.0,
        buffer_size: int = 100000,
        log_path: str = None,
    ):

        super(MinRedPPO, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            # create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.log_path = log_path
        self.action_trainer = action_trainer
        self.mf_trainer = mf_trainer
        self.method = method
        self.absolute_threshold = absolute_threshold
        self.wandb = wandb
        self.min_red_ent_coef = min_red_ent_coef
        self.buffer_size = buffer_size
        self.state_counts_buffer = deque(maxlen=1)

        def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Forward pass in all the networks (actor and critic)

            :param obs: Observation
            :param deterministic: Whether to sample or use deterministic actions
            :return: action, value and log probability of the action
            """
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            # Evaluate the values for the given observations
            values = self.value_net(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, log_prob, distribution.distribution.logits

        # override qnet_predict method for this object only
        self.policy.forward_w_logits = types.MethodType(forward, self.policy)

        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
        )

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        from stable_baselines3.common.policies import ActorCriticCnnPolicy
        import os
        act_policy_dir = os.path.join(self.log_path,"act_policy")
        self.policy.save(act_policy_dir)
        self.act_policy = ActorCriticCnnPolicy.load(act_policy_dir,device=self.device)
        def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
            """
            Forward pass in all the networks (actor and critic)

            :param obs: Observation
            :param deterministic: Whether to sample or use deterministic actions
            :return: action, value and log probability of the action
            """
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            # Evaluate the values for the given observations
            values = self.value_net(latent_vf)
            distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, log_prob, distribution.distribution.logits

        # override qnet_predict method for this object only
        self.act_policy.forward_w_logits = types.MethodType(forward, self.policy)

        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                # test ratio sholud be always one
                with th.no_grad():
                    _, test_old_log_prob,_ = self.act_policy.evaluate_actions(rollout_data.observations, actions)
                    test_ratio = th.exp(test_old_log_prob - rollout_data.old_log_prob)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)


                if self.method in ["random"]:
                    # use fixed random policy : stop training policy
                    if self._n_updates < 0:
                        self.policy.optimizer.step()
                else:
                    # use policy updated with pure curiosity reward
                    self.policy.optimizer.step()

                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

                # train action model
                replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
                with th.no_grad():
                    _, _, _, logits = self.act_policy.forward_w_logits(replay_data.observations)
                    act_pi = th.exp(logits)
                if self.method not in ['action']:
                    loss_item = self.action_trainer.train_step(replay_data, pi=act_pi, max_grad_norm=None)

                # print(loss_item)


                # train mask_factor model
                # print(self._n_updates)
                action_traniner_epoch = 5000
                if self._n_updates > action_traniner_epoch:
                    mf_trainnum_per_episode = 5
                    for _ in range(mf_trainnum_per_episode):
                        replay_data = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
                        with th.no_grad():
                            _, _, _, logits = self.act_policy.forward_w_logits(replay_data.observations)
                            act_pi = th.exp(logits)
                        if self.method not in ['action']:
                            self.mf_trainer.train_step(replay_data, pi=act_pi, action_module=self.action_trainer.action_model.q_net, max_grad_norm=None)

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        logger = self.logger
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates)
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        logit_vec = []
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs, logits = self.policy.forward_w_logits(obs_tensor)
                logit_vec.append(logits)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.replay_buffer.add(self._last_obs, new_obs, actions, rewards, dones,infos)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # zero advantage in last place
        rollout_buffer.advantages[-1] = 0

        callback.on_rollout_end()

        # wandb logging
        if self.wandb:
            wandb.log({"reward": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])}, step=self.num_timesteps)
            wandb.log({"ep_len": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])}, step=self.num_timesteps)
        return True

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                # record state_counts
                if len(self.state_counts_buffer) > 0:
                    fig = heatmap(self.state_counts_buffer[0])
                    self.logger.record("rollout/counts", Figure(fig, close=True),
                                       exclude=("stdout", "log", "json", "csv"))
                    plt.close()

                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.
        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            maybe_counts = info.get("counts")

            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
            if maybe_counts is not None:
                self.state_counts_buffer.extend([maybe_counts])
