from stable_baselines3.ppo import PPO
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from common.classify_points import classify_points, find_redundant_positions
import types
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

SelfPPO = TypeVar("SelfPPO", bound="PPO")
class Mfnet():
    def __init__(self,net):
        self.net = net

class MaskPPO(PPO):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        mf_model = None,
        mask_flag = False,
        mask_threshold = 2,
    ):
        super(MaskPPO, self).__init__(
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
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        self.mf_model = mf_model
        self.policy.mfnet = Mfnet(net=self.mf_model.to(device))

        self.mask_flag = mask_flag
        self.policy.mask_flag = mask_flag

        self.mask_threshold = mask_threshold
        self.policy.mask_threshold = mask_threshold

        def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            """
            Forward pass in all the networks (actor and critic)
            :param obs: Observation
            :param deterministic: Whether to sample or use deterministic actions
            :return: action, value and log probability of the action
            """
            # Preprocess the observation if needed
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            # Evaluate the values for the given observations
            values = self.value_net(latent_vf)


            if self.mask_flag:
                # Calculate Similarity_Factor_Matrix
                n_actions = self.action_space.n
                test_mf_model = self.mfnet.net
                with th.no_grad():
                    mf_predict_list = []
                    for aa in range(n_actions):
                        # all action is aa
                        test_actions = th.ones((len(obs))).to(th.int64) * aa
                        test_actions_onehot = th.nn.functional.one_hot(test_actions, n_actions).squeeze(dim=1)
                        sa = {"image": obs.float().to(self.device), "vector": test_actions_onehot.float().to(self.device)}
                        mf_predicted_aa = test_mf_model.q_net(sa)
                        mean_mf_predicted_aa = th.mean(mf_predicted_aa, dim=0)
                        mf_predict_list.append(mean_mf_predicted_aa.tolist())
                    SF_Matrix = np.diag(np.array(mf_predict_list)) - np.array(mf_predict_list)

                N_matrix = np.diag(np.array(mf_predict_list))
                classes,classes_actions = classify_points(SF_Matrix,threshold=self.mask_threshold)
                minred_actions_list, redundant_actions_list = find_redundant_positions(classes, classes_actions, N_matrix)
                distribution = self._get_action_dist_from_latent(latent_pi,redundant_actions_list=redundant_actions_list)
            else:

                distribution = self._get_action_dist_from_latent(latent_pi)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *self.action_space.shape))
            return actions, values, log_prob

        def _get_action_dist_from_latent(self, latent_pi: th.Tensor,redundant_actions_list=None) -> Distribution:
            """
            Retrieve action distribution given the latent codes.
            :param latent_pi: Latent code for the actor
            :return: Action distribution
            """
            mean_actions = self.action_net(latent_pi)

            if isinstance(self.action_dist, DiagGaussianDistribution):
                return self.action_dist.proba_distribution(mean_actions, self.log_std)
            elif isinstance(self.action_dist, CategoricalDistribution):
                # Here mean_actions are the logits before the softmax
                # mask redundant actions according to redundant_actions_list
                if redundant_actions_list:
                    mean_actions[:, redundant_actions_list] = -1e10
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            elif isinstance(self.action_dist, MultiCategoricalDistribution):
                # Here mean_actions are the flattened logits
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            elif isinstance(self.action_dist, BernoulliDistribution):
                # Here mean_actions are the logits (before rounding to get the binary actions)
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            elif isinstance(self.action_dist, StateDependentNoiseDistribution):
                return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
            else:
                raise ValueError("Invalid action distribution")

        # self.policy._predict = types.MethodType(_predict, self.policy)
        self.policy.forward = types.MethodType(forward, self.policy)
        self.policy._get_action_dist_from_latent = types.MethodType(_get_action_dist_from_latent, self.policy)

