import gym
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from min_red.action_model_trainer import ActionModelTrainer
from min_red.mf_model_trainer import MfModelTrainer
from min_red.config.parser_args import get_config
from min_red.config.config import Config
import argparse
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from common.format_string import pretty
import wandb
import os

from stable_baselines3.common.logger import configure
import torch

torch.set_num_threads(8)

curiosity_policy_path_dict = {"unlockpickupar-v0": "log/unlockpickupactionbonus-v0_PPO_n-1_2023-04-13-10-14-21/model614400",
                         "unlockpickupactionbonus-v0": "log/unlockpickupactionbonus-v0_PPO_n-1_2023-04-13-10-14-21/model614400",
                         "unlockpickupuncertaingoals-v0":"log/unlockpickupuncertaingoalsactionbonus-v0_PPO_n-1_2023-03-19-19-49-37/model409600"}
curiosity_policy_path_dict = {}
def eval_policy(env, model):
    obs = env.reset()
    traj_rewards = [0]
    while True:
        action, _state = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        env.render()
        time.sleep(0.03)
        traj_rewards[-1] += reward
        if done:
            obs = env.reset()
            m = input('Enter member idx: ')
            env.member = int(m)
            print(f"env member: {env.member}, R: {np.mean(traj_rewards)}")
            traj_rewards.append(0)


def train(config, log_path,logger):
    if config.is_atari:
        make_env = make_atari_stack_env
    else:
        make_env = make_vec_env

    env = make_env(config.env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)
    # env = gym.make(config.env_id)
    obs_shape = list(env.observation_space.shape)

    if config.algorithm.discrete:
        from min_red.min_red_ppo import MinRedPPO as Algorithm
        from stable_baselines3.dqn.policies import MultiInputPolicy as ActionModel
        # from min_red.dqn.policies import CnnPolicy as MfModel
        if len(obs_shape) > 2:
            s_space = (obs_shape[2], *obs_shape[:2])
            ssprime_shape = (2 * obs_shape[2], *obs_shape[:2])
            policy = 'CnnPolicy'
        else:
            s_space = (obs_shape[0], )
            ssprime_shape = (2 * obs_shape[0],)
            policy = 'MlpPolicy'
    else:
        pass

    # create action model obs space by extending env's obs space
    ssprime_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                       high=env.observation_space.high.max(),
                                       shape=ssprime_shape,
                                       dtype=env.observation_space.dtype)

    pi_obs_spce = gym.spaces.Box(low=0.,
                                 high=1.,
                                 shape=(env.action_space.n,),
                                 dtype=env.observation_space.dtype)

    new_ssprime_obs_space = gym.spaces.Dict({
        "image": ssprime_obs_space,
        "vector": pi_obs_spce,
    })

    action_model = ActionModel(observation_space=new_ssprime_obs_space,
                               action_space=env.action_space,
                               lr_schedule= lambda x: 3e-4).to(config.device)

    action_trainer = ActionModelTrainer(action_model=action_model,
                                        discrete=config.algorithm.discrete,
                                        new_logger=logger)

    #mask_factor_model: input: s + s' + action[one-hot vector]  output: Mask(s,a,b) = KL(P_a|P_b)

    s_obs_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                       high=env.observation_space.high.max(),
                                       shape=s_space,
                                       dtype=env.observation_space.dtype)

    sa_obs_space = gym.spaces.Dict({
        "image": s_obs_space,
        "vector": pi_obs_spce,
    })
    mf_model = ActionModel(observation_space=sa_obs_space,
                               action_space=env.action_space,
                               lr_schedule=lambda x: 3e-4).to(config.device)

    mf_trainer = MfModelTrainer(mf_model=mf_model,
                                discrete=config.algorithm.discrete,
                                new_logger=logger)
    if config.env_id in curiosity_policy_path_dict:
        curiosity_ppo_path = curiosity_policy_path_dict[config.env_id]
    else:
        curiosity_ppo_path = None

    model = Algorithm(policy, env, action_trainer=action_trainer, mf_trainer=mf_trainer, log_path=log_path, **config.algorithm.policy)
    if curiosity_ppo_path:
        curiosity_ppo = PPO.load(curiosity_ppo_path,device=config.device)
        model.policy.load_state_dict(curiosity_ppo.policy.state_dict())

    model.set_logger(logger)

    model.learn(**config.algorithm.learn)
    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model_path = os.path.join(log_path,"model")
        model.save(model_path,exclude=["action_trainer","mf_trainer"])

        mf_model_path = os.path.join(log_path, "mfmodel")
        mf_model.save(mf_model_path)

        action_model_path = os.path.join(log_path, "actionmodel")
        action_model.save(action_model_path)
    if config.play_model:
        eval_policy(env, model)


def bcast_config_vals(config):
    algorithm_config = Config(os.path.join(config.config_path, config.algorithm_type))
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.learn.total_timesteps = config.total_timesteps
    config.algorithm.policy["device"] = config.device
    config.algorithm.policy.method = config.method
    config.algorithm.policy.wandb = config.wandb
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)
    config = bcast_config_vals(config)
    if config.wandb:
        run = wandb.init(config=config)
    else:
        pretty(config)

    if 'n_actions' in config.env_kwargs.keys():
        n = config.env_kwargs.n_actions
    elif 'n_redundancies' in config.env_kwargs.keys():
        n = config.env_kwargs.n_redundancies
    else:
        n = -1
    experiment_name = "SF_Mask_"+ str(config.env_id) + '_' + str(config.method) + '_' + str(config.algorithm_type) + '_' + "n" + str(n) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = os.path.join("log", experiment_name)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    train(config,log_path,logger)

