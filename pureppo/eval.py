import gym
import envs
import numpy as np
import time
import argparse
from common.format_string import pretty
from common.parser_args import get_config
from common.config import Config
import os
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy,CnnPolicy
from torch.nn.modules.activation import Tanh,ReLU
from stable_baselines3.common.evaluation import evaluate_policy
torch.set_num_threads(8)

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


def eval(config, log_path):
    if config.is_atari:
        make_env = make_atari_stack_env
    else:
        make_env = make_vec_env
    env = make_env(config.env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)

    # if len(env.observation_space.shape) >=3:
    #     policy = 'CnnPolicy'
    # else:
    #     policy = 'MlpPolicy'
    # model = PPO(policy, env, tensorboard_log=log_path, **config.algorithm.policy)
    model = PPO.load(os.path.join(log_path,"model.zip"))

    mean,std = evaluate_policy(model,env)
    print("mean:"+str(mean)+" std:"+str(std))


def bcast_config_vals(config):
    algorithm_config = Config(os.path.join(config.config_path, config.algorithm_type))
    config.merge({"algorithm": algorithm_config}, override=False)
    config.algorithm.learn.total_timesteps = config.total_timesteps
    config.algorithm.policy["device"] = config.device
    if "activation_fn" in config.algorithm.policy.policy_kwargs:
        activation_fn = config.algorithm.policy.policy_kwargs["activation_fn"]
        if activation_fn == "ReLU":
            config.algorithm.policy.policy_kwargs["activation_fn"] = ReLU
        elif activation_fn == "Tanh":
            config.algorithm.policy.policy_kwargs["activation_fn"] = Tanh
        else:
            raise NotImplementedError
    # config.algorithm.policy.method = config.method
    # config.algorithm.policy.wandb = config.wandb
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    # get default parameters in this environment and override with extra_args
    config = get_config(args.f)
    # load default parameters in config_path/algorithm_type and override
    config = bcast_config_vals(config)
    pretty(config)

    if 'n_actions' in config.env_kwargs.keys():
        n = config.env_kwargs.n_actions
    elif 'n_redundancies' in config.env_kwargs.keys():
        n = config.env_kwargs.n_redundancies
    else:
        n = -1

    experiment_name_list_relu = ["HalfCheetah-v3_PPO_n-1_2023-02-23-17-29-36",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-17-29-46",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-17-29-57",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-17-30-07",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-17-30-17"]

    experiment_name_list_tanh = ["HalfCheetah-v3_PPO_n-1_2023-02-23-16-56-59",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-16-57-09",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-16-57-19",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-16-57-29",
                            "HalfCheetah-v3_PPO_n-1_2023-02-23-16-57-39"]
    experiment_name_list = experiment_name_list_tanh
    for experiment_name in experiment_name_list:
        # experiment_name = "HalfCheetah-v3_PPO_n-1_2023-02-23-17-29-36/"
        log_path = os.path.join("log", experiment_name)
        eval(config, log_path)
