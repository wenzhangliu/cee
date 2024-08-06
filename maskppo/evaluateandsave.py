import gym
import envs
import numpy as np
import time
import argparse
from common.format_string import pretty
from common.parser_args import get_config
from common.config import Config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy,CnnPolicy
from torch.nn.modules.activation import Tanh,ReLU
# from stable_baselines3.common.evaluation import evaluate_policy
from common.evaluation import evaluate_policy_and_save
import wandb
torch.set_num_threads(8)

# def eval_policy(env, model):
#     obs = env.reset()
#     traj_rewards = [0]
#     while True:
#         action, _state = model.predict(obs, deterministic=False)
#         next_obs, reward, done, info = env.step(action)
#         obs = next_obs
#         env.render()
#         time.sleep(0.03)
#         traj_rewards[-1] += reward
#         if done:
#             obs = env.reset()
#             m = input('Enter member idx: ')
#             env.member = int(m)
#             print(f"env member: {env.member}, R: {np.mean(traj_rewards)}")
#             traj_rewards.append(0)
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

def test(config,log_path):
    make_env = make_vec_env
    env = make_env(config.env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)
    # 获取目录下的所有文件
    file_list = os.listdir(log_path)

    # 筛选出所有扩展名为 ".zip" 的文件
    zip_file_list = [f for f in file_list if f.endswith('.zip')]

    for model_name in zip_file_list:
        model = PPO.load(os.path.join(log_path, model_name))
        mean, std = evaluate_policy_and_save(model, env, save_path=os.path.join(log_path, model_name.rstrip(".zip") + ".npy"), deterministic=True)
        print("model:" + model_name)
        print("mean:" + str(mean) + " std:" + str(std))


if __name__ == '__main__':
    experiment_name = "unlockpickupactionbonus-v0_PPO_n-1_2023-03-15-15-55-10"
    env_name = experiment_name.split("-")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    args.f = "pureppo/config/" + env_name
    # get default parameters in this environment and override with extra_args
    config = get_config(args.f)
    # load default parameters in config_path/algorithm_type and override
    config = bcast_config_vals(config)
    pretty(config)

    log_path = os.path.join("log", experiment_name)
    test(config, log_path)
