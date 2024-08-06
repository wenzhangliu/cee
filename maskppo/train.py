import time
import argparse
from common.format_string import pretty
from common.parser_args import get_config
from common.config import Config
import os
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn.modules.activation import Tanh,ReLU
from common.evaluation import evaluate_policy_and_save
from .maskppo import MaskPPO
torch.set_num_threads(8)  # 设置CPU多线程计算所占用的线程数

Env_mask_dict = {"GoToPositionBonus-v0":"Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "unlockpickupar-v0":"SF_Mask_unlockpickupar-v0_Nill_PPO_n-1_2024-04-18-15-20-52",
        "girdCL-v0": "Mask_girdCL-v0_random_PPO_n8_2024-04-15-10-44-14",
        "GoToR3BlueKeyAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToR3GreenBoxAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToR3PurpleBallAddPositionBonus-v0":"Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "PutNextLocalAR-v0":"SF_Mask_PutNextLocalPositionBonus-v0_Nill_PPO_n-1_2024-04-17-23-07-17",
        "GoToR3GreyKeyAddPositionBonus-v0":"Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToR3RedBoxAddPositionBonus-v0":"Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToDoorOpenR2AR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2AddPositionBonus-v0": "SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2GreyKeyAR-v0": "SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2GreenBoxAR-v0": "SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToR3BlueBallAddPositionBonus-v0": "Mask_GoToPositionBonus-v0_Nill_PPO_n-1_2024-04-16-10-45-41",
        "GoToDoorOpenR2RedBallAR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2RedBallAddPositionBonus-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2BlueBallAR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",
        "GoToDoorOpenR2GreenBallAR-v0":"SF_Mask_GoToDoorOpenR2PositionBonus-v0_Nill_PPO_n-1_2024-04-26-08-57-43",

}

def train(config, log_path, mask_path, mask_flag, mask_threshold):
    make_env = make_vec_env
    env = make_env(config.env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                   vec_env_kwargs=config.vec_env_kwargs, env_kwargs=config.env_kwargs)

    if len(env.observation_space.shape) >=3:
        policy = 'CnnPolicy'
    else:
        policy = 'MlpPolicy'

    mf_model_path = os.path.join(mask_path, "mfmodel")
    from stable_baselines3.dqn.policies import MultiInputPolicy as ActionModel
    mf_model = ActionModel.load(mf_model_path,device=config.device)

    model = MaskPPO(policy, env, tensorboard_log=log_path, mf_model=mf_model, mask_flag=mask_flag,
            mask_threshold=mask_threshold,**config.algorithm.policy, seed=1)

    model.learn(**config.algorithm.learn)
    print("Finished training...")
    if config.save_model:
        print("Saving model...")
        model_path = os.path.join(log_path,"model")
        model.save(model_path)
        # test_mf_model = ActionModel.load(mf_model_path)
    if config.play_model:
        save_path = os.path.join(log_path,"eval.npy")
        mean, std = evaluate_policy_and_save(model, env, save_path=save_path, deterministic=False)
        print("mean:" + str(mean) + " std:" + str(std))


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
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    parser.add_argument('--mask', type=str, default="True")
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    args, extra_args = parser.parse_known_args()
    # get default parameters in this environment and override with extra_args
    config = get_config(args.f)
    config = bcast_config_vals(config)
    pretty(config)
    if args.mask == "True":
        mask_flag = True
    else:
        mask_flag = False
    mask_threshold = args.mask_threshold
    goalstr = ''
    if 'goal' in config.env_kwargs.keys():
        goal = config.env_kwargs.goal
        goalstr = '_Goal'+str(goal)

    experiment_name = "Causal_" + str(config.env_id) + '_' +"mask"+ str(config.algorithm_type) + goalstr + '_' + "Mask" + str(mask_flag) \
            + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    log_path = os.path.join("log", experiment_name)
    mask_name = Env_mask_dict[config.env_id]
    mask_path = os.path.join("log",mask_name)
    train(config, log_path, mask_path, mask_flag, mask_threshold=mask_threshold)

