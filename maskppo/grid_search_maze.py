import os
import time
envs = ['girdCL-v0']
config_name = 'gridCL'
alg = 'PPO'
n_repeats = 1
total_timesteps = 500000
log_interval = 1
mask = ["True"]
mask_threshold = 0.5
n_actions = 6  # n_action number and the mask of pre-training n_action are consistent
# n_steps = 512
# n_redundancies = 8

for trials in range(n_repeats):
    for mask_flag in mask:
        for env in envs:
            cmd_line = f"python -m maskppo.train " \
                       f" --f maskppo/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --env_id {env}" \
                       f" --mask {mask_flag} " \
                       f" --mask_threshold {mask_threshold} " \
                       f" --env_kwargs.n_actions {n_actions} " \
                       f" --total_timesteps {total_timesteps} &"
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
# f" --algorithm.policy.n_steps {n_steps} " \

