import os
import time
envs = ['girdCL-v0']
config_name = 'gridCL'
alg = 'PPO'
methods = ['random']
n_repeats = 1
total_timesteps = 2000000
log_interval = 10  # (episodes)
#n_redundancies = 30
n_actions_list = [7]
# n_steps = 100

for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            for n_actions in n_actions_list:
                cmd_line = f"python -m min_red.train " \
                           f" --f min_red/config/{config_name} " \
                           f" --algorithm_type {alg} " \
                           f" --algorithm.learn.log_interval {log_interval} " \
                           f" --method {method} " \
                           f" --total_timesteps {total_timesteps}" \
                           f" --env_kwargs.n_actions {n_actions} " \
                           f" --wandb False & "
                print(cmd_line)
                os.system(cmd_line)
                time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
