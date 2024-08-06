import os
import time
envs = ['girdCL-v0']
config_name = 'gridCL'
alg = 'PPO'
n_repeats = 1
total_timesteps = 1000000
log_interval = 1
n_actions = 8
for trials in range(n_repeats):
    for env in envs:
        cmd_line = f"python -m pureppo.train " \
                   f" --f pureppo/config/{config_name} " \
                   f" --algorithm_type {alg} " \
                   f" --env_kwargs.n_actions {n_actions} " \
                   f" --total_timesteps {total_timesteps} &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(10)
# f" --algorithm.policy.n_steps {n_steps} " \

