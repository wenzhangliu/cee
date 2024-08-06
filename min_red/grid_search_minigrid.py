import os
import time
envs = ['unlockpickupar-v0']
config_name = 'unlockpickupar'
alg = 'PPO'
methods = ['Nill']
n_repeats = 1
abs_thresh = True
total_timesteps = 5000000
log_interval = 10

for trials in range(n_repeats):
    for env in envs:
        for method in methods:
            cmd_line = f"python -m min_red.train " \
                       f" --f min_red/config/{config_name} " \
                       f" --algorithm_type {alg} " \
                       f" --algorithm.learn.log_interval {log_interval} " \
                       f" --algorithm.policy.absolute_threshold {abs_thresh} " \
                       f" --method {method} " \
                       f" --total_timesteps {total_timesteps}" \
                       f" --wandb False & "
            print(cmd_line)
            os.system(cmd_line)
            time.sleep(10)
