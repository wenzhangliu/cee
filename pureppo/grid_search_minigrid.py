import os
import time
envs = ['unlockpickupar-v0']
config_name = 'unlockpickupar'
alg = 'PPO'
n_repeats = 1
total_timesteps = 500000
log_interval = 1

# n_steps = 512

for trials in range(n_repeats):
    for env in envs:
        cmd_line = f"python -m pureppo.train " \
                   f" --f pureppo/config/{config_name} " \
                   f" --algorithm_type {alg} " \
                   f" --log_interval {log_interval} " \
                   f" --total_timesteps {total_timesteps} &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
