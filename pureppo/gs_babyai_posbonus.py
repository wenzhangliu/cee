import os
import time

all_envs =  ['GoToRedBallGreyPositionBonus-v0','GoToR2PositionBonus-v0',"GoToDoorOpenR2PositionBonus-v0", "GoToR3BlueKeyAddPositionBonus-v0",
             "GoToR3GreyKeyAddPositionBonus-v0","GoToR3GreenBoxAddPositionBonus-v0","GoToR3RedBoxAddPositionBonus-v0",
             "PutNextLocalAR-v0","GoToDoorOpenR2GreyKeyAR-v0","GoToDoorOpenR2GreenBoxAR-v0","GoToDoorOpenR2GreyKeyAddPositionBonus-v0",
             "GoToDoorOpenR2RedBallAR-v0","GoToDoorOpenR2RedBallAddPositionBonus-v0","GoToDoorOpenR2PurpleBallAddPositionBonus-v0",
             "GoToDoorOpenR2GreenBallAR-v0" ]
#index= -1

# choose index
index = 14
if index > -1 :
    envs = [all_envs[index]]
config_name = 'babyaiposbonus'

alg = 'PPO'
n_repeats = 1
total_timesteps = 2000000
log_interval = 1

for trials in range(n_repeats):
    for env in envs:
        cmd_line = f"python -m pureppo.train " \
                   f" --f pureppo/config/{config_name} " \
                   f" --env_id {env}" \
                   f" --log_interval {log_interval}"\
                   f" --algorithm_type {alg} " \
                   f" --total_timesteps {total_timesteps} &"
        print(cmd_line)
        os.system(cmd_line)
        time.sleep(10)
#f" --algorithm.policy.n_steps {n_steps} " \
