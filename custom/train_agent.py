import gymnasium as gym
import os
import numpy as np
from customenv import CustomEnv
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

log_dir = "logs/PPO"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

print("Hello")
env = CustomEnv()
# check_env(env)

# episodes = 50

# for episode in range(episodes):
# 	term = False
# 	trunc = False
# 	obs = env.reset()
# 	while not term or not trunc:
# 		random_action = env.action_space.sample()
# 		print("action", random_action)
# 		obs, reward, term, trunc, info = env.step(random_action)
# 		print('reward', reward)
print(env.observation_space.shape)
print(env.action_space.n)
		
'''
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)


TIMESTEPS = 10000
for i in range (1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
'''