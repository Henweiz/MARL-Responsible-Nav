# Libraries
import numpy as np



def cal_speed_reward(env, delta=1, weight=0.5):

    agents = env.unwrapped.controlled_vehicles
    rewards = np.zeros(len(agents))

    for i in range(len(agents)):
        target_speed = agents[i].target_speed
        speed = agents[i].speed
        if speed > target_speed + delta:
            rewards[i] = -1 * weight
        elif speed < target_speed - delta:
            rewards[i] = -1 * weight
        else:
            rewards[i] = 1 * weight