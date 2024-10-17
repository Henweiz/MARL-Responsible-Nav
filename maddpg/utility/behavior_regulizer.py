# Libraries
import numpy as np

EPISILON = 1e-6


def cal_speed_reward(env, weight=1):

    agents = env.unwrapped.controlled_vehicles
    rewards = np.zeros(len(agents))

    '''
    for i in range(len(agents)):
        target_speed = agents[i].target_speed
        speed = agents[i].speed
        print("i = ", i)
        print("speed = ", speed)
        print("target_speed = ", target_speed)
        if speed > target_speed + delta:
            rewards[i] = -1 * weight
        elif speed < target_speed - delta:
            rewards[i] = -1 * weight
        else:
            rewards[i] = 1 * weight

    return rewards'''

    for i in range(len(agents)):

        speed = agents[i].speed
        #print("i = ", i)
        #print("speed = ", speed)
        if speed <= 9.0 + EPISILON and speed >= 7.0 - EPISILON:
            rewards[i] = 1 * weight

    return rewards