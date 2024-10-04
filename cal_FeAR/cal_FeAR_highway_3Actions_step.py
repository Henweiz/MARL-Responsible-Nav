# Libraries
import numpy as np
import copy


# General Settings
# DEFAULT_TARGET_SPEEDS = [20, 25, 30] # m/s
# DiscreteMetaAction = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
# dt = 1 / 15



def count_FeasibleActions(i, j, action, env, DiscreteMetaAction={0: "SLOWER", 1: "IDLE", 2: "FASTER"}, dt=1/15):
    '''
    Function that counts the number of feasible actions of agent "others" given the movement of agent "ego"

    return: the number of feasible actions of agent others given the movement of agent ego, int
    '''

    count = 0

    for action_j in range(len(DiscreteMetaAction)):

        action[j] = action_j
        environment = copy.deepcopy(env)

        _, _, _, _, info = environment.step(action)

        agent_i = environment.road.vehicles[i]
        agent_j = environment.road.vehicles[j]

        is_colliding, _, _ = agent_j._is_colliding(agent_i, dt)

        if is_colliding == False:
            count += 1

        del environment

    return count


def cal_FeAR_ij(i, j, action, MdR, before_action_env, epsilon = 1e-6):
    '''
    Function that calculates the FeAR value of agent i on agent j, i.e. FeAR_ij

    before_action_env: object, environment before taking the actions
    epsilon: float, add on denominator to avoid dividing by 0

    return: FrAR_ij, float 
    '''

    if i == j or action[i] == MdR[i]:
        return 0.0

    else:
        action_MdRi = copy.deepcopy(action)
        action_MdRi[i] = MdR[i]
            
        n_FeasibleActions_MdRi_j = count_FeasibleActions(i, j, action_MdRi, before_action_env)
        n_FeasibleActions_Actioni_j = count_FeasibleActions(i, j, action, before_action_env)

        FeAR_ij = np.clip( ( (n_FeasibleActions_MdRi_j - n_FeasibleActions_Actioni_j) / (n_FeasibleActions_MdRi_j + epsilon) ), -1, 1)


        return FeAR_ij


def cal_MdR(agents):
    '''
    Function that calculate the MdR of all agents in the environment

    DEFAULT_TARGET_SPEEDS = [20, 25, 30] m/s
    DiscreteMetaAction = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}

    return: list of intergers, len = n_agents
    '''

    MdR = []
    for agent in agents:
        MdR_i = 2 if agent.speed <= 23 else (0 if agent.speed >= 27 else 1)
        MdR.append(MdR_i)

    return MdR



if  __name__ == "__main__":

    MdR = cal_MdR(env.road.vehicles)

    before_action_env = copy.deepcopy(env)


    next_state, reward, termination, truncation, info = env.step(action)


    FeAR = np.zeros(n_agents)

    FeAR_weight = -5.0

    for i in range(n_agents): # number of intelligent vehicles
        FeAR_i = 0.0
        for j in range(n_agents): # number of total vehicles on the road
            FeAR_i += cal_FeAR_ij(i, j, info['action'], MdR, before_action_env)
        FeAR[i] = FeAR_i

    reward += FeAR_weight * FeAR

    del before_action_env



