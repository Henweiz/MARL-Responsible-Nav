# Libraries
import numpy as np
import copy


# General Settings
# DEFAULT_TARGET_SPEEDS = [20, 25, 30] # m/s
# DiscreteMetaAction = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
# dt = 1 / 15



def count_FeasibleActions(agent_i, agent_j, DiscreteMetaAction={0: "SLOWER", 1: "IDLE", 2: "FASTER"}, dt=1/15):
    '''
    Function that counts the number of feasible actions of agent "others" given the movement of agent "ego"

    return: the number of feasible actions of agent others given the movement of agent ego, int
    '''

    count = 0

    for action_j in range(3): # 3 is the number of actions

        copy_agent_j = copy.deepcopy(agent_j)
        copy_agent_j.act(DiscreteMetaAction[action_j])
        is_colliding, _, _ = copy_agent_j._is_colliding(agent_i, dt)

        if is_colliding == False:
            count += 1

        del copy_agent_j

    return count


def cal_FeAR_ij(i, j, before_action_agents, action, MdR, env, before_action_env, epsilon = 1e-6):
    '''
    Function that calculates the FeAR value of agent i on agent j, i.e. FeAR_ij

    before_action_env: object, environment before taking the actions
    epsilon: float, add on denominator to avoid dividing by 0

    return: FrAR_ij, float 
    '''

    if i == j or action[i] == MdR[i]:
        return 0.0

    else:
        agent_j = before_action_agents[j]

        agent_i_Actioni = env.road.vehicles[i]

        action_MdRi = action
        action_MdRi[i] = MdR[i]
        _, _, _, _, info = before_action_env.step(action_MdRi)
        agent_i_MdRi = before_action_env.road.vehicles[i]
            
        n_FeasibleActions_MdRi_j = count_FeasibleActions(agent_i_MdRi, agent_j)
        n_FeasibleActions_Actioni_j = count_FeasibleActions(agent_i_Actioni, agent_j)

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

    before_action_agents = copy.deepcopy(env.road.vehicles)
    MdR = cal_MdR(before_action_agents)

    before_action_env = copy.deepcopy(env)


    next_state, reward, termination, truncation, info = env.step(action)


    reward_array = np.array(list(reward.values()))
    FeAR = np.zeros(n_agents)

    FeAR_weight = -5.0

    for i in range(n_agents):
        FeAR_i = 0.0
        for j in range(n_agents):
            FeAR_i += cal_FeAR_ij(i, j, before_action_agents, info['action'], MdR, env, before_action_env)
        FeAR[i] = FeAR_i

    reward_array += FeAR_weight * FeAR

    del before_action_agents
    del before_action_env



