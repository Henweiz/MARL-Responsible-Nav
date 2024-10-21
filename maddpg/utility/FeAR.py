# Libraries
import numpy as np
import copy

from concurrent.futures import ProcessPoolExecutor



def count_FeasibleActions(i, j, action, env, trajectory_length, DiscreteMetaAction={0: "SLOWER", 1: "IDLE", 2: "FASTER"}):
    '''
    Function that counts the number of feasible actions of agent "others" given the movement of agent "ego"

    return: the number of feasible actions of agent others given the movement of agent ego, int
    '''
    
    if type(env.unwrapped.road.vehicles[j]) != type(env.unwrapped.controlled_vehicles[0]):

        environment = copy.deepcopy(env)

        for _ in range(trajectory_length):
            _, _, _, _, info = environment.step(tuple(action))

        agent_j = environment.unwrapped.road.vehicles[j]

        if agent_j.crashed == True:
            del environment
            return 0
        else:
            del environment
            return 1

    elif type(env.unwrapped.road.vehicles[j]) == type(env.unwrapped.controlled_vehicles[0]):

        count = 0
        index_j = env.unwrapped.controlled_vehicles.index(env.unwrapped.road.vehicles[j])

        for action_j in range(len(DiscreteMetaAction)):

            action[index_j] = action_j

            environment = copy.deepcopy(env)

            for _ in range(trajectory_length):
                _, _, _, _, info = environment.step(tuple(action))
                
            agent_j = environment.unwrapped.road.vehicles[j]

            if agent_j.crashed == False:
                del environment
                count += 1
            else:
                del environment

        return count

    else:
        raise Exception("Unknown type of vehicle.")

    return


def cal_FeAR_ij(i, j, action, MdR, before_action_env, trajectory_length, epsilon = 1e-6):
    '''
    Function that calculates the FeAR value of agent i on agent j, i.e. FeAR_ij

    before_action_env: object, environment before taking the actions
    epsilon: float, add on denominator to avoid dividing by 0

    return: FrAR_ij, float 
    '''

    action = list(action)
    agent_i = before_action_env.unwrapped.controlled_vehicles[i]
    agent_j = before_action_env.unwrapped.road.vehicles[j]

    if agent_i == agent_j or action[i] == MdR[i] or np.linalg.norm(np.array(agent_i.position) - np.array(agent_j.position)) > (10 * (trajectory_length + 1)):
        return 0.0

    else:
        action_MdRi = copy.deepcopy(action)
        action_MdRi[i] = MdR[i]
            
        n_FeasibleActions_MdRi_j = count_FeasibleActions(i, j, action_MdRi, before_action_env, trajectory_length)
        n_FeasibleActions_Actioni_j = count_FeasibleActions(i, j, action, before_action_env, trajectory_length)

        #print("n_FeasibleActions_MdRi_j=", n_FeasibleActions_MdRi_j)
        #print("n_FeasibleActions_Actioni_j", n_FeasibleActions_Actioni_j)

        FeAR_ij = np.clip( ( (n_FeasibleActions_MdRi_j - n_FeasibleActions_Actioni_j) / (n_FeasibleActions_MdRi_j + epsilon) ), -1, 1)

        del action_MdRi

        return FeAR_ij


def cal_MdR(agents, env, DEFAULT_TARGET_SPEEDS=[0, 4.5, 9]):
    '''
    Function that calculate the MdR of all agents in the environment

    DEFAULT_TARGET_SPEEDS = [0, 4.5, 9] m/s
    DiscreteMetaAction = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}

    return: list of intergers, len = n_agents
    '''

    MdR = []

    for agent in agents:

        MdR_i = 2 if agent.speed <= (DEFAULT_TARGET_SPEEDS[0] + 1.0) else (0 if agent.speed >= (DEFAULT_TARGET_SPEEDS[2] - 2.0) else 1)

        nearest_vehicle =  env.unwrapped.road.close_objects_to(agent, env.PERCEPTION_DISTANCE, count=1, see_behind=False, sort=True, vehicles_only=True)
        if nearest_vehicle != [] and np.linalg.norm(np.array(nearest_vehicle[0].position) - np.array(agent.position)) < 10:
            MdR_i = 0
        
        MdR.append(MdR_i)

    return MdR


def cal_FeAR(env, action_tuple, INIT_HP):
    '''
    Calculate FeAR
    '''
    
    #Get MdRs
    MdR = cal_MdR(env.unwrapped.controlled_vehicles, env)
    #print("MdR=", MdR)
                    
    before_action_env = env                
    FeAR = np.zeros(shape = (INIT_HP["N_AGENTS"],len(env.unwrapped.road.vehicles)))
    
    if len(env.unwrapped.controlled_vehicles) == 4:
        
        before_action_env0 = copy.deepcopy(before_action_env)
        before_action_env1 = copy.deepcopy(before_action_env)
        before_action_env2 = copy.deepcopy(before_action_env)
        before_action_env3 = copy.deepcopy(before_action_env)
       
        process_items = list(zip(range(4), [before_action_env0, before_action_env1, before_action_env2, before_action_env3]))
        
        for j in range(len(env.unwrapped.road.vehicles) - 1):

            with ProcessPoolExecutor(max_workers=4) as executor:
                results = [executor.submit(cal_FeAR_ij, process_items[i][0], j, action_tuple, MdR, process_items[i][1], INIT_HP["FeAR_trajectory_length"]) \
                            for i in range(len(process_items))]
                #print("results = ", results)

                for i, res in zip(range(4), results):
                    #print("FeAR[{},j]={}".format(i, res.result()))
                    FeAR[i, j] = res.result()
           
        del before_action_env0
        del before_action_env1
        del before_action_env2
        del before_action_env3
    
    else:
        for i in range(INIT_HP["N_AGENTS"]):
            for j in range(len(env.unwrapped.road.vehicles) - 1):
                FeAR[i,j] = cal_FeAR_ij(i, j, action_tuple, MdR, before_action_env, INIT_HP["FeAR_trajectory_length"])

    print(f"{FeAR}=")
   

    del before_action_env

    return FeAR