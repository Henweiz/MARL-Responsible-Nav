import gymnasium as gym
from gymnasium import spaces

import numpy as np
import math
import json
from . import grid_world
from . import custom_agent
from custom.custom_agent import CustomAgent
from . import Responsibility
from custom.grid_world import GWorld
# from stable_baselines3.common.env_checker import check_env
import numpy as np
rng = np.random.default_rng()
N_DISCRETE_ACTIONS = 9

inference = False
SaveImagestoFolder = 'GW_Snaps'
scenario_name = 'GameMap'
MdR4Agents_Default = 0 #Stay
ExhaustiveActions = False
Specific_MdR4Agents = [] #None
Scenario = grid_world.LoadJsonScenario(scenario_name="GameMap")
num_agents = Scenario['N_Agents']

ActionNames, ActionMoves = custom_agent.DefineActions()

print('N_Agents : ',num_agents)


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=-1.0, high=16.0,
                                            shape=(10, 16), dtype=np.float64)
        self.num_agents = 1
        self.prev_distance = []
    


    def step(self, action):
        done = False
        Action4Agents = self.World.SelectActionsForAll(defaultAction = self.defaultAction, InputActionID4Agents = self.SpecificAction4Agents)
#         print('SpecificAction Inputs 4Agents :', self.SpecificAction4Agents)
#         print('Actions chosen for Agents :',Action4Agents)

       # FeAR_vals,ValidMoves_MdR,ValidMoves_action1,ValidityOfMoves_Mdr,ValidityOfMoves_action1 =  Responsibility.FeAR(self.World, Action4Agents, self.MdR4Agents) 
        #FeAL_vals, ValidMoves_moveDeRigueur_FeAL, ValidMoves_action_FeAL, \
        #   ValidityOfMoves_Mdr_FeAL, ValidityOfMoves_action_FeAL =  Responsibility.FeAL(self.World, Action4Agents, self.MdR4Agents)
                
        Action4Agents[0] = (0, action)
#         print("Action4AGENTS: ----- ", Action4Agents)
        # print(self.apples)

        agent_crashes, restricted_moves, apples, apples_caught = self.World.UpdateGWorld(ActionID4Agents=Action4Agents, apples=self.apples, apple_eaters=[0])
        
        reward = 0
        info = {}
        terminated = False
        truncated = False
        distance = []
        
        apple_loc = next(iter(self.apples.values()))

        for loc in self.World.AgentLocations:
            distance.append(manhattan_dist(loc, apple_loc))
        

        # OBSERVATIONS   Shape (10,16)

        # observation[0,0] += 9
        # observation[0,15] += 9
        # observation[9,0] += 9
        # observation[9,15] += 9
        
        
        
        self.episode_length += 1

        if agent_crashes[0]:
            reward -= 20
            terminated = True

        if len(apples_caught) == 1:
            apple_id = apples_caught[0][1]
            # key = list(self.apples)[apple_idx]
            self.apples.pop(apple_id)
            self.apples_eaten += 1
            reward += 10
            if not self.apples:
                truncated = True
        
        #if restricted_moves[0]:
        #    reward -= 0.1
        #else:
        #    reward += 0.1

        if distance[0] < self.prev_distance[0]:
            reward += 0.1

        self.episode_reward += reward    
        observation = self.World.WorldState
        for loc in self.apples.values():
            observation[loc] += 9
        
        if terminated or truncated:
            info = {
                'episode': {
                    'r': self.episode_reward,  # Total reward for the episode
                    'l': self.episode_length    # Length of the episode
                },
                #"agent_mask_env": {restricted_moves}
                "restricted": {restricted_moves[0]}
            }
            #self.episode_reward = 0  # Reset for the next episode
            #self.episode_length = 0
        #inference = True

        self.prev_distance = distance
        
        if inference: print(observation, ", ")
        
            
        
        return observation, reward, terminated, truncated, info
        

    def reset(self, seed=None, options=None):
        self.Region = np.array(Scenario['Map']['Region'])
        self.Walls = Scenario['Map']['Walls']
        self.OneWays = Scenario['Map']['OneWays']
        self.World =  GWorld(self.Region, Walls= self.Walls, OneWays = self.OneWays) # Initialising GWorld from Matrix A
        self.AgentLocations = Scenario['AgentLocations'].copy()
        self.episode_reward = 0  # Reset for the next episode
        self.episode_length = 0


        self.AgentLocations = []
        for location in Scenario['AgentLocations']:
            self.AgentLocations.append(tuple(location))

        if len(self.AgentLocations) < num_agents:
            [locX,locY] = np.where(self.Region==1)

        LocIdxs = rng.choice(locX.shape[0], size=(num_agents-len(self.AgentLocations)), replace=False, shuffle=False)
        LocIdxs.sort()

        for Idx in LocIdxs:
            self.AgentLocations.append((locX[Idx],locY[Idx]))

        # Adding Agents
        PreviousAgentAdded = True
        for location in self.AgentLocations:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded: 
                Ag_i =  CustomAgent()
            PreviousAgentAdded = self.World.AddAgent(Ag_i,location, printStatus=False)

        PreviousAgentAdded = True
        while len(self.World.AgentList) < num_agents:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded: 
                Ag_i =  CustomAgent()
            Loc_i = (np.random.randint(self.Region.shape[0]),np.random.randint(self.Region.shape[1]))
            PreviousAgentAdded = self.World.AddAgent(Ag_i,Loc_i, printStatus=False)


        # Action Selection for Agents

        self.defaultAction = Scenario['defaultAction']
        self.SpecificAction4Agents = Scenario['SpecificAction4Agents']
#         print('SpecificAction4Agents :', self.SpecificAction4Agents)

        # Setting Policy for all Agents
        # The default Step and Direction Weights
        StepWeights=Scenario['StepWeights']
        DirectionWeights=Scenario['DirectionWeights']
        ListOfStepWeights = []
        ListOfDirectionWeights = []

        for ii in range(len(self.World.AgentList)):
            ListOfStepWeights.append(StepWeights)
            ListOfDirectionWeights.append(DirectionWeights)

        # Updating the list of stepweights based on specific weights for agents    
        for agentIDs,stepweights4agents in Scenario['SpecificStepWeights4Agents']:
            for agentID in agentIDs:
                ListOfStepWeights[agentID] = stepweights4agents

        # Updating the list of directionweights based on specific weights for agents            
        for agentIDs,directionweights4agents in Scenario['SpecificDirectionWeights4Agents']:
            for agentID in agentIDs:
                ListOfDirectionWeights[agentID] = directionweights4agents

        # Updating Agent Policies in World   
        for ii,ai in enumerate(self.World.AgentList):
            policy = custom_agent.GeneratePolicy(StepWeights=ListOfStepWeights[ii],DirectionWeights=ListOfDirectionWeights[ii])
            ai.UpdateActionPolicy(policy)

        #------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------

        # Move de Rigueur
    
        # MdR4Agents = [[1,4]]
        self.MdR4Agents = []

        #Setting the MdR for each Agent
        for ii in range(len(self.World.AgentList)):
            self.MdR4Agents.append([ii, MdR4Agents_Default])
            
        for agent,specific_mdr in Specific_MdR4Agents:
            self.MdR4Agents[agent] = [agent, specific_mdr]

        # print('MdR4Agents : ', self.MdR4Agents)



        # Observe (1) location of agent and all other agents, (2) map state, (3) apples/stars
        valid_locations = np.transpose(np.where(self.Region > 0))
        # print(valid_locations)
        # print(valid_locations.shape)
        #self.apples = valid_locations[[0, 14, 54, -1]]
        #self.apples = {"apple_1": (0,0),
        #               "apple_2": (0,15),
        #               "apple_3": (9,0),
        #               "apple_4": (9,15)}
        self.apples = {"apple_0": (9,15)}

            
        # self.apples = [(0,15),(9,0),(9,15),(0,0)]
        # self.apples = self.apples[0]

        # Observations   
        observation = self.World.WorldState
        self.apples_eaten = 0
        for loc in self.apples.values():
            observation[loc] += 9
        # observation[9,15] += 9
        # observation[9,0] += 9
        # observation[0,15] += 9
        # observation[0,0] += 9
        dist = []
        for loc in self.World.AgentLocations:
            dist.append(manhattan_dist(loc, next(iter(self.apples.values()))))
        self.prev_distance = dist



        return (observation, {})  # reward, done, info can't be included
    
#   def render(self, mode='human'):
#     pass
#   def close (self):
#     pass

def manhattan_dist(loc_1, loc_2):
    return sum(abs(a - b) for a, b in zip(loc_1, loc_2))

