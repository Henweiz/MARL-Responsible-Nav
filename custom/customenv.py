import gymnasium as gym
from gymnasium import spaces
import pygame

import pprint
import numpy as np
import math
import random
import json
import pygame
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
MdR4Agents_Default = 0 #Stay
ExhaustiveActions = False
Specific_MdR4Agents = [] #None
Scenario = grid_world.LoadJsonScenario(scenario_name="Level 3")
# Scenario = grid_world.LoadJsonScenario(scenario_name="GameMap")
num_agents = Scenario['N_Agents']

ActionNames, ActionMoves = custom_agent.DefineActions()

print('N_Agents : ',num_agents)
COLOR_MAP = {
    -1: (0, 0, 0),   # Black for inactive cells
    0: (255, 255, 255), # White for blank cells
    1: (255, 215, 0),  # Yellow for the learned agent
    2: (0, 0, 255),  # Blue for other agents
    3: (0, 0, 255),  # Blue for another agent
    9: (255, 0, 0), # Red for the apple
    11: (255, 0 ,0),
    12: (0, 0, 255),  
    13: (0, 0, 255)
}


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, render=False, fear=True):
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
        self.fear = fear
        self.window = None
        self.rendering = render

        if self.rendering:
            self.window_width = 800
            self.window_height = 500
            self.cell_size = 50

            self.clock = pygame.time.Clock()
            pygame.init()
            pygame.display.init()
            self.apple_image = pygame.image.load("apple.png")  # Load the apple image
            self.apple_image = pygame.transform.scale(self.apple_image, (self.cell_size, self.cell_size))


    


    def step(self, action):
        done = False
        self.MdR4Agents = []  # Resetting MdR4Agents
        

        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]

            # Updating Policies of Agents
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']
            if random.random() < 0.25:
                arr = [0, 0, 0, 1]
                agent_directionWeights = random.shuffle(arr)
                
            # Updating MdRs of Agents
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])

            policy = custom_agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)

        
        Action4Agents = self.World.SelectActionsForAll(defaultAction = self.defaultAction, InputActionID4Agents = self.SpecificAction4Agents)
        
        Action4Agents[0] = (0, action[0])
        
        # FeAR_vals,ValidMoves_MdR,ValidMoves_action1,ValidityOfMoves_Mdr,ValidityOfMoves_action1 =  Responsibility.FeAR_4_one_actor(self.World, Action4Agents, self.MdR4Agents) 
        RL_agentID = 0

        
        Action4Agents[RL_agentID] = (RL_agentID, action[0])

        max_distance = 5

        agents = self.close_agents(Action4Agents, RL_agentID, max_distance)

        if len(agents) <= 1 or (not self.fear):
            FeAR_vals = 0.0
        else:
            FeAR_vals,ValidMoves_MdR,ValidMoves_action1,ValidityOfMoves_Mdr,ValidityOfMoves_action1 =  Responsibility.FeAR_4_one_actor(self.World, agents, self.MdR4Agents, RL_agentID) 
        #FeAL_vals, ValidMoves_moveDeRigueur_FeAL, ValidMoves_action_FeAL, \
        #   ValidityOfMoves_Mdr_FeAL, ValidityOfMoves_action_FeAL =  Responsibility.FeAL(self.World, Action4Agents, self.MdR4Agents)

        agent_crashes, restricted_moves, apples, apples_caught = self.World.UpdateGWorld(ActionID4Agents=Action4Agents, apples=self.apples, apple_eaters=[0])
        
        reward = 0
        info = {}
        terminated = False
        truncated = False
        distance = []
        
        apple_loc = next(iter(self.apples.values()))

        for loc in self.World.AgentLocations:
            distance.append(manhattan_dist(loc, apple_loc))
                
        self.episode_length += 1

        if agent_crashes[RL_agentID]:
            reward -= 10
            terminated = True

        if len(apples_caught) == 1:
            apple_id = apples_caught[0][1]
            # key = list(self.apples)[apple_idx]
            self.apples.pop(apple_id)
            self.apples_eaten += 1
            reward += 20
            if not self.apples:
                truncated = True
        
        #if restricted_moves[0]:
        #    reward -= 0.1
        #else:
        #    reward += 0.1

        if distance[RL_agentID] < self.prev_distance[RL_agentID]:
            reward += 0.1

        self.episode_reward += reward    
        observation = self.World.WorldState
        for loc in self.apples.values():
            observation[loc] += 9
        info = {
            'episode': {
                'r': self.episode_reward,  # Total reward for the episode
                'l': self.episode_length    # Length of the episode
            },
            "restricted": restricted_moves[0],
            "fear": np.sum(FeAR_vals)
        }    

            #self.episode_reward = 0  # Reset for the next episode
            #self.episode_length = 0
        #inference = True

        self.prev_distance = distance
        self.observation = observation
        if inference: print(observation, ", ")

            
        self.observation = observation
        return observation, [reward], [terminated], truncated, info
        

    def reset(self, seed=None, options=None):
        self.Region = np.array(Scenario['Map']['Region'])
        self.Walls = Scenario['Map']['Walls']
        self.OneWays = Scenario['Map']['OneWays']
        self.World =  GWorld(self.Region, Walls= self.Walls, OneWays = self.OneWays) # Initialising GWorld from Matrix A
        self.AgentLocations = Scenario['AgentLocations'].copy()
        self.episode_reward = 0  # Reset for the next episode
        self.episode_length = 0


        # Dictionary of Policies
        self.policy_map = np.zeros(np.shape(self.Region), dtype=int)
        self.policies = Scenario['Policies']
        # print(f'policies = \n{pprint.pformat(self.policies)}')

        # Update PolicyMap
        policy_keys = self.policies.keys()
        # print(f'{policy_keys =}')
        for key in policy_keys:
            slicex = self.policies[key]['slicex']
            slicey = self.policies[key]['slicey']
            self.policy_map[slicex, slicey] = key
        # print(f'Region =\n {self.Region}')
        # print(f'policyMap =\n {self.policy_map}')

        # Dictionary of MdRs
        self.mdr_map = np.zeros(np.shape(self.Region), dtype=int)
        self.mdrs = Scenario['MdRs']
        # print(f'mdrs = \n{pprint.pformat(self.mdrs)}')

        # Update MdRMap
        mdrs_keys = self.mdrs.keys()
        # print(f'{mdrs_keys =}')
        for key in mdrs_keys:
            slicex = self.mdrs[key]['slicex']
            slicey = self.mdrs[key]['slicey']
            self.mdr_map[slicex, slicey] = key
        # print(f'Region =\n {self.Region}')
        # print(f'mdr_map =\n {self.mdr_map}')



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

#         # Setting Policy for all Agents
#         # The default Step and Direction Weights
#         StepWeights=Scenario['StepWeights']
#         DirectionWeights=Scenario['DirectionWeights']
#         ListOfStepWeights = []
#         ListOfDirectionWeights = []

#         for ii in range(len(self.World.AgentList)):
#             ListOfStepWeights.append(StepWeights)
#             ListOfDirectionWeights.append(DirectionWeights)

#         # Updating the list of stepweights based on specific weights for agents    
#         for agentIDs,stepweights4agents in Scenario['SpecificStepWeights4Agents']:
#             for agentID in agentIDs:
#                 ListOfStepWeights[agentID] = stepweights4agents

#         # Updating the list of directionweights based on specific weights for agents            
#         for agentIDs,directionweights4agents in Scenario['SpecificDirectionWeights4Agents']:
#             for agentID in agentIDs:
#                 ListOfDirectionWeights[agentID] = directionweights4agents

#         # Updating Agent Policies in World   
#         for ii,ai in enumerate(self.World.AgentList):
#             policy = custom_agent.GeneratePolicy(StepWeights=ListOfStepWeights[ii],DirectionWeights=ListOfDirectionWeights[ii])
#             ai.UpdateActionPolicy(policy)

        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']

            policy = custom_agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)


        #------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------

        # Move de Rigueur
    
        # MdR4Agents = [[1,4]]
        self.MdR4Agents = []

        # #Setting the MdR for each Agent
        # for ii in range(len(self.World.AgentList)):
        #     self.MdR4Agents.append([ii, MdR4Agents_Default])
            
        # for agent,specific_mdr in Specific_MdR4Agents:
        #     self.MdR4Agents[agent] = [agent, specific_mdr]

        # print('MdR4Agents : ', self.MdR4Agents)

        for ii in range(len(self.World.AgentList)):
            agent_location = self.World.AgentLocations[ii]
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])
            # print(f'{agent_location =}, {agent_mdr =}')


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


        self.observation = observation
        return (observation, {})  # reward, done, info can't be included
    
    def render(self, mode='human'):
        return self.render_frame()
    
    def render_frame(self):
        if self.window is None:
            self.window = pygame.display.set_mode((self.window_width, self.window_height))

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))

        for i, row in enumerate(self.observation):
            for j, cell_value in enumerate(row):
                color = COLOR_MAP.get(int(cell_value), (255, 255, 255))
                if color == (255, 0, 0):  
                    pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                    canvas.blit(self.apple_image, (j * self.cell_size, i * self.cell_size))  # Draw apple image
                elif color == (255, 215, 0):
                    pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                    pygame.draw.circle(canvas, color, (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2), self.cell_size // 2)
                elif color == (0, 0, 255):
                    pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                    pygame.draw.circle(canvas, color, (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2), self.cell_size // 2)
                else:
                    pygame.draw.rect(canvas, color, pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(10)

    def close (self):
        if self.window is not None and self.rendering:
            pygame.display.quit()
            pygame.quit()

    def close_agents(self, agents, agent_i, max_dist):
        agent_list = []
        for agentID, agent_action in agents:
            if agentID == agent_i:
                agent_list.append((agentID, agent_action))
                continue
            if manhattan_dist(self.World.AgentLocations[agent_i], self.World.AgentLocations[agentID]) <= max_dist:
                agent_list.append((agentID, agent_action))
        return agent_list


def manhattan_dist(loc_1, loc_2):
    return sum(abs(a - b) for a, b in zip(loc_1, loc_2))

