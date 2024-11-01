from pettingzoo import ParallelEnv, AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box
import functools
import random
import copy
import numpy as np
import pygame

from . import grid_world
from . import custom_agent
from custom.custom_agent import CustomAgent
from . import Responsibility
from custom.grid_world import GWorld



N_DISCRETE_ACTIONS = 9
N_INTELLIGENT_AGENTS = 2
four_corners = [(0,0), (9,0), (0,15), (9,15)]

inference = False

scenario_name = 'Level 3'
MdR4Agents_Default = 0 #Stay
Specific_MdR4Agents = [] #None
Scenario =  grid_world.LoadJsonScenario(scenario_name="Level 3")
total_num_agents = Scenario['N_Agents']
# assert num_agents == N_AGENTS

ActionNames, ActionMoves = custom_agent.DefineActions()

print('N_Agents : ',total_num_agents)
COLOR_MAP = {
    -2: (0, 0, 0),   # Black for inactive cells
    0: (255, 255, 255), # White for blank cells
    6: (255, 215, 0),  # Yellow for the learned agent
    10: (0, 0, 255),  # Blue for other agents
    # 6: (0, 255, 0),  # Green for another agent
    9: (255, 0, 0), # Gold for the apple
    15: (255, 215 ,0),
    19: (0, 0, 255)
    # 12: (0, 255, 0),


# FOR 2 INTELLIGENT AGENTS
    # -2: (0, 0, 0),   # Black for inactive cells
    # 0: (255, 255, 255), # White for blank cells
    # 2: (255, 215, 0),  # Yellow for the learned agent
    # 4: (0, 0, 255),  # Blue for other agents
    # 6: (0, 255, 0),  # Green for another agent
    # 8: (0, 255, 0),
    # 10: (255, 0, 0), # Gold for the apple
    # 12: (255, 215, 0),
    # 14: (0, 0, 255),  
    # 16: (0, 255, 0),
    # 18: (0, 255, 0)

# FOR 3 INTELLIGENT AGENTS
    # -3: (0, 0, 0),   # Black for inactive cells
    # 0: (255, 255, 255), # White for blank cells
    # 3: (255, 215, 0),  # Yellow for the learned agent
    # 6: (0, 0, 255),  # Blue for other agents
    # 9: (0, 255, 0),  # Green for another agent
    # 10: (255, 0, 0), # Gold for the apple
    # 13: (255, 215 ,0),
    # 16: (0, 0, 255),  
    # 19: (0, 255, 0)
}

class CustomMAEnv(ParallelEnv):
    

    def __init__(self, render=False, fear=True, seed=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        # super().__init__()
        self.possible_agents = ["agent_" + str(r) for r in range(N_INTELLIGENT_AGENTS)]
        self.agents = self.possible_agents[:]
        self.action_space = Discrete(N_DISCRETE_ACTIONS)
        self._observation_spaces = {
            agent: Box(low=-1.0, high=16.0, shape=(10, 16), dtype=np.float64) for agent in self.possible_agents
        }
        self.rendering = render
        self.fear = fear
        self.window = None
        self.rng = np.random.default_rng(seed=seed)

        if self.rendering:
            self.window_width = 800
            self.window_height = 500
            self.cell_size = 50

            self.clock = pygame.time.Clock()
            pygame.init()
            pygame.display.init()
            self.apple_image = pygame.image.load("apple.png")  # Load the apple image
            self.apple_image = pygame.transform.scale(self.apple_image, (self.cell_size, self.cell_size))


    # I do not know if I should keep this line:
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-1.0, high=16.0, shape=(10, 16), dtype=np.float64)

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(N_DISCRETE_ACTIONS)
    
    def render(self):
        """
        Renders the environment. In human mode, it can to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.window is None:
            self.window = pygame.display.set_mode((self.window_width, self.window_height))



        total_obs = np.zeros((10, 16))
        # Add each array element-wise
        for ob in self.observations.values():
            total_obs += ob

        # print(total_obs)

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))
        circles = [(255, 215, 0), (0, 0, 255), (0, 255, 0)]
        observation = self.observations[self.agents[0]]
        # for i, row in enumerate(observation):
        for i, row in enumerate(total_obs):
            for j, cell_value in enumerate(row):
                color = COLOR_MAP.get(int(cell_value), (255, 255, 255))
                if color == (255, 0, 0):  
                    pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                    canvas.blit(self.apple_image, (j * self.cell_size, i * self.cell_size))  # Draw apple image
                elif color in circles:
                    pygame.draw.rect(canvas, (255, 255, 255), pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                    pygame.draw.circle(canvas, color, (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2), self.cell_size // 2)
                else:
                    pygame.draw.rect(canvas, color, pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(5)
    
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncation
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.setup_env()
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncation = {agent: False for agent in self.agents}



        action_mask_dict = {a: self.get_action_mask(a) for a in self.agents}
        info = {"fear": 0.0}
        info.update(action_mask_dict)

        # self.state = observations
        self.observations = {agent: self.observation for agent in self.agents}
        observation = self.World.WorldState.copy()
        # for loc in self.apples.values():
        #     observation[loc] += 9
        for agent in self.agents:
            agent_id = int(agent[-1])
            if agent_id in [int(id[-1]) for id in self.apples.keys()]:
                for id, loc in self.apples.items():
                    if int(id[-1]) == agent_id:
                        observation[loc] += 9
                        break
            self.observations[agent] = observation
            observation = self.World.WorldState.copy()
        
        self.num_moves = 0
        self.prev_distance = {agent: None for agent in self.agents}
        # self.apples_eaten = 0

        return self.observations, info
    
    def step(self, actions):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncation
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if not actions:
            # self.agents = []
            return {}, {}, {}, {}, {}
        
        self.setup_step()
        self.num_moves += 1
        self.rewards = {agent: 0 for agent in self.agents}
        apple_rewarded = 0
        crash_count = 0
        
        for (idx, action) in enumerate(actions):
            # idx = int(agent[-1])
            # print(idx, ', ', action)
            self.Action4Agents[idx] = (idx, action)


        FeAR_dict = {a: 0.0 for a in self.agents}
        if self.fear:
            for agent in self.agents:
                agent_id = int(agent[-1])
                max_distance = 5
                agents = self.close_agents(self.Action4Agents, agent_id, max_distance)
                FeAR_vals, _, _, _, _ =  Responsibility.FeAR_4_one_actor(self.World, agents, self.MdR4Agents, agent_id) 
                FeAR_dict[agent] = np.sum(FeAR_vals)

        agent_crashes, restricted_moves, apples, apples_caught = self.World.UpdateGWorld(ActionID4Agents=self.Action4Agents, apples=self.apples, apple_eaters=[i for i in range(N_INTELLIGENT_AGENTS)])
        # print('step')
        #print(apples_caught)
        #print(np.sum(restricted_moves[:1]))
        for app in apples_caught:
            apple_key = app[1]
            agent_id = app[0]
            agent_key = f'agent_{agent_id}'
            apple_id = int(apple_key[-1])
            # print(apple_id)
            if apple_id == agent_id:
                if self.apples.__contains__(apple_key):
                    # fourth_loc = next(iter(set(four_corners) - set(self.apples.values())), None)
                    self.apples.pop(apple_key)
                    # self.apples[apple_key] = fourth_loc
                    # self.apples_eaten += 1            
                    self.rewards[agent_key] += 20
                    apple_rewarded += 1
                    if not self.apples: # or self.apples_eaten == 3:
                        for a in self.agents:            
                            self.rewards[a] += 20
                        self.truncation = {a: True for a in self.agents}
            
        distance = {}
        for i, agent in enumerate(self.agents):
            # if restricted_moves[i]:
            #     self.rewards[agent] -= 0.05
            if agent_crashes[i]:
                self.rewards[agent] -= 10    
                crash_count += 1
                self.truncation = {a: True for a in self.agents}
                self.terminations[agent] = True
                # self.agents = []
            closest_apple = None
            for apple_key, apple_loc in self.apples.items():
                apple_id = int(apple_key[-1])
                # if manhattan_dist(self.World.AgentLocations[i], apple_loc) < closest_apple:
                if apple_id == i:
                    closest_apple = manhattan_dist(self.World.AgentLocations[i], apple_loc)

            distance[agent] = closest_apple
            max_distance = 25
            if self.prev_distance[agent] is not None and distance[agent] is not None:
                if self.prev_distance[agent] > distance[agent]:
                    #normalized_distance = max(0, max_distance - closest_apple) / (2 * max_distance)
                    #self.rewards[agent] += normalized_distance
                    self.rewards[agent] += 1
            
        self.prev_distance = distance
        observation = self.World.WorldState.copy()
        # for loc in self.apples.values():
        #     observation[loc] += 9
        for agent in self.agents:
            agent_id = int(agent[-1])
            if agent_id in [int(id[-1]) for id in self.apples.keys()]:
                for id, loc in self.apples.items():
                    if int(id[-1]) == agent_id:
                        observation[loc] += 9
                        break
            
            all_ids = [1, 2, 3, 4]
            my_id = [agent_id+1]
            other_ids = iter(set(all_ids) - set(my_id))
            self.observations[agent] = observation

            for id in other_ids:
                self.observations[agent] = np.where(self.observations[agent] == id, 5, self.observations[agent])
            self.observations[agent] = np.where(self.observations[agent] == agent_id+1, 1, self.observations[agent])
            observation = self.World.WorldState.copy()

        action_mask_dict = {a: self.get_action_mask(a) for a in self.agents}
        info = {"fear": FeAR_dict,
                 "agent_crashes": crash_count,
                 "apples_caught": apple_rewarded
            }
        info.update(action_mask_dict)




        return self.observations, self.rewards, self.terminations, self.truncation, info



    def setup_env(self):
        self.Region = np.array(Scenario['Map']['Region'])
        self.Walls = Scenario['Map']['Walls']
        self.OneWays = Scenario['Map']['OneWays']
        self.World =  GWorld(self.Region, Walls= self.Walls, OneWays = self.OneWays) # Initialising GWorld from Matrix A
        self.AgentLocations = Scenario['AgentLocations'].copy()
        
        # Dictionary of Policies
        self.policy_map = np.zeros(np.shape(self.Region), dtype=int)
        self.policies = Scenario['Policies']

        # Update PolicyMap
        policy_keys = self.policies.keys()
        for key in policy_keys:
            slicex = self.policies[key]['slicex']
            slicey = self.policies[key]['slicey']
            self.policy_map[slicex, slicey] = key

        # Dictionary of MdRs
        self.mdr_map = np.zeros(np.shape(self.Region), dtype=int)
        self.mdrs = Scenario['MdRs']

        # Update MdRMap
        mdrs_keys = self.mdrs.keys()
        for key in mdrs_keys:
            slicex = self.mdrs[key]['slicex']
            slicey = self.mdrs[key]['slicey']
            self.mdr_map[slicex, slicey] = key

        
        self.AgentLocations = []
        for location in Scenario['AgentLocations']:
            self.AgentLocations.append(tuple(location))

        if len(self.AgentLocations) < total_num_agents:
            [locX,locY] = np.where(self.Region==1)

        
        LocIdxs = self.rng.choice(locX.shape[0], size=(total_num_agents-len(self.AgentLocations)), replace=False, shuffle=False)
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
        while len(self.World.AgentList) < total_num_agents:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded: 
                Ag_i =  CustomAgent()
            Loc_i = (np.random.randint(self.Region.shape[0]),np.random.randint(self.Region.shape[1]))
            PreviousAgentAdded = self.World.AddAgent(Ag_i,Loc_i, printStatus=False)


        # Action Selection for Agents
        self.defaultAction = Scenario['defaultAction']
        self.SpecificAction4Agents = Scenario['SpecificAction4Agents']

        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']

            policy = custom_agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)
        
        self.MdR4Agents = []

        for ii in range(len(self.World.AgentList)):
            agent_location = self.World.AgentLocations[ii]
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])
            # print(f'{agent_location =}, {agent_mdr =}')

        self.valid_locations = np.transpose(np.where(self.Region > 0))
        self.apples = {"apple_0": (9,0), "apple_1": (5, 10)} #, "apple_2": (0,15)}  Middle apple: (5, 10) 


        observation = self.World.WorldState
        # self.apples_eaten = 0


        self.observation = observation


    def setup_step(self):
        
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

        self.Action4Agents = self.World.SelectActionsForAll(defaultAction= self.defaultAction, InputActionID4Agents= self.SpecificAction4Agents)



    def close_agents(self, agents, agent_i, max_dist):
        agent_list = []
        for agentID, agent_action in agents:
            if agentID == agent_i:
                agent_list.append((agentID, agent_action))
                continue
            if manhattan_dist(self.World.AgentLocations[agent_i], self.World.AgentLocations[agentID]) <= max_dist:
                agent_list.append((agentID, agent_action))
        return agent_list
    

    def get_action_mask(self, agent):
        environment_map = self.Region
        agent_index = int(agent[-1])
        action_mask = np.ones(9, dtype=np.int8)  # Start with all actions allowed
        x, y = self.World.AgentLocations[agent_index]  # Get the agent's current position

        # Action1: Up 1
        if x - 1 < 0 or environment_map[x - 1][y] == 0:
            action_mask[1] = 0  # Block movement up by 1

        # Action2: Down 1
        if x + 1 >= len(environment_map) or environment_map[x + 1][y] == 0:
            action_mask[2] = 0  # Block movement down by 1

        # Action3: Left 1
        if y - 1 < 0 or environment_map[x][y - 1] == 0:
            action_mask[3] = 0  # Block movement left by 1

        # Action4: Right 1
        if y + 1 >= len(environment_map[0]) or environment_map[x][y + 1] == 0:
            action_mask[4] = 0  # Block movement right by 1

        # Action5: Up 2
        if x - 2 < 0 or environment_map[x - 2][y] == 0:
            action_mask[5] = 0  # Block movement up by 2

        # Action6: Down 2
        if x + 2 >= len(environment_map) or environment_map[x + 2][y] == 0:
            action_mask[6] = 0  # Block movement down by 2

        # Action7: Left 2
        if y - 2 < 0 or environment_map[x][y - 2] == 0:
            action_mask[7] = 0  # Block movement left by 2

        # Action8: Right 2
        if y + 2 >= len(environment_map[0]) or environment_map[x][y + 2] == 0:
            action_mask[8] = 0  # Block movement right by 2

        # Action0: Stay is always valid, so no need to mask it
        return {"action_mask": action_mask}

        


def manhattan_dist(loc_1, loc_2):
    return sum(abs(a - b) for a, b in zip(loc_1, loc_2))


