import numpy as np

np.random.seed(0)
rng = np.random.default_rng(seed=0)

import pprint
import matplotlib.pyplot as plt

import GWorld
import Agent
import Responsibility
import AnalysisOf_FeAR_Sims as FeARUI


N_APPLES = 50



class runGWorld:
    def __init__(self, game_level='GameMap_8'):
        self.scenario_name = game_level

        self.Scenario = GWorld.LoadJsonScenario(json_filename='Scenarios4GameOfFeAR_.json',
                                                scenario_name=self.scenario_name)
        self.N_Agents = self.Scenario['N_Agents']

        self.ActionNames, self.ActionMoves = Agent.DefineActions()

        self.region = np.array(self.Scenario['Map']['Region'])

        # Dictionary of Policies
        self.policy_map = np.zeros(np.shape(self.region), dtype=int)
        self.policies = self.Scenario['Policies']
        print(f'policies = \n{pprint.pformat(self.policies)}')

        # Update PolicyMap
        policy_keys = self.policies.keys()
        print(f'{policy_keys =}')
        for key in policy_keys:
            slicex = self.policies[key]['slicex']
            slicey = self.policies[key]['slicey']
            self.policy_map[slicex, slicey] = key
        print(f'Region =\n {self.region}')
        print(f'policyMap =\n {self.policy_map}')

        # Dictionary of MdRs
        self.mdr_map = np.zeros(np.shape(self.region), dtype=int)
        self.mdrs = self.Scenario['MdRs']
        print(f'mdrs = \n{pprint.pformat(self.mdrs)}')

        # Update MdRMap
        mdrs_keys = self.mdrs.keys()
        print(f'{mdrs_keys =}')
        for key in mdrs_keys:
            slicex = self.mdrs[key]['slicex']
            slicey = self.mdrs[key]['slicey']
            self.mdr_map[slicex, slicey] = key
        print(f'Region =\n {self.region}')
        print(f'mdr_map =\n {self.mdr_map}')

        # Running Simulation Cases !

        # Initialising World Map
        Walls = self.Scenario['Map']['Walls']
        OneWays = self.Scenario['Map']['OneWays']

        self.World = GWorld.GWorld(self.region, Walls=Walls, OneWays=OneWays)  # Initialising GWorld

        self.AgentLocations = []
        for location in self.Scenario['AgentLocations']:
            self.AgentLocations.append(tuple(location))

        # Adding nn Agents at sorted random positions
        if len(self.AgentLocations) < self.N_Agents:
            [locX, locY] = np.where(self.region == 1)
            LocIdxs = rng.choice(locX.shape[0], size=(self.N_Agents - len(self.AgentLocations)), replace=False,
                                 shuffle=False)
            LocIdxs.sort()
            for Idx in LocIdxs:
                self.AgentLocations.append((locX[Idx], locY[Idx]))

        # Adding Agents
        PreviousAgentAdded = True
        for location in self.AgentLocations:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded:
                Ag_i = Agent.Agent()
            PreviousAgentAdded = self.World.AddAgent(Ag_i, location, printStatus=False)

        PreviousAgentAdded = True
        while len(self.World.AgentList) < self.N_Agents:
            # Adding new Agents if Previous Agent was Added to the World
            if PreviousAgentAdded:
                Ag_i = Agent.Agent()
            Loc_i = (np.random.randint(self.region.shape[0]), np.random.randint(self.region.shape[1]))
            PreviousAgentAdded = self.World.AddAgent(Ag_i, Loc_i, printStatus=False)

        # Action Selection for Agents
        self.defaultAction = self.Scenario['defaultAction']
        self.SpecificAction4Agents = self.Scenario['SpecificAction4Agents']

        print('SpecificAction4Agents :', self.SpecificAction4Agents)

        # ------------------------------------------------------------------------------------------------------------------

        # Setting Policy for all Agents

        # Updating Agent Policies in World
        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']

            print(f'{agent_location =}, {agent_policy =}')
            print(f'{agent_stepWeights = }')
            print(f'{agent_directionWeights = }')

            policy = Agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)

        # ------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        # Move de Rigueur

        self.MdR4Agents = []

        for ii in range(len(self.World.AgentList)):
            agent_location = self.World.AgentLocations[ii]
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])
            print(f'{agent_location =}, {agent_mdr =}')

        print('MdR4Agents : ', self.MdR4Agents)
        mdr_string = FeARUI.get_mdr_string(self.MdR4Agents, return_names=True)
        print('MdRs: ', mdr_string)

        self.Action4Agents = self.World.SelectActionsForAll(defaultAction='stay')
        self.FeAR = []
        # self.FeAL = []

        valid_locations = np.transpose(np.where(self.region > 0))
        random_indices = np.random.choice(len(valid_locations), N_APPLES, replace=False)
        self.apples = valid_locations[random_indices]

        fig, self.ax = plt.subplots()

    def gworld_iteration(self):

        # Iterations

        # ------------------------------------------------------------------------------------------------------------------

        self.MdR4Agents = []  # Resetting MdR4Agents

        for ii, agent in enumerate(self.World.AgentList):
            agent_location = self.World.AgentLocations[ii]

            # Updating Policies of Agents
            agent_policy = str(self.policy_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_stepWeights = self.policies[agent_policy]['stepWeights']
            agent_directionWeights = self.policies[agent_policy]['directionWeights']

            print(f'{agent_location =}, {agent_policy =}')
            print(f'{agent_stepWeights = }')
            print(f'{agent_directionWeights = }')

            # Updating MdRs of Agents
            agent_mdr_key = str(self.mdr_map[agent_location[0], agent_location[1]]).zfill(2)
            agent_mdr = self.mdrs[agent_mdr_key]['mdr']
            self.MdR4Agents.append([ii, agent_mdr])
            print(f'{agent_location =}, {agent_mdr =}')
            print('MdR4Agents : ', self.MdR4Agents)
            mdr_string = FeARUI.get_mdr_string(self.MdR4Agents, return_names=True)
            print('MdRs: ', mdr_string)

            policy = Agent.GeneratePolicy(StepWeights=agent_stepWeights, DirectionWeights=agent_directionWeights)
            agent.UpdateActionPolicy(policy)

        # Select Actions for Agents based on defaultAction and SpecificAction4Agents
        self.Action4Agents = self.World.SelectActionsForAll(defaultAction=self.defaultAction,
                                                            InputActionID4Agents=self.SpecificAction4Agents)
        print('SpecificAction Inputs 4Agents :', self.SpecificAction4Agents)
        print('Actions chosen for Agents :', self.Action4Agents)

        # ------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

    def calculate_FeAR(self):
        # Responsibility
        # Calculate Responsibility Metric for the chosen Actions
        self.FeAR, _, _, _, _ = Responsibility.FeAR(self.World, self.Action4Agents, self.MdR4Agents)
        # self.FeAR, _, _, _, _ = Responsibility.FeAR_4_one_actor(self.World, self.Action4Agents, self.MdR4Agents,
        #                                                         actor_ii=0)
        # self.FeAL, _, _, _, _ = Responsibility.FeAL(self.World, self.Action4Agents, self.MdR4Agents)

    def gworld_update(self):
        # Update World with Selected Steps
        agent_crashes, restricted_moves, self.apples, apples_caught = \
            self.World.UpdateGWorld(ActionID4Agents=self.Action4Agents, apples=self.apples, apple_eaters=[0])

        if agent_crashes[0]:
            n_crashes = 1
        elif restricted_moves[0]:
            n_crashes = 1
        else:
            n_crashes = 0

        n_apples_caught = len(apples_caught)

        print(f'{n_apples_caught=}')

        return n_crashes, n_apples_caught



if __name__ == "__main__":
    example = runGWorld()
    example.calculate_FeAR()
    print()
    print("========================================================================================")
    print("The calculated FeAR score in this example is", example.FeAR)



