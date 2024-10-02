import os
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
from agilerl.algorithms.maddpg import MADDPG

import gymnasium as gym
import highway_env

class MADDPGAgent:
    def __init__(self,
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        num_envs,
        device,
        HPO=True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.NET_CONFIG = NET_CONFIG
        self.INIT_HP = INIT_HP
        self.num_envs = num_envs
        self.device = device
        self.pop = []
        self.loss = []
        self.HPO = HPO
        if (self.INIT_HP["POP_SIZE"] <= 1):
            self.HPO = False
        if self.HPO:
            self.pop = create_population("MADDPG",
                            self.state_dim,
                            self.action_dim,
                            self.one_hot,
                            self.NET_CONFIG,
                            self.INIT_HP,
                            population_size=self.INIT_HP["POP_SIZE"],
                            num_envs=self.num_envs,
                            device=self.device)
            
            # Instantiate a tournament selection object (used for HPO)
            self.tournament = TournamentSelection(
                tournament_size=2,  # Tournament selection size
                elitism=True,  # Elitism in tournament selection
                population_size=self.INIT_HP["POP_SIZE"],  # Population size
                eval_loop=1,  # Evaluate using last N fitness scores
            )

            # Instantiate a mutations object (used for HPO)
            self.mutations = Mutations(
                algo="MADDPG",
                no_mutation=0.2,  # Probability of no mutation
                architecture=0.2,  # Probability of architecture mutation
                new_layer_prob=0.2,  # Probability of new layer mutation
                parameters=0.2,  # Probability of parameter mutation
                activation=0,  # Probability of activation function mutation
                rl_hp=0.2,  # Probability of RL hyperparameter mutation
                rl_hp_selection=[
                    "lr",
                    "learn_step",
                    "batch_size",
                ],  # RL hyperparams selected for mutation
                mutation_sd=0.1,  # Mutation strength
                agent_ids=self.INIT_HP["AGENT_IDS"],
                arch=self.NET_CONFIG["arch"],
                rand_seed=1,
                device=self.device,
            )
        else:
            self.pop = []
            agent = MADDPG(
                state_dims=self.state_dim,
                action_dims=self.action_dim,
                one_hot=self.one_hot,
                n_agents=self.INIT_HP["N_AGENTS"],
                agent_ids=self.INIT_HP["AGENT_IDS"],
                O_U_noise=self.INIT_HP["O_U_NOISE"],
                expl_noise=self.INIT_HP["EXPL_NOISE"],
                vect_noise_dim=self.num_envs,
                mean_noise=self.INIT_HP["MEAN_NOISE"],
                theta=self.INIT_HP["THETA"],
                dt=self.INIT_HP["DT"],
                index=0,
                max_action=self.INIT_HP["MAX_ACTION"],
                min_action=self.INIT_HP["MIN_ACTION"],
                net_config=self.NET_CONFIG,
                batch_size=self.INIT_HP["BATCH_SIZE"],
                lr_actor=self.INIT_HP["LR_ACTOR"],
                lr_critic=self.INIT_HP["LR_CRITIC"],
                learn_step=self.INIT_HP["LEARN_STEP"],
                gamma=self.INIT_HP["GAMMA"],
                tau=self.INIT_HP["TAU"],
                discrete_actions=self.INIT_HP["DISCRETE_ACTIONS"],
                device=self.device
            )
            self.pop.append(agent)
            
        # Configure the multi-agent replay buffer
        self.memory = MultiAgentReplayBuffer(
            self.INIT_HP["MEMORY_SIZE"],
            field_names=["state", "action", "reward", "next_state", "done"],
            agent_ids=self.INIT_HP["AGENT_IDS"],
            device=self.device,
        )

    
    # Training loop
    def train(self, num_envs, evo_steps, learning_delay, env):
        pop_episode_scores = []
        total_steps = 0
        for agent in self.pop:  # Loop through population
            state, info = env.reset()
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            if self.INIT_HP["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(s, [-1], [-3])
                    for agent_id, s in state.items()
                }

            for idx_step in range(evo_steps // num_envs):
                #print("Step: ", idx_step)
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )
                state = [x.flatten() for x in state]
                state_dict = self.make_dict(state)
                # Get next action from agent
                cont_actions, discrete_action = agent.get_action(
                    states=state_dict,
                    training=True,
                    agent_mask=agent_mask,
                    env_defined_actions=None,
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                
                # Act in environment
                action_tuple  = tuple(action.values())
                action_tuple = tuple(x.item() for x in action_tuple)
                next_state, reward, termination, truncation, info = env.step(action_tuple)
                
                next_state_dict = self.make_dict([x.flatten() for x in next_state])
                reward_dict = self.make_dict(reward)
                #print(reward_dict)
                termination_dict = self.make_dict(termination)
            
                

                scores += np.sum(np.array(list(reward_dict.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Image processing if necessary for the environment
                if self.INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }
                
                cont_actions = {k: np.squeeze(v) for (k,v) in cont_actions.items()}
                # Save experiences to replay buffer
                self.memory.save_to_memory(
                    state_dict,
                    cont_actions,
                    reward_dict,
                    next_state_dict,
                    termination_dict,
                    is_vectorised=False,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(self.memory) >= agent.batch_size
                        and self.memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = self.memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        loss = agent.learn(experiences)
                        self.loss.append(loss)
                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(self.memory) >= agent.batch_size and self.memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = self.memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        loss = agent.learn(experiences)
                        self.loss.append(loss)

                state = next_state

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination_dict.values())).transpose()
                for i in range(num_envs):
                    if all(term_array) or truncation:
                        
                        reset_noise_indices.append(i)
                            
                        completed_episode_scores.append(scores[i])
                        agent.scores.append(scores[i])
                        scores[i] = 0
                        state, info = env.reset()
                '''
                for idx, d in enumerate((term_array)):
                    if np.any(d) or truncation:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)
                        '''
                
                agent.reset_action_noise(reset_noise_indices)
                

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Tournament selection and population mutation
        if self.HPO:
            _, pop = self.tournament.select(self.pop)
            self.pop = self.mutations.mutation(pop)

        # Update step counter
        for agent in self.pop:
            agent.steps.append(agent.steps[-1])
        
        return total_steps, pop_episode_scores

    def evaluate_agent(self, env, eval_steps=None, eval_loop=1):
        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=self.INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in self.pop
        ]
        return fitnesses

    # Save the most promising agent.
    def save_checkpoint(self, path, filename):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        if self.HPO:
            elite, _ = self.tournament.select(self.pop)
            elite.save_checkpoint(save_path)
        else:
            self.pop[0].save_checkpoint(save_path)
    
    # Load agents.
    def load_checkpoint(self, path):
        for agent in self.pop:
            agent.load_checkpoint(path)
            #agent.steps[-1] = 0
    
    # Check for reached the max steps number accross all agents.
    def reached_max_steps(self, max_steps):
        if max_steps == 0:
            return True
        
        return np.greater([agent.steps[-1] for agent in self.pop], max_steps).all()
    
    # Returns a list of number of steps taken by the agents.
    def agents_steps(self):
        return [agent.steps[-1] for agent in self.pop]
    
    def total_loss(self):
        # Get the last step (last dictionary in the list)
        last_step = self.loss[-1]
        
        # Calculate the total loss
        total_loss = sum(loss for loss, _ in last_step.values())
        return total_loss
    
    def make_dict(self,tuple):
        dict = {}
        for i in range(self.INIT_HP["N_AGENTS"]):
            dict[f'agent_{i}'] = tuple[i]
        return dict
