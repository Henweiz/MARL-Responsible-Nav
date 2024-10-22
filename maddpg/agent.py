import os
import numpy as np
import torch
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3

import pickle
import gymnasium as gym
import highway_env
import copy
#from utility.FeAR import count_FeasibleActions,cal_FeAR_ij,cal_MdR, cal_FeAR
from maddpg.utility.FeAR import cal_FeAR
from maddpg.utility.behavior_regulizer import cal_speed_reward



class MADDPGAgent:
    def __init__(self,
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        num_envs,
        device):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.NET_CONFIG = NET_CONFIG
        self.INIT_HP = INIT_HP
        self.num_envs = num_envs
        self.device = device
        self.loss = []
        self.agent = MADDPG(
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
            
        # Configure the multi-agent replay buffer
        self.memory = MultiAgentReplayBuffer(
            self.INIT_HP["MEMORY_SIZE"],
            field_names=["state", "action", "reward", "next_state", "done"],
            agent_ids=self.INIT_HP["AGENT_IDS"],
            device=self.device,
        )

    
    # Training loop
    def train(self, num_envs, env_steps, learning_delay, env, with_FeAR=True):
        total_steps = 0
        state, info = env.reset()
        scores = np.zeros(num_envs)
        completed_episode_scores = []
        steps = 0
        fear_score = 0
        
        for idx_step in range(env_steps):
            #Update observation to fit into our NN
            if self.NET_CONFIG["arch"] == "mlp":
                if self.INIT_HP["CUSTOM_ENV"]:
                    state_dict = {a: v.flatten() for a, v in state.items()}
                    # state = [val.flatten() for val in state.values()]
                else:
                    state = [x.flatten() for x in state]
            else:
                state = state[np.newaxis, :, :]        
                state_dict = self.make_dict(state)
            
            if self.NET_CONFIG["arch"] == "cnn":
                for i in range(self.INIT_HP["N_AGENTS"]):
                    obs = state_dict[f'agent_{i}']
                    state_dict[f'agent_{i}'] = obs[np.newaxis, :, :]
            
            if self.INIT_HP["CHANNELS_LAST"]:
                state_dict = {
                    agent_id: np.moveaxis(s, [-1], [-3])
                    for agent_id, s in state_dict.items()
                }

            # Get action mask from the environment
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            # Get next action from agent
            cont_actions, discrete_action = self.agent.get_action(
                states=state_dict,
                training=True,
                agent_mask=agent_mask
            )
            if self.agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            if with_FeAR:
                # Act in environment
                action_tuple  = tuple(action.values())
                action_tuple = tuple(x.item() for x in action_tuple)

                FeAR_weight = self.INIT_HP["FeAR_weight"]

                if self.INIT_HP["CUSTOM_ENV"]:
                    next_state, reward, termination, truncation, info = env.step(action_tuple)
                    FeAR = info["fear"]
                    
                    reward = {a: FeAR_weight * FeAR[a] + reward[a] for a in reward.keys()}
                    reward = np.array(list(reward.values()))
                    FeAR = np.sum(list(FeAR.values()))

                else:
                    FeAR = cal_FeAR(env, action_tuple, self.INIT_HP)
                    next_state, reward, termination, truncation, info = env.step(action_tuple)

                    speed_reward = cal_speed_reward(env)
                    reward = np.array(reward) + speed_reward + FeAR_weight * np.sum(FeAR, axis=1)
                    
                fear_score += FeAR
                reward = tuple(reward)

            else:
                # Act in environment
                action_tuple  = tuple(action.values())
                action_tuple = tuple(x.item() for x in action_tuple)

                next_state, reward, termination, truncation, info = env.step(action_tuple)

                if not self.INIT_HP["CUSTOM_ENV"]:
                    speed_reward = cal_speed_reward(env)
                    print("speed_reward = ", speed_reward)
                    reward = np.array(reward) + speed_reward
                reward = tuple(reward)



            if self.INIT_HP["CUSTOM_ENV"] and self.NET_CONFIG["arch"] == "mlp":
                next_state_dict = {a: v.flatten() for a, v in next_state.items()}
            
            # Flatten next state observation
            if self.NET_CONFIG["arch"] == "mlp" and not self.INIT_HP["CUSTOM_ENV"]:
                next_state_dict = self.make_dict([x.flatten() for x in next_state])
            # else:
            #     next_state_dict = self.make_dict(next_state[np.newaxis, :, :])
            #     next_state_dict = {
            #         agent_id: ns[np.newaxis, :, :]
            #         for agent_id, ns in next_state_dict.items()
            #     }

            reward_dict = self.make_dict(reward)

            scores += np.sum(np.array(list(reward_dict.values())).transpose(), axis=-1)

            total_steps += num_envs
            steps += num_envs

            # Image processing if necessary for the environment
            if self.INIT_HP["CHANNELS_LAST"]:
                next_state_dict = {
                    agent_id: np.moveaxis(ns, [-1], [-3])
                    for agent_id, ns in next_state_dict.items()
                }
            

            termination_dict = termination
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
            if self.agent.learn_step > num_envs:
                learn_step = self.agent.learn_step // num_envs
                if (
                    idx_step % learn_step == 0
                    and len(self.memory) >= self.agent.batch_size
                    and self.memory.counter > learning_delay
                ):
                    # Sample replay buffer
                    experiences = self.memory.sample(self.agent.batch_size)
                    # Learn according to agent's RL algorithm
                    loss = self.agent.learn(experiences)
                    self.loss.append(loss)
                    #print("loss=", loss)
            # Handle num_envs > learn step; learn multiple times per step in env
            elif (
                len(self.memory) >= self.agent.batch_size and self.memory.counter > learning_delay
            ):
                for _ in range(num_envs // self.agent.learn_step):
                    # Sample replay buffer
                    experiences = self.memory.sample(self.agent.batch_size)
                    # Learn according to agent's RL algorithm
                    loss = self.agent.learn(experiences)
                    self.loss.append(loss)
                    #print("loss=", loss)

            state = next_state
            print(info)

            # Return when the episode is finished
            reset_noise_indices = []
            term_array = np.array(list(termination_dict.values())).transpose()
            trunc_array = np.array(list(truncation.values())).transpose()
            for i in range(num_envs):
                if all(term_array) or all(trunc_array):
                    reset_noise_indices.append(i)
                        
                    completed_episode_scores.append(scores[i])
                    self.agent.scores.append(scores[i])

            
            self.agent.reset_action_noise(reset_noise_indices)
            if all(term_array) or all(trunc_array):
                break
            if idx_step == (env_steps -1):
                completed_episode_scores.append(scores[0])
                self.agent.scores.append(scores[0])
                
            self.agent.steps[-1] += steps

        # Update step counter
        self.agent.steps.append(self.agent.steps[-1])
        
        return total_steps, completed_episode_scores[-1], fear_score

    # Save the most promising agent.
    def save_checkpoint(self, path, filename):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        self.agent.save_checkpoint(save_path)
        memory_path = os.path.join(path, 'memory.pkl')
        with open(memory_path, 'wb') as f:
            pickle.dump(self.memory, f)
    
    # Load agents.
    def load_checkpoint(self, path, filename):
        load_path = os.path.join(path, filename)
        memory_path = os.path.join(path, "memory.pkl")
        self.agent.load_checkpoint(load_path)
            #agent.steps[-1] = 0
        with open(memory_path, 'rb') as f:
            self.memory = pickle.load(f)
    
    # Check for reached the max steps number accross all agents.
    def reached_max_steps(self, max_steps):
        if max_steps == 0:
            return True
        
        return np.greater(self.agent.steps[-1], max_steps).all()
    
    # Returns a list of number of steps taken by the agents.
    def agents_steps(self):
        return self.agent.steps[-1]
    
    def total_loss(self):
        # Get the last step (last dictionary in the list)
        try:
            last_step = self.loss[-1]
        # No learning happened yet, so we return 0    
        except:
            return 0
        
        # Calculate the total loss
        total_loss = sum(loss for loss, _ in last_step.values())
        return total_loss
    
    # Make a dictionary with agent ID and their corresponding tuple
    def make_dict(self, tuple):
        dict = {}
        for i in range(self.INIT_HP["N_AGENTS"]):
            dict[f'agent_{i}'] = tuple[i]
        return dict

