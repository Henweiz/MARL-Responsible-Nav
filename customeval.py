import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
from maddpg.agent import MADDPGAgent
from custom.customenv import CustomEnv
from custom.ma_customenv import CustomMAEnv

import os
# import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from tqdm import trange
from PIL import Image, ImageDraw
import supersuit as ss

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.algorithms.maddpg import MADDPG
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
import yaml
import util


def make_dict(tuple, n_agents):
    dict = {}
    for i in range(n_agents):
        dict[f'agent_{i}'] = tuple[i]
    return dict

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_episodes = 15
    
    # Load YAML config file
    config_path = "configs\custom.yaml"
    
    with open(config_path, 'r') as file:
        INIT_HP = yaml.safe_load(file)

    # Path & filename to save or load
    path = "./models/custom/multi/no_fear/MADDPG/"
    filename = "MADDPG_no_FeAR.pt"
    
    NET_CONFIG = util.get_net_config(INIT_HP["ARCH"])

    # Number of parallel environment
    num_envs = 1
    seed = 42

    # Define the simple spread environment as a parallel environment
    env, state_dim, action_dim, one_hot = util.create_custom_ma_env(INIT_HP["ARCH"], INIT_HP["WITH_FEAR"], seed)

    # Logger
    # Configure the multi-agent algo input arguments
    INIT_HP["MAX_ACTION"] = None
    INIT_HP["MIN_ACTION"] = None

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    agents = MADDPGAgent(state_dim, action_dim, one_hot, NET_CONFIG, INIT_HP, num_envs, device)

    agents.load_checkpoint(path, filename)
    print("Agent succesfully loaded!")

    crashes = 0
    destination_reached = 0

    for idx in range(eval_episodes):
        state, info = env.reset()
        termination = [False]
        truncation = False


        for _ in range(INIT_HP["TRAIN_STEPS"]):

            if NET_CONFIG["arch"] == "mlp":
                if INIT_HP["CUSTOM_ENV"]:
                    #state = [np.concatenate(state)] 
                    state_dict = {a: v.flatten() for a, v in state.items()}
                else:
                    state = [x.flatten() for x in state]
            else:
                state = state[np.newaxis, :, :]
            
           # state_dict = make_dict(state, 1)
            
            
            # Get next action from agent
            cont_actions, discrete_action = agents.agent.get_action(
                states=state_dict,
                training=False,
            )
            if agents.agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Act in environment
            action_tuple  = tuple(action.values())
            
            if INIT_HP["CUSTOM_ENV"]:
                next_state, reward, termination, truncation, info = env.step(action_tuple)

            
            # termination_dict = make_dict([termination], 1)

            state = next_state

            # Return when the episode is finished
            reset_noise_indices = []
            #term_array = np.array(list(termination.values())).transpose()
            #truncation = np.array(list(truncation.values())).transpose()

            for i in range(num_envs):
                if truncation:
                    reset_noise_indices.append(i)
                    
            # Render
            env.render()

            agents.agent.reset_action_noise(reset_noise_indices)
            crashes += info["agent_crashes"]
            destination_reached += info["apples_caught"]
            if truncation or any(termination):
                break

    print(f"Total destination reached: {destination_reached} across {eval_episodes} episodes")
    print(f"Total crashes: {crashes} across {eval_episodes} episodes")    
    env.close()
    


