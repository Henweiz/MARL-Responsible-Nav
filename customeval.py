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


def make_dict(tuple, n_agents):
    dict = {}
    for i in range(n_agents):
        dict[f'agent_{i}'] = tuple[i]
    return dict

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Define the initial hyperparameters
    INIT_HP = {
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CUSTOM_ENV": True,
        "ARCH": "mlp",
        "SEED": 31,
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 128,  # Batch size
        "O_U_NOISE": False,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.15,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.98,  # Discount factor
        "MEMORY_SIZE": 200000,  # Max memory buffer size
        "LEARN_STEP": 10,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 1,  # Policy frequnecy
        "POP_SIZE": 1,  # Population size, 1 if we do not want to use Hyperparameter Optimization
        "MAX_EPISODES": 1,
        "TRAIN_STEPS": 50,
        "LOAD_AGENT": True, # Load previous trained agent
        "SAVE_AGENT": False, # Save the agent
        "LOGGING": False,
        "RESUME": False,
        "RESUME_ID": "nhdokura"
    }

    # Path & filename to save or load
    path = "./models/custom/multi/MADDPG"
    filename = "MADDPG_2.pt"

    # Define the network configuration
    if INIT_HP["ARCH"] == "mlp":
        print("Using MLP architecture")
        NET_CONFIG = {
            "arch": "mlp",  # Network architecture
            "hidden_size": [128, 128],  # Actor hidden size
        }
    else:
        print("Using CNN architecture")
        NET_CONFIG = {
            "arch": "cnn",  # Network architecture
            "hidden_size": [128, 128],  # Actor hidden size
            "channel_size": [32, 64],
            "kernel_size": [2, 2],
            "stride_size": [2, 2]
        }

    # Number of parallel environment
    num_envs = 1

    # Define the simple spread environment as a parallel environment
    #env = gym.make("intersection-multi-agent-v1", render_mode=None, config = config2)
    #print(env.unwrapped.config)
    #env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    # env = CustomEnv(render=True, fear=True)
    env = CustomMAEnv(render=True, fear=False, seed=INIT_HP["SEED"])
    obs, info = env.reset()
    #env.num_agents = env.unwrapped.config['controlled_vehicles']

    env.agents = [f'agent_{i}' for i in range(env.num_agents)]
    # Logger
    # Configure the multi-agent algo input arguments
    if NET_CONFIG["arch"] == "mlp":
        # obs = obs.flatten()
        state_dim = [obs[agent].flatten().shape for agent in env.agents]
        print(state_dim)
        one_hot = False
    action_dim = [env.action_space.n for agent, _ in enumerate(env.agents)]
    print(action_dim)
    INIT_HP["DISCRETE_ACTIONS"] = True
    INIT_HP["MAX_ACTION"] = None
    INIT_HP["MIN_ACTION"] = None

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    agent = MADDPGAgent(state_dim, action_dim, one_hot, NET_CONFIG, INIT_HP, num_envs, device, HPO=True)

    if INIT_HP["LOAD_AGENT"]:
        agent.load_checkpoint(path, filename)
        print("Agent succesfully loaded!")

    

    for videos in range(30):
        state, info = env.reset()
        termination = [False]
        truncation = False
        for _ in range(INIT_HP["TRAIN_STEPS"]):
            # print("step")

            if NET_CONFIG["arch"] == "mlp":
                if INIT_HP["CUSTOM_ENV"]:
                    # state = [np.concatenate(state)] 
                    state_dict = {a: v.flatten() for a, v in state.items()}
                else:
                    state = [x.flatten() for x in state]
            else:
                state = state[np.newaxis, :, :]
            
                state_dict = make_dict(state, 1)
            
            
            # Get next action from agent
            cont_actions, discrete_action = agent.pop[0].get_action(
                states=state_dict,
                training=False,
            )
            if agent.pop[0].discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Act in environment
            action_tuple  = tuple(action.values())
            # print(action_tuple)
            
            if INIT_HP["CUSTOM_ENV"]:
                next_state, reward, termination, truncation, info = env.step(action_tuple)
            if INIT_HP["CUSTOM_ENV"] and NET_CONFIG["arch"] == "mlp":
                next_state_dict = {a: v.flatten() for a, v in next_state.items()}
            
            # termination_dict = make_dict([termination], 1)

            state = next_state

            # Return when the episode is finished
            reset_noise_indices = []
            term_array = np.array(list(termination.values())).transpose()
            truncation = np.array(list(truncation.values())).transpose()

            for i in range(num_envs):
                if all(term_array) or all(truncation):
                    reset_noise_indices.append(i)
                    
            # Render
            env.render()

            agent.pop[0].reset_action_noise(reset_noise_indices)
            if all(term_array) or all(truncation):
                break
        
    env.close()


