import os
#import imageio
import numpy as np
import torch
import yaml

from tqdm import trange
from PIL import Image, ImageDraw
import gymnasium as gym
import highway_env
import util

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper

from maddpg.agent import MADDPGAgent
from log import Logger
from custom.customenv import CustomEnv
from custom.ma_customenv import CustomMAEnv
from custom.grid_world import GWorld


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the initial hyperparameters
    LOGGING = {
        "LOAD_AGENT": False, # Load previous trained agent
        "SAVE_AGENT": False, # Save the agent
        "LOGGING": False,
        "RESUME": False,
        "RESUME_ID": "3dze0erh"
    }

    # Load YAML config file
    config_path = "configs\custom.yaml"
    
    with open(config_path, 'r') as file:
        INIT_HP = yaml.safe_load(file)

    INIT_HP.update(LOGGING)

    # Path & filename to save or load
    path = "./models/custom/multi/no_fear/MADDPG"
    filename = "MADDPG_no_FeAR.pt"

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
    
    # Logger
    if INIT_HP["LOGGING"]:
        config = INIT_HP
        if INIT_HP["RESUME"]:
            logger = Logger(filename, config, id=INIT_HP["RESUME_ID"])
        else:
            logger = Logger(filename, config)

    # Number of parallel environment
    num_envs = 1

    # Initialize the environment
    if INIT_HP["CUSTOM_ENV"]:
        env, state_dim, action_dim, one_hot = util.create_custom_ma_env(INIT_HP["ARCH"], INIT_HP["WITH_FEAR"], INIT_HP["SEED"])
    else:
        env_config = {
            "id": "intersection-multi-agent-v1",
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "absolute": True
                }
            },
            "action": {"type": "MultiAgentAction",
                    "action_config": {"type": "DiscreteMetaAction","longitudinal": True,
                        "lateral": False, "target_speed":[0,4.5,9]}
                    },
            "initial_vehicle_count": 10,
            "controlled_vehicles": 4,
            "collision_reward": -5,
            "high_speed_reward": 1,
            "arrived_reward": 1
        }
        env, state_dim, action_dim, one_hot = util.create_intersection_env(INIT_HP["ARCH"], env_config, INIT_HP["WITH_FEAR"], INIT_HP["SEED"])

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Idk why, but I have to assign it here instead of in the yaml file. Otherwise it breaks
    INIT_HP["MAX_ACTION"] = None
    INIT_HP["MIN_ACTION"] = None

    agents = MADDPGAgent(state_dim, action_dim, one_hot, NET_CONFIG, INIT_HP, num_envs, device)

    if INIT_HP["LOAD_AGENT"]:
        agents.load_checkpoint(path, filename)
        print("Agent succesfully loaded!")

    # Define training loop parameters
    episodes = INIT_HP["MAX_EPISODES"]  # Max episodes
    learning_delay = 0  # Steps before starting learning

    env_steps = INIT_HP["TRAIN_STEPS"] 

    total_steps = agents.agents_steps()

    # TRAINING LOOP
    print("Training...")
    pbar = trange(episodes, unit="episode")
    for i in range(episodes):
        steps, returns, fear = agents.train(num_envs, env_steps, learning_delay, env)

        total_steps += steps
        pbar.update(1)
        if INIT_HP["LOGGING"]:
            logger.log(returns, agents.total_loss(), steps, total_steps, fear)

        print(f"--- Episode: {i} ---")
        print(f"Steps {steps}")
        print(f"Return: {returns}")
        print(f"Loss: {agents.total_loss()}")
        print(f'Fear: {fear}')
    

    
    if INIT_HP["SAVE_AGENT"]:
        agents.save_checkpoint(path, filename)
        print("Succesfully saved the agent")

    pbar.close()
    env.close()