import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import MADDPGAgent

import highway_env

import os
import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from tqdm import trange
from PIL import Image, ImageDraw
import supersuit as ss

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.algorithms.maddpg import MADDPG
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper

from agent import MADDPGAgent

def make_dict(tuple, n_agents):
    dict = {}
    for i in range(n_agents):
        dict[f'agent_{i}'] = tuple[i]
    return dict

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Path & filename to save or load
    path = "./models/intersection"
    filename = "MADDPG_intersection_trained_agent.pt"

    # Number of parallel environment
    num_envs = 1

    config = {
    "id": "intersection-multi-agent-v0",
    "import_module": "highway_env",
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
            "absolute": True,
            "order": "shuffled"
        }
    },
    "initial_vehicle_count": 3,
    "controlled_vehicles": 2
    }

    # Define the simple spread environment as a parallel environment
    env = gym.make("intersection-multi-agent-v0", render_mode="human", config = config)
    print(env.unwrapped.config)
    #env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    obs, info = env.reset(seed=42)
    env.num_agents = env.unwrapped.config['controlled_vehicles']
    env.agents = [f'agent_{i}' for i in range(env.num_agents)]

    # Configure the multi-agent algo input arguments
    state_dim = [(25,1) for agent in env.agents]
    one_hot = False
    action_dim = [9 for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents
    
        # Instantiate an MADDPG object
    agent = MADDPG(
        state_dim,
        action_dim,
        one_hot,
        n_agents,
        agent_ids,
        max_action=None,
        min_action=None,
        discrete_actions=True,
        device=device,
    )

    # Load the previous trained agent.
    path = "./models/intersection/MADDPG_intersection_trained_agent.pt"
    agent.load_checkpoint(path)


    #env = RecordVideo(
    #    env, video_folder="intersection_maddpg/videos", episode_trigger=lambda e: True
    #)
    #env.unwrapped.set_record_video_wrapper(env)
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering

    for videos in range(5):
        done = truncated = False
        state, info = env.reset()
        while not (done or truncated):
            print("step")
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            state = [x.flatten() for x in state]
            state_dict = make_dict(state, n_agents)
                # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                states=state_dict,
                training=False,
                agent_mask=agent_mask
            )
            if agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

                
            # Act in environment
            action_tuple  = tuple(action.values())
            action_tuple = tuple(x.item() for x in action_tuple)
            next_state, reward, termination, truncation, info = env.step(action_tuple)
            state = next_state
            if all(termination):
                done = True
            
            # Render
            env.render()
    env.close()


