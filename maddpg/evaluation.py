import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import MADDPGAgent
from pickle import dump
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
from highway_env.envs.intersection_env import IntersectionEnv
import json



def make_dict(tuple, n_agents):
    dict = {}
    for i in range(n_agents):
        dict[f'agent_{i}'] = tuple[i]
    return dict

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    Args:
    x (float or np.ndarray): Input to the ReLU function.
    Returns:
    float or np.ndarray: Output after applying ReLU.
    """
    return np.maximum(0, x) 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = 10
    # Path & filename to save or load
    #path = "./models/intersection/"
    seed = 66
    #filename = "MADDPG_trained_4agent2000eps{}seed_wFeAR.pt".format(seed)
    filename = "MADDPG_trained_4agent2000eps_woFeAR_best.pt"

    # Number of parallel environment
    num_envs = 1

    config = {
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
               "action_config": {"type": "DiscreteMetaAction"}},
    "initial_vehicle_count": 5,
    "controlled_vehicles": 4,
    "collision_reward": -20,
    "high_speed_reward": 1,
    "arrived_reward": 10,
    "policy_frequency": 1
    }
    
    config2 = {
        "id": "intersection-multi-agent-v1",
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "OccupancyGrid",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20]
                    },
                    "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                    "grid_step": [5, 5],
                    "absolute": False
            }
        },
        "action": {"type": "MultiAgentAction",
               "action_config": {"type": "DiscreteMetaAction"}},
        "initial_vehicle_count": 8,
        "controlled_vehicles": 2
    }


    # Define the simple spread environment as a parallel environment
    #env = gym.make("intersection-multi-agent-v1", render_mode="human", config = config)
    env = IntersectionEnv(render_mode=None)
    env.unwrapped.config.update(config)
    print(env.unwrapped.config)
    #env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    obs, info = env.reset()
    env.num_agents = env.unwrapped.config['controlled_vehicles']
    env.agents = [f'agent_{i}' for i in range(env.num_agents)]
    net = "mlp"

    # Configure the multi-agent algo input arguments
    # Configure the multi-agent algo input arguments
    # Configure the multi-agent algo input arguments
    if net == "mlp":
        print(obs[0].shape)
        state_dim = [(obs[agent].flatten().shape[0], 1) for agent, _ in enumerate(env.agents)]
        print(state_dim)
        one_hot = False
    else:
        state_dim = [obs[agent].shape for agent, _ in enumerate(env.agents)]
        #state_dim = [np.moveaxis(np.zeros(state_dim[agent]), [-1], [-3]).shape for agent, _ in enumerate(env.agents)]
        print(state_dim)
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]
        one_hot = False

    action_dim = [env.action_space[agent].n for agent, _ in enumerate(env.agents)]



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
    #path = os.path.join(path, filename)
    paths = ["/Users/cemlevi/Desktop/marl_nav/MARL-Responsible-Nav/models/intersection/MADDPG_4agent1100eps_wFeAR_3_test3.pt"]
    for path in paths:
        print(f"{path=}")
        agent.load_checkpoint(path)


        #env = RecordVideo(
        #    env, video_folder="intersection_maddpg/videos", episode_trigger=lambda e: True
        #)
        #env.unwrapped.set_record_video_wrapper(env)
        env.unwrapped.config["simulation_frequency"] = 60  # Higher FPS for rendering

        num_crashes = []
        num_arrivals = []
        dist_to_dest = np.zeros((episodes,n_agents))
        average_min_distance = np.zeros((episodes,n_agents))
        
        for videos in range(episodes):
            
            average_min_distance_per_episode = []
            
            done = truncation = False
            #state, info = env.reset(seed=seed)
            state, info = env.reset()
            
            while not (done or truncation):
                #print("step")
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                if net == "mlp":
                    state = [x.flatten() for x in state]
                    state_dict = make_dict(state, n_agents)
                else:
                    state_dict = make_dict(state, n_agents)
                    state_dict = {
                            agent_id: np.moveaxis(s, [-1], [-3])
                            for agent_id, s in state_dict.items()
                    }
                
                    # Get next action from agent
                cont_actions, discrete_action = agent.get_action(
                    states=state_dict,
                    training=True,
                    agent_mask=agent_mask
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                    
                # Act in environment
                action_tuple  = tuple(action.values())
                action_tuple = tuple(x.item() for x in action_tuple)
                
                next_state, _, _, truncation, info = env.step(action_tuple)
                
                reward = info["agents_rewards"]
                termination = info["agents_terminated"]
                termination_bad = info["agents_terminated_bad"]
                termination_good = info["agents_terminated_good"]
                nearest_vehicles = [env.unwrapped.road.close_objects_to(agent, env.PERCEPTION_DISTANCE, count=1, see_behind=False, sort=True, vehicles_only=True) for agent in env.controlled_vehicles]
                min_distance_per_step = [np.linalg.norm(np.array(nearest_vehicle[0].position)) if nearest_vehicle != [] else 250 for nearest_vehicle in nearest_vehicles]
                average_min_distance_per_episode.append(min_distance_per_step)
                
                state = next_state
                
                if all(termination_good) or any(termination_bad):
                    num_crashes.append(sum(termination_bad))
                    num_arrivals.append(sum(termination_good))
                    distances = np.array([relu(25 - vehicle.lane.local_coordinates(vehicle.position)[0]) for vehicle in env.controlled_vehicles ])
                    dist_to_dest[videos] = distances
                    average_min_distance[videos] = np.average(np.array(average_min_distance_per_episode), axis=0)
                    done = True
                
                # Render
                #env.render()
        env.close()
        
        num_crashes = np.array(num_crashes)
        num_arrivals = np.array(num_arrivals)
        avg_crashes = np.average(num_crashes)
        avg_arrivals = np.average(num_arrivals)
        avg_distance_per_vec = np.average(dist_to_dest, axis=0)
        avg_distance = np.average(avg_distance_per_vec)
        avg_min_distance_per_vec = np.average(average_min_distance, axis=0)
        avg_min_distance = np.average(avg_min_distance_per_vec)
        
        stat_dict = {}
        name = path.split("w")[1].split(".")[0]
        stat_dict[name] = {"avg_crashes":avg_crashes,"avg_arrivals": avg_arrivals,
                        "avg_distance_per_vec": {f'agent{i}': x for i,x in enumerate(avg_distance_per_vec)},
                        "avg_distance":avg_distance,"avg_min_distance_per_vec": {f'agent{i}': x for i,x in enumerate(avg_min_distance_per_vec)}, 
                        "avg_min_distance":avg_min_distance}
        
    with open('eval_stats.json', 'w') as convert_file: 
        convert_file.write(json.dumps(stat_dict))


