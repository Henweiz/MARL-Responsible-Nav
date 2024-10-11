import os
#import imageio
import numpy as np
import torch

from tqdm import trange
from PIL import Image, ImageDraw
import gymnasium as gym
import highway_env

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper

from agent import MADDPGAgent
from log import Logger
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network configuration
    '''
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [64, 64],  # Actor hidden size
    }
    '''
    NET_CONFIG = {
        "arch": "cnn",  # Network architecture
        "hidden_size": [64, 64],  # Actor hidden size
        "channel_size": [32, 32],
        "kernel_size": [2, 2],
        "stride_size": [2, 2]
    }
    
   

    # Define the initial hyperparameters
    INIT_HP = {
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": True,
        "BATCH_SIZE": 128,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 800000,  # Max memory buffer size
        "LEARN_STEP": 50,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
        "POP_SIZE": 1,  # Population size, 1 if we do not want to use Hyperparameter Optimization
        "MAX_STEPS": 50000,
        "TRAIN_STEPS": 500,
        "LOAD_AGENT": False, # Load previous trained agent
        "SAVE_AGENT": False, # Save the agent
        "LOGGING": False,
        "RESUME": False,
        "RESUME_ID": "6v7adywb"
        "WITH_FEAR": True
    }
    
    # Path & filename to save or load
    path = "./models/intersection"
    filename = "MADDPG_trained_4agent3000step_wFeAR.pt"

    scores = []
    losses = []
    
    # Number of parallel environment
    num_envs = 1
    config = {
    "id": "intersection-multi-agent-v1",
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 10,
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
    "collision_reward": -10
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
                    "grid_size": [[-32, 32], [-32, 32]],
                    "grid_step": [2, 2],
                    "absolute": False
            }
        },
         "action": {"type": "MultiAgentAction",
               "action_config": {"type": "DiscreteMetaAction","longitudinal": True,
                "lateral": False, "target_speed":[0,4.5,9]}
               },
        "initial_vehicle_count": 10,
        "controlled_vehicles": 4,
        "collision_reward": -10
    }
    

    # Define the simple spread environment as a parallel environment
    env = gym.make("intersection-multi-agent-v1", render_mode=None, config = config2)
    print(env.unwrapped.config)
    #env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    obs, info = env.reset(seed=42)
    env.num_agents = env.unwrapped.config['controlled_vehicles']
    env.agents = [f'agent_{i}' for i in range(env.num_agents)]
    # Logger
    if INIT_HP["LOGGING"]:
        config = {
            "Architecture:": NET_CONFIG["arch"],
            "Hidden size": NET_CONFIG["hidden_size"],
            "Batch size": INIT_HP["BATCH_SIZE"],
            "Exploration noise": INIT_HP["EXPL_NOISE"],
            "LR Actor": INIT_HP["LR_ACTOR"],
            "LR Critic": INIT_HP["LR_CRITIC"],
            "Discount": INIT_HP["GAMMA"],
            "Memory size": INIT_HP["MEMORY_SIZE"],
            "Learn step": INIT_HP["LEARN_STEP"],
            "Tau": INIT_HP["TAU"],
            "Population size": INIT_HP["POP_SIZE"]
        }
        if INIT_HP["RESUME"]:
            logger = Logger(filename, config, id=INIT_HP["RESUME_ID"])
        else:
            logger = Logger(filename, config)

    # Configure the multi-agent algo input arguments
    if NET_CONFIG["arch"] == "mlp":
        print(obs[0].shape)
        state_dim = [(obs[agent].flatten().shape[0], 1) for agent, _ in enumerate(env.agents)]
        print(state_dim)
        one_hot = False
    else:
        state_dim = [obs[agent].shape for agent, _ in enumerate(env.agents)]
        #state_dim = [np.moveaxis(np.zeros(state_dim[agent]), [-1], [-3]).shape for agent, _ in enumerate(env.agents)]
        print(state_dim)
        one_hot = False
    try:
        action_dim = [env.action_space[agent].n for agent, _ in enumerate(env.agents)]
        print(action_dim)
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    agents = MADDPGAgent(state_dim, action_dim, one_hot, NET_CONFIG, INIT_HP, num_envs, device, HPO=True)

    if INIT_HP["LOAD_AGENT"]:
        agents.load_checkpoint(path, filename)

    # Define training loop parameters
    max_steps = INIT_HP["MAX_STEPS"]  # Max steps
    learning_delay = 0  # Steps before starting learning

    evo_steps = INIT_HP["TRAIN_STEPS"]  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = agents.pop[0].steps[-1]
    elite = None

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps - agents.agents_steps()[-1], unit="step")
    while not agents.reached_max_steps(max_steps):
        steps, pop_episode_scores = agents.train(num_envs, evo_steps, learning_delay, env, with_FeAR=INIT_HP["WITH_FEAR"])
        #fitnesses = agents.evaluate_agent(env, eval_steps)
        mean_scores = [
            np.mean(episode_scores) if len(episode_scores) > 0 else 0.0
            for episode_scores in pop_episode_scores
        ]

        total_steps += steps
        pbar.update(steps // len(agents.pop))
        if INIT_HP["LOGGING"]:
            logger.log(np.mean(mean_scores), agents.total_loss(), agents.agents_steps()[0])

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {agents.agents_steps()}")
        print(f"Scores: {mean_scores}")
        print(f"Loss: {agents.total_loss()}")
        #print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        scores.append(mean_scores)
        losses.append(agents.total_loss())

    
    pbar.close()
    env.close()
    if INIT_HP["SAVE_AGENT"]:
        agents.save_checkpoint(path, filename)
        print("Succesfully saved the agent")
    
    fig, (ax1, ax2) = plt.subplots(2)
    x_axis = np.arange(0, INIT_HP["MAX_STEPS"], INIT_HP["MAX_STEPS"]//10)
    fig.suptitle('Training results')
    ax1.plot(scores)
    ax2.plot(losses)
    ax1.set_xticks(x_axis) 
    ax2.set_xticks(x_axis)
    ax1.set_ylabel('Scores')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('steps')
    ax1.grid()
    ax2.grid()
    plt.savefig('./figures/train_results_{}.pdf'.format(filename), format = 'pdf')

