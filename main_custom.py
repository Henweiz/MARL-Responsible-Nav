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

from maddpg.agent import MADDPGAgent
from log import Logger
from custom.customenv import CustomEnv
from custom.ma_customenv import CustomMAEnv
from custom.grid_world import GWorld

def addDim(arr):
    arr = arr[np.newaxis, :, :]
    return arr


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the initial hyperparameters
    INIT_HP = {
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CUSTOM_ENV": True,
        "ARCH": "mlp",
        "SEED": 42,
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
        "MAX_EPISODES": 3000,
        "TRAIN_STEPS": 200,
        "LOAD_AGENT": False, # Load previous trained agent
        "SAVE_AGENT": True, # Save the agent
        "LOGGING": True,
        "RESUME": False,
        "RESUME_ID": "rghnyha9",
        "WITH_FEAR": False,
        "FeAR_weight": -2.0,
        "FeAR_trajectory_length": 5
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
    env = CustomMAEnv(fear=INIT_HP["WITH_FEAR"])
    obs, info = env.reset(INIT_HP["SEED"])
    #env.num_agents = env.unwrapped.config['controlled_vehicles']

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
            "Train step": INIT_HP["TRAIN_STEPS"],
            "Tau": INIT_HP["TAU"],
            "Population size": INIT_HP["POP_SIZE"]
        }
        if INIT_HP["RESUME"]:
            logger = Logger(filename, config, id=INIT_HP["RESUME_ID"])
        else:
            logger = Logger(filename, config)

    # Configure the multi-agent algo input arguments
    if NET_CONFIG["arch"] == "mlp":
        # obs = obs.flatten()
        state_dim = [obs[agent].flatten().shape for agent in env.agents]
        print(state_dim)
        one_hot = False
    else:
        obs = addDim(obs)
        state_dim = [obs.shape for agent, _ in enumerate(env.agents)]
        print(state_dim)
        one_hot = False
    action_dim = [env.action_space.n for agent, _ in enumerate(env.agents)]
    INIT_HP["DISCRETE_ACTIONS"] = True
    INIT_HP["MAX_ACTION"] = None
    INIT_HP["MIN_ACTION"] = None

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
        print("Agent succesfully loaded!")

    # Define training loop parameters
    episodes = INIT_HP["MAX_EPISODES"]  # Max steps
    learning_delay = 0  # Steps before starting learning

    evo_steps = INIT_HP["TRAIN_STEPS"]  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = agents.pop[0].steps[-1]

    # TRAINING LOOP
    print("Training...")
    pbar = trange(episodes, unit="episode")
    for i in range(episodes):
        steps, pop_episode_scores, fear = agents.train(num_envs, evo_steps, learning_delay, env)
        #fitnesses = agents.evaluate_agent(env, eval_steps)
        mean_scores = [
            np.mean(episode_scores) if len(episode_scores) > 0 else 0.0
            for episode_scores in pop_episode_scores
        ]

        total_steps += steps
        pbar.update(1)
        if INIT_HP["LOGGING"]:
            logger.log(np.mean(mean_scores), agents.total_loss(), steps, total_steps, fear)

        print(f"--- Episode: {i} ---")
        print(f"Steps {steps}")
        print(f"Scores: {mean_scores}")
        print(f"Loss: {agents.total_loss()}")
        print(f'Fear: {fear}')
    

    
    if INIT_HP["SAVE_AGENT"]:
        agents.save_checkpoint(path, filename)
        print("Succesfully saved the agent")

    pbar.close()
    env.close()

