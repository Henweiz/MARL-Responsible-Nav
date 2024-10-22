import numpy as np
from custom.ma_customenv import CustomMAEnv
from highway_env.envs.intersection_env import IntersectionEnv
import yaml

def addDim(arr):
    arr = arr[np.newaxis, :, :]
    return arr

def custom_state_dim(obs, arch, env_agents):
    if arch == "mlp":
        state_dim = [obs[agent].flatten().shape for agent in env_agents]
        one_hot = False
    else:
        obs = addDim(obs)
        state_dim = [obs.shape for agent, _ in enumerate(env_agents)]
        one_hot = False
    
    return state_dim, one_hot

def create_custom_ma_env(arch, fear=True, seed=42):
    env = CustomMAEnv(fear=fear, seed=seed)
    obs, _ = env.reset()

    env.agents = [f'agent_{i}' for i in range(env.num_agents)]

    state_dim, one_hot = custom_state_dim(obs, arch, env.agents)
    action_dim = [env.action_space.n for agent, _ in enumerate(env.agents)]

    return env, state_dim, action_dim, one_hot

def create_intersection_env(arch, config, fear=True, seed=42):
    env = IntersectionEnv(config=config)
    obs, info = env.reset(seed=66)
    env.num_agents = env.unwrapped.config['controlled_vehicles']
    env.agents = [f'agent_{i}' for i in range(env.num_agents)]

    if arch == "mlp":
        state_dim = [(obs[agent].flatten().shape[0], 1) for agent, _ in enumerate(env.agents)]
    else:
        state_dim = [obs[agent].shape for agent, _ in enumerate(env.agents)]
    one_hot = False
    action_dim = [env.action_space[agent].n for agent, _ in enumerate(env.agents)]

    return env, state_dim, action_dim, one_hot

def get_net_config(arch):
    # Define the network configuration
    if arch == "mlp":
        print("Using MLP architecture")
        path = "configs\mlp.yaml"
    else:
        print("Using CNN architecture")
        path = "configs\cnn.yaml"
     
    # Load YAML net config file
    with open(path, 'r') as file:
        NET_CONFIG = yaml.safe_load(file)    
    
    return NET_CONFIG
