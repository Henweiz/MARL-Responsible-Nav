import numpy as np
from custom.ma_customenv import CustomMAEnv

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