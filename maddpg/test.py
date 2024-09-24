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


# Define function to return image
def _label_with_episode_number(frame, episode_num):
    if frame is None:
        print("No frame found!")
        return None
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Define the simple spread environment
    env = simple_spread_v3.parallel_env(continuous_actions=False, max_cycles=30, render_mode="human")

    channels_last = False  # Needed for environments that use images as observations, like atari
    if channels_last:
        # Environment processing for image based observations
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
    env = PettingZooVectorizationParallelWrapper(env, n_envs=1)
    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]

    if channels_last:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

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
        max_action,
        min_action,
        discrete_actions,
        device=device,
    )

    # Load the previous trained agent.
    path = "./models/spread/MADDPG_trained_agent.pt"
    agent.load_checkpoint(path)

    eval_loop = 1  # Number of evaluation episodes
    max_steps = 100000

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    
    # Test loop for inference
    for ep in range(eval_loop):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            if channels_last:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                    for agent_id, s in state.items()
                }
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                state,
                training=False,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
            )
            if agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Save the frame for this step and append to frames list
            env.render()
            #frames.append(_label_with_episode_number(frame, episode_num=ep))

            # Take action in environment
            state, reward, termination, truncation, info = env.step(action)

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward for each agent
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    
    
    env.close()

    # Save the gif to specified path
    #gif_path = "./videos/"
    #os.makedirs(gif_path, exist_ok=True)
    #imageio.mimwrite(
     #   os.path.join("./videos/", "simple_spread.gif"), frames, duration=20
    #)
