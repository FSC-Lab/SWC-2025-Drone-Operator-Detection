import os
import csv

import imageio
import numpy as np
import torch
from agilerl.algorithms.matd3 import MATD3
from PIL import Image, ImageDraw

from Environment import search_env_v1

N = 80  # Number of Timesteps


# Define function to return image
def _label_with_episode_number(frame, episode_num):
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = search_env_v1.parallel_env(
        max_cycles=N, continuous_actions=True, render_mode="rgb_array"
    )
    env.reset()
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

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Instantiate an MADDPG object
    matd3 = MATD3(
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

    # Load the saved algorithm into the MADDPG object
    path = "./Models/MATD3/MATD3_trained_agent.pt"
    matd3.loadCheckpoint(path)

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = N  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards
    terminal_rewards = []  # List to store end of episode agent rewards

    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = matd3.getAction(
                state,
                epsilon=0,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
            )
            if matd3.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

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

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep+1}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    terminal_rewards = [reward_list[-1] for reward_list in indi_agent_rewards.items()]
    env.close()

    # Save the gif to specified path
    gif_path = "./Videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./Videos/", "MATD3_Search.gif"), frames, duration=10
    )
    
    # Save the algorithm fitness benchmarks
    data_path = "./Benchmarks/Test_Benchmarks/"
    os.makedirs(data_path, exist_ok=True)
    file_name = 'MATD3_Benchmarks_Search.csv'
    file_path = os.path.join(data_path, file_name)

    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Episode'] + [f'{agent_id} Terminal Reward' for agent_id in agent_ids]
        csvwriter.writerow(header)

        for i in range(episodes):
            csvwriter.writerow([i + 1] + [terminal_rewards[a][i] for a in range(len(agent_ids))])
