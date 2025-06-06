import os
import csv

import numpy as np
import torch
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation
from tqdm import trange

from Environment import search_env_v1

# EPISODE PARAMETER SCALES
E = 2250  # Number of episodes
N = 80    # Number of timesteps
P = 4     # Population Size of EHPO (keep at 4)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL MATD3 Demo =====")
    print(device)

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": P,
        "ALGO": "MATD3",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 128,  # Batch size
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.01,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": int(0.08 * E * N * P),  # Max memory buffer size
        "LEARN_STEP": 5,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
    }

    # Define the basedef environment as a parallel environment
    env = search_env_v1.parallel_env(continuous_actions=True)
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
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]
        
    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = initialPopulation(
        INIT_HP["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,
    )  # Evaluate using last N fitness scores

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_episodes = E  # Total episodes (default: 6000)
    max_steps = N  # Maximum steps to take in each episode
    epsilon = 1.0  # Starting epsilon value
    eps_end = 0.1  # Final epsilon value
    eps_decay = 0.995  # Epsilon decay
    evo_epochs = 20  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent
    
    # Store fitness Values
    fitness_scores = []
    fitness_avg_scores = []

    # Training loop
    for idx_epi in trange(max_episodes):
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            agent_reward = {agent_id: 0 for agent_id in env.agents}

            for _ in range(max_steps):
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )

                # Get next action from agent
                cont_actions, discrete_action = agent.getAction(
                    state, epsilon, agent_mask, env_defined_actions
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                next_state, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment
                
                # Save experiences to replay buffer
                memory.save2memory(state, cont_actions, reward, next_state, termination)

                # Collect the reward
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    experiences = memory.sample(
                        agent.batch_size
                    )  # Sample replay buffer
                    agent.learn(experiences)  # Learn according to agent's RL algorithm
                    
                state = next_state

                # Stop episode if any agents have terminated
                if any(truncation.values()) or any(termination.values()):
                    break

            # Save the total episode reward
            score = sum(agent_reward.values())
            agent.scores.append(score)

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]}'
            )
            
            fitness_scores.append([fitness for fitness in fitnesses])
            fitness_avg_scores.append([np.mean(agent.fitness[-100:]) for agent in pop])

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

    # Save the trained algorithm
    path = "./Models/MATD3"
    filename = "MATD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.saveCheckpoint(save_path)
    
    # Save the algorithm fitness benchmarks
    data_path = "./Benchmarks/Algorithm_Benchmarks/"
    os.makedirs(data_path, exist_ok=True)
    file_name = 'MATD3_Benchmarks.csv'
    file_path = os.path.join(data_path, file_name)

    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Episode'] + [f'Algorithm {j+1} Fitness' for j in range(P)] + [f'Algorithm {j+1} 100 Fitness Average' for j in range(P)]
        csvwriter.writerow(header)

        for i in range(E // 20):
            csvwriter.writerow([20 * (i + 1)] + fitness_scores[i] + fitness_avg_scores[i])
