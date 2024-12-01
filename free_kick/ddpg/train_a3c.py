import torch
import pygame
import sys
import csv
import signal
import numpy as np
from SoccerFreeKickEnv import SoccerFreeKickEnv
from a3c_agent import A3CAgent
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A3C training parameters
state_dim = SoccerFreeKickEnv().observation_space.shape[0]
action_dim = SoccerFreeKickEnv().action_space.shape[0]  # Should be 1
hidden_dim = 256
lr = 1e-4
gamma = 0.99
num_workers = 4
max_episode_steps = 100

agent = A3CAgent(state_dim, action_dim, hidden_dim, lr, gamma, device=device)

# Training parameters
num_episodes = 10000
log_interval = 10

# List to store results
manager = mp.Manager()
results = manager.list()

# Function to save results to CSV
def save_results(results):
    with open('a3c_training_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Goal Reached'])
        writer.writerows(results)
    print("Results saved to a3c_training_results.csv")

# Worker function
def worker(worker_id, agent, num_episodes, results):
    env = SoccerFreeKickEnv()
    for episode in range(num_episodes):
        state = env.reset()
        trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        total_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['next_states'].append(next_state)
            trajectory['dones'].append(done)
            total_reward += reward
            state = next_state

        agent.update(trajectory)
        results.append([episode + worker_id * num_episodes, total_reward, info.get("goal", False)])

        if (episode + 1) % log_interval == 0:
            print(f"Worker {worker_id}, Episode {episode + 1}")
        save_results(list(results))

    env.close()

# Handle keyboard interrupt
def signal_handler(sig, frame):
    print('Keyboard interrupt received. Saving results and exiting...')
    save_results(list(results))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Start workers
try:
    processes = []
    episodes_per_worker = num_episodes // num_workers
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, agent, episodes_per_worker, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Save results after all workers are done
    save_results(list(results))

except Exception as e:
    print(f"An error occurred: {e}")
    save_results(list(results))
    sys.exit(1)

finally:
    # Save results when training is complete or interrupted
    save_results(list(results))
