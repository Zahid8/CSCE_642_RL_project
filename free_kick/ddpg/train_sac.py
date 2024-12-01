import torch
import pygame
import sys
import csv
import signal
import numpy as np
from SoccerFreeKickEnv import SoccerFreeKickEnv
from sac_agent import SACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the environment
env = SoccerFreeKickEnv()

# SAC training parameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # Should be 1
hidden_dim = 256
lr = 3e-4
gamma = 0.99
tau = 1e-3
alpha = 0.2

agent = SACAgent(state_dim, action_dim, hidden_dim, lr, gamma, tau, alpha, device=device)

# Training parameters
num_episodes = 10000
log_interval = 100
batch_size = 64

# List to store results
results = []

# Function to save results to CSV
def save_results():
    with open('sac_training_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Goal Reached'])
        writer.writerows(results)
    print("Results saved to sac_training_results.csv")

# Handle keyboard interrupt
def signal_handler(sig, frame):
    print('Keyboard interrupt received. Saving results and exiting...')
    save_results()
    env.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)
            total_reward += reward
            state = next_state

            # Optional: Render the environment
            env.render()

        # Store results
        results.append([episode, total_reward, info.get("goal", False)])

        if episode % log_interval == 0:
            avg_reward = np.mean([r[1] for r in results[-log_interval:]])
            print(f"Episode {episode} \t Average Reward: {avg_reward}")

        # Save results periodically
        if episode % 10 == 0:
            save_results()

except Exception as e:
    print(f"An error occurred: {e}")
    save_results()
    env.close()
    sys.exit(1)

finally:
    # Save results when training is complete or interrupted
    save_results()
    env.close()
