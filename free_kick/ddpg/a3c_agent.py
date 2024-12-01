import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from a3c_model import ActorCritic
import torch.multiprocessing as mp
from collections import deque

class A3CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-4, gamma=0.99, device='cpu'):
        self.device = device
        self.gamma = gamma

        self.model = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std, _ = self.model(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -np.pi / 6, np.pi / 6)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action_clipped.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()[0]

    def evaluate_actions(self, states, actions):
        mean, std, values = self.model(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values.squeeze()

    def update(self, trajectory):
        states = torch.FloatTensor(np.array(trajectory['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(trajectory['actions'])).to(self.device)
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        next_states = torch.FloatTensor(np.array(trajectory['next_states'])).to(self.device)

        # Compute returns and advantages
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        _, _, values = self.model(states)
        values = values.squeeze()
        advantages = returns - values.detach()

        # Actor loss
        log_probs, entropy, _ = self.evaluate_actions(states, actions)
        actor_loss = -(log_probs * advantages).mean() - 0.001 * entropy.mean()

        # Critic loss
        critic_loss = advantages.pow(2).mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
