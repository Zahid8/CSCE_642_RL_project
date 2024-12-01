import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from a2c_model import Actor, Critic

class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3, gamma=0.99, device='cpu'):
        self.device = device
        self.gamma = gamma

        # Actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic network
        self.critic = Critic(state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.action_dim = action_dim

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -np.pi / 6, np.pi / 6)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action_clipped.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()[0]

    def evaluate_actions(self, states, actions):
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

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
        values = self.critic(states).squeeze()
        advantages = returns - values

        # Actor loss
        log_probs, entropy = self.evaluate_actions(states, actions)
        actor_loss = -(log_probs * advantages.detach()).mean() - 0.001 * entropy.mean()

        # Critic loss
        critic_loss = advantages.pow(2).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
