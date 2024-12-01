import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ddpg_model import Actor, Critic
from replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-3, gamma=0.99, tau=1e-3, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Actor network and target
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic network and target
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Exploration noise parameters
        self.noise_std = 0.2
        self.noise_clip = 0.5

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        if noise:
            action += np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action, -np.pi / 6, np.pi / 6)
        return action  # Returns a NumPy array

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)  # Shape: (batch_size, 1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)      # Shape: (batch_size, 1)
        
        # Compute target Q value
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        # Get current Q estimate
        current_Q = self.critic(state, action)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        
        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
