import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sac_model import Actor, QNetwork
from replay_buffer import ReplayBuffer

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=1e-3, alpha=0.2, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor network and target
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic networks
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Target networks
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        if evaluate:
            action = mean
        else:
            action = dist.sample()
        action_clipped = torch.clamp(action, -np.pi / 6, np.pi / 6)
        return action_clipped.cpu().detach().numpy()[0]

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action_mean, next_action_std = self.actor(next_state)
            next_dist = torch.distributions.Normal(next_action_mean, next_action_std)
            next_action = next_dist.sample()
            next_action = torch.clamp(next_action, -np.pi / 6, np.pi / 6)
            log_prob = next_dist.log_prob(next_action).sum(dim=1, keepdim=True)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # Current Q estimates
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Critic losses
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        # Optimize Critic1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Optimize Critic2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        action_mean, action_std = self.actor(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        sampled_action = dist.sample()
        sampled_action = torch.clamp(sampled_action, -np.pi / 6, np.pi / 6)
        log_prob = dist.log_prob(sampled_action).sum(dim=1, keepdim=True)
        q1 = self.critic1(state, sampled_action)
        q2 = self.critic2(state, sampled_action)
        q = torch.min(q1, q2)
        actor_loss = ((self.alpha * log_prob) - q).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
