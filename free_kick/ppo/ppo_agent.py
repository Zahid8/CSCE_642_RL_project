import torch
import torch.nn as nn
import torch.optim as optim
from model import PolicyNetwork, ValueNetwork
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, device='cpu'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.policy_old = PolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.value_function = ValueNetwork(state_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_function.parameters(), 'lr': lr}
        ])
        
        self.MseLoss = nn.MSELoss()
        
        self.memory = []
        self.memory_rewards_done = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_logits = self.policy_old(state)
            probs = F.softmax(action_logits, dim=-1)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        self.memory.append((state, action, log_prob))
        return action.item()
    
    def update(self):
        # Convert memory to batches
        states = torch.stack([m[0] for m in self.memory]).to(self.device)
        actions = torch.tensor([m[1] for m in self.memory]).to(self.device)
        old_log_probs = torch.stack([m[2] for m in self.memory]).to(self.device)
        
        # Compute rewards-to-go
        rewards = []
        discounted_reward = 0
        for reward, done in reversed(self.memory_rewards_done):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).float().to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Recompute values and advantages within each epoch to avoid graph reuse
            values = self.value_function(states).squeeze()
            advantages = rewards - values.detach()
            
            action_logits = self.policy(states)
            probs = F.softmax(action_logits, dim=-1)
            m = Categorical(probs)
            
            entropy = m.entropy().mean()
            new_log_probs = m.log_prob(actions)
            
            # Ratio for clipping
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) - 0.01 * entropy + self.MseLoss(values, rewards)
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Update the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = []
        self.memory_rewards_done = []
    
    def store_transition(self, reward, done):
        self.memory_rewards_done.append((reward, done))
    
    def clear_memory(self):
        self.memory = []
        self.memory_rewards_done = []
