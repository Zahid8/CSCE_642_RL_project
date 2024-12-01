import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)               # Shape: (1, state_dim)
        action = np.expand_dims(action, 0)             # Shape: (1, action_dim)
        reward = np.array([reward], dtype=np.float32)  # Shape: (1,)
        next_state = np.expand_dims(next_state, 0)     # Shape: (1, state_dim)
        done = np.array([done], dtype=np.float32)      # Shape: (1,)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.concatenate, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
