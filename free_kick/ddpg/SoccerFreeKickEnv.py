import gym
from gym import spaces
import numpy as np
import pygame

class SoccerFreeKickEnv(gym.Env):
    def __init__(self):
        super(SoccerFreeKickEnv, self).__init__()
        
        # Pygame setup
        pygame.init()
        self.screen_width = 600
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Soccer Free Kick Simulation")
        self.clock = pygame.time.Clock()
        
        # Goal area dimensions
        self.goal_width = 200
        self.goal_height = 10
        self.goal_position = np.array([self.screen_width / 2.0 - self.goal_width / 2.0, 0.0])
        
        # Define action space: continuous steering angle in radians (-30° to +30°)
        self.action_space = spaces.Box(low=np.array([-np.pi / 6]), high=np.array([np.pi / 6]), dtype=np.float32)
        
        # Define observation space
        # [ball_speed, ball_direction, agent_x, agent_y, opponent_x, opponent_y, ball_x, ball_y]
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        
        # Initialize positions
        self.reset()
        
        self.max_steps = 100
        self.current_step = 0

    def reset(self):
        # Agent positioned directly ahead of the goal
        self.agent_pos = np.array([self.screen_width / 2.0, 700.0])
        
        # Ball positioned close to the agent
        self.ball_pos = self.agent_pos.copy() + np.array([0.0, -10.0])
        self.ball_speed = 0.0
        self.ball_direction = 0.0  # Continuous angle in radians
        
        # Opponent positioned near the goal
        self.opponent_pos = np.array([self.screen_width / 2.0 + 50.0, 100.0])
        
        # Ball velocity components
        self.ball_velocity = np.array([0.0, 0.0])
        
        self.current_step = 0
        return self.get_state()

    def get_state(self):
        # Normalize the state
        normalized_ball_speed = self.ball_speed / 20.0  # Assuming max speed ~20
        normalized_ball_direction = (self.ball_direction + np.pi / 6) / (np.pi / 3)  # Map from [-π/6, π/6] to [0,1]
        normalized_agent_pos = self.agent_pos / np.array([self.screen_width, self.screen_height])
        normalized_opponent_pos = self.opponent_pos / np.array([self.screen_width, self.screen_height])
        normalized_ball_pos = self.ball_pos / np.array([self.screen_width, self.screen_height])
        
        state = np.array([
            normalized_ball_speed,
            normalized_ball_direction,
            normalized_agent_pos[0],
            normalized_agent_pos[1],
            normalized_opponent_pos[0],
            normalized_opponent_pos[1],
            normalized_ball_pos[0],
            normalized_ball_pos[1]
        ], dtype=np.float32)
        
        return state

    def step(self, action):
        done = False
        goal = False
        self.current_step += 1
        
        # Action corresponds to continuous direction angle
        self.ball_direction = action[0]
        kick_strength = 20.0  # Define a fixed kick strength
        
        # Update ball velocity based on action
        self.ball_velocity = np.array([
            kick_strength * np.sin(self.ball_direction),
            -kick_strength * np.cos(self.ball_direction)
        ])
        self.ball_speed = np.linalg.norm(self.ball_velocity)
        
        # Update ball position
        self.ball_pos += self.ball_velocity
        
        # Update ball speed (simulate friction)
        self.ball_speed *= 0.99
        if self.ball_speed < 0.1:
            self.ball_speed = 0.0
            self.ball_velocity = np.array([0.0, 0.0])
        
        # Move the opponent towards the ball if possible
        direction_to_ball = self.ball_pos - self.opponent_pos
        distance = np.linalg.norm(direction_to_ball)
        if distance > 0:
            direction_norm = direction_to_ball / distance
            self.opponent_pos += direction_norm * min(5.0, distance)  # Opponent speed
        
        # Update ball position with remaining velocity
        if self.ball_speed > 0:
            self.ball_pos += self.ball_velocity
        
        # Check for goal
        if self.ball_pos[1] <= self.goal_position[1] + self.goal_height:
            if self.goal_position[0] <= self.ball_pos[0] <= self.goal_position[0] + self.goal_width:
                reward = 100.0  # Goal scored
                done = True
                goal = True
                return self.get_state(), reward, done, {"goal": goal}
            else:
                # Ball missed the goal
                reward = -10.0
                done = True
                return self.get_state(), reward, done, {"goal": goal}
        
        # Check for opponent interception
        opponent_distance = np.linalg.norm(self.ball_pos - self.opponent_pos)
        if opponent_distance < 15.0:
            reward = -100.0  # Opponent intercepted
            done = True
            return self.get_state(), reward, done, {"goal": goal}
        
        # Time penalty
        reward = -1.0
        
        if self.current_step >= self.max_steps:
            done = True
        
        return self.get_state(), reward, done, {"goal": goal}

    def render(self, mode='human'):
        self.screen.fill((34, 139, 34))  # Green field
        
        # Draw goal area
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(
            self.goal_position[0], self.goal_position[1],
            self.goal_width, self.goal_height
        ))
        
        # Draw agent
        pygame.draw.circle(self.screen, (0, 0, 255), self.agent_pos.astype(int), 15)
        
        # Draw opponent
        pygame.draw.circle(self.screen, (255, 0, 0), self.opponent_pos.astype(int), 15)
        
        # Draw ball
        pygame.draw.circle(self.screen, (0, 0, 0), self.ball_pos.astype(int), 8)
        
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
