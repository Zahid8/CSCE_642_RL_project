import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control.locomotion import soccer as dm_soccer
import torch
import random
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from dm_control import viewer


class StepLoggingCallback(BaseCallback):
    """
    Custom callback for logging the agent's and ball's positions along with the total reward and episode duration at the end of each episode during training.
    """
    def __init__(self, verbose=0):
        super(StepLoggingCallback, self).__init__(verbose)
        self.step_num = 0
        self.episode_reward = 0.0
        self.last_agent_pos = np.array([0.0, 0.0, 0.0])
        self.last_ball_pos = np.zeros(3, dtype=np.float32)
        self.episode_start_time = None  # To track episode start time

    def _on_step(self) -> bool:
        """
        This method is called at each step of the environment.
        It accumulates rewards and updates positions. When an episode ends, it logs the required information.
        """
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        obs = self.locals.get('obs')  # Changed from 'obs_tensor' to 'obs'

        if rewards is not None and obs is not None:
            for i in range(len(rewards)):
                self.step_num += 1
                reward = rewards[i]
                observation = obs[i]

                # Initialize episode start time
                if self.episode_start_time is None:
                    self.episode_start_time = time.time()

                # Accumulate reward for the current episode
                self.episode_reward += reward

                # Agent's position is always [0, 0, 0] in ego frame
                agent_pos = np.array([0.0, 0.0, 0.0])
                print(f"Step {self.step_num} - Agent Position: {agent_pos}")

                # Extract ball's position from the flattened observation
                # Structure: [agent_pos (3), agent_vel (3), ball_pos (3), ball_vel (3)]
                if len(observation) == 12:
                    ball_pos = observation[6:9]
                    self.last_ball_pos = ball_pos
                else:
                    print(f"[WARNING] Unexpected observation length: {len(observation)}")
                    self.last_ball_pos = np.zeros(3)

                # Update the last agent position (remains constant in this setup)
                self.last_agent_pos = agent_pos

                # Check if the episode has ended
                if dones[i]:
                    # Calculate episode duration
                    episode_end_time = time.time()
                    episode_duration = episode_end_time - self.episode_start_time

                    # Log the Agent Position, Ball Position, Total Reward, and Duration for the episode
                    print(f"\n=== Episode {self.step_num} Summary ===")
                    print(f"Total Reward: {self.episode_reward:.2f}")
                    print(f"Agent Position: {self.last_agent_pos}")
                    print(f"Ball Position: {self.last_ball_pos}")
                    print(f"Episode Duration: {episode_duration:.2f} seconds\n")

                    # Reset the episode accumulators
                    self.episode_reward = 0.0
                    self.episode_start_time = None

        return True  # Continue training


class SoccerKickEnv(gym.Env):
    """
    Custom Gymnasium environment for the dm_control soccer simulation.
    This environment provides observations consisting of both the ball's and the agent's state.
    The reward function is designed to encourage the agent to kick the ball towards the goal.
    """
    def __init__(self):
        super(SoccerKickEnv, self).__init__()
        # Define team_size: number of players controlled by the agent
        self.team_size = 1  # Change to 2 if controlling two players

        # Load the dm_control soccer environment
        self.env = dm_soccer.load(
            team_size=self.team_size,
            time_limit=10.0,
            disable_walker_contacts=False,
            enable_field_box=True,
            terminate_on_goal=True,  # Terminate episode upon scoring a goal
            walker_type=dm_soccer.WalkerType.BOXHEAD
        )

        # Total number of players in the environment
        self.num_players = len(self.env.action_spec())  # Corrected from .shape[0]

        # Define observation space: agent position (3) + agent velocity (3) + ball position (3) + ball velocity (3)
        # Since agent position is [0,0,0], we still include it for consistency
        low = -np.inf * np.ones(12, dtype=np.float32)
        high = np.inf * np.ones(12, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Define action space based on the environment's action specification for one player
        # Assuming the agent controls only one player; adjust if controlling multiple
        action_spec = self.env.action_spec()[0]
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            dtype=np.float32
        )

        self._seed = None

        # Initialize the viewer to None
        self.viewer = None

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.backends.mps.is_available():
                # No additional seeding required for MPS as torch.manual_seed covers it
                pass
        timestep = self.env.reset()
        try:
            return self._flatten_observation(timestep.observation[0]), {}
        except ValueError as e:
            print(f"[ERROR] Failed to flatten observation during reset: {e}")
            raise e

    def step(self, action):
        """
        Executes the given action in the environment and returns the next observation, reward, done flag, etc.
        """
        # Ensure action is a single action (for one player)
        if isinstance(action, np.ndarray):
            agent_action = action
        else:
            agent_action = np.array(action, dtype=np.float32)

        # Clip action to ensure it stays within the action space
        agent_action = np.clip(agent_action, self.action_space.low, self.action_space.high)

        # Generate random actions for other players
        other_actions = []
        for i in range(self.num_players):
            if i < self.team_size:
                # Agent-controlled players
                other_actions.append(agent_action)
            else:
                # Opponent players: generate random actions
                opponent_action_spec = self.env.action_spec()[i]
                opponent_action = np.random.uniform(
                    opponent_action_spec.minimum,
                    opponent_action_spec.maximum,
                    size=opponent_action_spec.shape
                ).astype(np.float32)
                other_actions.append(opponent_action)

        # Step the environment with the combined actions
        timestep = self.env.step(other_actions)

        # Flatten the observation
        try:
            obs = self._flatten_observation(timestep.observation[0])
        except ValueError as e:
            print(f"[ERROR] Failed to flatten observation during step: {e}")
            raise e

        # Calculate reward
        reward = self._calculate_reward(timestep)

        # Determine if the episode is done
        done = timestep.last()  # True if the episode has ended

        # Gymnasium expects a 'truncated' flag; set to False as we are not handling truncations
        truncated = False

        return obs, reward, done, truncated, {}

    def _flatten_observation(self, observation):
        """
        Extracts and concatenates relevant parts of the observation.
        Includes both agent's and ball's state.
        """
        # Agent's position is fixed at [0,0,0] in ego frame
        agent_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Extract agent's velocity from 'sensors_velocimeter'
        agent_velocity = observation.get('sensors_velocimeter', np.zeros(3, dtype=np.float32))
        agent_velocity = np.asarray(agent_velocity, dtype=np.float32).flatten()
        if agent_velocity.shape != (3,):
            print(f"[ERROR] Unexpected agent_velocity shape: {agent_velocity.shape}")
            # Adjust to ensure it's 1D with 3 elements
            if agent_velocity.size >= 3:
                agent_velocity = agent_velocity[:3]
            else:
                agent_velocity = np.pad(agent_velocity, (0, 3 - agent_velocity.size), 'constant')
            print(f"[INFO] Adjusted agent_velocity: {agent_velocity}, shape: {agent_velocity.shape}")

        # Extract ball's position from 'ball_ego_position'
        ball_position = observation.get('ball_ego_position', np.zeros(3, dtype=np.float32))
        ball_position = np.asarray(ball_position, dtype=np.float32).flatten()
        if ball_position.shape != (3,):
            print(f"[ERROR] Unexpected ball_position shape: {ball_position.shape}")
            # Adjust to ensure it's 1D with 3 elements
            if ball_position.size >= 3:
                ball_position = ball_position[:3]
            else:
                ball_position = np.pad(ball_position, (0, 3 - ball_position.size), 'constant')
            print(f"[INFO] Adjusted ball_position: {ball_position}, shape: {ball_position.shape}")

        # Extract ball's velocity from 'ball_ego_linear_velocity'
        ball_velocity = observation.get('ball_ego_linear_velocity', np.zeros(3, dtype=np.float32))
        ball_velocity = np.asarray(ball_velocity, dtype=np.float32).flatten()
        if ball_velocity.shape != (3,):
            print(f"[ERROR] Unexpected ball_velocity shape: {ball_velocity.shape}")
            # Adjust to ensure it's 1D with 3 elements
            if ball_velocity.size >= 3:
                ball_velocity = ball_velocity[:3]
            else:
                ball_velocity = np.pad(ball_velocity, (0, 3 - ball_velocity.size), 'constant')
            print(f"[INFO] Adjusted ball_velocity: {ball_velocity}, shape: {ball_velocity.shape}")

        # Concatenate all components into a single observation array
        try:
            flattened_obs = np.concatenate([agent_position, agent_velocity, ball_position, ball_velocity]).astype(np.float32)
        except ValueError as e:
            print(f"[ERROR] Concatenation failed: {e}")
            print(f"agent_position: {agent_position}, shape: {agent_position.shape}")
            print(f"agent_velocity: {agent_velocity}, shape: {agent_velocity.shape}")
            print(f"ball_position: {ball_position}, shape: {ball_position.shape}")
            print(f"ball_velocity: {ball_velocity}, shape: {ball_velocity.shape}")
            raise e

        # Ensure the final observation has the correct shape
        if flattened_obs.shape != (12,):
            print(f"[ERROR] Flattened observation has incorrect shape: {flattened_obs.shape}")
            raise ValueError(f"Flattened observation has incorrect shape: {flattened_obs.shape}")

        return flattened_obs

    def _calculate_reward(self, timestep):
        """
        Calculates the reward based on the ball's distance to the goal, its velocity, and the distance to the agent.
        """
        observation = timestep.observation[0]

        # Extract ball's position and velocity from ego frame
        ball_position = observation.get('ball_ego_position', None)
        ball_velocity = observation.get('ball_ego_linear_velocity', None)

        # Define the goal position in ego frame
        goal_position = np.array([10.0, 0.0, 0.0], dtype=np.float32)  # Example goal position

        if ball_position is not None and ball_velocity is not None:
            # Flatten to ensure 1D arrays
            ball_position = np.asarray(ball_position, dtype=np.float32).flatten()
            ball_velocity = np.asarray(ball_velocity, dtype=np.float32).flatten()

            # Calculate the distance from the ball to the goal
            distance_to_goal = float(np.linalg.norm(ball_position - goal_position))

            # Calculate the distance from the agent to the ball
            distance_to_ball = float(np.linalg.norm(ball_position))  # Since agent is at [0,0,0]

            # Base reward: negative distance to goal to encourage minimizing it
            reward = -distance_to_goal

            # Additional reward: negative distance to ball to encourage minimizing it
            # You can adjust the scaling factor as needed
            alpha = 0.5  # Scaling factor for distance to ball
            reward -= alpha * distance_to_ball

            # Bonus reward for moving the ball towards the goal
            if ball_velocity[0] > 1.0:
                reward += 10.0  # Arbitrary bonus value

            # Additional small penalty for each step to encourage faster goals
            reward -= 0.01
        else:
            # No reward if ball data is missing
            reward = 0.0
            print("[WARNING] Ball data missing, no reward applied.")

        # Ensure the reward is a Python float
        return float(reward)

    def render(self, mode='human'):
        """
        Renders the environment using dm_control's viewer.
        """
        if mode == 'human':
            if self.viewer is None:
                # Initialize the viewer
                self.viewer = viewer.launch(self.env.physics, self.env.task)
            else:
                # Update the viewer with the current state
                self.viewer.render()
        else:
            super(SoccerKickEnv, self).render(mode=mode)  # Just in case

    def close(self):
        """
        Closes the environment and the viewer if it's open.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.env.close()


def flatten_observation(observation):
    """
    Converts dm_control observations to the flattened format used during training.
    """
    # Agent's position is fixed at [0, 0, 0] in ego frame
    agent_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Extract agent's velocity from 'sensors_velocimeter'
    agent_velocity = observation.get('sensors_velocimeter', np.zeros(3, dtype=np.float32))
    agent_velocity = np.asarray(agent_velocity, dtype=np.float32).flatten()
    if agent_velocity.shape != (3,):
        print(f"[ERROR] Unexpected agent_velocity shape: {agent_velocity.shape}")
        # Adjust to ensure it's 1D with 3 elements
        if agent_velocity.size >= 3:
            agent_velocity = agent_velocity[:3]
        else:
            agent_velocity = np.pad(agent_velocity, (0, 3 - agent_velocity.size), 'constant')
        print(f"[INFO] Adjusted agent_velocity: {agent_velocity}, shape: {agent_velocity.shape}")

    # Extract ball's position from 'ball_ego_position'
    ball_position = observation.get('ball_ego_position', np.zeros(3, dtype=np.float32))
    ball_position = np.asarray(ball_position, dtype=np.float32).flatten()
    if ball_position.shape != (3,):
        print(f"[ERROR] Unexpected ball_position shape: {ball_position.shape}")
        # Adjust to ensure it's 1D with 3 elements
        if ball_position.size >= 3:
            ball_position = ball_position[:3]
        else:
            ball_position = np.pad(ball_position, (0, 3 - ball_position.size), 'constant')
        print(f"[INFO] Adjusted ball_position: {ball_position}, shape: {ball_position.shape}")

    # Extract ball's velocity from 'ball_ego_linear_velocity'
    ball_velocity = observation.get('ball_ego_linear_velocity', np.zeros(3, dtype=np.float32))
    ball_velocity = np.asarray(ball_velocity, dtype=np.float32).flatten()
    if ball_velocity.shape != (3,):
        print(f"[ERROR] Unexpected ball_velocity shape: {ball_velocity.shape}")
        # Adjust to ensure it's 1D with 3 elements
        if ball_velocity.size >= 3:
            ball_velocity = ball_velocity[:3]
        else:
            ball_velocity = np.pad(ball_velocity, (0, 3 - ball_velocity.size), 'constant')
        print(f"[INFO] Adjusted ball_velocity: {ball_velocity}, shape: {ball_velocity.shape}")

    # Concatenate all components into a single observation array
    try:
        flattened_obs = np.concatenate([agent_position, agent_velocity, ball_position, ball_velocity]).astype(np.float32)
    except ValueError as e:
        print(f"[ERROR] Concatenation failed: {e}")
        print(f"agent_position: {agent_position}, shape: {agent_position.shape}")
        print(f"agent_velocity: {agent_velocity}, shape: {agent_velocity.shape}")
        print(f"ball_position: {ball_position}, shape: {ball_position.shape}")
        print(f"ball_velocity: {ball_velocity}, shape: {ball_velocity.shape}")
        raise e

    # Ensure the final observation has the correct shape
    if flattened_obs.shape != (12,):
        print(f"[ERROR] Flattened observation has incorrect shape: {flattened_obs.shape}")
        raise ValueError(f"Flattened observation has incorrect shape: {flattened_obs.shape}")

    return flattened_obs


def model_policy(time_step, model):
    """
    Policy function to interface between dm_control's viewer and the PPO model.
    """
    # Flatten the observation
    obs = flatten_observation(time_step.observation)

    # Predict the action using the PPO model
    action, _ = model.predict(obs, deterministic=True)

    return action.flatten()


def launch_evaluation_viewer(model, eval_env, num_episodes=10):
    """
    Launches the dm_control viewer with the trained PPO model's policy.
    """
    for episode in range(1, num_episodes + 1):
        print(f"Starting Evaluation Episode {episode}...")

        # Define a policy function that has access to the PPO model
        def policy(time_step):
            return model_policy(time_step, model)

        # Launch the viewer with the evaluation environment and the custom policy
        viewer.launch(eval_env, policy=policy)

        print(f"Completed Evaluation Episode {episode}.\n")


def set_seeds(seed: int):
    """
    Sets seeds for various random number generators to ensure reproducibility.
    """
    np.random.seed(seed)                # Seed NumPy
    torch.manual_seed(seed)             # Seed PyTorch (CPU and MPS)
    random.seed(seed)                   # Seed Python's built-in RNG


def evaluate_agent(model, env, num_episodes=10):
    """
    Evaluates the trained agent over a specified number of episodes and renders each episode.
    Prints the total reward and final positions for each.
    """
    print("\nStarting Evaluation...\n")
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        final_agent_pos = np.array([0.0, 0.0, 0.0])
        final_ball_pos = np.zeros(3, dtype=np.float32)
        episode_start_time = time.time()  # Start time for the episode
        while not done:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # Render the environment
            env.render(mode='human')
            # Extract positions from the current observation
            agent_pos = obs[0:3]
            ball_pos = obs[6:9]
            final_agent_pos = agent_pos
            final_ball_pos = ball_pos
        # Calculate episode duration
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        # After episode ends, print the final positions and total reward
        print(f"Evaluation Episode {ep}: Total Reward = {total_reward:.2f}")
        print(f"    Final Agent Position: {final_agent_pos}")
        print(f"    Final Ball Position: {final_ball_pos}")
        print(f"    Episode Duration: {episode_duration:.2f} seconds\n")
    print("Evaluation Complete.")


def main():
    # Instantiate the custom Gym environment
    custom_env = SoccerKickEnv()

    # Check if the environment is compatible with Gymnasium
    print("[INFO] Checking environment compatibility...")
    try:
        check_env(custom_env, warn=True)
    except Exception as e:
        print(f"[ERROR] Environment compatibility check failed: {e}")
        exit(1)

    # Create a vectorized environment with one instance and without additional wrappers
    vec_env = make_vec_env(lambda: SoccerKickEnv(), n_envs=1)

    # Determine the device for training
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    # Initialize the PPO model with appropriate hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.0,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42  # For reproducibility
    )

    # Set seeds for reproducibility
    set_seeds(42)

    # Initialize the custom callback for step-level logging
    step_logger = StepLoggingCallback()

    # Train the agent using PPO with the custom callback
    print("[INFO] Starting training...")
    try:
        model.learn(total_timesteps=100000, callback=step_logger)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        exit(1)

    # Save the trained model
    model.save("ppo_soccer_kick")
    print("[INFO] Training Complete and Model Saved as 'ppo_soccer_kick'.")

    # Instantiate the original dm_control environment for evaluation
    eval_env = dm_soccer.load(
        team_size=custom_env.team_size,  # Adjust as needed
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=True,
        walker_type=dm_soccer.WalkerType.BOXHEAD
    )

    # Load the trained model (optional if already in memory)
    model = PPO.load("ppo_soccer_kick", env=vec_env)

    # Launch the viewer for evaluation
    print("[INFO] Launching Evaluation Viewer...")
    launch_evaluation_viewer(model, eval_env, num_episodes=10)

    print("Training and Evaluation Complete.")


if __name__ == "__main__":
    main()
