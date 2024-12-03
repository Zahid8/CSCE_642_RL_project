import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from dm_control import suite, locomotion
import mujoco
import torch

# ------------------------------
# 1. Environment Wrapper
# ------------------------------
class DmControlHumanoidWrapper(gym.Env):
    """
    Gym wrapper for the dm_control humanoid environment tailored for a specific task.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DmControlHumanoidWrapper, self).__init__()

        # Load the dm_control humanoid environment with a specific task
        self.env = locomotion.load(
            domain_name='soccer',
            task_name='soccer'  # You can choose 'stand', 'walk', etc.
        )

        # Define action and observation spaces
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=action_spec.minimum,
            high=action_spec.maximum,
            shape=action_spec.shape,
            dtype=np.float32
        )

        # Reset the environment to get a sample observation
        time_step = self.env.reset()
        obs = self._get_obs(time_step)

        # Define observation space based on the actual observation shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,  # Correct shape based on actual observation
            dtype=np.float32
        )

        # Internal variables
        self._state = obs
        self._done = False
        self.random_state = None  # To store the random state

    def seed(self, seed=None):
        """
        Set the seed for the environment's random number generator.
        
        Args:
            seed (int): The seed value.
        
        Returns:
            list: A list containing the seed.
        """
        self.random_state = np.random.RandomState(seed)
        return [seed]

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.

        Accepts arbitrary keyword arguments to maintain compatibility
        with Stable Baselines3's environment wrappers.
        
        Args:
            **kwargs: Arbitrary keyword arguments (e.g., seed).
        
        Returns:
            np.ndarray: The initial observation.
        """
        # Extract the seed if provided
        seed = kwargs.get('seed', None)
        if seed is not None:
            # Initialize the random state with the seed
            self.random_state = np.random.RandomState(seed)
            # Pass the random state to dm_control's reset
            time_step = self.env.reset(random_state=self.random_state)
        else:
            # Reset without specifying a random state
            time_step = self.env.reset()

        self._done = False
        self._state = self._get_obs(time_step)
        return self._state

    def step(self, action):
        """
        Apply the action to the environment and return the result.
        
        Args:
            action (np.ndarray): The action to take.
        
        Returns:
            tuple: A tuple containing (observation, reward, done, info).
        """
        # Ensure action is in the correct shape
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Step the environment
        time_step = self.env.step(action)

        # Get observation
        obs = self._get_obs(time_step)

        # Get reward
        reward = self._get_reward(time_step, action)

        # Determine if the episode is done
        done = time_step.last()

        # Additional info (optional)
        info = {}
        if done:
            info['is_success'] = self._is_success(time_step)

        self._state = obs
        self._done = done

        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode to render with.
        """
        if mode == 'human':
            viewer = mujoco.viewer.launch(self.env.physics, width=800, height=600)
            viewer.render()

    def close(self):
        """
        Close the environment.
        """
        self.env.close()

    def _get_obs(self, time_step):
        """
        Extract and concatenate relevant state information.
        
        Args:
            time_step: The current time step from dm_control.
        
        Returns:
            np.ndarray: The concatenated observation vector.
        """
        if time_step is None:
            # Initial reset, define a dummy observation
            return np.zeros(1, dtype=np.float32)

        # Agent's joint positions and velocities
        joint_positions = self.env.physics.data.qpos.copy()
        joint_velocities = self.env.physics.data.qvel.copy()

        # Agent's orientation (quaternion)
        # Assuming the first 4 qpos elements represent the orientation quaternion
        agent_orientation = self.env.physics.data.qpos[:4].copy()

        # Concatenate all observations into a single vector
        obs = np.concatenate([
            joint_positions,
            joint_velocities,
            agent_orientation
        ]).astype(np.float32)

        return obs

    def _get_reward(self, time_step, action):
        """
        Compute the reward based on the current state and action.
        
        Args:
            time_step: The current time step from dm_control.
            action (np.ndarray): The action taken.
        
        Returns:
            float: The computed reward.
        """
        reward = 0.0

        # Define your reward structure here
        # For example, encourage the humanoid to stand upright
        # Reward can be based on the height of the torso, energy efficiency, etc.

        # Example: Reward for maintaining an upright position
        torso_height = self.env.physics.named.data.site_xpos['torso'].copy()[2]
        target_height = 1.0  # Target height in meters
        height_reward = -abs(torso_height - target_height)
        reward += height_reward

        # Example: Energy penalty to encourage efficient actions
        energy_penalty = np.sum(np.square(action))
        reward -= energy_penalty * 0.001

        # Time penalty to encourage faster task completion
        reward -= 0.01

        return reward

    def _is_success(self, time_step):
        """
        Determine if the task was successfully completed.
        
        Args:
            time_step: The current time step from dm_control.
        
        Returns:
            bool: True if the task was successful, False otherwise.
        """
        # Define success criteria based on the task
        # For the 'stand' task, success could be maintaining balance for the duration

        # Example: Success if torso height is within a certain range
        torso_height = self.env.physics.named.data.site_xpos['torso'].copy()[2]
        target_height = 1.0  # Target height in meters
        tolerance = 0.05  # Tolerance in meters

        success = abs(torso_height - target_height) < tolerance
        return success

# ------------------------------
# 2. Training the PPO Agent
# ------------------------------
def train_agent(total_timesteps=100000):
    """
    Train a PPO agent on the dm_control humanoid stand task.
    
    Args:
        total_timesteps (int): The total number of timesteps for training.
    """
    # Create the Gym environment
    def make_env():
        env = DmControlHumanoidWrapper()
        env = VecMonitor(DummyVecEnv([lambda: env]))
        return env

    env = make_env()

    # Define the PPO model
    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        tensorboard_log="./ppo_humanoid_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("ppo_humanoid_stand")
    print("Training completed and model saved.")

    env.close()

# ------------------------------
# 3. Evaluating the Trained Agent
# ------------------------------
def evaluate_agent(model_path="ppo_humanoid_stand", episodes=100, render=False):
    """
    Evaluate the trained PPO agent.
    
    Args:
        model_path (str): Path to the trained model.
        episodes (int): Number of evaluation episodes.
        render (bool): Whether to render the environment during evaluation.
    """
    # Load the trained model
    model = PPO.load(model_path)

    # Create the Gym environment
    env = DmControlHumanoidWrapper()

    success_count = 0

    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()

            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)

            # Take action in the environment
            obs, reward, done, info = env.step(action)

            # Check if task was successful
            if done and info.get('is_success', False):
                success_count += 1
                print(f"Episode {episode + 1}: Task completed successfully!")

        print(f"Episode {episode + 1} completed.")

    print(f"Success Rate: {success_count / episodes * 100}%")
    env.close()

# ------------------------------
# 4. Main Execution
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate PPO agent on dm_control humanoid stand task.")
    parser.add_argument('--train', action='store_true', help='Train the PPO agent.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained PPO agent.')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of timesteps for training.')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes for evaluation.')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation.')

    args = parser.parse_args()

    if args.train:
        train_agent(total_timesteps=args.timesteps)

    if args.evaluate:
        evaluate_agent(episodes=args.episodes, render=args.render)

    if not args.train and not args.evaluate:
        print("Please specify --train and/or --evaluate.")
