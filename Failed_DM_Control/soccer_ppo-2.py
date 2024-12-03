import gym
from gym import spaces
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import argparse
import time


class DmControlSoccerGymWrapper(gym.Env):
    """
    Gym wrapper for the dm_control.locomotion.soccer environment with Humanoid walkers.
    Controls the first agent and lets opponents act randomly.
    Implements a custom reward based on the ball's proximity to the goal.
    """

    def __init__(self, team_size=1, time_limit=10.0, walker_type=dm_soccer.WalkerType.HUMANOID):
        super(DmControlSoccerGymWrapper, self).__init__()

        # Initialize the dm_control environment
        self.env = dm_soccer.load(
            team_size=team_size,
            time_limit=time_limit,
            disable_walker_contacts=False,
            enable_field_box=True,
            terminate_on_goal=False,
            walker_type=walker_type
        )

        # Reset the environment to get a sample observation
        timestep = self.env.reset()

        # Determine number of agents
        if team_size == 1:
            self.num_agents = 2  # 1 vs 1
        else:
            self.num_agents = team_size * 2  # For 2 vs 2, etc.

        # Define observation space for Agent 0 (controlled agent)
        sample_observation = timestep.observation[0]
        obs_dim = 0
        for key, value in sample_observation.items():
            obs_dim += value.size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define action space based on the first agent's action spec
        action_spec = self.env.action_spec()
        single_agent_action_spec = action_spec[0]
        self.action_space = spaces.Box(
            low=single_agent_action_spec.minimum,
            high=single_agent_action_spec.maximum,
            dtype=np.float32
        )

        # Define goal position relative to the agent's frame
        # Adjust based on your environment's coordinate system
        self.GOAL_POSITION = np.array([10.0, 0.0, 0.0])  # Example values

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action: Action for the controlled agent.

        Returns:
            observation, reward, done, info
        """
        # Prepare actions for all agents
        actions = []
        for i in range(self.num_agents):
            if i == 0:
                # Controlled agent
                actions.append(action)
            else:
                # Opponent agents: random actions
                opponent_action = np.random.uniform(
                    low=self.action_space.low,
                    high=self.action_space.high,
                    size=self.action_space.shape
                )
                actions.append(opponent_action)

        # Step the environment
        timestep = self.env.step(actions)

        # Extract controlled agent's observation and environment reward
        obs_agent = timestep.observation[0]
        env_reward = timestep.reward[0]
        done = timestep.last()
        info = {}

        # Compute custom reward
        custom_reward = self.compute_custom_reward(obs_agent)

        # Combine environment reward with custom reward
        total_reward = env_reward + custom_reward

        # Flatten the observation
        obs_flat = self._flatten_observation(obs_agent)

        return obs_flat, total_reward, done, info

    def reset(self):
        """
        Reset the environment.

        Returns:
            observation
        """
        timestep = self.env.reset()
        obs_agent = timestep.observation[0]
        obs_flat = self._flatten_observation(obs_agent)
        return obs_flat

    def render(self, mode='human'):
        """
        Render the environment.
        """
        # Render using dm_control's built-in viewer
        self.env.physics.render(camera_id=0)

    def close(self):
        """
        Close the environment.
        """
        self.env.close()

    def _flatten_observation(self, observation):
        """
        Flatten the observation dictionary into a single numpy array.

        Args:
            observation: dict of observations.

        Returns:
            flattened numpy array.
        """
        obs_list = []
        for key in sorted(observation.keys()):
            obs_list.append(observation[key].flatten())
        obs_flat = np.concatenate(obs_list)
        return obs_flat

    def compute_custom_reward(self, observation):
        """
        Custom reward function that rewards the agent for moving the ball closer to the goal.

        Args:
            observation: The current observation for the agent.

        Returns:
            A float representing the custom reward.
        """
        # Extract ball position from observations
        ball_position = observation.get('ball_ego_position')  # Correct key

        if ball_position is None:
            # Handle cases where 'ball_ego_position' might not be in the observation
            print("Warning: 'ball_ego_position' not found in observation.")
            return 0.0

        # Compute the distance from the ball to the goal
        distance_to_goal = np.linalg.norm(ball_position - self.GOAL_POSITION)

        # Reward is higher when the ball is closer to the goal
        # You can adjust the scaling factor as needed
        reward = 1.0 / (distance_to_goal + 1e-6)  # Avoid division by zero

        return reward


def train(model_path, num_timesteps):
    """
    Train the PPO agent.

    Args:
        model_path: Directory to save the trained models.
        num_timesteps: Total number of training timesteps.
    """
    # Initialize the Gym environment
    env = DmControlSoccerGymWrapper(team_size=1, time_limit=10.0, walker_type=dm_soccer.WalkerType.HUMANOID)
    env = DummyVecEnv([lambda: env])  # Vectorized environment

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_soccer_tensorboard/",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2
    )

    # Create directory to save models
    os.makedirs(model_path, exist_ok=True)

    # Set up checkpoint callback to save every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_path,
        name_prefix="ppo_soccer_model"
    )

    # Set up evaluation callback
    eval_env = DmControlSoccerGymWrapper(team_size=1, time_limit=10.0, walker_type=dm_soccer.WalkerType.HUMANOID)
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=model_path,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(
        total_timesteps=num_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    # Save the final model
    model.save(os.path.join(model_path, "ppo_soccer_final_model"))

    # Close environments
    env.close()
    eval_env.close()
    print(f"Training completed. Model saved at {model_path}")


def evaluate(model_path, num_eval_episodes):
    """
    Evaluate the trained PPO agent.

    Args:
        model_path: Directory where the trained model is saved.
        num_eval_episodes: Number of episodes to evaluate.
    """
    # Path to the saved model
    final_model_path = os.path.join(model_path, "ppo_soccer_final_model.zip")
    if not os.path.exists(final_model_path):
        print(f"No trained model found at {final_model_path}. Please train the model first.")
        return

    # Load the model
    model = PPO.load(final_model_path)

    # Initialize the evaluation environment
    eval_env = DmControlSoccerGymWrapper(team_size=1, time_limit=10.0, walker_type=dm_soccer.WalkerType.HUMANOID)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Number of evaluation episodes
    total_rewards = []

    for episode in range(num_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Predict the action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, done, info = eval_env.step(action)

            # Accumulate the reward
            episode_reward += reward

            # Render the environment (optional)
            eval_env.envs[0].render()

            # Small delay for rendering smoothness
            time.sleep(0.02)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    # Calculate average reward
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_eval_episodes} Episodes: {average_reward}")

    # Close the evaluation environment
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train or evaluate a PPO agent on dm_control Soccer with Humanoid walkers.')
    parser.add_argument('--train', action='store_true', help='Train the PPO agent.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained PPO agent.')
    parser.add_argument('--model_path', type=str, default='./ppo_soccer_models/', help='Path to save/load the model.')
    parser.add_argument('--num_timesteps', type=int, default=1_000_000, help='Number of timesteps to train.')
    parser.add_argument('--num_eval_episodes', type=int, default=5, help='Number of episodes to evaluate.')

    args = parser.parse_args()

    if not args.train and not args.evaluate:
        print("Please specify --train and/or --evaluate")
        return

    if args.train:
        print("Starting training...")
        train(args.model_path, args.num_timesteps)

    if args.evaluate:
        print("Starting evaluation...")
        evaluate(args.model_path, args.num_eval_episodes)


if __name__ == "__main__":
    main()
