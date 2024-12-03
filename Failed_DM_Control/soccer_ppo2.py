# soccer_ppo2.py

import argparse
import os
import numpy as np
from dm_control import viewer
from stable_baselines3 import PPO
from dm_control.locomotion import soccer as dm_soccer

def load_trained_model(model_path):
    """
    Load the trained PPO model.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        PPO: Loaded PPO model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    model = PPO.load(model_path)
    return model

def flatten_observation(observation):
    """
    Flatten the observation into a single numpy array.
    Handles both dict and list/tuple observations.

    Args:
        observation (dict or list or tuple): Observation from the environment.

    Returns:
        np.ndarray: Flattened observation.
    """
    if isinstance(observation, (list, tuple)):
        flattened = np.concatenate([np.array(obs).flatten() for obs in observation])
        print(f"Flattened list/tuple observation shape: {flattened.shape}")
        return flattened
    elif isinstance(observation, dict):
        # Sort the keys to ensure consistent ordering
        flattened = np.concatenate([np.array(observation[key]).flatten() for key in sorted(observation.keys())])
        print(f"Flattened dict observation shape: {flattened.shape}")
        return flattened
    else:
        raise TypeError(f"Unsupported observation type: {type(observation)}")

def main(args):
    # Initialize the underlying dm_control environment with team_size=5
    dm_env = dm_soccer.load(
        team_size=5,  # Must match the training team_size
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.HUMANOID
    )

    if args.model_path:
        # Load the trained PPO model
        model = load_trained_model(args.model_path)

        # Define the policy function using the trained model
        def policy(timestep):
            """
            Policy function that uses the trained PPO model to select actions.

            Args:
                timestep (dm_env.TimeStep): The current timestep from the environment.

            Returns:
                np.ndarray: The action to take.
            """
            obs_data = timestep.observation
            print(f"Observation Type: {type(obs_data)}")
            
            if isinstance(obs_data, dict):
                print("Observation Keys and Shapes:")
                for key, value in obs_data.items():
                    print(f"  {key}: {np.array(value).shape}")
            elif isinstance(obs_data, list) or isinstance(obs_data, tuple):
                print("Observation Items and Shapes:")
                for i, item in enumerate(obs_data):
                    print(f"  Item {i}: {np.array(item).shape}")
            else:
                print(f"Observation Content: {obs_data}")
            
            try:
                obs_flat = flatten_observation(obs_data)
                print(f"Flattened Observation Shape: {obs_flat.shape}")
            except Exception as e:
                print(f"Error during observation flattening: {e}")
                raise e
            
            try:
                action, _states = model.predict(obs_flat, deterministic=True)
                print(f"Predicted Action: {action}")
            except Exception as e:
                print(f"Error during model prediction: {e}")
                raise e
            
            return action
    else:
        # Define a random policy if no model is provided
        def policy(timestep):
            """
            Random policy function that selects actions uniformly at random.

            Args:
                timestep (dm_env.TimeStep): The current timestep from the environment.

            Returns:
                list: A list of random actions.
            """
            # Access the action specifications from the environment
            action_spec = dm_env.action_spec()
            action = []
            for spec in action_spec:
                # Sample uniformly within the action specification
                a = spec.uniform_random()
                action.append(a)
            return action

    # Launch the viewer with the environment and the defined policy
    viewer.launch(dm_env, policy=policy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render dm_control Soccer environment with a policy.')
    parser.add_argument('--model_path', type=str, default="/Users/crossfire/Programming Projects/Classes/dm_control/RL_PROJECT/ppo_soccer_models/ppo_soccer_final_model.zip",
                        help='Path to the trained PPO model. If not provided, a random policy is used.')
    args = parser.parse_args()
    main(args)
