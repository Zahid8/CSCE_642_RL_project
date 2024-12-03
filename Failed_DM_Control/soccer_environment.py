import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

# Instantiate a 2-vs-2 BOXHEAD soccer environment
env = dm_soccer.load(team_size=2,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

# Retrieve action specifications for all players
action_specs = env.action_spec()

# Define a function to run each step with visualization
def step_through_env():
    timestep = env.reset()
    while not timestep.last():
        actions = []
        for action_spec in action_specs:
            # Generate random actions within the allowed action space for each player
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape)
            actions.append(action)
        timestep = env.step(actions)

        # # Print rewards and observations for each player
        # for i in range(len(action_specs)):
        #     print(
        #         f"Player {i}: reward = {timestep.reward[i]}, discount = {timestep.discount}, observations = {timestep.observation[i]}")

# Launch the viewer with your environment and step function
viewer.launch(environment_loader=lambda: env, policy=lambda _: step_through_env())
