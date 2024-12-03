import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

# Instantiate a 2-vs-2 BOXHEAD soccer environment
env = dm_soccer.load(team_size=1,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.HUMANOID)

# Retrieve action specifications for all players
action_specs = env.action_spec()

# Define a function to run each step with visualization
def random_policy(time_step):
    del time_step
    actions = []
    for action_spec in action_specs:
        # Generate random actions within the allowed action space for each player
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    print(actions)
    return actions

# Launch the viewer with your environment and step function
viewer.launch(env, policy=random_policy)