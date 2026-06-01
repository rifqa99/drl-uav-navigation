# Inside env/dynamics_continuous.py -> update_physics():

# The action elements are already bounded cleanly by the agent!
thrust = action[0] * 3.0  # Linear forward force scale maxes out at 3.0 N
# Rotational torque scale maps precisely to [-0.5, 0.5]
torque = action[1] * 0.5
