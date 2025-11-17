# Constants file for quarter-car suspension model
import numpy as np

# Simulation settings
T_START = 0.0
T_END = 3.0
T_EVAL = np.linspace(T_START, T_END, 2000)

# Initial state: [x_s, x_s_dot, x_u, x_u_dot]
X_INITIAL_STATE = [0.0, 0.0, 0.0, 0.0]

VELOCITY_THRESHOLD = 0.3     # velocity threshold [m/s] for piecewise passive (where damper transitions from soft to firm behaviour)
