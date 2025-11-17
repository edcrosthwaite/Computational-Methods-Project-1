# Constants file for quarter-car suspension model
import numpy as np

# Simulation settings
T_START = 0.0
T_END = 3.0
T_EVAL = np.linspace(T_START, T_END, 2000)

# Initial state: [x_s, x_s_dot, x_u, x_u_dot]
X_INITIAL_STATE = [0.0, 0.0, 0.0, 0.0]

VELOCITY_THRESHOLD = 0.1     # velocity threshold [m/s] for piecewise passive (where damper transitions from soft to firm behaviour)

K_S_GUESS = 20000.0  # initial guess for suspension stiffness [N/m] in ks tuning
