# params.py
# parameters + simulation settings for quarter-car suspension model
import numpy as np  # needed for t_eval

# Parameters 
m_s = 300.0      # sprung mass [kg]
m_u = 40.0       # unsprung mass [kg]
k_s = 28510.0    # suspension spring stiffness [N/m]
k_t = 250000.0   # tyre stiffness [N/m]

# Nonlinear passive / semi-active damper parameters
c_min = 500.0    # low damping [Ns/m]
c_max = 3000.0   # high damping [Ns/m]
v0    = 0.3     # velocity threshold [m/s] for piecewise passive (where damper transitions from soft to firm behaviour)

v = 5.0          # vehicle speed [m/s]

# Simulation settings
t_start = 0.0
t_end = 3.0
t_eval = np.linspace(t_start, t_end, 2000)

# Initial state: [x_s, x_s_dot, x_u, x_u_dot]
x0 = [0.0, 0.0, 0.0, 0.0]
