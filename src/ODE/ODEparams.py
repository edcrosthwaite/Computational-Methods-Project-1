# params.py
# parameters + simulation settings for quarter-car suspension model

# Parameters 
m_s = 300.0      # sprung mass [kg]
m_u = 40.0       # unsprung mass [kg]
k_s = 28510.0    # suspension spring stiffness [N/m]
k_t = 250000.0   # tyre stiffness [N/m]

# Nonlinear passive / semi-active damper parameters
c_min = 500.0    # low damping [Ns/m]
c_max = 3000.0   # high damping [Ns/m]

v = 5.0          # vehicle speed [m/s]
