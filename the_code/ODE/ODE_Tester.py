from params import SuspensionParams
from signals import half_sine_bump
from the_code.ODE.ODE_RK4 import rk4, rhs_passive
import numpy as np
import matplotlib.pyplot as plt

p = SuspensionParams()

# Road input
y_fun, bump_duration = half_sine_bump(amplitude=0.05, length=2.0, speed=20.0)

# Initial conditions
y0 = [0, 0, 0, 0]

# ---- RUN SIMULATION ----
t, Y = rk4(lambda t_, s_: rhs_passive(t_, s_, p, y_fun),
           (0.0, p.t_end),
           y0,
           p.dt)

# ---- CHECK OUTPUT ----
xs = Y[:,0]
xsdot = Y[:,1]
xu = Y[:,2]
xudot = Y[:,3]

# Plot to verify
plt.plot(t, xs, label='Sprung mass')
plt.plot(t, xu, label='Unsprung mass')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.show()
