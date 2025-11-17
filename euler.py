"""euler.py, uses Explicit Euler.

Choose what method (RK4, RK2, EC) to use
Should probably put params in a json file

into an external file? Give an option to ouput to external.

Step size must be small, or will diverge and be awful...

NOTE!!! Throughout, the unsprung (TYRE) is index 0, and Sprung (CAR) is index 1
"""

import numpy as np
import data_struc
import plot

def euler(pos_u, pos_s, k, h, c, t_max, mass_u, mass_s, vel_s=0, vel_u=0, acc_u=0, acc_s=0, f_in=0):
    """
    Uses the Euler-Cromer Method to calculate the motion of the system
    """

    # Calls data.init to create empty data array
    data = data_struc.data_init(t_max, h)

    # Index starts at 1 because it is used in time calculation, and time 0 is considered
    # as the initial conditions.
    index = 1
    index_max = np.ceil(t_max/h)

    # Place starting conditions in data
    data[0, 0, 0] = pos_u
    data[0, 1, 0] = vel_u
    data[0, 2, 0] = acc_u

    data[1, 0, 0] = pos_s
    data[1, 1, 0] = vel_s
    data[1, 2, 0] = acc_s

    data[2, 0, 0] = 0

    # Main simulation loop
    while index < index_max:

        # Unsprung (Tyre) acceleration calculation
        # Acceleration calculations are done first according to Euler-Cromer Method
        acc_u = (k[1]*(pos_s-pos_u) - k[0]*(pos_u-f_in) + c*(vel_s-vel_u)) / mass_u

        # Sprung (Car) acceleration calculation
        acc_s = (-k[1]*(pos_s-pos_u) - c*(vel_s-vel_u)) / mass_s

        # Velocity calculations performed second, acceleration is multiplied by timestep
        vel_u += acc_u * h
        vel_s += acc_s * h

        # Position calculations done last by again multiplying by timestep
        pos_s += vel_s * h
        pos_u += vel_u * h

        # Record exact time
        time = h * index

        # Record is appended with calculated position values
        data[0, 0, index] = pos_u
        data[0, 1, index] = vel_u
        data[0, 2, index] = acc_u

        data[1, 0, index] = pos_s
        data[1, 1, index] = vel_s
        data[1, 2, index] = acc_s

        data[2, 0, index] = time

        index += 1

    return data

spring_const = np.array([200000, 25000])

euler_data = euler(0.05, 0, spring_const, 0.00001, 500, 1, 40, 300, f_in=0)

plot.plot(euler_data, y=(0,0))
