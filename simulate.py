"""simulate.py

Simulates the suspension system using either rk4 or euler
Currently does not support an input force, must be next thing to do
JSON file still needs implementing

NOTE!!! Throughout, the unsprung (TYRE) is index 0, and Sprung (CAR) is index 1
"""

import numpy as np
import data_struc
import plot

# Parameters - put into a json file later
k = np.array([200000, 25000])
c = 2000
mass_u = 40
mass_s = 300
h = 0.01
t_max = 5


# Creates a function for the acceleration here so it can be called easier
def f(state, f_in):
    """Function that calculates the acceleration of the system
    Is passed the position and velocity at a given time, and returns
    the velocity and acceleration.
    """
    pos_u, pos_s, vel_u, vel_s = state

    # Sprung (Car) acceleration calculation
    acc_u = (k[1]*(pos_s-pos_u) - k[0]*(pos_u-f_in) + c*(vel_s-vel_u)) / mass_u
    acc_s = (-k[1]*(pos_s-pos_u) - c*(vel_s-vel_u)) / mass_s

    return np.array([vel_u, vel_s, acc_u, acc_s])


def simulate(pos_u=0, pos_s=0, vel_s=0, vel_u=0, acc_u=0, acc_s=0, sim="euler"):
    """
    Uses either the RK4 or Euler method to calculate the motion of the system
    """

    # Calls data.init to create empty data array
    data = data_struc.data_init(t_max, h)

    # Index starts at 1 because it is used in time calculation, and time 0 is considered
    # as the initial conditions.
    index = 1
    index_max = np.ceil(t_max/h)

    # Place starting conditions in data
    # Changed so that the first value is a header?
    data[0, 0, 0] = pos_u
    data[0, 1, 0] = vel_u
    data[0, 2, 0] = acc_u

    data[1, 0, 0] = pos_s
    data[1, 1, 0] = vel_s
    data[1, 2, 0] = acc_s

    # Main simulation loop
    while index < index_max:

        # state stores the pos_u, pos_s, vel_u and vel_s used in calculating acceleration
        state = np.array([pos_u, pos_s, vel_u, vel_s])

        # Calculate forcing input
        f_in = data[2, 1, index]

        k1 = f(state, f_in)             # Returns vel_u, vel_s, acc_u, acc_s

        if sim == "rk4":
            # RK4 method

            k2 = f((state + k1*h/2), f_in)
            k3 = f((state + k2*h/2), f_in)
            k4 = f((state + k3*h), f_in)

            # Velocity calculations performed second, acceleration is multiplied by timestep
            vel_u += (k1[2]/6 + k2[2]/3 + k3[2]/3 + k4[2]/6) * h
            vel_s += (k1[3]/6 + k2[3]/3 + k3[3]/3 + k4[3]/6) * h

            pos_u += (k1[0]/6 + k2[0]/3 + k3[0]/3 + k4[0]/6) * h
            pos_s += (k1[1]/6 + k2[1]/3 + k3[1]/3 + k4[1]/6) * h

        elif sim == "euler":
            # Euler method
            vel_u += k1[2] * h
            vel_s += k1[3] * h

            # Position calculations done last by again multiplying by timestep
            pos_u += vel_u * h
            pos_s += vel_s * h


        else:
            # Very simple error handling
            print("Sim type not recognised - please choose rk4 or euler")
            exit()

        # Record is appended with calculated position values
        # The acceleration values are recalculated here so stale values aren't being recorded
        data[0, 0, index] = pos_u
        data[0, 1, index] = vel_u
        data[0, 2, index] = f(state, f_in)[2]

        data[1, 0, index] = pos_s
        data[1, 1, index] = vel_s
        data[1, 2, index] = f(state, f_in)[3]

        index += 1

    return data

sim_data = simulate(sim="rk4")

plot.plot(sim_data)
