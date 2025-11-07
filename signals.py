# signals.py
# Function to represent the Road Input (Speed Bump)to the Suspension System

import numpy as np

def half_sine_bump(amplitude=0.05, length=0.5, speed=20.0):
    """
    Returns y(t) for a half-sine bump of given physical length [m],
    traversed at given speed [m/s]. Vertical amplitude in meters.
    The bump lasts for duration L / v, after which y=0.

    y(s) = A * sin(pi * x / L), 0 <= x <= L, with x = speed * t
    """
    duration = length / speed

    def y(t):
        t = np.asarray(t)
        x = speed * t
        y = np.zeros_like(t, dtype=float)
        mask = (t >= 0) & (t <= duration)
        # Map x in [0, L] -> half sine
        y[mask] = amplitude * np.sin(np.pi * x[mask] / length)
        return y
    return y, duration

