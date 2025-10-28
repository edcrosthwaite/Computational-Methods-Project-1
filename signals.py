
"""
Road input signals y(t) for the quarter-car model.
"""

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

def sine_sweep(amplitude=0.005, f_start=0.5, f_end=20.0, T=20.0):
    """
    Linear frequency sweep from f_start to f_end over time T.
    Returns y(t) with small amplitude (default 5 mm).
    """
    k = (f_end - f_start) / T  # Hz/s

    def y(t):
        t = np.asarray(t)
        # Instantaneous frequency f(t) = f_start + k t
        phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t**2)
        return amplitude * np.sin(phase)
    return y, T
