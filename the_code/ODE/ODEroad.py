# road.py
# road input function for quarter-car model
import numpy as np

# Road input, half-cosine bump
def road_input(t, v, h=0.05, L=1.0):
    """
    Half-cosine bump of height h and length L travelled at speed v.
    """
    x = v * t
    if 0 <= x <= L:
        return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
    return 0.0
