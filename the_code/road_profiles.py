# road_profiles.py
import numpy as np

def make_cosine_bump(h=0.05, L=1.0, v=10.0):
    """
    Factory that returns a function z_r(t) for a cosine speed bump.

    h : bump height [m]
    L : bump length along road [m]
    v : vehicle speed [m/s]
    """
    def z_r(t):
        x = v * t  # longitudinal position
        if 0 <= x <= L:
            return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
        return 0.0
    return z_r

def make_flat_road():
    def z_r(t):
        return 0.0
    return z_r
