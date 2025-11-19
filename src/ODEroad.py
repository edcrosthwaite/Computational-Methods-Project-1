"""
road.py

Contains the function defining the road displacement input (z_r or y) 
for the quarter-car suspension model. This function represents the road 
profile the vehicle is traveling over.
"""
import numpy as np
from src.iso8606_road import generate_iso8608_profile

# Road input, half-cosine bump
def road_input(t, v, h=0.05, L=2.0):
    """
    Calculates the vertical displacement (z_r or y) of a half-cosine bump 
    of height h and length L, traveled at a constant speed v.

    The half-cosine profile is a common choice for simulating isolated bumps 
    because it provides a smooth, continuous displacement, velocity, and 
    acceleration input, avoiding numerical spikes caused by sudden discontinuities.
    

    Args:
        t (float): Current time [s].
        v (float): Constant vehicle forward velocity [m/s].
        h (float): Total height of the bump [m]. Default is 0.05 m (50 mm).
        L (float): Total length (width) of the bump [m]. Default is 1.0 m.

    Returns:
        float: The vertical road displacement z_r at time t [m].
    """
    
    # Calculate the horizontal distance traveled by the vehicle since t=0.
    x = v * t
    
    # Check if the vehicle is currently positioned on the bump (0 <= x <= L).
    if 0 <= x <= L:
        # The half-cosine function defines the smooth shape of the bump.
        # z_r(x) = 0.5 * h * (1 - cos(2*pi*x / L))
        # It ensures z_r is 0 at x=0 and x=L, and peaks at h at x=L/2.
        return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
    
    # If the vehicle is before (x < 0) or after (x > L) the bump, 
    # the road displacement is zero.
    return 0.0

# --- 2) ISO 8608 road (pre-generated) ------------------------

# choose ISO road settings ONCE
ROAD_CLASS = "E"
ROAD_LENGTH = 200.0   # [m]
DX = 0.05             # [m]
SEED = 42

x_iso, z_iso = generate_iso8608_profile(
    class_label=ROAD_CLASS,
    length=ROAD_LENGTH,
    dx=DX,
    seed=SEED,
)

def road_iso(t: float, v: float) -> float:
    """
    ISO 8608 road elevation at time t for a vehicle moving at speed v.
    Uses interpolation on the precomputed (x_iso, z_iso).
    """
    x_pos = v * t

    if x_pos <= x_iso[0]:
        return float(z_iso[0])
    if x_pos >= x_iso[-1]:
        return float(z_iso[-1])

    return float(np.interp(x_pos, x_iso, z_iso))