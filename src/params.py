"""
Parameter definitions for the quarter-car suspension model.
All units SI unless stated otherwise.
"""

from dataclasses import dataclass

@dataclass
class SuspensionParams:
    # Masses
    ms: float = 300.0   # sprung mass [kg]
    mu: float = 40.0    # unsprung mass [kg]

    # Stiffness (initial guesses; ks may be updated by root finding)
    ks: float = 15000.0   # suspension spring stiffness [N/m] (20 kN/m)
    kt: float = 250000.0  # tyre stiffness [N/m] (190 kN/m)

    v: float = 5.0  # vehicle speed [m/s]

    v: float = 5.0  # vehicle speed [m/s]

    # Passive damper asymmetry (compression vs rebound)
    c_comp: float = 1000.0   # Ns/m (compression)
    c_reb: float = 1500.0    # Ns/m (rebound)

    # Semi-active damper bounds
    c_min: float = 250.0
    c_max: float = 3000.0

    # Passive piecewise damper parameters
    c_comp_low: float = 500.0    # Ns/m (low compression damping)
    c_comp_high: float = 1800.0  # Ns/m (high compression damping)
    c_reb_low: float = 1200.0     # Ns/m (low rebound damping)
    c_reb_high: float = 3000.0   # Ns/m (high rebound damping)

    # Constraints / targets (for post-processing checks)
    travel_limit: float = 0.075   # 75 mm
    tyre_defl_limit: float = 0.015  # 15 mm

    # Frequency targets (Hz)
    sprung_fn_band: tuple = (1.3, 1.6)   # comfort target band
    unsprung_fn_band: tuple = (10.0, 15.0)  # wheel hop target band

    # Simulation
    t_end: float = 5.0       # default simulation time [s]
    dt: float = 0.0005       # time step for fixed-step RK4 where used [s]

    # Random seed for reproducibility in demo regression
    seed: int = 42
