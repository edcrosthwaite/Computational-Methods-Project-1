"""
ODEinterpol.py

Variation of ODEmain.py for Interpolation to use
"""
import numpy as np
from scipy.integrate import solve_ivp

# Import constants (like time limits, initial state) and vehicle parameters
from src.constants import (
    VELOCITY_THRESHOLD, T_START, T_END_BUMP, T_EVAL_BUMP, X_INITIAL_STATE
)

# Import model components
from src.ODEroad import road_input # Road excitation profile
from src.ODEdampers import F_passive_piecewise, F_skyhook_clipped # Damping force models
from src.ODEodes import quarter_car_ode_passive, quarter_car_ode_skyhook # ODE system definitions
from src.params import SuspensionParams # Vehicle mass and stiffness parameters

# Initialise vehicle parameters instance
params = SuspensionParams()

def ode_bump(v: float = None):
    """
    Runs the numerical simulation for both damper configurations, calculates 
    performance metrics, and plots the comparative results.
    """
    # ----------------------------------------------------------------------
    # 1. ODE SOLVER EXECUTION
    # ----------------------------------------------------------------------

    if v is None:
        v = params.v        # Exists so interpolation can sweep through different v values

    # --- Passive simulation (Asymmetric Piecewise Damper) ---
    # The fun argument is a lambda function to pass all required parameters to the ODE solver.
    sol = solve_ivp(
        fun=lambda t, y: quarter_car_ode_passive(t, y, params.ms, params.mu, params.ks,
                                                params.c_comp_low, params.c_comp_high,
                                                params.c_reb_low, params.c_reb_high,
                                                VELOCITY_THRESHOLD, params.kt, v,
                                                z_r=road_input(t, v)),
        t_span=(T_START, T_END_BUMP), # Time range for integration
        y0=X_INITIAL_STATE,      # Initial state vector [x_s, x_s_dot, x_u, x_u_dot]
        t_eval=T_EVAL_BUMP,           # Specific time points at which to store the solution
        method='RK45'            # Runge-Kutta 4(5) is a robust integration method
    )

    # --- Skyhook simulation (Clipped Skyhook Semi-Active Damper) ---
    sol_sky = solve_ivp(
        fun=lambda t, y: quarter_car_ode_skyhook(t, y, params.ms, params.mu, params.ks,
                                                params.c_min, params.c_max, params.kt,
                                                v, z_r=road_input(t, v)),
        t_span=(T_START, T_END_BUMP),
        y0=X_INITIAL_STATE,
        t_eval=T_EVAL_BUMP,
        method='RK45'
    )

    # Report solver status to ensure successful integration
    print(f"Passive ODE solver success: {sol.success}, message: {sol.message}")
    print(f"Skyhook ODE solver success: {sol_sky.success}, message: {sol_sky.message}")

    # ----------------------------------------------------------------------
    # 2. EXTRACT AND PROCESS RESULTS
    # ----------------------------------------------------------------------

    # Extracting solutions from the solver object (sol.y[i, :] corresponds to state variable i)

    # Passive states (Sprung mass: 0, 1; Unsprung mass: 2, 3)
    x_s     = sol.y[0, :]
    x_s_dot = sol.y[1, :]
    x_u     = sol.y[2, :]
    x_u_dot = sol.y[3, :]

    # Skyhook states
    x_s_sky     = sol_sky.y[0, :]
    x_s_dot_sky = sol_sky.y[1, :]
    x_u_sky     = sol_sky.y[2, :]
    x_u_dot_sky = sol_sky.y[3, :]

    # ----------------------------------------------------------------------
    # 3. CALCULATE DERIVED QUANTITIES
    # ----------------------------------------------------------------------

    # --- Passive Derived Quantities ---
    # Suspension Travel (comfort/handling limit)
    travel_passive = x_s - x_u
    # Tyre Deflection (road holding/safety limit)
    # Relative Velocity (damper input)
    v_rel_passive  = x_s_dot - x_u_dot

    # Calculate Damping Force (F_d) for post-processing/acceleration calculation
    F_d_passive    = F_passive_piecewise(v_rel_passive, params.c_comp_low, params.c_comp_high,
                                        params.c_reb_low, params.c_reb_high, VELOCITY_THRESHOLD)

    # Sprung Mass Acceleration (primary metric for ride comfort)
    # x_s_ddot = (-k_s * (x_s - x_u) - F_d) / m_s
    acc_passive    = (-params.ks * travel_passive - F_d_passive) / params.ms

    # --- Skyhook Derived Quantities ---
    travel_skyhook = x_s_sky - x_u_sky

    # Calculate Damping Force (F_d) using the clipped skyhook control law
    F_d_sky        = F_skyhook_clipped(x_s_dot_sky, x_u_dot_sky, params.c_min, params.c_max)

    # Sprung Mass Acceleration (ride comfort metric)
    acc_skyhook    = (-params.ks * travel_skyhook - F_d_sky) / params.ms

    # RMS Acceleration (Overall Ride Comfort measure)
    rms_acc_passive    = np.sqrt(np.mean(acc_passive**2))
    rms_acc_skyhook    = np.sqrt(np.mean(acc_skyhook**2))

    # Peak Acceleration
    max_acc_passive = np.max(acc_passive)
    max_acc_skyhook = np.max(acc_skyhook)

    #return np.array([rms_acc_passive, rms_acc_skyhook])     # Defines values to be returned
    return np.array([max_acc_passive, max_acc_skyhook])     # Defines values to be returned

if __name__ == "__main__":
    ode_bump()
