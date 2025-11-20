"""
ODE_main.py

Main execution script for the quarter-car suspension simulation. 
This script handles initialization, runs the Ordinary Differential Equation (ODE) 
solvers for both the passive and semi-active (skyhook) models, processes the 
simulation results, calculates key performance metrics (RMS acceleration, travel, 
road holding), and generates comparative plots.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Import constants (like time limits, initial state) and vehicle parameters
from src.constants import (
    VELOCITY_THRESHOLD, T_START, T_END_ISO, T_EVAL_ISO, T_END_BUMP, T_EVAL_BUMP, X_INITIAL_STATE
)

# Import model components
from src.input_calculation import road_input, road_iso # Road excitation profile
from src.dampers import F_passive_piecewise, F_skyhook_clipped # Damping force models
from src.system_definition import quarter_car_ode_passive, quarter_car_ode_skyhook # ODE system definitions
from src.params import SuspensionParams # Vehicle mass and stiffness parameters

# Initialise vehicle parameters instance
params = SuspensionParams()

def ode(test_type: str = "bump"):
    """
    Runs the numerical simulation for both damper configurations, calculates 
    performance metrics, and plots the comparative results.
    """
    # ----------------------------------------------------------------------
    # 1. ODE SOLVER EXECUTION
    # ----------------------------------------------------------------------

    if test_type == "iso":
        # -------------------------------------------------------------
        # ALIGN INITIAL STATE WITH ROAD HEIGHT AT T_START
        # -------------------------------------------------------------
        z0 = road_iso(T_START, params.v)   # ← use ISO, or road_input if using bump

        X0 = np.array(X_INITIAL_STATE, dtype=float)
        X0[0] = z0   # sprung mass displacement = road height
        X0[2] = z0   # unsprung mass displacement = road height

        # --- Passive simulation (Asymmetric Piecewise Damper) ---
        # The fun argument is a lambda function to pass all required parameters to the ODE solver.
        sol = solve_ivp(
            fun=lambda t, y: quarter_car_ode_passive(t, y, params.ms, params.mu, params.ks, params.c_comp_low, params.c_comp_high, params.c_reb_low, params.c_reb_high, VELOCITY_THRESHOLD, params.kt, params.v, z_r=road_iso(t, params.v)),
            t_span=(T_START, T_END_ISO), # Time range for integration
            y0=X0,      # Initial state vector [x_s, x_s_dot, x_u, x_u_dot]
            t_eval=T_EVAL_ISO,           # Specific time points at which to store the solution
            method='RK45'            # Runge-Kutta 4(5) is a robust integration method
        )

        X0 = X_INITIAL_STATE
        X0[0] = z0   # sprung mass displacement = road height
        X0[2] = z0   # unsprung mass displacement = road height
    # --- Skyhook simulation (Clipped Skyhook Semi-Active Damper) ---
        sol_sky = solve_ivp(
            fun=lambda t, y: quarter_car_ode_skyhook(t, y, params.ms, params.mu, params.ks, params.c_min, params.c_max, params.kt, params.v, z_r=road_iso(t, params.v)),
            t_span=(T_START, T_END_ISO),
            y0=X0,
            t_eval=T_EVAL_ISO,
            method='RK45'
        )
    
    # Bump test type
    else:
        sol = solve_ivp(
        fun=lambda t, y: quarter_car_ode_passive(t, y, params.ms, params.mu, params.ks, params.c_comp_low, params.c_comp_high, params.c_reb_low, params.c_reb_high, VELOCITY_THRESHOLD, params.kt, params.v, z_r=road_input(t, params.v)),
        t_span=(T_START, T_END_BUMP), # Time range for integration
        y0=X_INITIAL_STATE,      # Initial state vector [x_s, x_s_dot, x_u, x_u_dot]
        t_eval=T_EVAL_BUMP,           # Specific time points at which to store the solution
        method='RK45'            # Runge-Kutta 4(5) is a robust integration method
        )

        # --- Skyhook simulation (Clipped Skyhook Semi-Active Damper) ---
        sol_sky = solve_ivp(
            fun=lambda t, y: quarter_car_ode_skyhook(t, y, params.ms, params.mu, params.ks, params.c_min, params.c_max, params.kt, params.v, z_r=road_input(t, params.v)),
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
    t = sol.t
    x_s     = sol.y[0, :]
    x_s_dot = sol.y[1, :]
    x_u     = sol.y[2, :]
    x_u_dot = sol.y[3, :]

    # Skyhook states
    t_sky       = sol_sky.t
    x_s_sky     = sol_sky.y[0, :]
    x_s_dot_sky = sol_sky.y[1, :]
    x_u_sky     = sol_sky.y[2, :]
    x_u_dot_sky = sol_sky.y[3, :]

    # Road profiles for both simulations (at the evaluated time points)
    if test_type == "bump":
        z_r_passive = np.array([road_input(ti, params.v) for ti in t])
        z_r_sky = np.array([road_input(ti, params.v) for ti in t_sky])
    
    else:
        z_r_passive = np.array([road_iso(ti, params.v) for ti in t])
        z_r_sky = np.array([road_iso(ti, params.v) for ti in t_sky])

    # ----------------------------------------------------------------------
    # 3. CALCULATE DERIVED QUANTITIES
    # ----------------------------------------------------------------------

    # --- Passive Derived Quantities ---
    # Suspension Travel (comfort/handling limit)
    travel_passive = x_s - x_u 
    # Tyre Deflection (road holding/safety limit)
    tyre_passive   = x_u - z_r_passive
    # Relative Velocity (damper input)
    v_rel_passive  = x_s_dot - x_u_dot
    
    # Calculate Damping Force (F_d) for post-processing/acceleration calculation
    F_d_passive    = F_passive_piecewise(v_rel_passive, params.c_comp_low, params.c_comp_high, params.c_reb_low, params.c_reb_high, VELOCITY_THRESHOLD)
    
    # Sprung Mass Acceleration (primary metric for ride comfort)
    # x_s_ddot = (-k_s * (x_s - x_u) - F_d) / m_s
    acc_passive    = (-params.ks * travel_passive - F_d_passive) / params.ms

    # --- Skyhook Derived Quantities ---
    travel_skyhook = x_s_sky - x_u_sky
    tyre_skyhook   = x_u_sky - z_r_sky
    vel_rel_sky    = x_s_dot_sky - x_u_dot_sky
    
    # Calculate Damping Force (F_d) using the clipped skyhook control law
    F_d_sky        = F_skyhook_clipped(x_s_dot_sky, x_u_dot_sky, params.c_min, params.c_max)
    
    # Sprung Mass Acceleration (ride comfort metric)
    acc_skyhook    = (-params.ks * travel_skyhook - F_d_sky) / params.ms

    # ----------------------------------------------------------------------
    # 4. PERFORMANCE METRICS (Objective Evaluation)
    # ----------------------------------------------------------------------

    # --- Passive Metrics ---
    # Max Suspension Travel (Clearance check)
    max_travel_passive = np.max(np.abs(travel_passive))
    # Max Tire Deflection (Road Holding check)
    max_tyre_passive   = np.max(np.abs(tyre_passive))
    # RMS Acceleration (Overall Ride Comfort measure)
    rms_acc_passive    = np.sqrt(np.mean(acc_passive**2))

    # --- Skyhook Metrics ---
    max_travel_skyhook = np.max(np.abs(travel_skyhook))
    max_tyre_skyhook   = np.max(np.abs(tyre_skyhook))
    rms_acc_skyhook    = np.sqrt(np.mean(acc_skyhook**2))

    # ----------------------------------------------------------------------
    # 5. PRINT RESULTS
    # ----------------------------------------------------------------------

    print("=== PASSIVE (piecewise nonlinear) (BUMP) ===")
    print(f"Max travel:        {max_travel_passive*1000:.2f} mm")
    print(f"Max tyre defl:     {max_tyre_passive*1000:.2f} mm")
    print(f"RMS acceleration:  {rms_acc_passive:.3f} m/s^2")

    print("\n=== SKYHOOK (clipped) (BUMP) ===")
    print(f"Max travel:        {max_travel_skyhook*1000:.2f} mm")
    print(f"Max tyre defl:     {max_tyre_skyhook*1000:.2f} mm")
    print(f"RMS acceleration:  {rms_acc_skyhook:.3f} m/s^2")

    # ----------------------------------------------------------------------
    # 6. PLOTTING (Visualization)
    # ----------------------------------------------------------------------
    
    # NOTE: The original plotting code (showing displacements for passive then skyhook)
    # has been replaced by the consolidated comparison plot below for better analysis.

    # --- Consolidated Comparison Plot ---
    # A single figure comparing key performance metrics side-by-side.
    plt.figure(figsize=(10, 8))

    # 1. Sprung Mass Acceleration Comparison (Comfort)
    plt.subplot(3, 1, 1)
    plt.plot(t, acc_passive, 'r', label='Passive Acceleration')
    plt.plot(t_sky, acc_skyhook, 'b', label='Skyhook Acceleration')
    plt.ylabel(r'Acc ($\ddot{x}_s$) [m/s$^2$]')
    plt.title('Performance Comparison: Passive vs. Clipped Skyhook')
    plt.legend()
    plt.grid(True)

    # 2. Suspension Travel Comparison (Handling/Limits)
    plt.subplot(3, 1, 2)
    plt.plot(t, travel_passive * 1000, 'r', label='Passive Travel')
    plt.plot(t_sky, travel_skyhook * 1000, 'b', label='Skyhook Travel')
    plt.ylabel('Travel [mm]')
    plt.legend()
    plt.grid(True)

    # 3. Tire Deflection Comparison (Road Holding)
    plt.subplot(3, 1, 3)
    plt.plot(t, tyre_passive * 1000, 'r', label='Passive Tyre Defl.')
    plt.plot(t_sky, tyre_skyhook * 1000, 'b', label='Skyhook Tyre Defl.')
    plt.xlabel('Time [s]')
    plt.ylabel('Tyre Defl. [mm]')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ode()