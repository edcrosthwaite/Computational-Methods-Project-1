"""
Nonlinear Passive Damper Simulation and System Identification Script
--------------------------------------------------------------------
This script models a quarter-car suspension system utilizing a designer's
piecewise, nonlinear passive damper. It executes a full computational workflow:

1. System Identification: Generates synthetic, noisy force-velocity (F-v) data,
   simulating real-world measurement error.
2. Numerical Stabilization: Fits a C^2 continuous (twice-differentiable)
   UnivariateSpline to the noisy data. This is crucial to replace the analytically
   correct but numerically unstable (discontinuous derivative) piecewise law.
3. Dynamic Simulation: Solves the Quarter-Car ODE using the robust spline model.
4. Performance Analysis: Calculates the three main design metrics (Ride Comfort,
   Safety, Road Holding) to evaluate the final damper design.

Inputs:
- Suspension parameters (params.py)
- Simulation constants (constants.py)
- Piecewise law function (F_passive_piecewise)

Outputs:
- Console output: F-v data range, interpolation/regression comparison, final
  suspension performance metrics.
- Plot: F-v scatter plot of noisy data with the true piecewise law and the
  fitted smooth spline.
"""

# --- Imports and Parameter Initialization ---
import numpy as np
import matplotlib.pyplot as plt

# Core numerical tools
from fit_damper_advanced import compare_interpolation_methods
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp

# Project-specific imports
from constants import (T_START, T_END, T_EVAL, VELOCITY_THRESHOLD, X_INITIAL_STATE)
from ODEroad import road_input
from ODEdampers import F_passive_piecewise
from params import SuspensionParams

# Initialize parameters object for access to system values
params = SuspensionParams()

def piecewise_interpolation():
    # --- Step 1: Analytical Law and Synthetic Data Generation ---

    # Define the velocity domain for the F-v characteristic
    v_min = -1.20  # m/s
    v_max =  1.20  # m/s
    n_points = 61  # Resolution for sampling the characteristic
    v_data = np.linspace(v_min, v_max, n_points)

    # Generate the True Analytical Law (The 'Designer Blueprint')
    # F_data uses the four piecewise damping coefficients to define the ideal, noise-free curve.
    F_data = F_passive_piecewise(
        v_data, 
        params.c_comp_low, 
        params.c_comp_high, 
        params.c_reb_low, 
        params.c_reb_high, 
        VELOCITY_THRESHOLD
    )

    print("Synthetic F–v dataset created.")

    # Inject Simulated Experimental Noise
    # Adds 5% Gaussian noise relative to the peak force. This simulates measurement
    # error, requiring a robust system identification step (fitting/filtering).
    rng = np.random.default_rng(seed=42)  # For reproducibility
    noise_level = 0.05
    F_noisy = F_data + noise_level * np.max(np.abs(F_data)) * rng.standard_normal(size=n_points)


    # --- Step 2: System Identification and Numerical Stabilization ---

    # Compare interpolation/regression methods to justify the final choice
    results = compare_interpolation_methods(v_data, F_noisy)
    print("\n=== Regression / Interpolation Comparison (Model Choice) ===")
    for method, metrics in results.items():
        if "error" in metrics:
            print(f"{method:20s} -> ERROR: {metrics['error']}")
        else:
            print(f"{method:20s} -> RMSE = {metrics['rmse']:.2f} N,  Smoothness = {metrics['smoothness']:.3f}")

    # Critical Step: Fit the Smoothed UnivariateSpline Model
    # F_d_spline_smooth is a callable function object that replaces the piecewise law.
    # The smoothing factor 's' forces the spline to filter out the noise and non-differentiable
    # corners, guaranteeing the C^2 continuity needed for the RK45 solver's stability.
    F_d_spline_smooth = UnivariateSpline(v_data, F_noisy, s=0.02 * len(v_data))


    # --- Step 3: Model Justification and Visualization ---

    # Define a high-resolution velocity array for smooth plotting lines
    v_plot = np.linspace(v_data.min(), v_data.max(), 200)

    # Recalculate true law force on the high-resolution array for a smooth plot line
    F_plot_true = F_passive_piecewise(
        v_plot, params.c_comp_low, params.c_comp_high, params.c_reb_low, params.c_reb_high, VELOCITY_THRESHOLD
    )

    plt.figure(figsize=(6,4))
    # Scatter plot shows the noisy input data
    plt.scatter(v_data, F_noisy, s=20, color='tab:orange', label='Synthetic data (F_noisy)')
    # Plot of the original, ideal design (discontinuous)
    plt.plot(v_plot, F_plot_true, 'b-', linewidth=2, label='True Piecewise Law')
    # Plot of the stable, fitted model (C^2 continuous)
    plt.plot(v_plot, F_d_spline_smooth(v_plot), 'r--', linewidth=2, label='Smoothed Spline Fit')
    plt.xlabel(r'Relative velocity $v_{\mathrm{rel}}$ [m/s]')
    plt.ylabel(r'Damper force $F_{\mathrm{d}}$ [N]')
    plt.title('System Identification: Stabilizing the Nonlinear Damper Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # --- Step 4: Dynamic Model Definition (The ODE) ---
    def quarter_car_ode_passive_spline(t, state, m_s, m_u, k_s, k_t, v, F_d_model):
        """
        Quarter-car ODE using the spline-fitted damper model.
        state = [x_s, x_s_dot, x_u, x_u_dot]
        F_d_model is the callable UnivariateSpline function object.
        """
        x_s, x_s_dot, x_u, x_u_dot = state

        z_r = road_input(t, v)

        x_su  = x_s - x_u
        v_rel = x_s_dot - x_u_dot
        
        # CORE IMPLEMENTATION: Dynamically retrieve damper force from the smooth spline.
        # This line implements the numerical model into the physical system.
        F_d   = F_d_model(v_rel) 

        # Equations of Motion (Newton's Second Law):
        x_s_ddot = (-k_s * x_su - F_d) / m_s      # Sprung Mass (Chassis) acceleration
        x_ur     = x_u - z_r
        x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u  # Unsprung Mass (Wheel) acceleration

        return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


    # --- Step 5: Time-Domain Simulation (Solving the ODE) ---

    # Integrate the system using the stable spline function
    sol_spline = solve_ivp(
        # The lambda function passes the smooth spline F_d_spline_smooth to the ODE
        fun=lambda t, y: quarter_car_ode_passive_spline(t, y, params.ms, params.mu, params.ks, params.kt, params.v, F_d_spline_smooth),
        t_span=(T_START, T_END),
        y0=X_INITIAL_STATE,
        t_eval=T_EVAL,
        method='RK45'  # Robust and efficient explicit Runge-Kutta method
    )


    # --- Step 6: Performance Analysis and Metric Extraction ---

    # Extract primary state variables from the solution
    t_spl     = sol_spline.t
    x_s_spl   = sol_spline.y[0, :]
    x_u_spl   = sol_spline.y[2, :]

    # Calculate required derived quantities: road input and deflections
    z_r_spl   = np.array([road_input(ti, params.v) for ti in t_spl])
    travel_spl = x_s_spl - x_u_spl        # Suspension Travel
    tyre_spl   = x_u_spl - z_r_spl        # Tyre Deflection

    # Calculate Sprung Mass Acceleration (required for RMS metric)
    v_rel_spl = sol_spline.y[1, :] - sol_spline.y[3, :]
    F_d_spl_t = F_d_spline_smooth(v_rel_spl)
    acc_spl   = (-params.ks * (x_s_spl - x_u_spl) - F_d_spl_t) / params.ms 

    # Calculate Final Engineering Metrics (Design Evaluation)
    max_travel_spl = np.max(np.abs(travel_spl)) # Max Suspension Deflection (Safety/Durability Metric)
    max_tyre_spl   = np.max(np.abs(tyre_spl))   # Max Tyre Deflection (Road Holding Metric)
    rms_acc_spl    = np.sqrt(np.mean(acc_spl**2)) # RMS Acceleration (Ride Comfort Metric)

    print("\n=== PASSIVE DAMPER PERFORMANCE ANALYSIS ===")
    print(f"Max travel (Safety):        {max_travel_spl*1000:.2f} mm")
    print(f"Max tyre defl (Road Holding): {max_tyre_spl*1000:.2f} mm")
    print(f"RMS acceleration (Comfort):  {rms_acc_spl:.3f} m/s^2")

    print("\nPassive damper simulation and analysis complete.")

if __name__ == "__main__":
    piecewise_interpolation()