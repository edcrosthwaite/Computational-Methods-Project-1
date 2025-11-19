"""
Skyhook Damper Regression and Analysis Script
--------------------------------------------
This script simulates the semi-active quarter-car model using a clipped Skyhook 
control strategy. It then performs data post-processing to extract the Damper Force 
vs. Relative Velocity (F-v) characteristic. A key step involves applying linear 
regression to the noisy data points, verifying that the intended minimum (c_min) 
and maximum (c_max) damping coefficients are accurately recovered.

Inputs:
- Suspension parameters (params.py)
- Simulation constants (constants.py)
- ODE function (quarter_car_ode_skyhook)

Outputs:
- Console output: ODE success status, F-v data range, regression results (slopes, errors).
- Plot: F-v scatter plot of noisy data with fitted linear regression lines.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Imports from project modules ---

# Core simulation settings (T_START, T_END, T_EVAL, X_INITIAL_STATE)
from constants import (T_START, T_END, T_EVAL, X_INITIAL_STATE)

# Suspension parameters (masses, stiffness, damping bounds)
from params import SuspensionParams 

# The Skyhook ODE function (defines the system dynamics)
from ODEodes import quarter_car_ode_skyhook

# (HIGH MODULARITY NOTE): To avoid duplicating the control logic (np.where) 
# in Step 2, the 'get_c_eff' helper function should be imported from ODEdampers.py:
# from ODEdampers import get_c_eff 


# Initialize parameters object for access to system values (m_s, c_min, etc.)
params = SuspensionParams()

def skyhook_interpolation():
    # ---------- Step 1: Time-Domain Simulation (Solving the ODE) ----------

    # Solve the Ordinary Differential Equations (ODEs) for the Skyhook system.
    # The 'fun' lambda wraps the state vector 'y' and time 't' for the solver, 
    # while passing all system parameters (ms, mu, etc.) to the ODE function.
    sol_sky = solve_ivp(
        fun=lambda t, y: quarter_car_ode_skyhook(t, y, 
                                                params.ms, params.mu, params.ks, 
                                                params.c_min, params.c_max, params.kt, 
                                                params.v),
        t_span=(T_START, T_END),        # Time span for integration
        y0=X_INITIAL_STATE,             # Initial conditions for all states
        t_eval=T_EVAL,                  # Specific time points to return results
        method='RK45'                   # Runge-Kutta 4(5) is a robust solver
    )

    print("Skyhook ODE success:", sol_sky.success, sol_sky.message)


    # ---------- Step 2: Extract F–v Data and Calculate Damper Force History ----------

    # Unpack all four state variables (position and velocity for sprung/unsprung masses)
    # The second dimension [1, :] and [3, :] extracts the velocity arrays.
    t_sky       = sol_sky.t
    x_s_sky     = sol_sky.y[0, :]
    x_s_dot_sky = sol_sky.y[1, :]
    x_u_sky     = sol_sky.y[2, :]
    x_u_dot_sky = sol_sky.y[3, :]

    # Relative damper velocity, v_rel = x_s_dot - x_u_dot [m/s]
    v_rel_sky = x_s_dot_sky - x_u_dot_sky

    # Effective damping coefficient at each time step (same logic as ODE).
    # This logic implements the clipped Skyhook rule: c_eff = c_max if (x_s_dot * v_rel > 0), else c_min.
    # (MODULARITY WARNING): This 'np.where' logic is a duplication of the control law
    # defined in the ODE. The high-grade solution is to import and use 'get_c_eff' here.
    c_eff_sky = np.where(x_s_dot_sky * v_rel_sky > 0.0, params.c_max, params.c_min)

    # Damper force time history is calculated by the linear law: F_d = c_eff * v_rel [N]
    F_d_sky = c_eff_sky * v_rel_sky

    print(f"v_rel_sky range: {v_rel_sky.min():.2f} to {v_rel_sky.max():.2f} [m/s]")
    print(f"F_d_sky range:   {F_d_sky.min():.2f} to {F_d_sky.max():.2f} [N]")


    # ---------- Step 3: Split Data and Introduce Simulated Measurement Noise ----------

    v_sky = v_rel_sky
    F_sky = F_d_sky

    # Create boolean masks to separate data points into the two operating modes (c_min and c_max)
    mask_soft = c_eff_sky == params.c_min
    mask_firm = c_eff_sky == params.c_max

    # Initialize a random number generator for reproducibility
    rng = np.random.default_rng(42)
    noise_level = 0.05  # 5% peak noise (common for experimental measurement error)

    # Introduce random noise (Gaussian distribution) to the force data.
    # The noise magnitude is scaled by 5% of the overall maximum force observed.
    F_sky_noisy = F_sky + noise_level * np.max(np.abs(F_sky)) * rng.standard_normal(F_sky.shape)

    # Apply the masks to the noisy data to separate the Soft and Firm branches.
    F_soft = F_sky_noisy[mask_soft]
    F_firm = F_sky_noisy[mask_firm]
    v_soft = v_sky[mask_soft]
    v_firm = v_sky[mask_firm]

    print(f"Soft samples: {v_soft.size}, Firm samples: {v_firm.size}")


    # ---------- Step 4: Linear Regression Analysis (Model Identification) ----------

    # np.polyfit(x, y, 1) performs a linear regression (1st order polynomial) 
    # and returns the coefficients [slope 'a', intercept 'b'].

    # Fit F ≈ a*v + b for the Soft mode (expected slope: c_min)
    coef_soft = np.polyfit(v_soft, F_soft, 1)
    a_soft, b_soft = coef_soft

    # Fit F ≈ a*v + b for the Firm mode (expected slope: c_max)
    coef_firm = np.polyfit(v_firm, F_firm, 1)
    a_firm, b_firm = coef_firm

    # Calculate the regression error: |Fitted Slope - True Coefficient|
    error_soft = abs(a_soft - params.c_min)
    error_firm = abs(a_firm - params.c_max)


    # Enhanced printout for robust analytical reporting
    print("\n=== Skyhook damper regression ===")
    print(f"Soft branch: fitted slope = {a_soft:.1f} Ns/m (True: {params.c_min:.1f}), Intercept = {b_soft:.1f} N")
    print(f"   --> Regression Error: {error_soft:.2f} Ns/m (Measures fidelity to c_min)")
    print(f"Firm branch: fitted slope = {a_firm:.1f} Ns/m (True: {params.c_max:.1f}), Intercept = {b_firm:.1f} N")
    print(f"   --> Regression Error: {error_firm:.2f} Ns/m (Measures fidelity to c_max)")


    # ---------- Step 5: Plotting the F–v Characteristics ----------

    # Create a smooth line array for plotting the fitted models
    v_line = np.linspace(v_sky.min(), v_sky.max(), 200)

    plt.figure(figsize=(6,4))

    # Scatter plot of the noisy, soft-mode data
    plt.scatter(v_soft, F_soft, s=10, alpha=0.5, color='tab:blue', label=r'Soft ($c_{\mathrm{min}}$)')
    # Scatter plot of the noisy, firm-mode data
    plt.scatter(v_firm, F_firm, s=10, alpha=0.5, color='tab:red', label=r'Firm ($c_{\mathrm{max}}$)')

    # Plot the fitted linear model for the Soft branch (y = a_soft * x + b_soft)
    plt.plot(v_line, a_soft*v_line + b_soft, 'b--', linewidth=2, label=f'Soft fit ({a_soft:.0f} Ns/m)')
    # Plot the fitted linear model for the Firm branch (y = a_firm * x + b_firm)
    plt.plot(v_line, a_firm*v_line + b_firm, 'r--', linewidth=2, label=f'Firm fit ({a_firm:.0f} Ns/m)')

    # Axis and plot labels using LaTeX formatting for clarity
    plt.xlabel(r'Relative velocity $v_{\mathrm{rel}}$ [m/s]')
    plt.ylabel(r'Damper force $F_{\mathrm{d}}$ [N]')
    plt.title(r'Semi-active skyhook damper: $F-v$ characteristics (with noise)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('skyhook_f_v_regression.png') # Uncomment to save the final figure
    plt.show()

    # Final confirmation that all steps ran successfully
    print("\nSkyhook F-v regression analysis complete.")

if __name__ == "__main__":
    skyhook_interpolation()

