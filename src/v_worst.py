import numpy as np 
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from ODEroad import road_input
from piecewise_interpol import quarter_car_ode_passive_spline
from constants import (T_START, T_END, T_EVAL, X_INITIAL_STATE, VELOCITY_THRESHOLD)
from ODEdampers import F_passive_piecewise

from params import SuspensionParams

params = SuspensionParams()

# Create F-v data points for the default damper
v_data = np.linspace(-1.5, 1.5, 61) # Velocity range for F-v curve
F_data = F_passive_piecewise(
    v_data, 
    params.c_comp_low, 
    params.c_comp_high, 
    params.c_reb_low, 
    params.c_reb_high, 
    VELOCITY_THRESHOLD
)

# Add small noise and fit the C2 continuous spline for numerical stability
rng = np.random.default_rng(seed=42)
noise_level = 0.05
F_noisy = F_data + noise_level * np.max(np.abs(F_data)) * rng.standard_normal(size=len(v_data))
F_d_spline_smooth = UnivariateSpline(v_data, F_noisy, s=0.02 * len(v_data))

# ==============================================================================
# 4. VELOCITY SWEEP EXECUTION
# ==============================================================================

# Define the speed range to sweep (1 m/s to 15 m/s)
v_sweep = np.linspace(1.0, 15.0, 30) 
rms_acc_vs_v = []

print(f"\n--- Starting Velocity Sweep for k_s = {params.ks:.0f} N/m ---")

# Loop through each velocity value and run a full simulation
for v_val in v_sweep:
    
    # 1. Solve ODE for the current velocity v_val
    sol = solve_ivp(
        # The ODE passes the fixed k_s and the looping v_val
        fun=lambda t, y: quarter_car_ode_passive_spline(t, y, params.ms, params.mu, params.ks, params.kt, v_val, F_d_spline_smooth),
        t_span=(T_START, T_END),
        y0=X_INITIAL_STATE,
        t_eval=T_EVAL,
        method='RK45'
    )
    
    # 2. Calculate Sprung Mass Acceleration
    x_s, x_u = sol.y[0, :], sol.y[2, :]
    x_s_dot, x_u_dot = sol.y[1, :], sol.y[3, :]

    travel = x_s - x_u
    v_rel = x_s_dot - x_u_dot
    
    # Calculate acceleration using the system equation: a_s = (-k_s * travel - F_d) / m_s
    F_d_t = F_d_spline_smooth(v_rel)
    acc = (-params.ks * travel - F_d_t) / params.ms 
    
    # Calculate RMS Acceleration (The discomfort metric)
    rms_acc = np.sqrt(np.mean(acc**2))
    rms_acc_vs_v.append(rms_acc)

# --- Find and Report Worst-Case Speed (v_worst) ---
rms_acc_array = np.array(rms_acc_vs_v)
v_worst_index = np.argmax(rms_acc_array)
v_worst = v_sweep[v_worst_index]
max_rms_acc = rms_acc_array[v_worst_index]

print(f"\nRESULTS:")
print(f"==========================================")
print(f"Critical Speed (v_worst) found: {v_worst:.2f} m/s")
print(f"Maximum RMS Acc at v_worst: {max_rms_acc:.3f} m/s^2")
print(f"==========================================")

# --- Plot the Sweep ---
plt.figure(figsize=(8, 5))
plt.plot(v_sweep, rms_acc_vs_v, 'b-', linewidth=2)
plt.title(f'RMS Acceleration vs. Vehicle Speed (k$_s$={params.ks:.0f} N/m)')
plt.xlabel(r'Vehicle Speed $v$ [m/s]')
plt.ylabel(r'RMS Sprung Mass Acceleration $a_{\mathrm{s,rms}}$ [m/s$^2$]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Conclusion for Next Step ---
# Use the reported v_worst value (e.g., 1.45 m/s) in the final Pareto Sweep.

