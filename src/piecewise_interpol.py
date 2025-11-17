# 1. Synthetic F-v data for passive damper
import numpy as np
import matplotlib.pyplot as plt

from fit_damper_advanced import compare_interpolation_methods
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.integrate import solve_ivp

from constants import (T_START, T_END, T_EVAL, VELOCITY_THRESHOLD, X_INITIAL_STATE)

from ODEdampers import F_passive_piecewise
from testing_methods.ODEparams import c_min, c_max, m_s, m_u, k_s, k_t  

# Velocity range and resolution
v_min = -1.20  # m/s
v_max =  1.20  # m/s
n_points = 61 # gives 0.04 m/s spacing

# Velocity samples
v_data = np.linspace(v_min, v_max, n_points)

# Corresponding damper forces from piecewise law (no noise)
F_data = F_passive_piecewise(v_data, c_min, c_max, VELOCITY_THRESHOLD)

print("Synthetic F–v dataset created.")
print("v_data range: ", v_data.min(), "to", v_data.max(), "[m/s]")
print("F_data range: ", F_data.min(), "to", F_data.max(), "[N]")

# adding random noise to the force data
rng = np.random.default_rng(seed=42)  # for reproducibility
noise_level = 0.05  # 5% noise

F_noisy = F_data + noise_level * np.max(np.abs(F_data)) * rng.standard_normal(size=n_points)

# Comparing interpolation/regression methods
# ---------- Step 4: compare interpolation / regression methods ----------
results = compare_interpolation_methods(v_data, F_noisy)
'''
print("\n=== Regression / Interpolation Comparison ===")
for method, metrics in results.items():
    if "error" in metrics:
        print(f"{method:20s} -> ERROR: {metrics['error']}")
    else:
        print(f"{method:20s} -> RMSE = {metrics['rmse']:.2f} N,  Smoothness = {metrics['smoothness']:.3f}")
'''
# Building cubic-spline fitted damper model for use in simulations

'''
# Fitting spline to noisy 'measured' data
F_d_spline = CubicSpline(v_data, F_noisy, extrapolate=True)
'''

# Potential smoothing fix...
F_d_spline_smooth = UnivariateSpline(v_data, F_noisy, s=0.02 * len(v_data))

# Plotting noisy data, true law, and fitted spline  
v_plot = np.linspace(v_data.min(), v_data.max(), 61)

plt.figure(figsize=(6,4))
plt.scatter(v_data, F_noisy, s=20, color='tab:orange', label='Synthetic data')
plt.plot(v_plot, F_data, 'b-', linewidth=2, label='True piecewise law')
plt.plot(v_plot, F_d_spline_smooth(v_plot), 'r--', linewidth=2, label='Cubic spline fit')
plt.xlabel('Relative velocity [m/s]')
plt.ylabel('Damper force [N]')
plt.title('Cubic Spline Fit to Nonlinear Passive Damper')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

v = 5.0 # m/s

def road_input(t, v, h=0.05, L=1.0):
    """
    Half-cosine bump of height h and length L travelled at speed v.
    """
    x = v * t
    if 0.0 <= x <= L:
        return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
    return 0.0

def quarter_car_ode_passive_spline(t, state, m_s, m_u, k_s, k_t, v, F_d_model):
    """
    Quarter-car with damper force from fitted F_d_model(v_rel).
    state = [x_s, x_s_dot, x_u, x_u_dot]
    """
    x_s, x_s_dot, x_u, x_u_dot = state

    z_r = road_input(t, v)

    x_su  = x_s - x_u
    v_rel = x_s_dot - x_u_dot
    F_d   = F_d_model(v_rel)  # <-- spline-based damper force

    x_s_ddot = (-k_s * x_su - F_d) / m_s

    x_ur     = x_u - z_r
    x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]

# Solve with spline-based damper
sol_spline = solve_ivp(
    fun=lambda t, y: quarter_car_ode_passive_spline(t, y, m_s, m_u, k_s, k_t, v, F_d_spline_smooth),
    t_span=(T_START, T_END),
    y0=X_INITIAL_STATE,
    t_eval=T_EVAL,
    method='RK45'
)

# Extract and compute basic performance metrics
t_spl     = sol_spline.t
x_s_spl   = sol_spline.y[0, :]
x_sdot_spl= sol_spline.y[1, :]
x_u_spl   = sol_spline.y[2, :]
x_udot_spl= sol_spline.y[3, :]

z_r_spl   = np.array([road_input(ti, v) for ti in t_spl])
travel_spl = x_s_spl - x_u_spl
tyre_spl   = x_u_spl - z_r_spl

v_rel_spl = x_sdot_spl - x_udot_spl
F_d_spl_t = F_d_spline_smooth(v_rel_spl)
acc_spl   = (-k_s * (x_s_spl - x_u_spl) - F_d_spl_t) / m_s

max_travel_spl = np.max(np.abs(travel_spl))
max_tyre_spl   = np.max(np.abs(tyre_spl))
rms_acc_spl    = np.sqrt(np.mean(acc_spl**2))

print("\n=== PASSIVE (spline-fitted damper in ODE) ===")
print(f"Max travel:        {max_travel_spl*1000:.2f} mm")
print(f"Max tyre defl:     {max_tyre_spl*1000:.2f} mm")
print(f"RMS acceleration:  {rms_acc_spl:.3f} m/s^2")
