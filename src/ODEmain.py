# main.py

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from constants import (
    VELOCITY_THRESHOLD, T_START, T_END, T_EVAL, X_INITIAL_STATE
)

from ODEroad import road_input
from ODEdampers import F_passive_piecewise
from ODEodes import quarter_car_ode_passive, quarter_car_ode_skyhook
from params import SuspensionParams

params = SuspensionParams()


# Passive simulation
sol = solve_ivp(
    fun=lambda t, y: quarter_car_ode_passive(t, y, params.ms, params.mu, params.ks, params.c_min, params.c_max, VELOCITY_THRESHOLD, params.kt, params.v),
    t_span=(T_START, T_END),
    y0=X_INITIAL_STATE,
    t_eval=T_EVAL,
    method='RK45'
)

# Skyhook simulation
sol_sky = solve_ivp(
    fun=lambda t, y: quarter_car_ode_skyhook(t, y, params.ms, params.mu, params.ks, params.c_min, params.c_max, params.kt, params.v),
    t_span=(T_START, T_END),
    y0=X_INITIAL_STATE,
    t_eval=T_EVAL,
    method='RK45'
)

print(f"Passive ODE solver success: {sol.success}, message: {sol.message}")
print(f"Skyhook ODE solver success: {sol_sky.success}, message: {sol_sky.message}")

# Extracting solutions

# Passive
t = sol.t
x_s     = sol.y[0, :]
x_s_dot = sol.y[1, :]
x_u     = sol.y[2, :]
x_u_dot = sol.y[3, :]

# Skyhook
t_sky       = sol_sky.t
x_s_sky     = sol_sky.y[0, :]
x_s_dot_sky = sol_sky.y[1, :]
x_u_sky     = sol_sky.y[2, :]
x_u_dot_sky = sol_sky.y[3, :]

# Road profiles
z_r_passive = np.array([road_input(ti, params.v) for ti in t])
z_r_sky     = np.array([road_input(ti, params.v) for ti in t_sky])

# Derived quantities

# Passive
travel_passive = x_s - x_u
tyre_passive   = x_u - z_r_passive
v_rel_passive  = x_s_dot - x_u_dot
F_d_passive    = F_passive_piecewise(v_rel_passive, params.c_min, params.c_max, VELOCITY_THRESHOLD)
acc_passive    = (-params.ks * (x_s - x_u) - F_d_passive) / params.ms

# Skyhook
travel_skyhook = x_s_sky - x_u_sky
tyre_skyhook   = x_u_sky - z_r_sky
vel_rel_sky    = x_s_dot_sky - x_u_dot_sky
c_eff_sky      = np.where(x_s_dot_sky * vel_rel_sky > 0, params.c_max, params.c_min)
F_d_sky        = c_eff_sky * vel_rel_sky
acc_skyhook    = (-params.ks * (x_s_sky - x_u_sky) - F_d_sky) / params.ms

# Performance metrics 

# Passive
max_travel_passive = np.max(np.abs(travel_passive))
max_tyre_passive   = np.max(np.abs(tyre_passive))
rms_acc_passive    = np.sqrt(np.mean(acc_passive**2))

# Skyhook
max_travel_skyhook = np.max(np.abs(travel_skyhook))
max_tyre_skyhook   = np.max(np.abs(tyre_skyhook))
rms_acc_skyhook    = np.sqrt(np.mean(acc_skyhook**2))

# --------------- Print results --------------

print("=== PASSIVE (piecewise nonlinear) ===")
print(f"Max travel:        {max_travel_passive*1000:.2f} mm")
print(f"Max tyre defl:     {max_tyre_passive*1000:.2f} mm")
print(f"RMS acceleration:  {rms_acc_passive:.3f} m/s^2")

print("\n=== SKYHOOK (clipped) ===")
print(f"Max travel:        {max_travel_skyhook*1000:.2f} mm")
print(f"Max tyre defl:     {max_tyre_skyhook*1000:.2f} mm")
print(f"RMS acceleration:  {rms_acc_skyhook:.3f} m/s^2")

# --- Plot displacements with road profile ---
plt.figure(figsize=(8,5))

plt.subplot(2,1,1)
plt.plot(t, z_r_passive*1000, 'k--', label='Road input (z_r)')
plt.plot(t, x_u*1000, label='Unsprung mass (x_u)')
plt.plot(t, x_s*1000, label='Sprung mass (x_s)')
plt.ylabel('Displacement [mm]')
plt.legend()
plt.grid(True)
plt.title('Quarter-Car Vertical Displacements (RK45)')

# --- Plot tyre deflection (x_u - z_r) ---
plt.subplot(2,1,2)
plt.plot(t, (x_u - z_r_passive)*1000, color='tab:orange', label='Tyre deflection (x_u - z_r)')
plt.xlabel('Time [s]')
plt.ylabel('Tyre Deflection [mm]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Plot displacements with road profile (skyhook) ---
plt.figure(figsize=(8,5))

plt.subplot(2,1,1)
plt.plot(t_sky, z_r_passive*1000, 'k--', label='Road input (z_r)')
plt.plot(t_sky, x_u_sky*1000, label='Unsprung mass (x_u)')
plt.plot(t_sky, x_s_sky*1000, label='Sprung mass (x_s)')
plt.ylabel('Displacement [mm]')
plt.legend()
plt.grid(True)
plt.title('Quarter-Car Vertical Displacements (RK45)')

# --- Plot tyre deflection (x_u - z_r) ---
plt.subplot(2,1,2)
plt.plot(t_sky, (x_u_sky - z_r_sky)*1000, color='tab:orange', label='Tyre deflection (x_u - z_r)')
plt.xlabel('Time [s]')
plt.ylabel('Tyre Deflection [mm]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
