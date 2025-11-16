import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ODEparams import (
    m_s, m_u, k_s, k_t,
    c_min, c_max, v
)

# Road input, same as in main ODE code
def road_input(t, v, h=0.05, L=1.0):
    x = v * t
    if 0.0 <= x <= L:
        return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
    return 0.0

# Skyhook ODE copied from main file
def quarter_car_ode_skyhook(t, state, m_s, m_u, k_s, c_min, c_max, k_t, v):
    x_s, x_s_dot, x_u, x_u_dot = state

    z_r = road_input(t, v)

    x_su  = x_s - x_u
    v_rel = x_s_dot - x_u_dot

    # Clipped skyhook logic
    if x_s_dot * v_rel > 0:
        c_eff = c_max
    else:
        c_eff = c_min

    F_d = c_eff * v_rel

    x_s_ddot = (-k_s * x_su - F_d) / m_s

    x_ur     = x_u - z_r
    x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]

# Running a skyhook simulation

t_start = 0.0
t_end   = 3.0
t_eval  = np.linspace(t_start, t_end, 2000)
x0      = [0.0, 0.0, 0.0, 0.0]

sol_sky = solve_ivp(
    fun=lambda t, y: quarter_car_ode_skyhook(t, y, m_s, m_u, k_s, c_min, c_max, k_t, v),
    t_span=(t_start, t_end),
    y0=x0,
    t_eval=t_eval,
    method='RK45'
)

print("Skyhook ODE success:", sol_sky.success, sol_sky.message)

# ---------- Step 2: extract F–v data for semi-active (skyhook) ----------

# Unpack states
t_sky       = sol_sky.t
x_s_sky     = sol_sky.y[0, :]
x_s_dot_sky = sol_sky.y[1, :]
x_u_sky     = sol_sky.y[2, :]
x_u_dot_sky = sol_sky.y[3, :]

# Relative damper velocity
v_rel_sky = x_s_dot_sky - x_u_dot_sky

# Effective damping coefficient at each time step (same logic as ODE)
c_eff_sky = np.where(x_s_dot_sky * v_rel_sky > 0.0, c_max, c_min)

# Damper force time history
F_d_sky = c_eff_sky * v_rel_sky

print("v_rel_sky range: ", v_rel_sky.min(), "to", v_rel_sky.max(), "[m/s]")
print("F_d_sky range:   ", F_d_sky.min(),   "to", F_d_sky.max(),   "[N]")

# ---------- Step 3: split into soft / firm branches ----------

v_sky = v_rel_sky
F_sky = F_d_sky

mask_soft = c_eff_sky == c_min
mask_firm = c_eff_sky == c_max

# Adding noise to simulate experimental data

rng = np.random.default_rng(42)
noise_level = 0.05  # 5% noise relative to peak force

F_sky_noisy = F_sky + noise_level * np.max(np.abs(F_sky)) * rng.standard_normal(F_sky.shape)

# Split noisy forces into branches
F_soft = F_sky_noisy[mask_soft]
F_firm = F_sky_noisy[mask_firm]
v_soft = v_sky[mask_soft]
v_firm = v_sky[mask_firm]


v_soft, F_soft = v_sky[mask_soft], F_sky[mask_soft]
v_firm, F_firm = v_sky[mask_firm], F_sky[mask_firm]

print(f"Soft samples: {v_soft.size}, Firm samples: {v_firm.size}")

# ---------- Step 4: linear regression on each branch ----------

# Fit F ≈ a*v + b for each mode
coef_soft = np.polyfit(v_soft, F_soft, 1)
coef_firm = np.polyfit(v_firm, F_firm, 1)

a_soft, b_soft = coef_soft
a_firm, b_firm = coef_firm

print("\n=== Skyhook damper regression ===")
print(f"Soft branch: fitted slope = {a_soft:.1f} Ns/m, intercept = {b_soft:.1f} N (true c_min = {c_min})")
print(f"Firm branch: fitted slope = {a_firm:.1f} Ns/m, intercept = {b_firm:.1f} N (true c_max = {c_max})")

# ---------- Step 5: plot F–v data and fitted lines ----------

v_line = np.linspace(v_sky.min(), v_sky.max(), 200)

plt.figure(figsize=(6,4))
plt.scatter(v_soft, F_soft, s=10, alpha=0.5, color='tab:blue', label='Soft (c_min)')
plt.scatter(v_firm, F_firm, s=10, alpha=0.5, color='tab:red', label='Firm (c_max)')

# Fitted lines
plt.plot(v_line, a_soft*v_line + b_soft, 'b--', linewidth=2, label=f'Soft fit ({a_soft:.0f} Ns/m)')
plt.plot(v_line, a_firm*v_line + b_firm, 'r--', linewidth=2, label=f'Firm fit ({a_firm:.0f} Ns/m)')

plt.xlabel('Relative velocity v_rel [m/s]')
plt.ylabel('Damper force F_d [N]')
plt.title('Semi-active skyhook damper: F–v characteristics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
