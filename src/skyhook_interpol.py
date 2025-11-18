import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from constants import (
    T_START, T_END, T_EVAL, X_INITIAL_STATE
)

from ODEodes import quarter_car_ode_skyhook

from params import SuspensionParams

params = SuspensionParams()


sol_sky = solve_ivp(
    fun=lambda t, y: quarter_car_ode_skyhook(t, y, params.ms, params.mu, params.ks, params.c_min, params.c_max, params.kt, params.v),
    t_span=(T_START, T_END),
    y0=X_INITIAL_STATE,
    t_eval=T_EVAL,
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
c_eff_sky = np.where(x_s_dot_sky * v_rel_sky > 0.0, params.c_max, params.c_min)

# Damper force time history
F_d_sky = c_eff_sky * v_rel_sky

print(f"v_rel_sky range: {v_rel_sky.min():.2f} to {v_rel_sky.max():.2f} [m/s]")
print(f"F_d_sky range:   {F_d_sky.min():.2f} to {F_d_sky.max():.2f} [N]")

# ---------- Step 3: split into soft / firm branches ----------

v_sky = v_rel_sky
F_sky = F_d_sky

mask_soft = c_eff_sky == params.c_min
mask_firm = c_eff_sky == params.c_max

# Adding noise to simulate experimental data
rng = np.random.default_rng(42)
noise_level = 0.05  # 5% noise relative to peak force

F_sky_noisy = F_sky + noise_level * np.max(np.abs(F_sky)) * rng.standard_normal(F_sky.shape)

# Split noisy forces into branches
F_soft = F_sky_noisy[mask_soft]
F_firm = F_sky_noisy[mask_firm]
v_soft = v_sky[mask_soft]
v_firm = v_sky[mask_firm]

print(f"Soft samples: {v_soft.size}, Firm samples: {v_firm.size}")

# ---------- Step 4: linear regression on each branch ----------

# Fit F ≈ a*v + b for each mode
coef_soft = np.polyfit(v_soft, F_soft, 1)
coef_firm = np.polyfit(v_firm, F_firm, 1)

a_soft, b_soft = coef_soft
a_firm, b_firm = coef_firm

# Calculate regression errors (the difference between fitted slope and true coefficient)
error_soft = abs(a_soft - params.c_min)
error_firm = abs(a_firm - params.c_max)

print("\n=== Skyhook damper regression ===")
print(f"Soft branch: fitted slope = {a_soft:.1f} Ns/m (True: {params.c_min:.1f}), Intercept = {b_soft:.1f} N")
print(f"   --> Regression Error: {error_soft:.2f} Ns/m")
print(f"Firm branch: fitted slope = {a_firm:.1f} Ns/m (True: {params.c_max:.1f}), Intercept = {b_firm:.1f} N")
print(f"   --> Regression Error: {error_firm:.2f} Ns/m")
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
plt.title('Semi-active skyhook damper: F–v characteristics (with noise)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
