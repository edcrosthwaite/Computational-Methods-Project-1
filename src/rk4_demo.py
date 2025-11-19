import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
m_s = 300.0    # sprung mass [kg]
m_u = 40.0     # unsprung mass [kg]
k_s = 28000.0  # suspension spring stiffness [N/m]
k_t = 250000.0 # tyre stiffness [N/m]
c_s = 1500.0   # damping [Ns/m]
v = 5.0       # vehicle speed [m/s]

# --- Road input (cosine bump) --- change clash royale
def road_input(t, v, h=0.05, L=1.0):
    x = v * t
    if 0 <= x <= L:
        return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
    return 0.0

# --- Quarter car ODE (passive only) ---
def quarter_car_ode(t, state):
    x_s, x_s_dot, x_u, x_u_dot = state

    z_r = road_input(t, v)
    x_su = x_s - x_u
    x_su_dot = x_s_dot - x_u_dot

    # Equations of motion
    x_s_ddot = (-k_s * x_su - c_s * x_su_dot) / m_s
    x_ur = x_u - z_r
    x_u_ddot = (k_s * x_su + c_s * x_su_dot - k_t * x_ur) / m_u

    return np.array([x_s_dot, x_s_ddot, x_u_dot, x_u_ddot])

# --- Fixed-step RK4 integrator ---
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# --- Simulation setup ---
t0, tf, h = 0.0, 2.0, 0.0005   # start, end, timestep
t_points = np.arange(t0, tf, h)

y = np.zeros((len(t_points), 4))  # state history
y[0, :] = [0, 0, 0, 0]            # initial conditions

# --- Time integration loop ---
for i in range(len(t_points)-1):
    t = t_points[i]
    y[i+1, :] = rk4_step(quarter_car_ode, t, y[i, :], h)

# --- Extract results ---
x_s = y[:, 0]
x_s_dot = y[:, 1]
x_u = y[:, 2]
x_u_dot = y[:, 3]

# Road input over time
z_r = np.array([road_input(t, v) for t in t_points])

# Derived quantities
travel = x_s - x_u
tyre_defl = x_u - z_r
acc_s = (-k_s * (x_s - x_u) - c_s * (x_s_dot - x_u_dot)) / m_s

# Performance metrics
max_travel = np.max(np.abs(travel))
max_tyre = np.max(np.abs(tyre_defl))
rms_acc = np.sqrt(np.mean(acc_s**2))

# --- Print results ---
print("=== PASSIVE RK4 DEMO RESULTS ===")
print(f"Vehicle speed:       {v:.1f} m/s")
print(f"Max suspension travel: {max_travel*1000:.1f} mm")
print(f"Max tyre deflection:   {max_tyre*1000:.1f} mm")
print(f"RMS sprung acceleration: {rms_acc:.3f} m/s^2")

# --- Plot displacements with road profile ---
plt.figure(figsize=(8,5))

plt.subplot(2,1,1)
plt.plot(t_points, z_r*1000, 'k--', label='Road input (z_r)')
plt.plot(t_points, x_u*1000, label='Unsprung mass (x_u)')
plt.plot(t_points, x_s*1000, label='Sprung mass (x_s)')
plt.ylabel('Displacement [mm]')
plt.legend()
plt.grid(True)
plt.title('Quarter-Car Vertical Displacements (RK4)')

# --- Plot tyre deflection (x_u - z_r) ---
plt.subplot(2,1,2)
plt.plot(t_points, (x_u - z_r)*1000, color='tab:orange', label='Tyre deflection (x_u - z_r)')
plt.xlabel('Time [s]')
plt.ylabel('Tyre Deflection [mm]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()