import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ================================================================
# PARAMETERS
# ================================================================

def parameters():
    """Return dictionary of all model parameters."""
    return {
        "m_s": 300.0,
        "m_u": 40.0,
        "k_s": 20000.0,
        "k_t": 200000.0,
        "c_base": 1500.0,   # passive damping
        "c_comp": 3000.0,   # skyhook compression damping
        "c_reb": 1500.0,    # skyhook rebound damping
    }


# ================================================================
# ROAD PROFILE
# ================================================================
# Half-cosine bump road input (industry standard)
def road_speed_bump(t, v, h=0.05, L=1.0):
    x = v * t
    if 0 <= x <= L:
        return 0.5 * h * (1 - np.cos(2 * np.pi * x / L))
    return 0.0

# Pothole road input (avoids numerical stifness that a sharp step causes)
def road_pothole(t, v, depth=0.04, L_total=0.6, ramp=0.1):
    x = v * t

    x_start = 1.0           # start location of pothole
    x_drop  = x_start + ramp
    x_rise  = x_start + L_total - ramp
    x_end   = x_start + L_total

    if x < x_start:
        return 0.0
    elif x_start <= x < x_drop:
        # ramp down
        return -depth * (x - x_start) / ramp
    elif x_drop <= x <= x_rise:
        # flat bottom
        return -depth
    elif x_rise < x <= x_end:
        # ramp up
        return -depth * (1 - (x - x_rise) / ramp)
    else:
        return 0.0

# ================================================================
# PASSIVE SUSPENSION MODEL
# ================================================================

def passive_ode(t, x, p, v, road_fn):
    """Quarter-car passive suspension ODE."""
    x_s, x_s_dot, x_u, x_u_dot = x

    z_r = road_fn(t, v)

    rel_pos = x_s - x_u
    rel_vel = x_s_dot - x_u_dot

    x_s_ddot = (-p["k_s"] * rel_pos - p["c_base"] * rel_vel) / p["m_s"]
    x_ur = x_u - z_r
    x_u_ddot = (p["k_s"] * rel_pos + p["c_base"] * rel_vel - p["k_t"] * x_ur) / p["m_u"]

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


# ================================================================
# SKYHOOK SUSPENSION MODEL
# ================================================================

def skyhook_ode(t, x, p, v, road_fn):
    """Skyhook-controlled quarter-car ODE."""
    x_s, x_s_dot, x_u, x_u_dot = x

    z_r = road_fn(t, v)

    rel_pos = x_s - x_u
    rel_vel = x_s_dot - x_u_dot

    # Skyhook logic
    if x_s_dot * rel_vel > 0:
        c_eff = p["c_comp"]
    else:
        c_eff = p["c_reb"]

    x_s_ddot = (-p["k_s"] * rel_pos - c_eff * rel_vel) / p["m_s"]
    x_ur = x_u - z_r
    x_u_ddot = (p["k_s"] * rel_pos + c_eff * rel_vel - p["k_t"] * x_ur) / p["m_u"]

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


# ================================================================
# SIMULATION WRAPPER
# ================================================================

def simulate(ode, p, v, t_end=5.0, x0=None, road_fn=road_speed_bump):
    """Run simulation for any ODE model."""
    if x0 is None:
        x0 = [0, 0, 0, 0]

    t_eval = np.linspace(0, t_end, 2000)

    sol = solve_ivp(
        lambda t, y: ode(t, y, p, v, road_fn),
        (0, t_end),
        x0,
        t_eval=t_eval,
        method='RK45'
    )

    return sol


# ================================================================
# PERFORMANCE METRICS
# ================================================================

def metrics(sol, p, v, road_fn, damping_mode="passive"):
    """Compute max travel, tyre deflection, RMS acceleration."""
    
    t = sol.t
    x_s     = sol.y[0]
    x_s_dot = sol.y[1]
    x_u     = sol.y[2]
    x_u_dot = sol.y[3]

    z_r = np.array([road_fn(ti, v) for ti in t])

    travel = x_s - x_u
    tyre   = x_u - z_r
    rel_vel = x_s_dot - x_u_dot

    # Skyhook needs exact time-varying damping
    if damping_mode == "skyhook":
        c_eff = np.where(x_s_dot * rel_vel > 0, p["c_comp"], p["c_reb"])
    else:
        c_eff = p["c_base"]

    acc = (-p["k_s"] * (x_s - x_u) - c_eff * rel_vel) / p["m_s"]

    rms = np.sqrt(np.mean(acc**2))

    return {
        "max_travel_mm":  np.max(np.abs(travel)) * 1000,
        "max_tyre_mm":    np.max(np.abs(tyre))   * 1000,
        "rms_acc":        rms
    }

# Speed Sweep Function
def speed_sweep(ode, p, speeds, t_end=5.0, road_fn=road_speed_bump):
    """Sweep over vehicle speeds and compute RMS acceleration."""
    results = {}
    for v in speeds:
        sol = simulate(ode, p, v, t_end, road_fn=road_fn)
        metrics_v = metrics(sol, p, v, road_fn, damping_mode="passive" if ode==passive_ode else "skyhook")
        results[v] = metrics_v
    return results



# PLOTTING RESULTS
def plot_results(sol, p, v, road_fn, title="Suspension Response"):
    t = sol.t
    x_s     = sol.y[0]
    x_s_dot = sol.y[1]
    x_u     = sol.y[2]
    x_u_dot = sol.y[3]

    z_r = np.array([road_fn(ti, v) for ti in t])

    travel = x_s - x_u
    tyre   = x_u - z_r
    acc    = (-p["k_s"]*(x_s - x_u) - p["c_base"]*(x_s_dot - x_u_dot)) / p["m_s"]

    # --- Plot ---
    plt.figure(figsize=(10,6))
    plt.suptitle(title)

    plt.subplot(3,1,1)
    plt.plot(t, travel*1000)
    plt.ylabel("Travel [mm]")
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, tyre*1000)
    plt.ylabel("Tyre Defl [mm]")
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, acc)
    plt.ylabel("Accel [m/s²]")
    plt.xlabel("Time [s]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ================================================================
# MAIN SIMULATION PIPELINE
# ================================================================

def main():
    p = parameters()
    v = 20.0

    # Simulate passive and skyhook
    sol_passive = simulate(passive_ode, p, v)
    sol_sky     = simulate(skyhook_ode, p, v)

    # Evaluate metrics
    passive_results = metrics(sol_passive, p, v, road_fn=road_speed_bump, damping_mode="passive")
    skyhook_results = metrics(sol_sky, p, v, road_fn=road_speed_bump, damping_mode="skyhook")

    print("\n=== PASSIVE ===")
    for k, v in passive_results.items():
        print(f"{k:18s}: {v}")

    print("\n=== SKYHOOK ===")
    for k, v in skyhook_results.items():
        print(f"{k:18s}: {v}")

    plot_results(sol_passive, p, v, road_speed_bump, title="Passive Suspension")
    plot_results(sol_sky, p, v, road_speed_bump, title="Skyhook Suspension")

p = parameters()
speeds = [5, 10, 15, 20, 25, 30]

speed_sweep_passive = speed_sweep(passive_ode, p, speeds)
speed_sweep_skyhook = speed_sweep(skyhook_ode, p, speeds)

print("\n=== SPEED SWEEP PASSIVE ===")
for vv, rr in speed_sweep_passive.items():
    print(f"{vv} m/s -> {rr}")

print("\n=== SPEED SWEEP SKYHOOK ===")
for vv, rr in speed_sweep_skyhook.items():
    print(f"{vv} m/s -> {rr}")


# ------------------------------------------------------------
# Run the script
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
