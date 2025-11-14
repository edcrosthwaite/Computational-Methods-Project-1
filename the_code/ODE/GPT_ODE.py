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
        "k_t": 190000.0,
        "c_base": 1500.0,   # passive damping
        "c_comp": 3000.0,   # skyhook compression damping
        "c_reb": 1500.0,    # skyhook rebound damping
    }


# ================================================================
# ROAD PROFILE
# ================================================================

def road_profile(t, v):
    """Half-sine bump road input."""
    A = 0.02
    freq = 1.0
    return A * np.sin(2 * np.pi * freq * t)


# ================================================================
# PASSIVE SUSPENSION MODEL
# ================================================================

def passive_ode(t, x, p, v):
    """Quarter-car passive suspension ODE."""
    x_s, x_s_dot, x_u, x_u_dot = x

    z_r = road_profile(t, v)

    rel_pos = x_s - x_u
    rel_vel = x_s_dot - x_u_dot

    x_s_ddot = (-p["k_s"] * rel_pos - p["c_base"] * rel_vel) / p["m_s"]
    x_ur = x_u - z_r
    x_u_ddot = (p["k_s"] * rel_pos + p["c_base"] * rel_vel - p["k_t"] * x_ur) / p["m_u"]

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


# ================================================================
# SKYHOOK SUSPENSION MODEL
# ================================================================

def skyhook_ode(t, x, p, v):
    """Skyhook-controlled quarter-car ODE."""
    x_s, x_s_dot, x_u, x_u_dot = x

    z_r = road_profile(t, v)

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

def simulate(ode, p, v, t_end=5.0, x0=None):
    """Run simulation for any ODE model."""
    if x0 is None:
        x0 = [0, 0, 0, 0]

    t_eval = np.linspace(0, t_end, 2000)

    sol = solve_ivp(
        lambda t, y: ode(t, y, p, v),
        (0, t_end),
        x0,
        t_eval=t_eval,
        method='RK45'
    )

    return sol


# ================================================================
# PERFORMANCE METRICS
# ================================================================

def metrics(sol, p, v, damping_mode="passive"):
    """Compute max travel, tyre deflection, RMS acceleration."""
    
    t = sol.t
    x_s     = sol.y[0]
    x_s_dot = sol.y[1]
    x_u     = sol.y[2]
    x_u_dot = sol.y[3]

    z_r = np.array([road_profile(ti, v) for ti in t])

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
    passive_results = metrics(sol_passive, p, v, damping_mode="passive")
    skyhook_results = metrics(sol_sky, p, v, damping_mode="skyhook")

    print("\n=== PASSIVE ===")
    for k, v in passive_results.items():
        print(f"{k:18s}: {v}")

    print("\n=== SKYHOOK ===")
    for k, v in skyhook_results.items():
        print(f"{k:18s}: {v}")


# ------------------------------------------------------------
# Run the script
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
