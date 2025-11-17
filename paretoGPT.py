"""
CMM3: Sensitivity analysis, Pareto identification, Monte Carlo, and 2D sweeps
for nonlinear piecewise-passive and semi-active (skyhook) quarter-car models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------

@dataclass
class Params:
    # Masses
    ms: float = 300.0   # sprung mass [kg]
    mu: float = 40.0    # unsprung mass [kg]

    # Springs
    ks: float = 28510.0  # suspension spring stiffness [N/m]
    kt: float = 250000.0 # tyre stiffness [N/m]

    # Passive damper (nonlinear piecewise)
    c_comp: float = 3000.0  # compression damping [Ns/m]
    c_reb: float = 1500.0   # rebound damping [Ns/m]
    v0: float = 0.05        # velocity threshold [m/s] for nonlinearity (if you want it)

    # Semi-active (skyhook) damper
    c_min: float = 500.0    # low damping [Ns/m]
    c_max: float = 3000.0   # high damping [Ns/m]
    v_switch: float = 0.05  # switching velocity threshold [m/s] for skyhook logic

    # Operating conditions
    v: float = 5.0         # vehicle speed [m/s]

    # Simulation options
    t_end: float = 2.0      # total simulation time [s]
    n_steps: int = 2000     # number of time steps


# ---------------------------------------------------------------------
# Road input
# ---------------------------------------------------------------------

def road_input(t: float, v: float, h: float = 0.05, L: float = 1.0) -> float:
    """
    Half-sine bump of height h over length L travelled at speed v.
    """
    x = v * t
    if 0.0 <= x <= L:
        return 0.5 * h * (1.0 - np.cos(2.0 * np.pi * x / L))
    return 0.0


# ---------------------------------------------------------------------
# Quarter-car ODEs
# ---------------------------------------------------------------------

def quarter_car_ode_passive(t: float, state: np.ndarray, p: Params) -> np.ndarray:
    """
    Nonlinear piecewise passive damper model.

    State = [x_s, x_s_dot, x_u, x_u_dot]
    """
    x_s, x_s_dot, x_u, x_u_dot = state

    z_r = road_input(t, p.v)
    x_su = x_s - x_u
    v_rel = x_s_dot - x_u_dot  # relative suspension velocity

    # Nonlinear piecewise damping: different coeffs in compression vs rebound
    if v_rel >= 0.0:
        c_eff = p.c_comp  # compression
    else:
        c_eff = p.c_reb   # rebound

    # Equations of motion
    x_s_ddot = (-p.ks * x_su - c_eff * v_rel) / p.ms
    x_u_ddot = (p.ks * x_su + c_eff * v_rel - p.kt * (x_u - z_r)) / p.mu

    return np.array([x_s_dot, x_s_ddot, x_u_dot, x_u_ddot])


def quarter_car_ode_skyhook(t: float, state: np.ndarray, p: Params) -> np.ndarray:
    """
    Semi-active clipped skyhook damper.

    State = [x_s, x_s_dot, x_u, x_u_dot]
    """
    x_s, x_s_dot, x_u, x_u_dot = state

    z_r = road_input(t, p.v)
    x_su = x_s - x_u
    v_rel = x_s_dot - x_u_dot  # relative suspension velocity

    # Clipped skyhook logic:
    # If sprung mass velocity and relative velocity have same sign,
    # use high damping, else low damping.
    if x_s_dot * v_rel > 0.0 and abs(v_rel) > p.v_switch:
        c_eff = p.c_max
    else:
        c_eff = p.c_min

    x_s_ddot = (-p.ks * x_su - c_eff * v_rel) / p.ms
    x_u_ddot = (p.ks * x_su + c_eff * v_rel - p.kt * (x_u - z_r)) / p.mu

    return np.array([x_s_dot, x_s_ddot, x_u_dot, x_u_ddot])


# ---------------------------------------------------------------------
# Generic simulation wrapper
# ---------------------------------------------------------------------

def run_quarter_car(model_fn: Callable[[float, np.ndarray, Params], np.ndarray],
                    p: Params) -> Dict[str, float]:
    """
    Integrate the given quarter-car model and compute performance metrics:
    - rms_acc      : RMS sprung mass acceleration [m/s^2]
    - peak_travel  : max |x_s - x_u| [m]
    - peak_tyre    : max |x_u - z_r| [m]
    - settling_time: approximate time for travel to settle within a small band
    """
    t_span = (0.0, p.t_end)
    t_eval = np.linspace(t_span[0], t_span[1], p.n_steps)

    # initial conditions: no deflection, no velocity
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    sol = solve_ivp(
        fun=lambda t, y: model_fn(t, y, p),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        vectorized=False
    )

    t = sol.t
    x_s = sol.y[0, :]
    x_s_dot = sol.y[1, :]
    x_u = sol.y[2, :]

    # Road profile over time
    z_r = np.array([road_input(ti, p.v) for ti in t])

    # Metrics
    travel = x_s - x_u
    tyre_defl = x_u - z_r

    # Acceleration via numerical differentiation
    x_s_ddot = np.gradient(x_s_dot, t)
    rms_acc = float(np.sqrt(np.mean(x_s_ddot**2)))
    peak_travel = float(np.max(np.abs(travel)))
    peak_tyre = float(np.max(np.abs(tyre_defl)))

    # Crude "settling time": first time after bump where travel stays within small band
    band = 0.002  # 2 mm band, tweak if needed
    settling_time = float(p.t_end)
    for i in range(len(t)):
        if np.all(np.abs(travel[i:]) < band):
            settling_time = float(t[i])
            break

    return {
        "rms_acc": rms_acc,
        "peak_travel": peak_travel,
        "peak_tyre": peak_tyre,
        "settling_time": settling_time,
    }


def simulate_passive(p: Params) -> Dict[str, float]:
    return run_quarter_car(quarter_car_ode_passive, p)


def simulate_skyhook(p: Params) -> Dict[str, float]:
    return run_quarter_car(quarter_car_ode_skyhook, p)


# ---------------------------------------------------------------------
# Sensitivity analysis (1D, one parameter at a time)
# ---------------------------------------------------------------------

def sensitivity_analysis_structured(
    simulate_fn: Callable[[Any], Dict[str, float]],
    p_base: Any,
    param_ranges: Dict[str, Tuple[float, float]],
    samples: int = 20
):
    """
    One-at-a-time sensitivity analysis.

    For each parameter in `param_ranges`, scale it between (smin, smax) around p_base
    and record how the key performance metrics change.
    """
    results = {}
    for param, (smin, smax) in param_ranges.items():
        scales = np.linspace(smin, smax, samples)
        metrics = {
            'rms_acc': [],
            'peak_travel': [],
            'peak_tyre': [],
            'settling_time': [],
        }
        for s in scales:
            p = copy.deepcopy(p_base)
            setattr(p, param, getattr(p_base, param) * s)
            out = simulate_fn(p)
            metrics['rms_acc'].append(out['rms_acc'])
            metrics['peak_travel'].append(out['peak_travel'])
            metrics['peak_tyre'].append(out['peak_tyre'])
            metrics['settling_time'].append(out.get('settling_time', 0.0))
        results[param] = {'scales': scales, 'metrics': metrics}
    return results


# ---------------------------------------------------------------------
# Pareto front identification (2D: RMS vs travel)
# ---------------------------------------------------------------------

def pareto_frontier_identification(rms_arr: np.ndarray, travel_arr: np.ndarray) -> np.ndarray:
    """
    Identify Pareto-efficient points for a 2-objective minimisation problem:
    minimise RMS acceleration and peak travel.

    Returns points as [rms_acc, peak_travel].
    """
    pts = np.column_stack([np.ravel(rms_arr), np.ravel(travel_arr)])
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)

    for i, p in enumerate(pts):
        if is_pareto[i]:
            dominated = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            dominated[i] = False
            is_pareto[dominated] = False
    return pts[is_pareto]


# ---------------------------------------------------------------------
# Monte Carlo robustness analysis
# ---------------------------------------------------------------------

def robustness_analysis_monte_carlo(
    simulate_fn: Callable[[Any], Dict[str, float]],
    p_nominal: Any,
    n_samples: int = 1000,
    rng_seed: int = 42
):
    """
    Monte Carlo robustness analysis around a nominal parameter set.

    The parameter object p_nominal is deep-copied each iteration and scaled by
    Gaussian factors representing uncertainty/tolerances.
    """
    rng = np.random.default_rng(rng_seed)
    results = {'rms_acc': [], 'peak_travel': [], 'peak_tyre': []}

    for _ in range(n_samples):
        p = copy.deepcopy(p_nominal)
        # Adjust these lines to match your parameter names if needed
        p.ks     *= rng.normal(1.0, 0.05)
        p.c_comp *= rng.normal(1.0, 0.10)
        p.c_reb  *= rng.normal(1.0, 0.10)
        p.ms     *= rng.normal(1.0, 0.10)
        p.mu     *= rng.normal(1.0, 0.10)

        out = simulate_fn(p)
        results['rms_acc'].append(out['rms_acc'])
        results['peak_travel'].append(out['peak_travel'])
        results['peak_tyre'].append(out['peak_tyre'])

    summary = {}
    for k, v in results.items():
        arr = np.array(v)
        std = float(arr.std())
        summary[k] = {
            'mean': float(arr.mean()),
            'std': std,
            '95ci': 1.96 * std,  # ± interval width assuming normal distribution
        }
    return {'raw': results, 'summary': summary}


# ---------------------------------------------------------------------
# 2D parametric sweep + Pareto wrapper
# ---------------------------------------------------------------------

def parametric_sweep_2d(
    simulate_fn: Callable[[Any], Dict[str, float]],
    p_base: Any,
    param_x: str,
    values_x: np.ndarray,
    param_y: str,
    values_y: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Perform a 2D parametric sweep over two parameters and collect performance metrics.
    """
    values_x = np.asarray(values_x)
    values_y = np.asarray(values_y)

    nx = len(values_x)
    ny = len(values_y)

    rms_arr = np.zeros((nx, ny))
    travel_arr = np.zeros_like(rms_arr)
    tyre_arr = np.zeros_like(rms_arr)

    for i, vx in enumerate(values_x):
        for j, vy in enumerate(values_y):
            p = copy.deepcopy(p_base)
            setattr(p, param_x, vx)
            setattr(p, param_y, vy)

            out = simulate_fn(p)

            rms_arr[i, j] = out['rms_acc']
            travel_arr[i, j] = out['peak_travel']
            tyre_arr[i, j] = out['peak_tyre']

    X, Y = np.meshgrid(values_y, values_x)  # X ~ param_y, Y ~ param_x (for plotting)
    return {
        'X': X,
        'Y': Y,
        'rms': rms_arr,
        'travel': travel_arr,
        'tyre': tyre_arr,
    }


def pareto_from_sweep(
    sweep_result: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the output of parametric_sweep_2d, compute the Pareto front in
    (rms_acc, peak_travel) space.
    """
    rms = sweep_result['rms']
    travel = sweep_result['travel']

    pareto_pts = pareto_frontier_identification(rms, travel)
    rms_all = rms.ravel()
    travel_all = travel.ravel()
    return pareto_pts, rms_all, travel_all


# ---------------------------------------------------------------------
# Example usage / quick demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Base parameters (you can tweak)
    p_base_passive = Params()
    p_base_skyhook = Params()

    # 2D sweep ranges (example: ks vs c_comp for passive, ks vs c_min for skyhook)
    ks_values = np.linspace(20000.0, 40000.0, 9)
    c_comp_values = np.linspace(1000.0, 5000.0, 9)
    c_min_values = np.linspace(200.0, 2000.0, 9)

    # Passive sweep + Pareto
    sweep_passive = parametric_sweep_2d(
        simulate_fn=simulate_passive,
        p_base=p_base_passive,
        param_x="ks",
        values_x=ks_values,
        param_y="c_comp",
        values_y=c_comp_values,
    )
    pareto_passive, rms_all_passive, travel_all_passive = pareto_from_sweep(sweep_passive)

    # Skyhook sweep + Pareto
    sweep_skyhook = parametric_sweep_2d(
        simulate_fn=simulate_skyhook,
        p_base=p_base_skyhook,
        param_x="ks",
        values_x=ks_values,
        param_y="c_min",
        values_y=c_min_values,
    )
    pareto_skyhook, rms_all_skyhook, travel_all_skyhook = pareto_from_sweep(sweep_skyhook)

    # Plot Pareto fronts (RMS acc vs peak travel)
    plt.figure()
    plt.scatter(rms_all_passive, travel_all_passive, alpha=0.2, label="Passive (all)")
    plt.scatter(pareto_passive[:, 0], pareto_passive[:, 1], marker='o', label="Passive Pareto")

    plt.scatter(rms_all_skyhook, travel_all_skyhook, alpha=0.2, label="Skyhook (all)")
    plt.scatter(pareto_skyhook[:, 0], pareto_skyhook[:, 1], marker='x', label="Skyhook Pareto")

    plt.xlabel("RMS sprung acceleration [m/s²]")
    plt.ylabel("Max suspension travel [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
