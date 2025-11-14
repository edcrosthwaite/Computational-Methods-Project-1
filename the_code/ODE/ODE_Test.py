"""
Quarter-car ODE solver demo:
- Runs passive and skyhook simulations over several speeds
- Computes travel, tyre deflection, RMS acceleration
- Plots example time histories
"""

import numpy as np
import matplotlib.pyplot as plt

from params import SuspensionParams
from signals import half_sine_bump
from the_code.ODE.ODE_RK4 import rk4, rhs_passive, rhs_skyhook  # adjust module name if needed


def compute_metrics(t, Y, y_fun, params: SuspensionParams) -> dict:
    """
    Extract useful signals and performance metrics from the ODE solution.

    Returns a dict with:
        xs, xsdot, xu, xudot, travel, tyre_defl, acc_s,
        max_travel, max_tyre_defl, rms_acc, peak_acc
    """
    xs    = Y[:, 0]
    xsdot = Y[:, 1]
    xu    = Y[:, 2]
    xudot = Y[:, 3]

    # Road displacement over time
    y_vals = np.array([y_fun(ti) for ti in t])

    # Suspension travel and tyre deflection
    travel    = xs - xu
    tyre_defl = xu - y_vals

    # Sprung-mass acceleration via numerical differentiation
    acc_s = np.gradient(xsdot, params.dt)

    max_travel    = float(np.max(np.abs(travel)))
    max_tyre_defl = float(np.max(np.abs(tyre_defl)))
    rms_acc       = float(np.sqrt(np.mean(acc_s**2)))
    peak_acc      = float(np.max(np.abs(acc_s)))

    return {
        "xs": xs,
        "xsdot": xsdot,
        "xu": xu,
        "xudot": xudot,
        "travel": travel,
        "tyre_defl": tyre_defl,
        "acc_s": acc_s,
        "max_travel": max_travel,
        "max_tyre_defl": max_tyre_defl,
        "rms_acc": rms_acc,
        "peak_acc": peak_acc,
    }


def run_simulation(
    speed: float,
    mode: str,
    params: SuspensionParams,
    bump_height: float = 0.05,
    bump_length: float = 2.0,
):
    """
    Run one simulation at a given speed and mode: 'passive' or 'skyhook'.

    Returns (t, Y, metrics).
    """
    # Road input y(t): half-sine bump
    y_fun, _ = half_sine_bump(amplitude=bump_height, length=bump_length, speed=speed)

    # Initial conditions: at rest
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    if mode == "passive":
        f = lambda t_, s_: rhs_passive(t_, s_, params, y_fun)
    elif mode == "skyhook":
        f = lambda t_, s_: rhs_skyhook(t_, s_, params, y_fun)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t, Y = rk4(f, (0.0, params.t_end), y0, params.dt)
    metrics = compute_metrics(t, Y, y_fun, params)
    return t, Y, metrics


if __name__ == "__main__":
    p = SuspensionParams()

    speeds = [10.0, 20.0, 30.0]  # m/s

    for mode in ["passive", "skyhook"]:
        print(f"\n=== {mode.upper()} SUSPENSION ===")
        for U in speeds:
            t, Y, m = run_simulation(U, mode, p)

            print(
                f"Speed = {U:4.1f} m/s | "
                f"max travel = {m['max_travel']*1000:5.1f} mm | "
                f"max tyre defl = {m['max_tyre_defl']*1000:5.1f} mm | "
                f"RMS acc = {m['rms_acc']:4.2f} m/s^2"
            )

            # Plot an example response at 20 m/s for each mode
            if U == 20.0:
                plt.figure()
                plt.plot(t, m["xs"], label="Sprung mass")
                plt.plot(t, m["xu"], label="Unsprung mass")
                plt.xlabel("Time [s]")
                plt.ylabel("Displacement [m]")
                plt.title(f"{mode.capitalize()} suspension, U = {U} m/s")
                plt.legend()
                plt.grid(True)
                plt.show()
