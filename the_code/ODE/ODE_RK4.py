"""
Quarter-car model (2-DoF) and RK4 integrator.

State vector:
    state = [xs, xsdot, xu, xudot]

where:
    xs     = sprung mass displacement [m]
    xsdot  = sprung mass velocity [m/s]
    xu     = unsprung mass displacement [m]
    xudot  = unsprung mass velocity [m/s]

Equations of motion:
    ms*xsddot = -ks*(xs - xu) - Fd(v_rel)
    mu*xuddot =  ks*(xs - xu) + Fd(v_rel) - kt*(xu - y)

with:
    v_rel = xsdot - xudot
    Fd    = damper force (passive or semi-active strategy)
    y(t)  = road displacement input [m]
"""

import numpy as np
from typing import Callable, Tuple
from damping import passive_damper_force, skyhook_on_off
from params import SuspensionParams


def rhs_passive(
    t: float,
    state: np.ndarray,
    params: SuspensionParams,
    y_fun: Callable[[float], float],
) -> np.ndarray:
    """
    Right-hand side for the passive suspension model.

    Uses asymmetric passive damping:
        c_comp for compression (v_rel > 0),
        c_reb  for rebound    (v_rel < 0).

    Parameters
    ----------
    t : float
        Current time [s].
    state : np.ndarray
        Current state vector [xs, xsdot, xu, xudot].
    params : SuspensionParams
        Physical and numerical parameters (masses, stiffnesses, damping, etc.).
    y_fun : Callable[[float], float]
        Road displacement function y(t) [m].

    Returns
    -------
    np.ndarray
        Time derivative of the state: [xsdot, xsddot, xudot, xuddot].
    """
    xs, xsdot, xu, xudot = state
    y = float(y_fun(t))

    # Relative velocity across the damper
    v_rel = xsdot - xudot

    # Passive damper force (asymmetric compression/rebound)
    Fd = passive_damper_force(v_rel, params.c_comp, params.c_reb)

    # Spring forces
    Fs = params.ks * (xs - xu)     # suspension spring
    Ft = params.kt * (xu - y)      # tyre spring

    # Accelerations from Newton's second law
    xsddot = (-Fs - Fd) / params.ms
    xuddot = ( Fs + Fd - Ft) / params.mu

    return np.array([xsdot, xsddot, xudot, xuddot], dtype=float)


def rhs_skyhook(
    t: float,
    state: np.ndarray,
    params: SuspensionParams,
    y_fun: Callable[[float], float],
) -> np.ndarray:
    """
    Right-hand side for the semi-active skyhook suspension model.

    Uses an on-off skyhook logic to switch the damper coefficient
    between c_min and c_max depending on the sign of the product
    between sprung velocity and relative velocity.

    Parameters
    ----------
    t : float
        Current time [s].
    state : np.ndarray
        Current state vector [xs, xsdot, xu, xudot].
    params : SuspensionParams
        Physical and numerical parameters (masses, stiffnesses, damping, etc.).
    y_fun : Callable[[float], float]
        Road displacement function y(t) [m].

    Returns
    -------
    np.ndarray
        Time derivative of the state: [xsdot, xsddot, xudot, xuddot].
    """
    xs, xsdot, xu, xudot = state
    y = float(y_fun(t))

    # Relative velocity across the damper
    v_rel = xsdot - xudot

    # Skyhook control: choose effective damping c(t)
    c_eff = skyhook_on_off(xsdot, v_rel, params.c_min, params.c_max)

    # Damper force under skyhook strategy
    Fd = c_eff * v_rel

    # Spring forces
    Fs = params.ks * (xs - xu)     # suspension spring
    Ft = params.kt * (xu - y)      # tyre spring

    # Accelerations from Newton's second law
    xsddot = (-Fs - Fd) / params.ms
    xuddot = ( Fs + Fd - Ft) / params.mu

    return np.array([xsdot, xsddot, xudot, xuddot], dtype=float)


def rk4(
    f: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step 4th-order Runge--Kutta integrator.

    Parameters
    ----------
    f : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side function f(t, y) returning dy/dt.
    t_span : (float, float)
        Tuple (t0, tf) specifying start and end times [s].
    y0 : np.ndarray
        Initial state vector at t0.
    dt : float
        Time step [s].

    Returns
    -------
    t : np.ndarray
        Time vector from t0 to tf with uniform spacing dt.
    Y : np.ndarray
        Array of state vectors at each time step, shape (n_steps, len(y0)).
    """
    t0, tf = t_span

    # Number of steps (inclusive of t0)
    n_steps = int(np.floor((tf - t0) / dt)) + 1

    # Time vector with exactly uniform spacing dt
    t = t0 + np.arange(n_steps) * dt

    # Storage for solution
    Y = np.zeros((n_steps, len(y0)), dtype=float)
    Y[0] = np.array(y0, dtype=float)

    yi = np.array(y0, dtype=float)
    ti = t0

    for i in range(1, n_steps):
        k1 = f(ti, yi)
        k2 = f(ti + 0.5 * dt, yi + 0.5 * dt * k1)
        k3 = f(ti + 0.5 * dt, yi + 0.5 * dt * k2)
        k4 = f(ti + dt,       yi + dt * k3)

        yi = yi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Y[i] = yi
        ti += dt

    return t, Y
