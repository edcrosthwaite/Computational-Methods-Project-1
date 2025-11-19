"""
passive_eval.py

Evaluation utilities for the nonlinear piecewise-passive suspension.

This module provides a single high-level function, :func:`evaluate_passive`,
which takes a baseline set of quarter-car parameters, scales the passive damper
coefficients by a user-defined factor, runs a time-domain simulation, and
extracts scalar performance metrics.

These metrics are used in the CMM3 design project to:
    - quantify ride comfort (RMS sprung-mass acceleration),
    - quantify suspension travel usage, and
    - quantify tyre deflection as a proxy for road-holding / contact-patch variation.

All quantities are expressed in SI units unless stated otherwise.
"""

import numpy as np
from dataclasses import replace

from params import SuspensionParams
from passive_sim import run_passive_sim


def evaluate_passive(params: SuspensionParams, damping_scale: float) -> tuple[float, float, float]:
    """
    Evaluate the performance of the *nonlinear piecewise passive* suspension
    for a given global damping scale factor.

    The baseline damper is defined by four coefficients in ``SuspensionParams``:
        - ``c_comp_low``  : low-speed compression damping      [N·s/m]
        - ``c_comp_high`` : high-speed compression damping     [N·s/m]
        - ``c_reb_low``   : low-speed rebound damping          [N·s/m]
        - ``c_reb_high``  : high-speed rebound damping         [N·s/m]

    This function:
        1. Creates a **scaled copy** of the parameter set in which all four
           coefficients are multiplied by ``damping_scale``. The shape of the
           piecewise damper characteristic is preserved; only the overall
           damping level is changed.
        2. Runs a single time-domain simulation of the quarter-car model using
           :func:`run_passive_sim`.
        3. Post-processes the simulated time histories to obtain three scalar
           performance metrics.

    Parameters
    ----------
    params : SuspensionParams
        Baseline suspension parameters (masses, stiffnesses, damping
        coefficients, simulation settings, etc.). This object is **not**
        modified in-place; a scaled copy is created internally.
    damping_scale : float
        Dimensionless global scale factor applied to all four passive damper
        coefficients. For example:
            - ``damping_scale = 0.5`` → 50% softer damping,
            - ``damping_scale = 1.0`` → baseline damping,
            - ``damping_scale = 1.5`` → 50% stiffer damping.

    Returns
    -------
    rms_acc : float
        Root-mean-square (RMS) sprung-mass vertical acceleration [m/s²].
        This is the primary **ride comfort** metric; lower values indicate
        smoother ride.
    max_travel : float
        Maximum absolute suspension deflection [m], defined as
        ``x_s - x_u``, where ``x_s`` is the sprung-mass displacement and
        ``x_u`` is the unsprung-mass displacement. This is used to check
        against suspension travel / packaging limits.
    max_tyre_def : float
        Maximum absolute tyre deflection [m], defined as
        ``x_u - z_r``, where ``z_r`` is the road displacement input.
        This quantity is used as a proxy for dynamic tyre load variation
        and road-holding capability.

    Notes
    -----
    - All differentiation is performed numerically using ``numpy.gradient``,
      which assumes uniform time steps (as provided by the solver).
    - The function is intentionally side-effect free: the input ``params``
      object is left unchanged, making it safe to call repeatedly inside
      parameter sweeps (e.g. Pareto analyses).
    """

    # ---------------------------------------------------------------------
    # 1. Create a scaled copy of the suspension parameters
    #    (do NOT modify the original dataclass instance).
    # ---------------------------------------------------------------------
    p_scaled = replace(
        params,
        c_comp_low=damping_scale * params.c_comp_low,
        c_comp_high=damping_scale * params.c_comp_high,
        c_reb_low=damping_scale * params.c_reb_low,
        c_reb_high=damping_scale * params.c_reb_high,
    )

    # ---------------------------------------------------------------------
    # 2. Run the nonlinear passive simulation for this damping level.
    #    The solver returns time histories of sprung/unsprung motion and
    #    the road input.
    # ---------------------------------------------------------------------
    t, x_s, x_u, z_r = run_passive_sim(p_scaled)

    # ---------------------------------------------------------------------
    # 3. Post-processing: compute accelerations and relative deflections.
    # ---------------------------------------------------------------------

    # Sprung-mass velocity and acceleration (numerical differentiation)
    x_s_dot = np.gradient(x_s, t)
    x_s_ddot = np.gradient(x_s_dot, t)

    # Suspension deflection (travel) and tyre deflection
    travel = x_s - x_u           # [m] suspension displacement
    tyre_def = x_u - z_r         # [m] tyre compression relative to road

    # ---------------------------------------------------------------------
    # 4. Scalar performance metrics for design evaluation.
    # ---------------------------------------------------------------------

    # Ride comfort: RMS of sprung-mass acceleration
    rms_acc = np.sqrt(np.mean(x_s_ddot**2))

    # Packaging constraint: maximum suspension travel magnitude
    max_travel = np.max(np.abs(travel))

    # Road-holding constraint: maximum tyre deflection magnitude
    max_tyre_def = np.max(np.abs(tyre_def))

    return rms_acc, max_travel, max_tyre_def
