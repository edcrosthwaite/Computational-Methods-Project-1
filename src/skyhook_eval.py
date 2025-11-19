"""
skyhook_eval.py

Evaluation utilities for the semi-active **clipped skyhook** suspension.

This module provides a single high-level function, :func:`evaluate_skyhook`,
which takes a baseline set of quarter-car parameters, scales the semi-active
damper limits (c_min and c_max) by a user-defined factor, runs a skyhook
simulation, and extracts scalar performance metrics.

These metrics are used in the CMM3 design project to:
    - quantify ride comfort (RMS sprung-mass acceleration),
    - quantify suspension travel usage, and
    - quantify tyre deflection as a proxy for road-holding / contact-patch variation.

The implementation mirrors :mod:`passive_eval` so that **passive** and
**semi-active** configurations can be compared directly on the same basis.

All quantities are expressed in SI units unless stated otherwise.
"""

import numpy as np
from dataclasses import replace

from params import SuspensionParams
from skyhook_sim import run_skyhook_sim


def evaluate_skyhook(params: SuspensionParams, damping_scale: float) -> tuple[float, float, float]:
    """
    Evaluate the performance of the **clipped skyhook** semi-active suspension
    for a given global damping scale factor.

    The skyhook damper is defined by two bounds in ``SuspensionParams``:
        - ``c_min`` : off-state / low damping level [N·s/m]
        - ``c_max`` : on-state / high damping level [N·s/m]

    The clipped skyhook control law (implemented in :mod:`ODEdampers`) switches
    between these two values depending on the sprung-mass and relative velocities.
    In this function, both bounds are scaled by a common factor ``damping_scale``,
    preserving the controller logic while adjusting the overall damping level.

    This function:
        1. Creates a **scaled copy** of the parameter set in which both
           ``c_min`` and ``c_max`` are multiplied by ``damping_scale``.
        2. Runs a single time-domain simulation of the quarter-car model using
           :func:`run_skyhook_sim`.
        3. Post-processes the simulated time histories to obtain three scalar
           performance metrics.

    Parameters
    ----------
    params : SuspensionParams
        Baseline suspension parameters (masses, stiffnesses, damping bounds,
        simulation settings, etc.). This dataclass instance is **not** modified
        in-place; a scaled copy is created internally.
    damping_scale : float
        Dimensionless global scale factor applied to both skyhook damping
        limits. For example:
            - ``damping_scale = 0.5`` → both c_min and c_max halved
            - ``damping_scale = 1.0`` → baseline skyhook tuning
            - ``damping_scale = 1.5`` → both c_min and c_max increased by 50 %

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
        This metric is used as a proxy for dynamic tyre load variation and
        thus road-holding capability.

    Notes
    -----
    - All differentiation is performed numerically using ``numpy.gradient``,
      which assumes a uniformly spaced time vector (as supplied by the solver).
    - The function is side-effect free: it returns new values without modifying
      the original ``params`` object, making it safe to call repeatedly inside
      parameter sweeps (e.g. Pareto analyses comparing different skyhook gains).
    """

    # ---------------------------------------------------------------------
    # 1. Create a scaled copy of the suspension parameters.
    #
    #    Both semi-active damping bounds are multiplied by the same factor
    #    to preserve the controller logic (i.e. the relative "gap" between
    #    c_min and c_max) while varying the overall damping level.
    # ---------------------------------------------------------------------
    p_scaled = replace(
        params,
        c_min=damping_scale * params.c_min,
        c_max=damping_scale * params.c_max,
    )

    # ---------------------------------------------------------------------
    # 2. Run the clipped skyhook simulation for this damping level.
    #    The solver returns time histories of sprung/unsprung motion and
    #    the road input.
    # ---------------------------------------------------------------------
    t, x_s, x_u, z_r = run_skyhook_sim(p_scaled)

    # ---------------------------------------------------------------------
    # 3. Post-processing: compute accelerations and relative deflections.
    # ---------------------------------------------------------------------

    # Sprung-mass velocity and acceleration (numerical differentiation)
    x_s_dot = np.gradient(x_s, t)
    x_s_ddot = np.gradient(x_s_dot, t)

    # Suspension deflection (travel) and tyre deflection
    travel = x_s - x_u        # [m] suspension displacement
    tyre_def = x_u - z_r      # [m] tyre compression relative to road

    # ---------------------------------------------------------------------
    # 4. Scalar performance metrics used in the Pareto analysis.
    # ---------------------------------------------------------------------

    # Ride comfort: RMS of sprung-mass acceleration
    rms_acc = np.sqrt(np.mean(x_s_ddot**2))

    # Packaging constraint: maximum suspension travel magnitude
    max_travel = np.max(np.abs(travel))

    # Road-holding constraint: maximum tyre deflection magnitude
    max_tyre_def = np.max(np.abs(tyre_def))

    return rms_acc, max_travel, max_tyre_def
