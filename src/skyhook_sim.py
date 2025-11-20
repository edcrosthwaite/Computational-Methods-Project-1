"""
skyhook_sim.py

Time-domain simulation of the quarter-car model with a **clipped skyhook**
semi-active damper.

This module provides a single high-level function, :func:`run_skyhook_sim`,
which wraps the skyhook ODE definition in :mod:`ODEodes`, configures the
numerical solver (:func:`scipy.integrate.solve_ivp`), and returns the
simulated time histories required for performance evaluation and Pareto
analysis in the CMM3 design project.

The model represents:
    - a sprung mass (quarter vehicle body) connected to
    - an unsprung mass (wheel/axle) via a suspension spring and semi-active damper,
    - with the unsprung mass supported by a linear tyre stiffness on a
      prescribed road profile (half-cosine bump).

The **clipped skyhook** control law is implemented inside
:func:`quarter_car_ode_skyhook`. It switches the damper coefficient between
a low value (c_min) and a high value (c_max) based on a velocity-dependent
logic that approximates an ideal skyhook damper while enforcing passivity.

Simulation settings (time span, evaluation grid, and initial conditions)
are provided centrally by :mod:`constants` so that passive and semi-active
cases are directly comparable.

All units are SI unless otherwise noted.
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.system_definition import quarter_car_ode_skyhook
from src.input_calculation import road_input
from src.params import SuspensionParams
from src.constants import (
    T_START,
    T_END,
    T_EVAL_BUMP,
    X_INITIAL_STATE,
)


def run_skyhook_sim(params: SuspensionParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the quarter-car model with the **clipped skyhook** semi-active
    damper for a given set of suspension parameters.

    The equations of motion are defined in :func:`quarter_car_ode_skyhook`,
    which uses:
        - baseline suspension stiffness (ks),
        - tyre stiffness (kt),
        - sprung/unsprung masses (ms, mu),
        - semi-active damping bounds (c_min, c_max),
        - vehicle forward speed (v),
      together with the skyhook switching logic to compute the damper force.

    This function configures and calls the ODE solver, then post-processes
    the solution to compute the corresponding road input history. The outputs
    are later used to derive scalar performance metrics (ride comfort,
    suspension travel, tyre deflection) in the Pareto analysis.

    Parameters
    ----------
    params : SuspensionParams
        Dataclass instance containing all physical and numerical parameters
        for the simulation, including:
            - ms, mu : sprung and unsprung masses [kg]
            - ks     : suspension stiffness [N/m]
            - kt     : tyre stiffness [N/m]
            - c_min  : skyhook low damping bound [N·s/m]
            - c_max  : skyhook high damping bound [N·s/m]
            - v      : vehicle forward speed [m/s]

        The simulation time span, evaluation grid, and initial conditions are
        taken from :mod:`constants` to ensure that passive and semi-active
        simulations use an identical time base.

    Returns
    -------
    t : np.ndarray
        1D array of time instants [s] at which the state has been evaluated.
        This matches the grid defined by ``constants.T_EVAL``.
    x_s : np.ndarray
        1D array of sprung-mass vertical displacement [m] relative to the
        inertial reference.
    x_u : np.ndarray
        1D array of unsprung-mass (wheel/axle) vertical displacement [m].
    z_r : np.ndarray
        1D array of road displacement input [m] computed via
        :func:`road_input` at each time step and the specified vehicle speed.

    Notes
    -----
    - The initial state ``X_INITIAL_STATE`` is typically all zeros, corresponding
      to static equilibrium before the bump is encountered.
    - The same time vector ``T_EVAL`` is used for all simulations to make
      performance metric calculation and plotting directly comparable between
      passive and skyhook configurations.
    - The details of the skyhook control law (clipping, switching condition)
      are encapsulated in :func:`quarter_car_ode_skyhook`; this wrapper is
      intentionally kept agnostic to the control implementation.
    """

    # ---------------------------------------------------------------------
    # 1. Configure and run the ODE solver.
    #
    #    quarter_car_ode_skyhook expects:
    #        t, state,
    #        ms, mu, ks,
    #        c_min, c_max,
    #        kt, v
    #
    #    The time span, evaluation grid, and initial state are shared with
    #    the passive case to ensure fair, like-for-like comparison.
    # ---------------------------------------------------------------------
    sol = solve_ivp(
        fun=lambda t, state: quarter_car_ode_skyhook(
            t,
            state,
            params.ms,
            params.mu,
            params.ks,
            params.c_min,
            params.c_max,
            params.kt,
            params.v,
            z_r=road_input(t, params.v)
        ),
        t_span=(T_START, T_END),
        y0=X_INITIAL_STATE,
        t_eval=T_EVAL_BUMP,
        method="RK45",
    )

    # Extract state histories from the solution object.
    t = sol.t
    x_s = sol.y[0]  # sprung-mass displacement [m]
    x_u = sol.y[2]  # unsprung-mass displacement [m]

    # ---------------------------------------------------------------------
    # 2. Reconstruct the road input applied during the simulation.
    #
    #    road_input(t, v) implements the half-cosine bump profile as a
    #    function of time and vehicle forward speed. This is recomputed here
    #    for convenience so that downstream code receives all relevant
    #    signals with consistent dimensions.
    # ---------------------------------------------------------------------
    z_r = np.array([road_input(tt, params.v) for tt in t])

    return t, x_s, x_u, z_r
