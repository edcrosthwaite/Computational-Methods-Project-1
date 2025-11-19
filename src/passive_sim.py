"""
passive_sim.py

Time-domain simulation of the quarter-car model with a nonlinear
piecewise-passive damper.

This module provides a single high-level function, :func:`run_passive_sim`,
which wraps the ODE definition in :mod:`ODEodes`, configures the numerical
solver (:func:`scipy.integrate.solve_ivp`), and returns the simulated time
histories required for performance evaluation in the CMM3 design project.

The model represents:
    - a sprung mass (vehicle body quarter) connected to
    - an unsprung mass (wheel/axle) via a suspension spring and nonlinear damper,
    - with the unsprung mass supported by a linear tyre stiffness on a
      prescribed road profile (half-cosine bump).

Simulation settings (time span, evaluation grid, and initial conditions)
are provided centrally by :mod:`constants` so that passive and semi-active
cases are directly comparable.

All units are SI unless otherwise noted.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ODEodes import quarter_car_ode_passive
from ODEroad import road_input
from params import SuspensionParams
from constants import (
    T_START,
    T_END,
    T_EVAL,
    X_INITIAL_STATE,
    VELOCITY_THRESHOLD,
)


def run_passive_sim(params: SuspensionParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the quarter-car model with the **nonlinear piecewise passive**
    damper for a given set of suspension parameters.

    The equations of motion are defined in :func:`quarter_car_ode_passive`,
    which implements a four-coefficient piecewise damper model:
        - low-/high-speed compression coefficients
        - low-/high-speed rebound coefficients

    This function configures and calls the ODE solver, then post-processes the
    solution to compute the corresponding road input history. The outputs are
    later used to derive scalar performance metrics (ride comfort, suspension
    travel, tyre deflection) in the Pareto analysis.

    Parameters
    ----------
    params : SuspensionParams
        Dataclass instance containing all physical and numerical parameters
        for the simulation, including:
            - ms, mu : sprung and unsprung masses [kg]
            - ks     : suspension stiffness [N/m]
            - kt     : tyre stiffness [N/m]
            - c_comp_low, c_comp_high,
              c_reb_low, c_reb_high : passive damper coefficients [N·s/m]
            - v      : vehicle forward speed [m/s]

        The simulation time span and initial conditions are not taken from
        this dataclass but from the :mod:`constants` module to ensure that
        passive and semi-active simulations use an identical time grid.

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
      to the static equilibrium position before the bump is encountered.
    - The same time vector ``T_EVAL`` is used for all simulations to make
      metric comparisons and plotting straightforward.
    - The velocity threshold used to switch between low- and high-speed
      damping regimes is provided by ``VELOCITY_THRESHOLD`` from
      :mod:`constants`, ensuring consistency with the ODE definition.
    """

    # ---------------------------------------------------------------------
    # 1. Configure and run the ODE solver.
    #
    #    quarter_car_ode_passive expects:
    #        t, state, ms, mu, ks,
    #        c_comp_low, c_comp_high,
    #        c_reb_low, c_reb_high,
    #        v0 (velocity threshold),
    #        kt, v (vehicle speed)
    #
    #    The time span, evaluation grid, and initial state are shared with
    #    the semi-active (skyhook) case to ensure fair comparison.
    # ---------------------------------------------------------------------
    sol = solve_ivp(
        fun=lambda t, y: quarter_car_ode_passive(
            t,
            y,
            params.ms,
            params.mu,
            params.ks,
            params.c_comp_low,
            params.c_comp_high,
            params.c_reb_low,
            params.c_reb_high,
            VELOCITY_THRESHOLD,
            params.kt,
            params.v,
        ),
        t_span=(T_START, T_END),
        y0=X_INITIAL_STATE,
        t_eval=T_EVAL,
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
