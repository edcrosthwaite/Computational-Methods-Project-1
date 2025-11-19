"""
odes.py

Contains the two primary Ordinary Differential Equation (ODE) functions 
required by the scipy.integrate.solve_ivp solver for the quarter-car model.
Each function defines the time derivative of the state vector (velocity and 
acceleration) for a specific damper configuration.
"""
import numpy as np
from src.ODEroad import road_input, road_iso
from src.ODEdampers import F_passive_piecewise, F_skyhook_clipped


# --------------- ODEs -----------------------

def quarter_car_ode_passive(t, state, m_s, m_u, k_s, c_comp_low, c_comp_high, c_reb_low, c_reb_high, v0, k_t, v, z_r):
    """
    Defines the first-order differential equations for the quarter-car model 
    with a nonlinear, asymmetric piecewise passive damper.

    The function computes the time derivative of the state vector x_dot, as 
    required by ODE solvers.

    Args:
        t (float): Current time [s]. (Provided automatically by solver).
        state (np.ndarray): State vector [x_s, x_s_dot, x_u, x_u_dot]. (Provided by solver).
        m_s (float): Sprung mass (body mass) [kg].
        m_u (float): Unsprung mass (axle/wheel mass) [kg].
        k_s (float): Suspension spring stiffness [N/m].
        c_comp_low (float): Low-speed compression damping [N s / m].
        c_comp_high (float): High-speed compression damping [N s / m].
        c_reb_low (float): Low-speed rebound damping [N s / m].
        c_reb_high (float): High-speed rebound damping [N s / m].
        v0 (float): Damping velocity threshold [m/s].
        k_t (float): Tire stiffness [N/m].
        v (float): Vehicle forward velocity [m/s] (used for road input).

    Returns:
        list: The derivative of the state vector: [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot].
    """
    # 1. Extract states
    x_s, x_s_dot, x_u, x_u_dot = state

    # 2. External Input
    # Calculate the current road displacement (z_r) based on time and vehicle speed.
    '''z_r = road_input(t, v)'''

    # 3. Intermediate Quantities
    # x_su: Suspension deflection
    x_su = x_s - x_u
    # v_rel: Relative velocity between sprung and unsprung masses
    v_rel = x_s_dot - x_u_dot

    # 4. Damping Force Calculation (F_d)
    # The force is determined by the nonlinear piecewise function using four coefficients.
    F_d = F_passive_piecewise(v_rel, c_comp_low, c_comp_high, c_reb_low, c_reb_high, v0)

    # 5. Equations of Motion (F = ma)

    # Sprung Mass (x_s_ddot): Force from suspension spring and damper.
    # m_s * x_s_ddot = -k_s*(x_s - x_u) - F_d
    x_s_ddot = (-k_s * x_su - F_d) / m_s

    # Unsprung Mass (x_u_ddot): Force from suspension, damper, and tire.
    # Tire deflection (x_u - z_r)
    x_ur = x_u - z_r
    # m_u * x_u_ddot = k_s*(x_s - x_u) + F_d - k_t*(x_u - z_r)
    x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u

    # 6. Return State Derivatives
    # Returns [d(x_s)/dt, d(x_s_dot)/dt, d(x_u)/dt, d(x_u_dot)/dt]
    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


def quarter_car_ode_skyhook(t, state, m_s, m_u, k_s, c_min, c_max, k_t, v, z_r):
    """
    Defines the first-order differential equations for the quarter-car model 
    with a semi-active clipped skyhook damper.

    Args:
        t (float): Current time [s].
        state (np.ndarray): State vector [x_s, x_s_dot, x_u, x_u_dot].
        m_s (float): Sprung mass (body mass) [kg].
        m_u (float): Unsprung mass (axle/wheel mass) [kg].
        k_s (float): Suspension spring stiffness [N/m].
        c_min (float): Minimum damping coefficient (off-state) [N s / m].
        c_max (float): Maximum damping coefficient (on-state) [N s / m].
        k_t (float): Tire stiffness [N/m].
        v (float): Vehicle forward velocity [m/s].

    Returns:
        list: The derivative of the state vector: [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot].
    """
    # 1. Extract states
    x_s, x_s_dot, x_u, x_u_dot = state

    # 2. External Input
    '''z_r = road_input(t, v)'''

    # 3. Intermediate Quantities
    x_su = x_s - x_u
    # v_rel is primarily used inside the damper function, but defined here for potential debugging
    v_rel = x_s_dot - x_u_dot

    # 4. Damping Force Calculation (F_d)
    # The force is determined by the semi-active clipped skyhook control law.
    # The function uses x_s_dot and v_rel to switch the damping coefficient.
    F_d = F_skyhook_clipped(x_s_dot, x_u_dot, c_min, c_max)

    # 5. Equations of Motion (F = ma)
    # Sprung Mass (x_s_ddot): EOM is identical to the passive case, only F_d differs.
    x_s_ddot = (-k_s * x_su - F_d) / m_s

    # Unsprung Mass (x_u_ddot): EOM is identical to the passive case, only F_d differs.
    x_ur = x_u - z_r
    x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u

    # 6. Return State Derivatives
    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]