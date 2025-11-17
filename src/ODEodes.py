# odes.py
# the two ODE functions for quarter-car suspension model
from src.ODEroad import road_input
from src.ODEdampers import F_passive_piecewise

# --------------- ODEs -----------------------

def quarter_car_ode_passive(t, state, m_s, m_u, k_s, c_comp_low, c_comp_high, c_reb_low, c_reb_high, v0, k_t, v):
    """
    Quarter-car with nonlinear passive (piecewise) damper.
    state = [x_s, x_s_dot, x_u, x_u_dot]
    returns [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]
    """
    x_s, x_s_dot, x_u, x_u_dot = state

    # Road input
    z_r = road_input(t, v)

    # Relative quantities
    x_su = x_s - x_u
    v_rel = x_s_dot - x_u_dot

    # Nonlinear passive damper force
    F_d = F_passive_piecewise(v_rel, c_comp_low, c_comp_high, c_reb_low, c_reb_high, v0)

    # Equations of motion
    x_s_ddot = (-k_s * x_su - F_d) / m_s

    x_ur = x_u - z_r
    x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


def quarter_car_ode_skyhook(t, state, m_s, m_u, k_s, c_min, c_max, k_t, v):
    """
    Quarter-car with semi-active clipped skyhook damper.
    state = [x_s, x_s_dot, x_u, x_u_dot]
    """
    x_s, x_s_dot, x_u, x_u_dot = state

    # Road input
    z_r = road_input(t, v)

    # Relative quantities
    x_su = x_s - x_u
    v_rel = x_s_dot - x_u_dot

    # Clipped skyhook: use c_max when damping reduces body motion, else c_min
    if x_s_dot * v_rel > 0:
        c_eff = c_max
    else:
        c_eff = c_min
    F_d = c_eff * v_rel

    # Equations of motion
    x_s_ddot = (-k_s * x_su - F_d) / m_s

    x_ur = x_u - z_r
    x_u_ddot = (k_s * x_su + F_d - k_t * x_ur) / m_u

    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]
