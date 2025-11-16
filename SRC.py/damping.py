
"""
Damper laws: passive piecewise-linear and semi-active skyhook (on-off).
"""

def passive_damper_force(v_rel, c_comp, c_reb):
    """
    Passive asymmetric damper: F = c * v_rel
    Compression when v_rel < 0 (damper shortening) -> use c_comp
    Rebound when v_rel >= 0 -> use c_reb
    """
    c = c_comp if v_rel < 0.0 else c_reb
    return c * v_rel

def skyhook_on_off(v_s, v_rel, c_min, c_max):
    """
    On-off skyhook logic commonly used:
    If v_s * v_rel > 0 => enable high damping (c_max)
    else low damping (c_min)

    v_s   : sprung mass absolute velocity
    v_rel : relative velocity across damper = xsdot - xudot
    """
    if v_s * v_rel > 0.0:
        return c_max
    return c_min
