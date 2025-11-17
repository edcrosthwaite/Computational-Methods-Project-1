# dampers.py
# damper model for quarter-car suspension model
import numpy as np

def F_passive_piecewise(v_rel, c_min, c_max, v0):
    """
    Nonlinear passive damper:
    - low damping c_min for small |v_rel|
    - high damping c_max for large |v_rel|
    Works with scalars or numpy arrays.
    """
    c_eff = np.where(np.abs(v_rel) < v0, c_min, c_max)
    return c_eff * v_rel
