# dampers.py
# damper model for quarter-car suspension model
import numpy as np

def F_passive_piecewise(v_rel, c_comp_low, c_comp_high, c_reb_low, c_reb_high, v0):
    """
    Calculates the damping force for an asymmetric and digressive damper.
    """
    
    # 1. Determine coefficients for the compression stroke (v_rel < 0)
    # c_comp_low for low speed, c_comp_high for high speed
    c_eff_comp = np.where(np.abs(v_rel) < v0, c_comp_low, c_comp_high)
    
    # 2. Determine coefficients for the rebound stroke (v_rel > 0)
    # c_reb_low for low speed, c_reb_high for high speed
    c_eff_rebound = np.where(np.abs(v_rel) < v0, c_reb_low, c_reb_high)
    
    # 3. Choose the final effective coefficient based on the direction (v_rel sign)
    # If v_rel is negative, use compression coeffs; else (positive), use rebound coeffs.
    c_eff = np.where(v_rel < 0, c_eff_comp, c_eff_rebound)
    
    # Calculate Damping Force
    F_d = c_eff * v_rel
    
    return F_d
