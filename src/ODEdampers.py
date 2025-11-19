"""
dampers.py

Contains utility functions for calculating the damping force (F_d) in the 
quarter-car suspension model. This module implements both passive and semi-active 
damper control laws using vectorized NumPy operations for speed.
"""
import numpy as np

def F_passive_piecewise(v_rel, c_comp_low, c_comp_high, c_reb_low, c_reb_high, v0):
    """
    Calculates the damping force for an asymmetric and digressive (piecewise) damper.
    
    This function implements a highly realistic passive damper model where the 
    damping coefficient (c_eff) changes based on:
    1. The direction of motion (Asymmetry: Rebound vs. Compression).
    2. The speed of motion (Digression: Low speed vs. High speed, separated by v0).

    Args:
        v_rel (np.ndarray): Relative velocity (x_s_dot - x_u_dot) [m/s].
        c_comp_low (float): Low-speed damping coefficient for the Compression stroke.
        c_comp_high (float): High-speed damping coefficient for the Compression stroke.
        c_reb_low (float): Low-speed damping coefficient for the Rebound stroke.
        c_reb_high (float): High-speed damping coefficient for the Rebound stroke.
        v0 (float): Threshold velocity that separates the low- and high-speed regions [m/s].
        
    Returns:
        np.ndarray: The resulting damping force F_d [N].
    """
    
    # 1. Determine coefficients for the compression stroke (v_rel < 0)
    # Logic: If |v_rel| is below the threshold v0, use c_comp_low; otherwise, use c_comp_high.
    c_eff_comp = np.where(np.abs(v_rel) < v0, c_comp_low, c_comp_high)
    
    # 2. Determine coefficients for the rebound stroke (v_rel > 0)
    # Logic: Same speed check, but uses the separate rebound coefficients.
    c_eff_rebound = np.where(np.abs(v_rel) < v0, c_reb_low, c_reb_high)
    
    # 3. Final Selection: Choose the effective coefficient based on the direction (sign of v_rel)
    # Compression occurs when v_rel is negative (damper piston moves into the body).
    # If v_rel < 0, use c_eff_comp; else (v_rel >= 0, rebound or zero), use c_eff_rebound.
    c_eff = np.where(v_rel < 0, c_eff_comp, c_eff_rebound)
    
    # Calculate Damping Force F_d = c_eff * v_rel
    F_d = c_eff * v_rel
    
    return F_d

def F_skyhook_clipped(x_s_dot, x_u_dot, c_min, c_max):
    """
    Calculates the semi-active clipped skyhook damping force.

    The skyhook strategy aims to damp the sprung mass velocity (x_s_dot) as if 
    it were connected to a damper anchored in the sky (an inertial reference).
    This control law is clipped (semi-active) because the damper can only dissipate 
    energy (c_eff must be non-negative).

    Args:
        x_s_dot (np.ndarray): Absolute sprung mass velocity [m/s].
        x_u_dot (np.ndarray): Absolute unsprung mass velocity [m/s].
        c_min (float): Minimum damping coefficient (off-state, near zero) [N s / m].
        c_max (float): Maximum damping coefficient (on-state) [N s / m].
        
    Returns:
        np.ndarray: The resulting damping force F_d [N].
    """
    v_rel = x_s_dot - x_u_dot # Relative velocity
    
    # Skyhook Condition: Damping is 'ON' (c_max) when the product is positive.
    # Condition: x_s_dot * v_rel > 0
    # This means the relative motion (v_rel) is in the same direction as the 
    # sprung mass motion (x_s_dot), so damping will oppose body movement.
    c_eff = np.where(x_s_dot * v_rel > 0, c_max, c_min)
    
    # The semi-active force is then applied using the selected coefficient.
    F_d = c_eff * v_rel
    
    return F_d