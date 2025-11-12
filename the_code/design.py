
"""
Root-finding utilities for tuning ks to target sprung natural frequency,
and helper for estimating coupled natural frequencies.
"""
import numpy as np
from typing import Tuple
from params import SuspensionParams

def sprung_fn_simple(ks: float, ms: float) -> float:
    """ Single-DOF estimate of sprung natural frequency [Hz]. """
    return (1.0/(2.0*np.pi)) * np.sqrt(ks/ms)

def unsprung_fn_simple(kt: float, mu: float) -> float:
    """ Single-DOF estimate of unsprung (wheel-hop) frequency [Hz]. """
    return (1.0/(2.0*np.pi)) * np.sqrt(kt/mu)

def tune_ks_for_target_fn(params: SuspensionParams, f_target: float) -> float:
    """
    Solve for ks such that sprung_fn_simple(ks, ms) == f_target.
    Closed form: ks = (2*pi*f)^2 * ms
    """
    ks = (2.0*np.pi*f_target)**2 * params.ms
    return ks

def coupled_natural_frequencies(params: SuspensionParams, c_lin: float = 1000.0) -> Tuple[float,float]:
    """
    Estimate coupled natural frequencies by linearizing about y=0 with a linear damper c_lin.
    Returns two natural frequencies (Hz) from eigenvalues of the state matrix (ignoring damping for frequency estimate).
    """
    # 2-DOF undamped stiffness & mass matrices
    M = np.array([[params.ms, 0.0],
                  [0.0, params.mu]])
    K = np.array([[ params.ks, -params.ks],
                  [-params.ks, params.ks + params.kt]])

    # Solve generalized eigenvalue problem |K - w^2 M| = 0
    # Extract positive eigenvalues -> w (rad/s) -> f (Hz)
    evals, _ = np.linalg.eig(np.linalg.inv(M) @ K)
    w = np.sqrt(np.sort(evals))
    f = w/(2.0*np.pi)
    return f[0], f[1]
