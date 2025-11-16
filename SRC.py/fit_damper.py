
"""
Regression / interpolation example:
Generate synthetic damper F-v data with asymmetry + noise,
then fit an interpolant F(v) for use in simulations (optional).
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class DamperFit:
    v_grid: np.ndarray
    F_grid: np.ndarray

    def F(self, v_rel: float) -> float:
        # Piecewise linear interpolation (manual to avoid external dependencies)
        v = self.v_grid
        F = self.F_grid
        # Clamp outside range
        if v_rel <= v[0]:
            return F[0]
        if v_rel >= v[-1]:
            return F[-1]
        # Find interval
        i = np.searchsorted(v, v_rel) - 1
        t = (v_rel - v[i])/(v[i+1]-v[i])
        return F[i]*(1-t) + F[i+1]*t

def synth_damper_data(c_comp: float, c_reb: float, n: int = 41, v_max: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    v = np.linspace(-v_max, v_max, n)
    F_true = np.where(v < 0.0, c_comp*v, c_reb*v)
    noise = 0.05 * F_true  # 5% noise
    F_noisy = F_true + noise * rng.normal(size=n)
    return v, F_noisy, F_true

def fit_piecewise_interpolant(v, F) -> DamperFit:
    # Sort by v just in case (should already be sorted)
    idx = np.argsort(v)
    v_sorted = v[idx]
    F_sorted = F[idx]
    return DamperFit(v_sorted, F_sorted)
