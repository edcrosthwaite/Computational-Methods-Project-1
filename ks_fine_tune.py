"""
Design utilities: natural frequencies and Newton–Raphson tuning of ks
based on the coupled 2-DOF body mode.
"""
import os
import numpy as np
from the_code.params import SuspensionParams


def body_mode_frequency(ks: float, params: SuspensionParams) -> float:
    """
    Compute the lower (body) natural frequency [Hz] of the 2-DOF quarter-car
    for a given suspension stiffness ks.
    """
    # Mass and stiffness matrices
    M = np.array([[params.ms, 0.0],
                  [0.0,        params.mu]])

    K = np.array([[ ks,        -ks],
                  [-ks,  ks + params.kt]])

    # Generalised eigenproblem: M^{-1} K φ = λ φ, with λ = ω^2
    evals, _ = np.linalg.eig(np.linalg.inv(M) @ K)

    # Sort, take the lower mode, ensure real part
    evals_sorted = np.sort(evals.real)
    omega_body = np.sqrt(evals_sorted[0])        # rad/s
    f_body = omega_body / (2.0 * np.pi)          # Hz
    return f_body


def tune_ks_newton(
    params: SuspensionParams,
    f_target: float,
    k0: float | None = None,
    tol: float = 1e-4,
    max_iter: int = 20,
    dk: float = 100.0,
) -> float:
    """
    Newton–Raphson tuning of ks so that the coupled body mode frequency
    matches f_target. Derivative dF/dks is approximated by finite difference.
    F(ks) = f_body(ks) - f_target.
    """
    if k0 is None:
        k0 = params.ks  # start from current guess

    k = k0
    for _ in range(max_iter):
        f = body_mode_frequency(k, params)
        F = f - f_target

        if abs(F) < tol:
            return k

        # Finite-difference derivative dF/dk ≈ [F(k+dk) - F(k)] / dk
        f_plus = body_mode_frequency(k + dk, params)
        F_plus = f_plus - f_target
        dFdk = (F_plus - F) / dk

        if abs(dFdk) < 1e-12:
            # avoid division by near-zero slope
            break

        k = k - F / dFdk

    return k  # return last iterate even if not fully converged


if __name__ == "__main__":
    p = SuspensionParams()
    f_target = 1.45  # target body mode frequency [Hz]

    ks_tuned = tune_ks_newton(p, f_target, k0=p.ks)
    f_body_final = body_mode_frequency(ks_tuned, p)

    print(f"Initial ks guess: {p.ks:.1f} N/m")
    print(f"Tuned ks (Newton): {ks_tuned:.1f} N/m")
    print(f"Resulting body mode frequency: {f_body_final:.4f} Hz")
