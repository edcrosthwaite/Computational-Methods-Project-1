"""
Design utilities for suspension tuning.

This script computes the natural frequencies of the quarter-car model 
(a 2-DOF system) and uses the Newton–Raphson root-finding algorithm to 
iteratively adjust the suspension stiffness (ks) until the system's 
primary mode frequency (the 'body mode') matches a desired target frequency.
This ensures the suspension is tuned based on the full coupled dynamics.
"""
import numpy as np
from src.params import SuspensionParams
import src.constants

def body_mode_frequency(ks: float, params: SuspensionParams) -> float:
    """
    Compute the lower (body) natural frequency [Hz] of the 2-DOF quarter-car
    for a given suspension stiffness ks. This frequency dictates the vehicle's
    fundamental ride comfort characteristic.

    Args:
        ks (float): The current suspension spring stiffness [N/m] being tested.
        params (SuspensionParams): Vehicle parameters (masses, tire stiffness).

    Returns:
        float: The body mode natural frequency [Hz].
    """
    # Mass (M) and Stiffness (K) matrices for the undamped 2-DOF quarter-car model.
    # The system is M*x_ddot + K*x = 0, where x = [x_s, x_u]^T.
    # M = [[m_s, 0], [0, m_u]]
    M = np.array([[params.ms, 0.0],
                  [0.0, params.mu]])

    # K = [[ks, -ks], [-ks, ks + kt]]
    # Note: k_t is the tire stiffness.
    K = np.array([[ ks, -ks],
                  [-ks, ks + params.kt]])

    # Generalised Eigenproblem: M^{-1} K * phi = lambda * phi
    # The eigenvalues (lambda) correspond to omega^2 (angular frequency squared).
    A = np.linalg.inv(M) @ K # A = M^{-1} K
    evals, _ = np.linalg.eig(A)

    # Sort eigenvalues. The lower eigenvalue corresponds to the body mode.
    # We take the real part, as frequencies must be real numbers in a physical system.
    evals_sorted = np.sort(evals.real)
    omega2_body = evals_sorted[0] # The lowest eigenvalue is omega_body^2
    
    omega_body = np.sqrt(omega2_body)        # Convert back to angular frequency (rad/s)
    f_body = omega_body / (2.0 * np.pi)      # Convert to frequency in Hertz (Hz)
    return f_body


def tune_ks_newton(
    params: SuspensionParams,
    f_target: float,
    k0: float = src.constants.K_INITIAL_GUESS,
    tol: float = 1e-4,
    max_iter: int = 20,
    dk: float = 100.0,
) -> float:
    """
    Uses the Newton–Raphson iterative method to find the suspension stiffness (ks) 
    that achieves the desired target body mode frequency (f_target).

    The function seeks the root of the error function: F(ks) = f_body(ks) - f_target.

    Args:
        params (SuspensionParams): Vehicle parameters.
        f_target (float): The desired body natural frequency [Hz].
        k0 (float): Initial guess for the stiffness [N/m].
        tol (float): Convergence tolerance for the error F(ks) [Hz].
        max_iter (int): Maximum number of iterations to prevent infinite loops.
        dk (float): Step size for the finite difference derivative approximation.

    Returns:
        float: The tuned suspension stiffness ks [N/m].
    """

    k = k0
    for i in range(max_iter):
        # 1. Evaluate the function F(k) = f_actual - f_target
        f = body_mode_frequency(k, params)
        F = f - f_target

        if abs(F) < tol:
            # Check for convergence: If the error (F) is within tolerance, we found the root.
            print(f"Converged in {i+1} iterations.")
            return k

        # 2. Approximate the derivative F'(k) = dF/dk using finite difference.
        # F'(k) ≈ [F(k+dk) - F(k)] / dk
        f_plus = body_mode_frequency(k + dk, params)
        F_plus = f_plus - f_target
        dFdk = (F_plus - F) / dk

        if abs(dFdk) < 1e-12:
            # Stop if the slope is near zero (flat line); prevents division by zero.
            print("Warning: Near-zero slope encountered. Stopping early.")
            break

        # 3. Newton–Raphson Update Rule: k_new = k_old - F(k) / F'(k)
        k = k - F / dFdk

    return k  # return last iterate even if not fully converged


# Utility function to encapsulate the tuning process
def root_finding_entry(p: SuspensionParams, f_target: float) -> tuple[float, float]:
    """
    Runs the Newton-Raphson tuning for a single target frequency.

    Args:
        p (SuspensionParams): Vehicle parameters.
        f_target (float): The desired body natural frequency [Hz].

    Returns:
        tuple[float, float]: (Tuned ks, Resulting body mode frequency)
    """
    # Start the tuning process, using the current ks as the initial guess
    ks_tuned = tune_ks_newton(p, f_target, k0=p.ks)
    # Verify the final frequency achieved by the tuned stiffness
    f_body_final = body_mode_frequency(ks_tuned, p)

    return ks_tuned, f_body_final
    
def ks_rootfinding():
    """
    Main execution function to demonstrate stiffness tuning for a list of target frequencies.
    """
    p = SuspensionParams()

    # List of target body natural frequencies (Hz) for the design study
    f_target_list = [1.3, 1.45, 1.6]
    results = {}
    for f_target in f_target_list:
        ks_tuned, f_body_final = root_finding_entry(p, f_target)
        results[f_target] = (ks_tuned, f_body_final)

        print(f"----------------------------------------")
        print(f"\nFor target frequency {f_target:.3f} Hz:\n")
        print(f"Initial ks guess: {p.ks:.1f} N/m")
        print(f"Tuned ks (Newton): {ks_tuned:.1f} N/m")
        print(f"Resulting body mode frequency: {f_body_final:.4f} Hz")
        
if __name__ == "__main__":
    ks_rootfinding()