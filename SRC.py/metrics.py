 
"""
Performance metrics and constraint checks.
"""
import numpy as np

def rms(x, t):
    x = np.asarray(x)
    t = np.asarray(t)
    # Trapezoidal integration of x^2 over t / duration
    x2 = x**2
    integral = np.trapz(x2, t)
    duration = t[-1] - t[0]
    return np.sqrt(integral / duration)

def compute_outputs(t, Y, y_vec):
    """
    Given time vector t, states Y = [xs, xsdot, xu, xudot],
    and road displacement y(t), return performance signals and metrics.
    """
    xs = Y[:,0]
    xsdot = Y[:,1]
    xu = Y[:,2]
    xudot = Y[:,3]

    # Accelerations by finite difference (central)
    dt = t[1] - t[0]
    xsddot = np.gradient(xsdot, dt)

    # Tyre deflection, suspension travel
    travel = xs - xu
    tyre_defl = xu - y_vec

    out = {
        "xs": xs,
        "xsdot": xsdot,
        "xsddot": xsddot,
        "xu": xu,
        "xudot": xudot,
        "travel": travel,
        "tyre_defl": tyre_defl,
        "rms_acc": rms(xsddot, t),
        "rms_tyre": rms(tyre_defl, t),
        "peak_travel": float(np.max(np.abs(travel))),
        "peak_tyre": float(np.max(np.abs(tyre_defl))),
    }
    return out
