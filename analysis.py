
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Callable
from params import SuspensionParams
from signals import half_sine_bump
from model import rk4, rhs_passive, rhs_skyhook
from metrics import compute_outputs

def sine_input(A: float, f_hz: float):
    """ road input y(t) = A sin(2π f t) """
    w = 2.0*np.pi*f_hz
    def y(t):
        t = np.asarray(t)
        return A*np.sin(w*t)
    return y

def freq_response_passive(params: SuspensionParams,
                          freqs: np.ndarray,
                          A: float = 0.005,
                          cycles_settle: int = 10,
                          cycles_meas: int = 10,
                          dt: float = None) -> Dict[str, np.ndarray]:
    """
    Estimate a discrete frequency response by simulating a steady sine at each frequency.
    Returns arrays of RMS body acc and ratio to input amplitude (a kind of transmissibility proxy).
    """
    if dt is None:
        dt = params.dt
    rms_acc = []
    for f in freqs:
        y_fun = sine_input(A, f)
        T = 1.0/f
        t_end = (cycles_settle + cycles_meas)*T
        t, Y = rk4(lambda t,s: rhs_passive(t,s,params,y_fun), (0.0, t_end), [0,0,0,0], dt)
        # discard transients
        t0 = cycles_settle*T
        mask = t >= t0
        y_vec = y_fun(t[mask])
        out = compute_outputs(t[mask], Y[mask,:], y_vec)
        rms_acc.append(out["rms_acc"])
    rms_acc = np.array(rms_acc)
    # "Transmissibility-like" proxy: RMS(acc) normalized by input amplitude (m)
    # Not dimensionless; this is a convenience metric to locate resonances.
    proxy = rms_acc / A
    return {"freqs": freqs, "rms_acc": rms_acc, "proxy": proxy}

def pareto_sweep_passive(params: SuspensionParams,
                         ks_scale: np.ndarray,
                         c_scale: np.ndarray,
                         bump_speed: float = 20.0,
                         A: float = 0.05,
                         L: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep ks and damper scales to map comfort vs travel/tyre metrics for passive system.
    Returns arrays shaped (len(ks_scale), len(c_scale)) for:
      rms_acc, peak_travel, peak_tyre
    """
    y_fun, duration = half_sine_bump(amplitude=A, length=L, speed=bump_speed)
    t_end = max(3*duration, 2.0)
    dt = params.dt

    rms_acc = np.zeros((len(ks_scale), len(c_scale)))
    peak_travel = np.zeros_like(rms_acc)
    peak_tyre = np.zeros_like(rms_acc)

    for i, s_k in enumerate(ks_scale):
        for j, s_c in enumerate(c_scale):
            p = SuspensionParams(**vars(params))
            p.ks = params.ks * s_k
            p.c_comp = params.c_comp * s_c
            p.c_reb = params.c_reb * s_c
            t, Y = rk4(lambda t, s: rhs_passive(t, s, p, y_fun), (0.0, t_end), [0,0,0,0], dt)
            out = compute_outputs(t, Y, y_fun(t))
            rms_acc[i,j] = out["rms_acc"]
            peak_travel[i,j] = out["peak_travel"]
            peak_tyre[i,j] = out["peak_tyre"]
    return rms_acc, peak_travel, peak_tyre

def proxy_weighting(xsddot: np.ndarray, t: np.ndarray) -> float:
    """
    A simple *proxy* comfort weighting emphasizing ~4–8 Hz content.
    NOT ISO 2631-1. Implemented via a zero-phase band-emphasis in frequency domain.
    Returns weighted RMS.
    """
    x = np.asarray(xsddot)
    t = np.asarray(t)
    dt = t[1]-t[0]
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Emphasis window: triangular gain peaking in 4–8 Hz band
    g = np.ones_like(freqs)
    # linearly ramp from 1 at 1 Hz to 2 at 6 Hz, back to 1 at 12 Hz
    g += np.clip((freqs-1.0)/(6.0-1.0), 0, 1)  # up-ramp
    g -= np.clip((freqs-6.0)/(12.0-6.0), 0, 1) # down-ramp

    Xw = X * g
    xw = np.fft.irfft(Xw, n=n)
    # weighted RMS
    duration = t[-1]-t[0]
    rms = np.sqrt(np.trapz(xw**2, t)/duration)
    return float(rms)

def export_pareto_csv(path: str, ks_scale: np.ndarray, c_scale: np.ndarray,
                      rms_acc: np.ndarray, peak_travel: np.ndarray, peak_tyre: np.ndarray):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ks_scale","c_scale","rms_acc","peak_travel","peak_tyre"])
        for i, s_k in enumerate(ks_scale):
            for j, s_c in enumerate(c_scale):
                w.writerow([s_k, s_c, rms_acc[i,j], peak_travel[i,j], peak_tyre[i,j]])

def plot_xy(x, y, xlabel, ylabel, title, path):
    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
