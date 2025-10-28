
"""
Main driver:
1) Root-find ks to hit target sprung frequency mid-point (1.45 Hz)
2) Simulate passive damper over a 50 mm bump at 10, 20, 30 m/s
3) Simulate semi-active skyhook for comparison (20 m/s case)
4) Summarise RMS, peak metrics; check constraints
5) Demonstrate regression/interpolation on damper data (saves a plot)
All figures saved under ./outputs
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from params import SuspensionParams
from design import tune_ks_for_target_fn, coupled_natural_frequencies, sprung_fn_simple, unsprung_fn_simple
from signals import half_sine_bump, sine_sweep
from model import rk4, rhs_passive, rhs_skyhook
from metrics import compute_outputs
from fit_damper import synth_damper_data, fit_piecewise_interpolant
from io_utils import ensure_dir, save_json

def run():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    ensure_dir(out_dir)
    p = SuspensionParams()

    # 1) Root-find ks for target sprung frequency (mid of 1.3–1.6 Hz)
    f_target = 0.5*(p.sprung_fn_band[0] + p.sprung_fn_band[1])  # 1.45 Hz
    ks_new = tune_ks_for_target_fn(p, f_target)
    p.ks = ks_new

    f1, f2 = coupled_natural_frequencies(p)
    f_s_est = sprung_fn_simple(p.ks, p.ms)
    f_u_est = unsprung_fn_simple(p.kt, p.mu)

    print(f"Chosen ks to target {f_target:.3f} Hz (sprung): ks = {p.ks:.1f} N/m")
    print(f"Coupled natural frequencies (Hz): f1={f1:.3f}, f2={f2:.3f}")
    print(f"Single-DOF est: sprung={f_s_est:.3f} Hz, unsprung={f_u_est:.3f} Hz")

    # 2) Passive simulations over the bump at different speeds
    speeds = [10.0, 20.0, 30.0]
    results_summary = {"passive": {}, "skyhook": {}}
    for v in speeds:
        y_fun, duration = half_sine_bump(amplitude=0.05, length=0.5, speed=v)
        t_end = max(3*duration, 2.0)  # simulate beyond bump
        t, Y = rk4(lambda t, s: rhs_passive(t, s, p, y_fun), (0.0, t_end), [0,0,0,0], p.dt)
        y_vec = y_fun(t)
        out = compute_outputs(t, Y, y_vec)
        results_summary["passive"][f"bump_{int(v)}ms"] = {
            "rms_acc": out["rms_acc"],
            "rms_tyre": out["rms_tyre"],
            "peak_travel": out["peak_travel"],
            "peak_tyre": out["peak_tyre"]
        }

        # Plot example time histories (xs acc, travel, tyre defl)
        fig = plt.figure()
        plt.plot(t, out["xsddot"])
        plt.xlabel("t [s]")
        plt.ylabel("Sprung acc [m/s^2]")
        plt.title(f"Passive: Sprung acceleration – bump @ {int(v)} m/s")
        fig.savefig(os.path.join(out_dir, f"passive_acc_{int(v)}ms.png"), dpi=160)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(t, out["travel"])
        plt.xlabel("t [s]")
        plt.ylabel("Suspension travel [m]")
        plt.title(f"Passive: Travel – bump @ {int(v)} m/s")
        fig.savefig(os.path.join(out_dir, f"passive_travel_{int(v)}ms.png"), dpi=160)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(t, out["tyre_defl"])
        plt.xlabel("t [s]")
        plt.ylabel("Tyre deflection [m]")
        plt.title(f"Passive: Tyre deflection – bump @ {int(v)} m/s")
        fig.savefig(os.path.join(out_dir, f"passive_tyre_{int(v)}ms.png"), dpi=160)
        plt.close(fig)

    # 3) Semi-active skyhook on 20 m/s bump
    y_fun, duration = half_sine_bump(amplitude=0.05, length=0.5, speed=20.0)
    t_end = max(3*duration, 2.0)
    t, Y = rk4(lambda t, s: rhs_skyhook(t, s, p, y_fun), (0.0, t_end), [0,0,0,0], p.dt)
    y_vec = y_fun(t)
    out = compute_outputs(t, Y, y_vec)
    results_summary["skyhook"]["bump_20ms"] = {
        "rms_acc": out["rms_acc"],
        "rms_tyre": out["rms_tyre"],
        "peak_travel": out["peak_travel"],
        "peak_tyre": out["peak_tyre"]
    }

    fig = plt.figure()
    plt.plot(t, out["xsddot"])
    plt.xlabel("t [s]")
    plt.ylabel("Sprung acc [m/s^2]")
    plt.title("Skyhook: Sprung acceleration – bump @ 20 m/s")
    fig.savefig(os.path.join(out_dir, f"skyhook_acc_20ms.png"), dpi=160)
    plt.close(fig)

    # 4) Save summary JSON
    save_json(results_summary, os.path.join(out_dir, "summary.json"))

    # 5) Regression / interpolation demo on damper force-velocity
    v, F_noisy, F_true = synth_damper_data(p.c_comp, p.c_reb, seed=p.seed)
    fit = fit_piecewise_interpolant(v, F_noisy)

    v_dense = np.linspace(v.min(), v.max(), 401)
    F_fit = np.array([fit.F(vi) for vi in v_dense])

    fig = plt.figure()
    plt.plot(v, F_noisy, 'o', label="noisy data")
    plt.plot(v, F_true, '-', label="true (unknown)")
    plt.plot(v_dense, F_fit, '--', label="piecewise fit")
    plt.xlabel("v_rel [m/s]")
    plt.ylabel("Damper force [N]")
    plt.title("Damper F–v regression/interpolation (demo)")
    plt.legend()
    fig.savefig(os.path.join(out_dir, "damper_fit.png"), dpi=160)
    plt.close(fig)

    # Print console summary for convenience
    print("=== Summary (key metrics) ===")
    for k, v in results_summary["passive"].items():
        print(f"Passive {k}: rms_acc={v['rms_acc']:.3f}, peak_travel={v['peak_travel']:.4f}, peak_tyre={v['peak_tyre']:.4f}")
    sk = results_summary["skyhook"]["bump_20ms"]
    print(f"Skyhook 20 m/s: rms_acc={sk['rms_acc']:.3f}, peak_travel={sk['peak_travel']:.4f}, peak_tyre={sk['peak_tyre']:.4f}")

if __name__ == "__main__":
    run()


# === EXTENDED ANALYSIS ===
from design import tune_ks_for_target_fn
from params import SuspensionParams
import json
from analysis import freq_response_passive, pareto_sweep_passive, export_pareto_csv, plot_xy, proxy_weighting

# Rebuild params and ks identical to earlier run
out_dir = os.path.join(os.path.dirname(__file__), "outputs")
p = SuspensionParams()
f_target = 0.5*(p.sprung_fn_band[0] + p.sprung_fn_band[1])
p.ks = tune_ks_for_target_fn(p, f_target)

# Load results_summary from JSON
with open(os.path.join(out_dir, "summary.json"), "r") as _f:
    results_summary = json.load(_f)

# A) Frequency response (discrete)
freqs = np.linspace(0.5, 20.0, 30)
fr = freq_response_passive(p, freqs, A=0.002, cycles_settle=5, cycles_meas=5)
fig = plt.figure()
plt.plot(fr["freqs"], fr["proxy"])
plt.xlabel("Frequency [Hz]")
plt.ylabel("RMS(acc)/A [s^-2]")
plt.title("Passive discrete frequency response (proxy)")
fig.savefig(os.path.join(out_dir, "freq_response_proxy.png"), dpi=160)
plt.close(fig)

# B) Parameter sweep and Pareto charts
ks_scale = np.linspace(0.8, 1.2, 5)     # ±30% ks
c_scale  = np.linspace(0.7, 1.3, 5)     # 0.5× to 1.5× damping
rms_acc, peak_travel, peak_tyre = pareto_sweep_passive(p, ks_scale, c_scale, bump_speed=20.0)

# Export CSV for tables
export_pareto_csv(os.path.join(out_dir, "pareto_passive.csv"), ks_scale, c_scale, rms_acc, peak_travel, peak_tyre)

# Simple Pareto scatter: comfort vs travel (flattened grid)
x = rms_acc.flatten()
y = peak_travel.flatten()
fig = plt.figure()
plt.plot(x, y, 'o')
plt.xlabel("RMS body acc [m/s^2]")
plt.ylabel("Peak travel [m]")
plt.title("Passive Pareto: comfort vs travel")
fig.savefig(os.path.join(out_dir, "pareto_comfort_vs_travel.png"), dpi=160)
plt.close(fig)

# Comfort vs tyre
y2 = peak_tyre.flatten()
fig = plt.figure()
plt.plot(x, y2, 'o')
plt.xlabel("RMS body acc [m/s^2]")
plt.ylabel("Peak tyre deflection [m]")
plt.title("Passive Pareto: comfort vs tyre")
fig.savefig(os.path.join(out_dir, "pareto_comfort_vs_tyre.png"), dpi=160)
plt.close(fig)

# C) Weighted (proxy) comfort metric on the 20 m/s bump (passive vs skyhook)
y_fun, duration = half_sine_bump(amplitude=0.05, length=0.5, speed=20.0)
t_end = max(3*duration, 2.0)
t_p, Y_p = rk4(lambda t, s: rhs_passive(t, s, p, y_fun), (0.0, t_end), [0,0,0,0], p.dt)
t_s, Y_s = rk4(lambda t, s: rhs_skyhook(t, s, p, y_fun), (0.0, t_end), [0,0,0,0], p.dt)

y_vec = y_fun(t_p)
out_p = compute_outputs(t_p, Y_p, y_vec)
out_s = compute_outputs(t_s, Y_s, y_vec)

w_rms_p = proxy_weighting(out_p["xsddot"], t_p)
w_rms_s = proxy_weighting(out_s["xsddot"], t_s)

with open(os.path.join(out_dir, "weighted_comfort_proxy.txt"), "w") as f:
    f.write(f"Weighted (proxy) RMS acc – Passive: {w_rms_p:.4f} m/s^2\n")
    f.write(f"Weighted (proxy) RMS acc – Skyhook: {w_rms_s:.4f} m/s^2\n")

# D) Compact results table (Markdown)
md = []
md.append("| Case | RMS acc [m/s^2] | Peak travel [m] | Peak tyre [m] |")
md.append("|------|------------------|------------------|---------------|")
for k, v in results_summary["passive"].items():
    md.append(f"| Passive {k} | {v['rms_acc']:.3f} | {v['peak_travel']:.4f} | {v['peak_tyre']:.4f} |")
sk = results_summary["skyhook"]["bump_20ms"]
md.append(f"| Skyhook bump_20ms | {sk['rms_acc']:.3f} | {sk['peak_travel']:.4f} | {sk['peak_tyre']:.4f} |")

with open(os.path.join(out_dir, "results_table.md"), "w") as f:
    f.write("\n".join(md))

print("Extended analyses complete.")
