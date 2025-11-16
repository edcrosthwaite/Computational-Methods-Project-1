"""Sensitivity analysis, Pareto identification, and Monte Carlo skeleton."""
from __future__ import annotations
import numpy as np, copy
from typing import Callable, Dict, Any

def sensitivity_analysis_structured(simulate_fn: Callable[[Any], Dict[str, float]], p_base: Any, param_ranges: Dict[str, tuple], samples: int = 20):
    results = {}
    for param, (smin, smax) in param_ranges.items():
        scales = np.linspace(smin, smax, samples)
        metrics = {'rms_acc': [], 'peak_travel': [], 'peak_tyre': [], 'settling_time': []}
        for s in scales:
            p = copy.deepcopy(p_base)
            setattr(p, param, getattr(p_base, param) * s)
            out = simulate_fn(p)
            metrics['rms_acc'].append(out['rms_acc']); metrics['peak_travel'].append(out['peak_travel']); metrics['peak_tyre'].append(out['peak_tyre']); metrics['settling_time'].append(out.get('settling_time', 0.0))
        results[param] = {'scales': scales, 'metrics': metrics}
    return results

def pareto_frontier_identification(rms_arr: np.ndarray, travel_arr: np.ndarray):
    pts = np.column_stack([np.ravel(rms_arr), np.ravel(travel_arr)])
    n = len(pts); is_pareto = np.ones(n, dtype=bool)
    for i, p in enumerate(pts):
        if is_pareto[i]:
            dominated = np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)
            dominated[i] = False
            is_pareto[dominated] = False
    return pts[is_pareto]

def robustness_analysis_monte_carlo(simulate_fn: Callable[[Any], Dict[str, float]], p_nominal: Any, n_samples: int = 1000, rng_seed: int = 42):
    rng = np.random.default_rng(rng_seed)
    results = {'rms_acc': [], 'peak_travel': [], 'peak_tyre': []}
    for _ in range(n_samples):
        p = copy.deepcopy(p_nominal)
        p.ks *= rng.normal(1.0, 0.05); p.c_comp *= rng.normal(1.0, 0.10); p.c_reb *= rng.normal(1.0, 0.10)
        p.ms *= rng.normal(1.0, 0.10); p.mu *= rng.normal(1.0, 0.10)
        out = simulate_fn(p)
        results['rms_acc'].append(out['rms_acc']); results['peak_travel'].append(out['peak_travel']); results['peak_tyre'].append(out['peak_tyre'])
    summary = {}
    for k, v in results.items():
        arr = np.array(v)
        summary[k] = {'mean': float(arr.mean()), 'std': float(arr.std()), '95ci': 1.96 * float(arr.std())}
    return {'raw': results, 'summary': summary}
