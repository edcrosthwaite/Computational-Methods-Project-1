"""Damper characterization: compare interpolation/regression methods."""
from __future__ import annotations
import numpy as np
from typing import Dict
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def compare_interpolation_methods(v_data: np.ndarray, F_data: np.ndarray) -> Dict[str, Dict]:
    v = np.asarray(v_data); F = np.asarray(F_data)
    if v.ndim != 1 or F.ndim != 1 or v.size != F.size:
        raise ValueError('v_data and F_data must be 1D and same length')
    results = {}
    f_lin = interp1d(v, F, kind='linear', fill_value='extrapolate')
    f_cubic = CubicSpline(v, F, extrapolate=True)
    f_smooth = UnivariateSpline(v, F, s=0.1)
    poly = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(v.reshape(-1,1), F)
    f_poly = lambda vv: poly.predict(np.atleast_1d(vv).reshape(-1,1))
    ransac = RANSACRegressor(random_state=42).fit(v.reshape(-1,1), F)
    f_ransac = lambda vv: ransac.predict(np.atleast_1d(vv).reshape(-1,1))
    methods = {'piecewise_linear': f_lin, 'cubic_spline': f_cubic, 'smoothing_spline': f_smooth, 'polynomial_deg3': f_poly, 'ransac': f_ransac}
    v_test = np.linspace(v.min(), v.max(), 500)
    for name, fn in methods.items():
        try:
            F_pred = fn(v)
            rmse = float(np.sqrt(np.mean((F_pred - F)**2)))
            F_pred_test = fn(v_test)
            dd = np.gradient(np.gradient(F_pred_test))
            smoothness = float(np.std(dd))
            results[name] = {'rmse': rmse, 'smoothness': smoothness}
        except Exception as e:
            results[name] = {'error': str(e)}
    return results

if __name__ == '__main__':
    v = np.linspace(-1,1,21)
    F = 1500.0 * np.tanh(3.0 * v) + 50.0 * v
    print(compare_interpolation_methods(v, F))
