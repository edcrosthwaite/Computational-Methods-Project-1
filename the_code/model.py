
"""
Quarter-car model (2-DoF): states [xs, xsdot, xu, xudot].
Equations:
ms*xsddot = -ks*(xs-xu) - Fd(v_rel)
mu*xuddot =  ks*(xs-xu) + Fd(v_rel) - kt*(xu - y)

where v_rel = xsdot - xudot and Fd depends on damping strategy.
"""

import numpy as np
from typing import Callable
from damping import passive_damper_force, skyhook_on_off

def rhs_passive(t, state, params, y_fun: Callable[[float], float]):
    xs, xsdot, xu, xudot = state
    y = float(y_fun(t))
    v_rel = xsdot - xudot
    Fd = passive_damper_force(v_rel, params.c_comp, params.c_reb)
    # Forces
    Fs = params.ks * (xs - xu)
    Ft = params.kt * (xu - y)

    xsddot = (-Fs - Fd) / params.ms
    xuddot = ( Fs + Fd - Ft) / params.mu
    return np.array([xsdot, xsddot, xudot, xuddot])

def rhs_skyhook(t, state, params, y_fun: Callable[[float], float]):
    xs, xsdot, xu, xudot = state
    y = float(y_fun(t))
    v_rel = xsdot - xudot
    c = skyhook_on_off(xsdot, v_rel, params.c_min, params.c_max)
    Fd = c * v_rel
    Fs = params.ks * (xs - xu)
    Ft = params.kt * (xu - y)

    xsddot = (-Fs - Fd) / params.ms
    xuddot = ( Fs + Fd - Ft) / params.mu
    return np.array([xsdot, xsddot, xudot, xuddot])

def rk4(f, t_span, y0, dt):
    """
    Fixed-step RK4 integrator.
    f: function(t, y) -> dy/dt
    t_span: (t0, tf)
    y0: initial state vector
    dt: time step
    Returns t, Y array
    """
    t0, tf = t_span
    n = int(np.ceil((tf - t0)/dt)) + 1
    t = np.linspace(t0, tf, n)
    y = np.zeros((n, len(y0)), dtype=float)
    y[0] = y0
    ti = t0
    yi = np.array(y0, dtype=float)
    for i in range(1, n):
        k1 = f(ti, yi)
        k2 = f(ti + 0.5*dt, yi + 0.5*dt*k1)
        k3 = f(ti + 0.5*dt, yi + 0.5*dt*k2)
        k4 = f(ti + dt, yi + dt*k3)
        yi = yi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        y[i] = yi
        ti = t[i]
    return t, y
