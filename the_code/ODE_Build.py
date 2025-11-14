import numpy as np # arrays, maths etc
from scipy.integrate import solve_ivp # ODE solver
import matplotlib.pyplot as plt # plotting

# Parameters
m_s = 300.0  # sprung mass [kg]
m_u = 40.0   # unsprung mass [kg]
k_s = 20000.0  # suspension spring stiffness [N/m]
k_t = 190000.0 # tyre stiffness [N/m]
c_s = 1500.0  # suspension damping [Ns/m]
c_comp = 3000.0  # damper compression damping [Ns/m]
c_reb = 1500.0   # damper rebound damping [Ns/m]

v = 20.0  # vehicle speed [m/s]

# Road input: half-sine bump
def road_input(t, v):
    '''Simple placeholder road profile, can replace with more complex function later'''
    A = 0.02    # bump amplitude [m]
    freq = 1.0   # bump frequency [Hz]
    z_r = A * np.sin(2 * np.pi * freq * t)
    return z_r

# ODE for passive quarter-car model
def quarter_car_ode(t, state, m_s, m_u, k_s, c_s, k_t, v):
    '''
    state = [x_s, x_s_dot, x_u, x_u_dot]
    returns [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]
    '''
    x_s, x_s_dot, x_u, x_u_dot = state

    # Road input at time t 
    z_r = road_input(t, v)

    # Relative displacements and velocities
    x_su = x_s - x_u    # suspension deflection
    x_su_dot = x_s_dot - x_u_dot  # suspension deflection rate

    # Equations of motion
    x_s_ddot = (-k_s * x_su - c_s * x_su_dot) / m_s
    
    x_ur = x_u - z_r  # tyre deflection
    x_u_ddot = (k_s * x_su + c_s * x_su_dot - k_t * x_ur) / m_u

    # Return first-order system 
    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]


# ODE for skyhook suspension model
def quarter_car_skyhook(t, state, m_s, m_u, k_s, c_comp, c_reb, k_t, v):
    x_s, x_s_dot, x_u, x_u_dot = state

    # Road input at time t
    z_r = road_input(t, v)

    # Relative displacements and velocities
    x_su = x_s - x_u    # suspension deflection
    x_su_dot = x_s_dot - x_u_dot  # suspension deflection rate

    # Decide effective damping based on skyhook logic
    if x_s_dot * x_su_dot > 0:  # same sign
        c_eff = c_comp # rebound damping
    else:
        c_eff = c_reb  # compression damping

    # Equations of motion
    x_s_ddot = (-k_s * x_su - c_eff * x_su_dot) / m_s
    
    x_ur = x_u - z_r  # tyre deflection
    x_u_ddot = (k_s * x_su + c_eff * x_su_dot - k_t * x_ur) / m_u

    # Return first-order system 
    return [x_s_dot, x_s_ddot, x_u_dot, x_u_ddot]

''' Test the ODE implementation '''
'''
if __name__ == "__main__":
    test_state = [0.0, 0.0, 0.0, 0.0]
    print(quarter_car_ode(0.0, test_state, m_s, m_u, k_s, c_s, k_t, v))
'''

# Simulation settings
t_start = 0.0
t_end = 5.0 # total simulation time [s]
t_eval = np.linspace(t_start, t_end, 2000) # time points at which we want results

# Initial state: [x_s, x_s_dot, x_u, x_u_dot] = [0, 0, 0, 0
x0 = [0.0, 0.0, 0.0, 0.0] # starts everything at rest

# Solving the passive ODE
sol = solve_ivp(
    fun=lambda t, y: quarter_car_ode(t, y, m_s, m_u, k_s, c_s, k_t, v),
    t_span=(t_start, t_end),
    y0=x0,
    t_eval=t_eval,
    method='RK45'   
)

# Solving the skyhook ODE
sol_sky = solve_ivp(
    fun=lambda t, y: quarter_car_skyhook(t, y, m_s, m_u, k_s, c_comp, c_reb, k_t, v),
    t_span=(t_start, t_end),
    y0=x0,
    t_eval=t_eval,
    method='RK45'   
)

'''Wrap ODE in a lambda to pass extra parameters, t_span tells SciPy start and end time
y0 is initial state, t_eval is time points we want results for and method='RK45' is a common ODE solver'''

# Sanity check print
print(f"ODE solver success: {sol.success}, message: {sol.message}")

# Extracting solution (passive)
t = sol.t
x_s     = sol.y[0, :]  # sprung mass displacement
x_s_dot = sol.y[1, :]  # sprung mass velocity
x_u     = sol.y[2, :]  # unsprung mass displacement
x_u_dot = sol.y[3, :]  # unsprung mass velocity
'''Just pulling the columns out of sol.y for easier access'''

# Extracting solution (skyhook)
t_sky = sol_sky.t
x_s_sky     = sol_sky.y[0, :]  # sprung mass displacement
x_s_dot_sky = sol_sky.y[1, :]  # sprung mass velocity
x_u_sky     = sol_sky.y[2, :]  # unsprung mass displacement
x_u_dot_sky = sol_sky.y[3, :]  # unsprung mass velocity
'''Just pulling the columns out of sol.y for easier access'''

# Road inputs for each
z_r_passive = np.array([road_input(ti, v) for ti in t])  # road profile over time
z_r_sky     = np.array([road_input(ti, v) for ti in t_sky])  # road profile over time

# Derived quantities (passive)
travel_passive = x_s - x_u  # suspension travel
tyre_passive = x_u - z_r_passive  # tyre deflection
acc_passive = ( -k_s * (x_s - x_u) - c_s * (x_s_dot - x_u_dot) ) / m_s  # sprung mass acceleration

# Derived quantities (skyhook)
travel_skyhook = x_s_sky - x_u_sky  # suspension travel
tyre_skyhook = x_u_sky - z_r_sky  # tyre deflection
vel_rel_sky = x_s_dot_sky - x_u_dot_sky
c_eff_sky = np.where(x_s_dot_sky * vel_rel_sky > 0, c_comp, c_reb)
acc_skyhook = (-k_s * (x_s_sky - x_u_sky) - c_eff_sky * vel_rel_sky) / m_s  # sprung mass acceleration
'''This gives us the three key metrics for the report'''

# Performance metrics (passive)
max_travel_passive = np.max(np.abs(travel_passive))
max_tyre_passive = np.max(np.abs(tyre_passive))
rms_acc_passive = np.sqrt(np.mean(acc_passive**2))

# Performance metrics (skyhook)
max_travel_skyhook = np.max(np.abs(travel_skyhook))
max_tyre_skyhook = np.max(np.abs(tyre_skyhook))
rms_acc_skyhook = np.sqrt(np.mean(acc_skyhook**2))

# Printing results
print("=== PASSIVE ===")
print("Max travel:        ", max_travel_passive*1000, "mm")
print("Max tyre defl:     ", max_tyre_passive*1000, "mm")
print("RMS acceleration:  ", rms_acc_passive, "m/s^2")

print("\n=== SKYHOOK ===")
print("Max travel:        ", max_travel_skyhook*1000, "mm")
print("Max tyre defl:     ", max_tyre_skyhook*1000, "mm")
print("RMS acceleration:  ", rms_acc_skyhook, "m/s^2")


'''
# Plotting results
plt.figure(figsize=(10,5))
plt.plot(t, suspension_travel*1000, label="Suspension Travel (mm)")
plt.plot(t, tyre_deflection*1000, label="Tyre Deflection (mm)")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (mm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, x_s_ddot, label="Sprung Mass Acceleration (m/s²)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
'''
