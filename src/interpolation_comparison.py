"""
ODE_Interpol.py

Uses interpolation to present how peak acceleration varies with velocity for each damping system
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from src.ODEinterpol import ode_bump

def interpolation_comparison():
    """
    Runs ODEs N times and measures the peak acceleration while varying velocity.
    Then plots acceleration and interpolates between points.
    """
    n = 5       # Sets number of experiments to perform
    values = np.empty((n, 2))       # Create empty array to hold results
    v_values = np.linspace(1, 20, n)    # Sets velocity values to be used in experiments

    # Perform experiments N times
    for i in range(n):
        values[i] = ode_bump(v = v_values[i])

    values = np.rot90(values, k=-1)     # Reorders the matrix for plotting
    values = np.fliplr(values)
    labels = [["Passive", "r"], ["Skyhook", "b"]]

    # Used to plot both passive and skyhook results
    for i in range(2):
        x = v_values
        y = values[i]

        # Set number of points to use for interpolation
        # num gives number of interpolated points between each measured point
        x_int = np.linspace(0, max(v_values), num=n*10)

        # Generate spline interpolant
        f_spline  = interpolate.splrep(x, y, s=0)
        # Evaluate spline at the desired points
        y_int_spline = interpolate.splev(x_int, f_spline, der=0)

        plt.plot(x, y, "h", ms=10, label=labels[i][0], color=labels[i][1])
        plt.plot(x_int, y_int_spline, "o", ms=1, color=labels[i][1])
        plt.xlabel("Velocity [ms-1]")
        plt.ylabel("Peak Acceleration [ms-2]")

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    interpolation_comparison()
