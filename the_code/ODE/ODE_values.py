import matplotlib.pyplot as plt
from params import SuspensionParams

p = SuspensionParams()
speeds = [10.0, 20.0, 30.0]  # m/s

for mode in ["passive", "skyhook"]:
    print(f"\n=== {mode.upper()} SUSPENSION ===")
    for U in speeds:
        t, Y, m = run_simulation(U, mode, p)

        print(f"Speed = {U:4.1f} m/s | "
              f"max travel = {m['max_travel']*1000:5.1f} mm | "
              f"max tyre defl = {m['max_tyre_defl']*1000:5.1f} mm | "
              f"RMS acc = {m['rms_acc']:4.2f} m/s^2")

        # Example: plot body displacement for one case (say 20 m/s)
        if U == 20.0:
            plt.figure()
            plt.plot(t, m["xs"], label="Sprung mass")
            plt.plot(t, m["xu"], label="Unsprung mass")
            plt.title(f"{mode.capitalize()} suspension, U = {U} m/s")
            plt.xlabel("Time [s]")
            plt.ylabel("Displacement [m]")
            plt.legend()
            plt.grid(True)
            plt.show()
