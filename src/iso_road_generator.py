import numpy as np
import matplotlib.pyplot as plt
from src.constants import CLASS_GQ

def generate_iso8608_profile(
    class_label: str = "E",
    length: float = 200.0,
    dx: float = 0.05,
    w: float = 2.0,
    n0: float = 0.1,
    n_min: float = 0.01,
    seed: int | None = 23,
):
    """
    Generate a 1D ISO 8608-style road profile z(x).

    Returns:
        x : positions along road [m]
        z : elevation [m]
    """

    # Use uppercase internally so "a", "A" both work
    class_label = class_label.upper()

    if class_label not in CLASS_GQ:
        raise ValueError(f"Unknown class '{class_label}'. Use A–F only.")

    Gq_n0 = CLASS_GQ[class_label]

    rng = np.random.default_rng(seed)

    # Spatial grid
    x = np.arange(0.0, length, dx)
    N = x.size

    # Spatial frequency grid [cycles/m]
    n = np.fft.rfftfreq(N, d=dx)
    n_nonzero = n.copy()
    if n.size > 1:
        n_nonzero[0] = n[1] if n[1] > 0.0 else n0
    else:
        n_nonzero[0] = n0

    # ISO 8608 PSD
    Gq = Gq_n0 * (n_nonzero / n0) ** (-w)

    # Low-frequency cut-off
    Gq[n < n_min] = 0.0

    # One-sided spectrum amplitudes
    dn = n[1] - n[0] if n.size > 1 else 0.0
    amplitude = np.sqrt(2.0 * Gq * dn)

    # Random phases
    phi = 2.0 * np.pi * rng.random(n.size)
    Zk = amplitude * np.exp(1j * phi)

    # Zero mean elevation
    Zk[0] = 0.0

    # Inverse FFT → spatial profile
    z = np.fft.irfft(Zk, n=N)
    z -= np.mean(z)

    # SCALE FACTOR
    SCALE = 150  # Scale factor of 150 to emulate an intensely rough road (Edinburgh Streets)
    z *= SCALE

    return x, z


if __name__ == "__main__":
    # === change this line to try different classes ===
    # e.g. "A", "B", "C", "D", "E", "F" (case-insensitive)
    road_class = "E"

    x, z = generate_iso8608_profile(
        class_label=road_class,
        length=200.0,
        dx=0.05,
        seed=42,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(x, z)
    plt.title(f"ISO 8608 Class {road_class.upper()} Road Profile (200 m)")
    plt.xlabel("Distance x [m]")
    plt.ylabel("Road elevation z(x) [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
