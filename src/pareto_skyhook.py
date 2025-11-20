"""
skyhook_pareto.py

Pareto analysis for the **clipped skyhook** semi-active suspension.

This module performs a parametric sweep over a **global skyhook damping
scale factor**, evaluates performance metrics for each configuration, and
identifies the **Pareto-optimal** designs.

The three scalar objectives considered are:
    1. RMS sprung-mass acceleration       [m/s²]  → ride comfort (minimise)
    2. Maximum suspension travel          [m]     → packaging / bump-stop risk (minimise)
    3. Maximum tyre deflection            [m]     → road-holding / tyre load variation (minimise)

The results are used in the CMM3 design project to:
    - quantify how changes in skyhook gain affect comfort and constraints,
    - compare semi-active performance against the nonlinear passive baseline,
    - support selection of a final skyhook damping level based on a clear
      comfort–travel–tyre trade-off.

All units are SI unless otherwise stated.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.params import SuspensionParams
from src.skyhook_eval import evaluate_skyhook
from src.params import SuspensionParams

params = SuspensionParams()


def skyhook_suspension_pareto():
    def pareto_front(costs: np.ndarray) -> np.ndarray:
        """
        Compute the Pareto front for a set of designs in a **minimisation** problem.

        A design *j* is said to **dominate** design *i* if:
            - it is no worse than design i in all objectives, and
            - it is strictly better than design i in at least one objective.

        This function returns a boolean mask indicating which designs are
        **non-dominated** (i.e. Pareto-optimal) within the input set.

        Parameters
        ----------
        costs : np.ndarray
            2D array of shape (N, M), where:
                - N is the number of designs (e.g. different damping scales),
                - M is the number of objectives (here M = 3).
            Each row ``costs[i, :]`` contains the objective values for design i,
            with all objectives assumed to be **minimised**.

        Returns
        -------
        is_efficient : np.ndarray
            1D boolean array of length N. ``is_efficient[i]`` is True if design i
            is Pareto-efficient (non-dominated), and False otherwise.

        Notes
        -----
        - This implementation is O(N²), which is perfectly adequate for the
        relatively small design sets used in the project (N ≲ 50).
        - The function is generic and can be reused for both skyhook and passive
        damping Pareto analyses.
        """
        N = costs.shape[0]
        is_efficient = np.ones(N, dtype=bool)

        for i in range(N):
            if not is_efficient[i]:
                # Already known to be dominated; skip further checks.
                continue

            # Design j dominates design i if:
            #   costs[j] <= costs[i] in all objectives AND
            #   costs[j]  < costs[i] in at least one objective.
            dominates_i = np.all(costs <= costs[i], axis=1) & np.any(
                costs < costs[i], axis=1
            )

            if np.any(dominates_i & (np.arange(N) != i)):
                is_efficient[i] = False

        return is_efficient


    def sweep_skyhook(
        params: SuspensionParams,
        g_min: float = 0.05,
        g_max: float = 2.5,
        n: int = 25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform a parametric sweep of the **skyhook damping scale** and evaluate
        the corresponding performance metrics.

        For each value of the global scale factor ``g`` in [g_min, g_max], this
        function:
            1. Scales the skyhook damping bounds (c_min, c_max) by ``g``,
            2. Runs a clipped skyhook simulation via :func:`evaluate_skyhook`,
            3. Stores the resulting performance metrics.

        Parameters
        ----------
        params : SuspensionParams
            Baseline suspension parameter set (masses, stiffnesses, damping,
            simulation settings). This object is treated as read-only; scaled
            copies are created internally to avoid side effects.
        g_min : float, optional
            Lower bound of the damping scale range (dimensionless). A value
            less than 1.0 corresponds to a globally softer skyhook damper.
            Default is 0.05 to allow exploration of very soft settings.
        g_max : float, optional
            Upper bound of the damping scale range (dimensionless). A value
            greater than 1.0 corresponds to a globally stiffer skyhook damper.
            Default is 2.5.
        n : int, optional
            Number of sampling points between g_min and g_max (inclusive).
            Default is 25.

        Returns
        -------
        g_values : np.ndarray
            1D array of length N = n containing the sampled skyhook damping
            scale factors used in the sweep.
        metrics : np.ndarray
            2D array of shape (N, 3) where each row corresponds to:
                [RMS acceleration, max travel, max tyre deflection]
            in SI units, as returned by :func:`evaluate_skyhook`.

        Notes
        -----
        - This function is typically followed by :func:`pareto_front` to identify
        the subset of non-dominated skyhook designs.
        - Printing inside the loop provides a basic progress indicator when
        running longer sweeps.
        """
        g_values = np.linspace(g_min, g_max, n)
        metrics: list[tuple[float, float, float]] = []

        for g in g_values:
            #print(f"Running skyhook sweep at damping scale g = {g:.3f}")
            rms_acc, max_travel, max_tyre_def = evaluate_skyhook(params, g)
            metrics.append((rms_acc, max_travel, max_tyre_def))

        return g_values, np.array(metrics)


    def plot_results(
        g_values: np.ndarray,
        metrics: np.ndarray,
        is_eff: np.ndarray,
    ) -> None:
        """
        Generate summary plots for the skyhook damping Pareto sweep.

        Two visualisations are produced:

        1. **Objective space:** RMS acceleration vs. maximum suspension travel,
        coloured by damping scale g, with Pareto-optimal points highlighted.
        This shows the comfort–travel trade-off explicitly.

        2. **Design space:** Each metric plotted as a function of g. This helps
        interpret how sensitive each performance indicator is to changes in
        the skyhook damping level and where "reasonable" operating regions lie.

        Parameters
        ----------
        g_values : np.ndarray
            1D array of damping scale factors (output from :func:`sweep_skyhook`).
        metrics : np.ndarray
            2D array of objective values of shape (N, 3), where columns are:
                0 → RMS acceleration [m/s²]
                1 → Max suspension travel [m]
                2 → Max tyre deflection [m]
        is_eff : np.ndarray
            1D boolean array indicating which designs lie on the Pareto front
            (output from :func:`pareto_front`).

        Returns
        -------
        None
            The function creates and displays Matplotlib figures but does not
            return any values.
        """
        rms = metrics[:, 0]
        travel = metrics[:, 1]
        tyre = metrics[:, 2]

        # ------------------------------------------------------------------
        # 1. Pareto view in objective space:
        #    RMS acceleration vs suspension travel, coloured by g.
        # ------------------------------------------------------------------
        plt.figure()
        sc = plt.scatter(rms, travel, c=g_values, alpha=0.4, label="all designs")
        plt.scatter(
            rms[is_eff],
            travel[is_eff],
            facecolors="none",
            edgecolors="k",
            s=80,
            label="Pareto front",
        )

        plt.title("Skyhook Suspension Pareto Front")
        plt.xlabel("RMS sprung acceleration [m/s²]")
        plt.ylabel("Max suspension travel [m]")
        cbar = plt.colorbar(sc)
        cbar.set_label("skyhook damping scale g")
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/skyhook_pareto_objective_space.png", dpi=300)
        plt.show()

        """
        # ------------------------------------------------------------------
        # 2. Metrics vs damping scale: design space perspective.
        # ------------------------------------------------------------------
        plt.figure()
        plt.plot(g_values, rms, "-o", label="RMS accel [m/s²]")
        plt.plot(g_values, travel, "-o", label="Max travel [m]")
        plt.plot(g_values, tyre, "-o", label="Max tyre defl [m]")
        plt.xlabel("skyhook damping scale g [-]")
        plt.ylabel("metric value [SI units]")
        plt.legend()
        plt.tight_layout()
        plt.show()
        """

    '''if __name__ == "__main__":'''
        # ------------------------------------------------------------------
        # Example usage:
        #   - create a baseline parameter set,
        #   - sweep the skyhook damping scale,
        #   - compute the Pareto front,
        #   - print key results and generate plots.
        # ------------------------------------------------------------------

    print(f"Running passive sweep at damping scale 0.05 < g < 2.5 with {25} samples")
    g_values, metrics = sweep_skyhook(params, g_min=0.05, g_max=2.5, n=25)
    is_eff = pareto_front(metrics)

    print(f"{'g':<10} {'RMS Acc':<12} {'Travel (mm)':<15} {'Tyre Def (mm)':<15}")
    print("-" * 55)

    # Loop through data and print formatted rows
    for g, row in zip(g_values[is_eff], metrics[is_eff]):
        if 0.05 <= g <= 2.5:
            rms_acc = row[0]
            max_travel_mm = row[1] * 1000  # Convert m to mm
            max_tyre_mm = row[2] * 1000    # Convert m to mm
    
            # Print with 2 decimal places (.2f)
            print(f"{g:<10.2f} {rms_acc:<12.2f} {max_travel_mm:<15.2f} {max_tyre_mm:<15.2f}")

    plot_results(g_values, metrics, is_eff)

if __name__ == "__main__":
    skyhook_suspension_pareto()