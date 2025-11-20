"""
passive_pareto.py

Pareto analysis for the nonlinear piecewise-passive suspension.

This module performs a parametric sweep over a **global damping scale factor**
applied to the four-coefficient passive damper, evaluates performance metrics
for each configuration, and identifies the **Pareto-optimal** designs.

The three scalar objectives considered are:
    1. RMS sprung-mass acceleration       [m/s²]  → ride comfort (minimise)
    2. Maximum suspension travel          [m]     → packaging / bump-stop risk (minimise)
    3. Maximum tyre deflection            [m]     → road-holding / tyre load variation (minimise)

The outputs are used in the CMM3 design project to visualise the trade-offs
between comfort and mechanical constraints for the **passive** suspension
baseline, and to compare against the semi-active (skyhook) configuration.

All units are SI unless otherwise stated.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.params import SuspensionParams
from src.passive_eval import evaluate_passive
from src.params import SuspensionParams

params = SuspensionParams()

def passive_suspension_pareto():
    
    def pareto_front(costs: np.ndarray) -> np.ndarray:
        """
        Compute the Pareto front for a set of designs in a **minimisation** problem.

        A design A is said to **dominate** design B if:
            - A is no worse than B in all objectives, and
            - A is strictly better than B in at least one objective.

        This function returns a boolean mask indicating which designs are
        **non-dominated** (i.e. Pareto-optimal) within the input set.

        Parameters
        ----------
        costs : np.ndarray
            2D array of shape (N, M), where:
                - N is the number of designs (e.g. different damping scales),
                - M is the number of objectives (here M = 3).
            Each row `costs[i, :]` contains the objective values for design i,
            with all objectives assumed to be **minimised**.

        Returns
        -------
        is_eff : np.ndarray
            1D boolean array of length N. `is_eff[i]` is True if design i is
            Pareto-efficient (non-dominated) and False otherwise.

        Notes
        -----
        - This is an O(N²) algorithm, which is perfectly adequate for the
        relatively small design sets used in this project (typically N ≤ 50).
        - The function is generic and can be reused for both passive and
        semi-active (skyhook) Pareto analyses.
        """
        N = costs.shape[0]
        is_eff = np.ones(N, dtype=bool)

        for i in range(N):
            if not is_eff[i]:
                # Already known to be dominated; skip further checks.
                continue

            # A design j dominates design i if:
            #   costs[j] <= costs[i] in all objectives AND
            #   costs[j]  < costs[i] in at least one objective.
            dominates_i = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)

            if np.any(dominates_i & (np.arange(N) != i)):
                is_eff[i] = False

        return is_eff


    def sweep_passive(
        params: SuspensionParams,
        g_min: float = 0.3,
        g_max: float = 2.0,
        n: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform a parametric sweep of the **passive damping scale** and evaluate
        the corresponding performance metrics.

        For each value of the global scale factor ``g`` in [g_min, g_max], this
        function:
            1. Scales all four passive damper coefficients by ``g``,
            2. Runs a nonlinear passive simulation via :func:`evaluate_passive`,
            3. Stores the resulting performance metrics.

        Parameters
        ----------
        params : SuspensionParams
            Baseline suspension parameter set (masses, stiffnesses, damping,
            simulation settings). This object is treated as read-only; scaled
            copies are created internally to avoid side effects.
        g_min : float, optional
            Lower bound of the damping scale range (dimensionless). A value
            less than 1.0 corresponds to a globally softer passive damper.
            Default is 0.3.
        g_max : float, optional
            Upper bound of the damping scale range (dimensionless). A value
            greater than 1.0 corresponds to a globally stiffer passive damper.
            Default is 2.0.
        n : int, optional
            Number of sampling points between g_min and g_max (inclusive).
            Default is 20.

        Returns
        -------
        g_values : np.ndarray
            1D array of length N = n containing the sampled damping scale
            factors used in the sweep.
        metrics : np.ndarray
            2D array of shape (N, 3) where each row corresponds to:
                [RMS acceleration, max travel, max tyre deflection]
            in SI units, as returned by :func:`evaluate_passive`.

        Notes
        -----
        - This function is typically followed by :func:`pareto_front` to identify
        the subset of non-dominated passive designs.
        - Printing inside the loop provides a basic progress indicator when
        running longer sweeps.
        """
        g_values = np.linspace(g_min, g_max, n)
        metrics_list: list[tuple[float, float, float]] = []

        for g in g_values:
            print(f"Running passive sweep at damping scale g = {g:.3f}")
            metrics_list.append(evaluate_passive(params, g))

        metrics = np.array(metrics_list)
        return g_values, metrics


    def plot_results(
        g_values: np.ndarray,
        metrics: np.ndarray,
        is_eff: np.ndarray,
    ) -> None:
        """
        Generate summary plots for the passive damping Pareto sweep.

        Two visualisations are produced:

        1. **Objective space:** RMS acceleration vs. maximum suspension travel,
        coloured by damping scale g, with Pareto-optimal points highlighted.
        This shows the comfort–travel trade-off.

        2. **Design space:** Each metric plotted as a function of g. This helps
        interpret how sensitive each performance indicator is to changes in
        the global damping level.

        Parameters
        ----------
        g_values : np.ndarray
            1D array of damping scale factors (output from :func:`sweep_passive`).
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
        tyre_def = metrics[:, 2]

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

        plt.xlabel("RMS sprung acceleration [m/s²]")
        plt.ylabel("Max suspension travel [m]")
        cbar = plt.colorbar(sc)
        cbar.set_label("passive damping scale g")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------------
        # 2. Metrics vs damping scale: design space perspective.
        # ------------------------------------------------------------------
        plt.figure()
        plt.plot(g_values, rms, "-o", label="RMS accel [m/s²]")
        plt.plot(g_values, travel, "-o", label="Max travel [m]")
        plt.plot(g_values, tyre_def, "-o", label="Max tyre defl [m]")
        plt.xlabel("passive damping scale g [-]")
        plt.ylabel("metric value [SI units]")
        plt.legend()
        plt.tight_layout()
        plt.show()


    '''if __name__ == "__main__":'''
        # ------------------------------------------------------------------
        # Example usage:
        #   - create a baseline parameter set,
        #   - sweep the passive damping scale,
        #   - compute the Pareto front,
        #   - print key results and generate plots.
        # ------------------------------------------------------------------

    g_values, metrics = sweep_passive(params, g_min=0.05, g_max=2.5, n=25)
    is_eff = pareto_front(metrics)

    print("Pareto-optimal passive damping scales g:")
    print(g_values[is_eff])
    print("Corresponding metrics [RMS acc, max travel, max tyre def]:")
    print(metrics[is_eff])

    plot_results(g_values, metrics, is_eff)

if __name__ == "__main__":
    passive_suspension_pareto()