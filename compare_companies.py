"""Fetch and plot valuation metrics for selected semiconductor companies."""

from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import yfinance as yf


CompanyMetrics = Tuple[str, float, float]


def fetch_metrics(ticker_map: Dict[str, str]) -> List[CompanyMetrics]:
    """Return (name, EV/EBITDA, EBITDA margin %) tuples for the requested tickers."""
    metrics: List[CompanyMetrics] = []
    for name, symbol in ticker_map.items():
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:  # pragma: no cover - network/service exceptions
            print(f"Skipping {name} ({symbol}): unable to download data ({exc}).", file=sys.stderr)
            continue

        ev_to_ebitda = info.get("enterpriseToEbitda")
        ebitda_margin = info.get("ebitdaMargins")

        if ev_to_ebitda is None or ebitda_margin is None:
            print(f"Skipping {name} ({symbol}): missing EV/EBITDA or EBITDA margin.", file=sys.stderr)
            continue

        metrics.append((name, float(ev_to_ebitda), float(ebitda_margin) * 100.0))

    return metrics


def plot_metrics(metrics: Iterable[CompanyMetrics], *, output_path: str | None = None) -> None:
    """Plot EV/EBITDA against EBITDA margin as a scatter chart."""
    names, ev_to_ebitda, ebitda_margin = zip(*metrics)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ev_to_ebitda, ebitda_margin, s=80, color="#1f77b4")

    for name, x, y in zip(names, ev_to_ebitda, ebitda_margin):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))

    ax.set_xlabel("EV/EBITDA")
    ax.set_ylabel("EBITDA Margin (%)")
    ax.set_title("EV/EBITDA vs EBITDA Margin")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved chart to {output_path}")
    else:
        plt.show()


def main() -> int:
    companies = {
        "Infineon": "IFX.DE",
        "STMicroelectronics": "STM",
        "ASML": "ASML",
        "ON Semiconductor": "ON",
    }

    metrics = fetch_metrics(companies)
    if not metrics:
        print("Unable to retrieve metrics for the requested tickers.", file=sys.stderr)
        return 1

    plot_metrics(metrics, output_path="outputs/ev_ebitda_scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
