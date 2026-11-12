"""Compare American-call price convergence using the Problem 1 parameters."""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from CRR_american import BBSR_american_call, CRR_american_call


OUTPUT_DIR = Path(__file__).with_name("plots")
# Adjacent even/odd counts expose the oscillation of the unsmoothed tree.
STEP_COUNTS = tuple(
    steps + offset
    for steps in (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
    for offset in (0, 1)
)
OPTION_INPUTS = {
    "stock_spot": 100.0,
    "strike": 100.0,
    "vol": 0.35,
    "ttm": 5.0,
    "rate": 0.05,
    "dividend": 0.08,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, nargs="+", default=STEP_COUNTS)
    parser.add_argument("--reference-steps", type=int, default=40_000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    step_counts = sorted(set(args.n_steps))
    if min(step_counts) < 4:
        parser.error("--n-steps must be at least 4 for the smoothing implementation")
    if args.reference_steps <= max(step_counts):
        parser.error("--reference-steps must exceed every plotted step count")

    reference_price, *_ = BBSR_american_call(
        **OPTION_INPUTS, n_steps=args.reference_steps
    )
    prices_by_method = {"CRR": [], "CRR + smoothing": [], "BBSR": []}
    for n_steps in step_counts:
        price, *_ = CRR_american_call(**OPTION_INPUTS, n_steps=n_steps)
        prices_by_method["CRR"].append(price)
        price, *_ = CRR_american_call(
            **OPTION_INPUTS, n_steps=n_steps, smoothing=True
        )
        prices_by_method["CRR + smoothing"].append(price)
        price, *_ = BBSR_american_call(**OPTION_INPUTS, n_steps=n_steps)
        prices_by_method["BBSR"].append(price)

    figure = Figure(figsize=(13, 5.5), layout="constrained")
    price_ax, error_ax = figure.subplots(1, 2)
    for method, prices in prices_by_method.items():
        price_ax.semilogx(step_counts, prices, marker=".", label=method)
        errors = np.abs(np.asarray(prices) - reference_price)
        # Exact matches have no finite position on a logarithmic error axis.
        errors[errors == 0] = np.nan
        error_ax.loglog(step_counts, errors, marker=".", label=method)

    price_ax.axhline(
        reference_price, color="black", linestyle="--", label="BBSR reference"
    )
    price_ax.set_title("Price convergence")
    price_ax.set_ylabel("American call price")
    error_ax.set_title("Absolute error against reference")
    error_ax.set_ylabel("Absolute price difference")
    for ax in (price_ax, error_ax):
        ax.set_xlabel("Base step count N (BBSR uses N and 2N trees)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    figure.suptitle(
        "American call: CRR / CRR + smoothing / BBSR\n"
        "S = K = 100, volatility = 35%, T = 5 years, r = 5%, q = 8%"
    )
    figure.supxlabel(
        f"Reference: BBSR at N = {args.reference_steps:,}, "
        f"price = {reference_price:.8f}.",
        fontsize=9,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "tree_convergence.png"
    figure.savefig(path, dpi=150)
    print(f"BBSR reference (N={args.reference_steps:,}): {reference_price:.10f}")
    print(path)


if __name__ == "__main__":
    main()
