"""Check CRR price and Greek convergence for odd/even American-call trees."""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from tree_american import crr_american_call, cent_fdm


OUTPUT_DIR = Path(__file__).with_name("plots")
OPTION_INPUTS = {
    "stock_spot": 100.0,
    "strike": 100.0,
    "vol": 0.35,
    "ttm": 5.0,
    "rate": 0.05,
    "dividend": 0.08,
}
OPTION_DESCRIPTION = (
    "American call: S = K = 100, volatility = 35%, T = 5 years, r = 5%, q = 8%"
)


def _convergence_diagnostics(step_counts, estimates, *, offset=0):
    """Use base counts 2**k; offset=1 keeps all estimates on odd trees."""
    counts = step_counts - offset
    next_counts = 2 * step_counts - offset
    following_counts = 4 * step_counts - offset
    differences = np.array(
        [estimates[next_n] - estimates[n] for n, next_n in zip(counts, next_counts)]
    )
    next_differences = np.array(
        [
            estimates[following_n] - estimates[next_n]
            for next_n, following_n in zip(next_counts, following_counts)
        ]
    )
    ratios = np.full_like(differences, np.nan)
    # A zero denominator leaves a gap rather than an infinite plotted ratio.
    np.divide(differences, next_differences, out=ratios, where=next_differences != 0)
    return differences, ratios, counts * differences


def _plot_convergence(step_counts, estimates, name, output_dir):
    figure = Figure(figsize=(15, 5), layout="constrained")
    axes = figure.subplots(1, 3)
    titles = (
        rf"{name} increment: $D_k = X_{{N_{{k+1}}}} - X_{{N_k}}$",
        r"Increment ratio: $D_k / D_{k+1}$",
        r"Scaled increment: $N_k D_k$",
    )
    labels = (r"$D_k$ (signed, symlog scale)", r"$D_k / D_{k+1}$", r"$N_k D_k$")
    for offset, label, marker in (
        (0, r"Even: $N_k = 2^k$", "o"),
        (1, r"Odd: $N_k = 2^k - 1$", "s"),
    ):
        counts = step_counts - offset
        diagnostics = _convergence_diagnostics(step_counts, estimates, offset=offset)
        for ax, values in zip(axes, diagnostics):
            ax.plot(counts, values, marker=marker, label=label)

    for ax, title, label in zip(axes, titles, labels):
        ax.set_xscale("log", base=2)
        ax.set_xticks(step_counts, [str(n) for n in step_counts], rotation=45)
        ax.set_xlabel(r"Actual tree step count $N_k$")
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_yscale("symlog", linthresh=1e-6)
    axes[0].axhline(0, color="black", linestyle="--", label="Limit: 0")
    axes[0].relim()
    axes[0].autoscale_view()
    axes[1].axhline(2, color="black", linestyle="--", label="Constant-C limit: 2")
    for ax in axes:
        ax.legend()
    figure.suptitle(
        f"CRR {name} asymptotic\n{OPTION_DESCRIPTION}"
    )
    figure.supxlabel(
        r"Each curve compares $N_k$, $N_{k+1}$, and $N_{k+2}$ within its parity."
        "\n"
        r"If $X_{N_k} = X + C/N_k + o(N_k^{-1})$ with constant $C \ne 0$: "
        r"$D_k/D_{k+1} \to 2$, $N_kD_k \to -C/2$. "
        r"The coefficient $C$ may differ between even and odd sequences.",
        fontsize=10,
    )
    suffix = f"_{name.lower()}"
    path = output_dir / f"crr_asymptotic{suffix}.png"
    figure.savefig(path, dpi=150)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-exp", type=int, default=10, help="first exponent k")
    parser.add_argument("--max-exp", type=int, default=15, help="last exponent k")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.start_exp < 2:
        parser.error("--start-exp must be at least 2 so every tree has 3+ steps")
    if args.max_exp < args.start_exp:
        parser.error("--max-exp must be at least --start-exp")

    step_counts = np.array([2**k for k in range(args.start_exp, args.max_exp + 1)])
    # Two more exponents supply both increments within each parity sequence.
    required_counts = sorted(
        {
            int(factor * n - offset)
            for n in step_counts
            for factor in (1, 2, 4)
            for offset in (0, 1)
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for n in required_counts:
        print(f"Calculating CRR at N={n:,}...", flush=True)
        results[n] = list(crr_american_call(**OPTION_INPUTS, n_steps=n))
        results[n].append(cent_fdm(**OPTION_INPUTS, n_steps=n, bump_param='vol'))
        results[n].append(cent_fdm(**OPTION_INPUTS, n_steps=n, bump_param='rate'))
    for index, name in enumerate(("Price", "Delta", "Gamma", "Theta", "Vega", "Rho")):
        estimates = {n: result[index] for n, result in results.items()}
        _plot_convergence(step_counts, estimates, name, args.output_dir)


if __name__ == "__main__":
    main()
