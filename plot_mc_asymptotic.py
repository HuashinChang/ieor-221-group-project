"""Plot Asian-call MC prices and pointwise asymptotic confidence intervals.

Omit --av and --cv to export all six variance-reduction combinations.
Path budgets include both antithetic paths and control-variate pilot paths.
"""

import argparse
import csv
from pathlib import Path

import torch
from matplotlib.figure import Figure
from scipy.stats import norm

from mc_asian import MCPricer


OUTPUT_DIR = Path(__file__).with_name("plots")
CSV_DIR = Path(__file__).with_name("csv")


def _control_variate(value):
    value = value.lower().replace("-", "_")
    return {"price": "stock_prices", "stock_price": "stock_prices"}.get(value, value)


def _export_results(rows, method, label, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    filename = f"mc_asym_{method}_{str(args.ci).replace('.', '')}"
    csv_path = args.csv_dir / f"{filename}.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow((
            "N paths", "Price", "MC SE",
            f"{100 * args.ci:g}% CI lower",
            f"{100 * args.ci:g}% CI upper",
        ))
        for n_paths, *values in rows:
            writer.writerow((n_paths, *(f"{value:.4f}" for value in values)))

    path_counts, prices, _, lower_bounds, upper_bounds = zip(*rows)
    figure = Figure(figsize=(9, 6), layout="constrained")
    ax = figure.subplots()
    ax.errorbar(
        path_counts, prices, yerr=[row[4] - row[1] for row in rows],
        marker="o", capsize=3, label="MC price",
    )
    ax.fill_between(
        path_counts, lower_bounds, upper_bounds, alpha=0.25,
        label=f"Pointwise asymptotic {100 * args.ci:g}% CI",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(path_counts, [f"$2^{{{n.bit_length() - 1}}}$" for n in path_counts])
    ax.set_xlabel("Total simulated paths (including CV pilot paths)")
    ax.set_ylabel("Asian call price")
    ax.set_title(
        f"{label}: MC price convergence\n"
        "S = K = 100, volatility = 35%, r = 5%, q = 8%, maturity = 125/250 years"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    png_path = args.output_dir / f"{filename}.png"
    figure.savefig(png_path, dpi=150)
    print(csv_path)
    print(png_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="MC seed (default: 0)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--start-exp", type=int, default=10,
        help="first exponent k for N = 2**k total paths (default: 10)",
    )
    parser.add_argument(
        "--end-exp", type=int, default=20,
        help="last exponent k, inclusive (default: 20)",
    )
    parser.add_argument(
        "--av", type=str.lower, choices=("true", "false"),
        help="select antithetic variates; omit to run both choices",
    )
    parser.add_argument(
        "--cv", type=_control_variate,
        choices=("none", "stock_prices", "geom_asian"),
        help="select CV; price and geom-asian are aliases; omit to run all three",
    )
    parser.add_argument(
        "--ci", "--ci-level", dest="ci", type=float, default=0.99,
        help="confidence level, between 0 and 1 (default: 0.99)",
    )
    parser.add_argument(
        "--pilot-usage", type=float, default=0.1,
        help="fraction of independent samples used for CV fitting (default: 0.1)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    args = parser.parse_args(argv)

    if not 0 < args.ci < 1:
        parser.error("--ci must be between 0 and 1, exclusive")
    if not 0 < args.pilot_usage < 1:
        parser.error("--pilot-usage must be between 0 and 1, exclusive")
    if args.start_exp < 1:
        parser.error("--start-exp must be at least 1")
    if args.end_exp < args.start_exp:
        parser.error("--end-exp must be at least --start-exp")

    av_choices = (False, True) if args.av is None else (args.av == "true",)
    cv_choices = (None, "stock_prices", "geom_asian") if args.cv is None else (
        None if args.cv == "none" else args.cv,
    )
    # Validate the smallest independent sample set across the selected methods.
    min_samples = 2**args.start_exp // (2 if any(av_choices) else 1)
    pilot_samples = int(min_samples * args.pilot_usage) if any(cv_choices) else 0
    if min_samples - pilot_samples < 2 or (any(cv_choices) and pilot_samples < 2):
        parser.error(
            "increase --start-exp or adjust --pilot-usage to leave at least two "
            "independent observations in each pricing and CV pilot sample set"
        )

    device = torch.device(args.device)
    ci_multiplier = norm.ppf((1 + args.ci) / 2)
    print(
        f"seed={args.seed}; device={device}; CI={args.ci:g}; "
        f"pilot_usage={args.pilot_usage:g}", flush=True,
    )
    for antithetic in av_choices:
        for ctrl_var in cv_choices:
            method = f"av_{str(antithetic).lower()}_cv_{ctrl_var or 'none'}"
            label = f"AV={antithetic}, CV={ctrl_var or 'none'}"
            pricer = MCPricer(
                stock_spot=100, dividend=0.08, vol=0.35, rate=0.05,
                seed=args.seed, device=device, day_count=250, verbose=False,
            )
            rows = []
            for exponent in range(args.start_exp, args.end_exp + 1):
                n_paths = 2**exponent
                # A pair is one independent observation, with two simulated paths.
                pricer.n_samples = n_paths // 2 if antithetic else n_paths
                price, standard_error = pricer.run_mc(
                    strike=100, dtm=125, antithetic=antithetic,
                    ctrl_var=ctrl_var, pilot_usage=args.pilot_usage,
                )
                half_width = ci_multiplier * standard_error
                lower, upper = price - half_width, price + half_width
                rows.append((n_paths, price, standard_error, lower, upper))
                print(
                    f"{label}; N={n_paths:,}: price={price:.4f}, "
                    f"SE={standard_error:.4f}, {100 * args.ci:g}% "
                    f"CI=[{lower:.4f}, {upper:.4f}]", flush=True,
                )
            _export_results(rows, method, label, args)


if __name__ == "__main__":
    main()
