"""Plot Asian-call MC prices and pointwise asymptotic 99% confidence intervals."""

import argparse
import csv
from pathlib import Path

import torch
from matplotlib.figure import Figure
from scipy.stats import norm

from mc_asian import MCPricer


OUTPUT_DIR = Path(__file__).with_name("plots")


def _export_results(rows, method, label, ci, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"mc_asym_{method}_0{str(ci)[2:]}.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow((
            "N paths", "Price", "MC SE", 
            f"{100*ci}% CI lower",
            f"{100*ci}% CI upper"
        ))
        for n_paths, *values in rows:
            writer.writerow((n_paths, *(f"{value:.4f}" for value in values)))

    path_counts, prices, _, lower_bounds, upper_bounds = zip(*rows)
    figure = Figure(figsize=(9, 6), layout="constrained")
    ax = figure.subplots()
    ax.errorbar(
        path_counts, prices, yerr=[norm.ppf(2 * ci - 1) * row[2] for row in rows],
        marker="o", capsize=3, label="MC price",
    )
    ax.fill_between(
        path_counts, lower_bounds, upper_bounds, alpha=0.25,
        label=f"Pointwise asymptotic {100*ci}% CI",
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
    png_path = output_dir / f"mc_asym_{method}_0{str(ci)[2:]}.png"
    figure.savefig(png_path, dpi=150)
    print(csv_path)
    print(png_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-exp", type=int, default=10,
        help="first exponent k for N = 2**k total paths (default: 10)",
    )
    parser.add_argument(
        "--end-exp", type=int, default=20,
        help="last exponent k, inclusive (default: 20)",
    )
    parser.add_argument(
        "--av", type=str.lower, choices=("true", "false"), default="false",
        help="use antithetic variates (default: false)",
    )
    parser.add_argument(
        "--cv", choices=("geom-asian", "price"),
        help="control variate: geometric Asian payoff or terminal stock price; "
        "omit to disable CV",
    )
    parser.add_argument(
        "--ci", type=float, default= 0.99,
        help="confident level (default: 0.99)"
    )
    args = parser.parse_args(argv)
    antithetic = args.av == "true"
    # CV uses 10% of independent samples for fitting its coefficient.
    min_exp = (5 if args.cv else 1) + int(antithetic)
    if args.start_exp < min_exp:
        parser.error(
            f"--start-exp must be at least {min_exp} for this method "
            "to leave at least two independent samples in each used sample set"
        )
    if args.end_exp < args.start_exp:
        parser.error("--end-exp must be at least --start-exp")
    if args.ci >= 1 or args.ci < 0:
        parser.error("--ci must be in [0, 1)") 

    ctrl_var = {"geom-asian": "geom_asian", "price": "stock_prices"}.get(args.cv)
    method = "av" if antithetic else "plain"
    label = "Antithetic MC" if antithetic else "Plain MC"
    if args.cv:
        cv_suffix = f"cv_{args.cv.replace('-', '_')}"
        method = f"av_{cv_suffix}" if antithetic else cv_suffix
        control_label = "geometric Asian payoff" if args.cv == "geom-asian" else (
            "terminal stock price"
        )
        label += f" + CV ({control_label})"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pricer = MCPricer(
        stock_spot=100, dividend=0.08, vol=0.35, rate=0.05,
        seed=0, device=device,
    )
    print(f"{label}; seed=0; device={device}", flush=True)
    rows = []
    for exponent in range(args.start_exp, args.end_exp + 1):
        n_paths = 2**exponent
        # A pair is one independent observation, with two simulated paths.
        pricer.n_samples = n_paths // 2 if antithetic else n_paths
        price, standard_error = pricer.run_mc(
            strike=100, dtm=125, antithetic=antithetic, ctrl_var=ctrl_var,
        )
        CI_MULTIPLIER = norm.ppf(2 * args.ci - 1)
        half_width = CI_MULTIPLIER * standard_error
        lower, upper = price - half_width, price + half_width
        rows.append((n_paths, price, standard_error, lower, upper))
        print(
            f"N={n_paths:,}: price={price:.4f}, SE={standard_error:.4f}, "
            f"{100 * args.ci}% CI=[{lower:.4f}, {upper:.4f}]",
            flush=True,
        )

    _export_results(rows, method, label, args.ci, OUTPUT_DIR)


if __name__ == "__main__":
    main()
