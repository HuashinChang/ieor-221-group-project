"""Plot pointwise chi-square CIs for r_est, as in tests/test_mc_std.py.

r_est is the empirical standard deviation of repeated MC prices divided by
the root mean square of their reported standard errors. Each method gets a
separate PNG in plots/ and CSV in csv/. Path budgets include both antithetic paths
and any control-variate pilot paths.
"""

import argparse
import csv
import math
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import chi2

from mc_asian import MCPricer


OUTPUT_DIR = Path(__file__).with_name("plots")
CSV_DIR = Path(__file__).with_name("csv")


def _control_variate(value):
    value = value.lower().replace("-", "_")
    return {"price": "stock_prices", "stock_price": "stock_prices"}.get(value, value)


def _estimate_ratio(pricer, seeds, antithetic, ctrl_var, ci):
    prices = []
    variances = []
    for seed in tqdm(seeds, unit="rep"):
        pricer.seed = seed
        price, standard_error = pricer.run_mc(
            strike=100, dtm=125, antithetic=antithetic, ctrl_var=ctrl_var,
            pilot_usage=0.1,
        )
        prices.append(price)
        variances.append(standard_error**2)

    empirical_std = float(np.std(prices, ddof=1))
    native_std = math.sqrt(float(np.mean(variances)))
    if not math.isfinite(empirical_std) or not math.isfinite(native_std):
        raise ValueError("MC prices or standard errors produced nonfinite statistics")
    if native_std <= 0:
        raise ValueError("The RMS native standard error must be positive")

    r_est = empirical_std / native_std
    df = len(seeds) - 1
    alpha = (1 - ci) / 2
    lower = r_est * math.sqrt(df / chi2.ppf(1 - alpha, df=df))
    upper = r_est * math.sqrt(df / chi2.ppf(alpha, df=df))
    return empirical_std, native_std, r_est, lower, upper


def _export_results(rows, method, label, args, ci):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv_dir / f"mc_std_ci_0{str(100*ci)[2:]}_{method}.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    path_counts = [row["n_paths"] for row in rows]
    estimates = [row["r_est"] for row in rows]
    lower_bounds = [row["ci_lower"] for row in rows]
    upper_bounds = [row["ci_upper"] for row in rows]
    figure = Figure(figsize=(9, 6), layout="constrained")
    ax = figure.subplots()
    ax.plot(path_counts, estimates, marker="o", label="r_est")
    ax.vlines(
        path_counts, lower_bounds, upper_bounds, color="tab:blue",
        label=f"Pointwise {100 * args.ci:g}% CI",
    )
    ax.fill_between(path_counts, lower_bounds, upper_bounds, alpha=0.2)
    ax.axhline(1, color="black", linestyle=":", label="r_est = 1")
    threshold = 1 + args.width / 2
    ax.axhline(
        threshold, color="tab:red", linestyle="--",
        label=f"Upper-CI threshold = {threshold:g}",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(
        path_counts, [f"$2^{{{n.bit_length() - 1}}}$" for n in path_counts],
    )
    ax.set_xlabel("Total simulated paths per replicate (including CV pilot paths)")
    ax.set_ylabel("r_est = empirical price SD / RMS native SE")
    ax.set_title(
        f"{label}: standard-error calibration ({args.n_rep} replicates)\n"
        "S = K = 100, volatility = 35%, r = 5%, q = 8%, maturity = 125/250 years"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    png_path = args.output_dir / f"mc_std_ci_0{str(100*ci)[2:]}_{method}.png"
    figure.savefig(png_path, dpi=150)
    print(csv_path)
    print(png_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="root seed (default: 0)")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--width", type=float, default=0.2,
        help="require CI upper bound < 1 + width / 2 (default: 0.2)",
    )
    parser.add_argument(
        "--ci", "--ci-level", dest="ci", type=float, default=0.9,
        help="confidence level, between 0 and 1 (default: 0.9)",
    )
    parser.add_argument(
        "--start-exp", type=int, default=15,
        help="first exponent k for 2**k total paths (default: 15)",
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
        "--n-rep", type=int, default=500,
        help="independent MC replicates per path count (default: 500)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    args = parser.parse_args(argv)

    if not 0 < args.ci < 1:
        parser.error("--ci must be between 0 and 1, exclusive")
    if not math.isfinite(args.width) or args.width <= 0:
        parser.error("--width must be finite and positive")
    if args.n_rep < 2:
        parser.error("--n-rep must be at least 2")
    if args.end_exp < args.start_exp:
        parser.error("--end-exp must be at least --start-exp")

    av_choices = (False, True) if args.av is None else (args.av == "true",)
    cv_choices = (None, "stock_prices", "geom_asian") if args.cv is None else (
        None if args.cv == "none" else args.cv,
    )
    # CV needs at least two pilot observations; AV uses two paths per observation.
    min_exp = (5 if any(cv_choices) else 1) + int(any(av_choices))
    if args.start_exp < min_exp:
        parser.error(
            f"--start-exp must be at least {min_exp} for the selected methods"
        )

    device = torch.device(args.device)
    root_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    seeds = torch.randint(
        0, 2**63 - 1, (args.n_rep,), generator=root_gen, device="cpu",
    ).tolist()
    print(
        f"seed={args.seed}; device={device}; replicates={args.n_rep}; "
        f"CI={args.ci:g}; width={args.width:g}", flush=True,
    )
    for antithetic in av_choices:
        for ctrl_var in cv_choices:
            method = f"av_{str(antithetic).lower()}_cv_{ctrl_var or 'none'}"
            label = f"AV={antithetic}, CV={ctrl_var or 'none'}"
            pricer = MCPricer(
                stock_spot=100, dividend=0.08, vol=0.35, rate=0.05,
                device=device, day_count=250, verbose=False,
            )
            rows = []
            for exponent in range(args.start_exp, args.end_exp + 1):
                n_paths = 2**exponent
                pricer.n_samples = n_paths // 2 if antithetic else n_paths
                empirical_std, native_std, r_est, lower, upper = _estimate_ratio(
                    pricer, seeds, antithetic, ctrl_var, args.ci,
                )
                threshold = 1 + args.width / 2
                rows.append({
                    "n_paths": n_paths,
                    "n_samples": pricer.n_samples,
                    "n_rep": args.n_rep,
                    "seed": args.seed,
                    "device": str(device),
                    "av": antithetic,
                    "cv": ctrl_var or "none",
                    "ci_level": args.ci,
                    "width": args.width,
                    "s_emp": empirical_std,
                    "s_native": native_std,
                    "r_est": r_est,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "upper_threshold": threshold,
                    "upper_below_threshold": bool(upper < threshold),
                })
                print(
                    f"{label}; N={n_paths:,}: r_est={r_est:.4f}, "
                    f"{args.ci:.1%} CI=[{lower:.4f}, {upper:.4f}], "
                    f"upper < {threshold:g}: {upper < threshold}", flush=True,
                )
            _export_results(rows, method, label, args, args.ci)


if __name__ == "__main__":
    main()
