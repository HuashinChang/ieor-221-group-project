import argparse
from pathlib import Path

import torch
import math
from matplotlib.figure import Figure

from mc_asian import MCPricer


STRIKES = {"ITM": 90, "ATM": 100, "OTM": 110}
VARIANCE_REDUCTION = (
    ("Plain", False, None),
    ("Antithetic", True, None),
    ("Stock control", False, "stock_prices"),
    ("Geometric Asian control", False, "geom_asian"),
    ("Antithetic + stock control", True, "stock_prices"),
    ("Antithetic + geometric Asian control", True, "geom_asian"),
)
OUTPUT_DIR = Path(__file__).with_name("plots")

def plot_variance_reduction(mean_ratios, output_dir):
    figure = Figure(figsize=(12, 6.5), layout="constrained")
    ax = figure.subplots()
    policies = [policy for policy, _, _ in VARIANCE_REDUCTION]
    bar_height = 0.24

    for index, (moneyness, ratios) in enumerate(mean_ratios.items()):
        positions = [i + (index - 1) * bar_height for i in range(len(policies))]
        bars = ax.barh(
            positions,
            [ratios[policy] for policy in policies],
            height=bar_height,
            label=f"{moneyness} (strike {STRIKES[moneyness]})",
        )
        ax.bar_label(bars, fmt="%.2fx", padding=3)

    ax.set_yticks(range(len(policies)), labels=policies)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=max(max(ratios.values()) for ratios in mean_ratios.values()) * 2)
    ax.axvline(1, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Asian call: average variance reduction at equal path budgets")
    ax.set_xlabel(
        "Mean of (Plain SE / method SE)^2 across total path counts\n"
        "1x = Plain MC; larger ratios mean greater variance reduction"
    )
    ax.grid(True, axis="x", which="both", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()
    path = output_dir / "mc_variance_reduction.png"
    figure.savefig(path, dpi=150)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-exp", type=int, default=10, help="first exponent k")
    parser.add_argument("--max-exp", type=int, default=15, help="last exponent k")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.start_exp < 1:
        parser.error("--start-exp must be at least 1")
    if args.max_exp < args.start_exp:
        parser.error("--max-exp must be at least --start-exp")

    total_path_counts = [2**n for n in range(args.start_exp, args.max_exp+1)]

    device = torch.device(args.device)
    pricer = MCPricer(
        stock_spot=100, dividend=0.08, vol=0.35, rate=0.05, seed=42, device=device
    )
    dtm = 125
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_ratios = {}

    for moneyness, strike in STRIKES.items():
        figure = Figure(figsize=(8, 5.5), layout="constrained")
        ax = figure.subplots()
        ses_by_policy = {}
        for policy, antithetic, ctrl_var in VARIANCE_REDUCTION:
            ses = []
            for total_paths in total_path_counts:
                # Each antithetic sample uses two paths within the same budget.
                pricer.n_samples = total_paths // 2 if antithetic else total_paths
                _, mc_se = pricer.run_mc(
                    strike, dtm, antithetic=antithetic, ctrl_var=ctrl_var
                )
                ses.append(mc_se)
            ses_by_policy[policy] = ses
            ax.loglog(total_path_counts, ses, marker="o", label=policy)

        # Square standard-error ratios to compare estimator variances, then average.
        plain_ses = ses_by_policy["Plain"]
        mean_ratios[moneyness] = {
            policy: math.fsum(
                (plain_se / method_se) ** 2
                for plain_se, method_se in zip(plain_ses, ses)
            ) / len(total_path_counts)
            for policy, ses in ses_by_policy.items()
        }

        ax.set_title(f"{moneyness} Asian call (strike {strike})")
        ax.set_xlabel("Total simulated paths")
        x_label = [f"2**{str(int(math.log2(N)))}" for N in total_path_counts]
        ax.set_xticks(ticks=total_path_counts, labels=x_label)
        ax.set_ylabel("MC price standard error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        path = args.output_dir / f"mc_se_{moneyness.lower()}.png"
        figure.savefig(path, dpi=150)
        print(path)

    plot_variance_reduction(mean_ratios, args.output_dir)


if __name__ == "__main__":
    main()
