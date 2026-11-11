from pathlib import Path

import torch
import math
from matplotlib.figure import Figure

from mc_asian import MCPricer


TOTAL_PATH_COUNTS = (2**10, 2**12, 2**14, 2**16, 2**18, 2**20)
STRIKES = {"ITM": 90, "ATM": 100, "OTM": 110}
VARIANCE_REDUCTION = (
    ("Plain", False, None),
    ("Antithetic", True, None),
    ("Stock control", False, "stock_prices"),
    ("Geometric Asian control", False, "geom_asian"),
    ("Antithetic + stock control", True, "stock_prices"),
    ("Antithetic + geometric Asian control", True, "geom_asian"),
)
OUTPUT_DIR = Path(__file__).with_name("mc_std_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def main(output_dir=OUTPUT_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pricer = MCPricer(
        stock_spot=100, dividend=0.08, vol=0.35, rate=0.05, seed=42, device=device
    )
    dtm = 125
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_ratios = {}

    for moneyness, strike in STRIKES.items():
        figure = Figure(figsize=(8, 5.5), layout="constrained")
        ax = figure.subplots()
        stds_by_policy = {}
        for policy, antithetic, ctrl_var in VARIANCE_REDUCTION:
            stds = []
            for total_paths in TOTAL_PATH_COUNTS:
                # Each antithetic sample uses two paths within the same budget.
                pricer.n_samples = total_paths // 2 if antithetic else total_paths
                _, mc_std = pricer.run_mc(
                    strike, dtm, antithetic=antithetic, ctrl_var=ctrl_var
                )
                stds.append(mc_std)
            stds_by_policy[policy] = stds
            ax.loglog(TOTAL_PATH_COUNTS, stds, marker="o", label=policy)

        # Square standard-error ratios to compare estimator variances, then average.
        plain_stds = stds_by_policy["Plain"]
        mean_ratios[moneyness] = {
            policy: math.fsum(
                (plain_std / method_std) ** 2
                for plain_std, method_std in zip(plain_stds, stds)
            ) / len(TOTAL_PATH_COUNTS)
            for policy, stds in stds_by_policy.items()
        }

        ax.set_title(f"{moneyness} Asian call (strike {strike})")
        ax.set_xlabel("Total simulated paths")
        x_label = [f"2**{str(int(math.log2(N)))}" for N in TOTAL_PATH_COUNTS]
        ax.set_xticks(ticks=TOTAL_PATH_COUNTS, labels=x_label)
        ax.set_ylabel("MC price standard error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        path = output_dir / f"mc_std_{moneyness.lower()}.png"
        figure.savefig(path, dpi=150)
        print(path)

    plot_variance_reduction(mean_ratios, output_dir)


if __name__ == "__main__":
    main()
