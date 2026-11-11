from pathlib import Path

import torch
from matplotlib.figure import Figure

from mc_asion import MCPricer


PATH_COUNTS = (2**12, 2**14, 2**16, 2**18, 2**20)
STRIKES = {"ITM": 90, "ATM": 100, "OTM": 110}
POLICIES = ("plain", "antithetic", "ctrl_var")
OUTPUT_DIR = Path(__file__).with_name("mc_std_plots")


def main(output_dir=OUTPUT_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pricer = MCPricer(
        stock_spot=100, dividend=0.08, vol=0.35, rate=0.05, seed=42, device=device
    )
    dtm = 125
    output_dir.mkdir(parents=True, exist_ok=True)

    for moneyness, strike in STRIKES.items():
        figure = Figure(figsize=(7, 4.5), layout="constrained")
        ax = figure.subplots()
        for policy in POLICIES:
            stds = []
            for total_paths in PATH_COUNTS:
                # Compare at equal simulated path counts, including both paths per pair.
                pricer.n_paths = (
                    total_paths // 2 if policy == "antithetic" else total_paths
                )
                _, mc_std = pricer.run_mc(strike, dtm, policy)
                stds.append(mc_std)
            ax.loglog(PATH_COUNTS, stds, marker="o", label=policy)

        ax.set_title(f"{moneyness} Asian call (strike {strike})")
        ax.set_xlabel("Total simulated paths")
        ax.set_ylabel("MC price standard error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        path = output_dir / f"mc_std_{moneyness.lower()}.png"
        figure.savefig(path, dpi=150)
        print(path)


if __name__ == "__main__":
    main()
