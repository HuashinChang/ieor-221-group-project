from pathlib import Path

import torch
from matplotlib.figure import Figure

from mc_asion import MCPricer


SAMPLE_COUNTS = (2**12, 2**14, 2**16, 2**18, 2**20)
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


def main(output_dir=OUTPUT_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pricer = MCPricer(
        stock_spot=100, dividend=0.08, vol=0.35, rate=0.05, seed=42, device=device
    )
    dtm = 125
    output_dir.mkdir(parents=True, exist_ok=True)

    for moneyness, strike in STRIKES.items():
        figure = Figure(figsize=(8, 5.5), layout="constrained")
        ax = figure.subplots()
        for policy, antithetic, ctrl_var in VARIANCE_REDUCTION:
            stds = []
            for n_samples in SAMPLE_COUNTS:
                # Each antithetic pair is one independent sample (two paths).
                pricer.n_samples = n_samples
                _, mc_std = pricer.run_mc(
                    strike, dtm, antithetic=antithetic, ctrl_var=ctrl_var
                )
                stds.append(mc_std)
            ax.loglog(SAMPLE_COUNTS, stds, marker="o", label=policy)

        ax.set_title(f"{moneyness} Asian call (strike {strike})")
        ax.set_xlabel("Independent samples (N; antithetic uses 2N paths)")
        ax.set_ylabel("MC price standard error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        path = output_dir / f"mc_std_{moneyness.lower()}.png"
        figure.savefig(path, dpi=150)
        print(path)


if __name__ == "__main__":
    main()
