from argparse import ArgumentParser
import torch
from pathlib import Path
import csv


def get_p2_parser():
    parser = ArgumentParser(
        add_help=False,
        description=(
            "Price the arithmetic Asian call with six MC configurations: plain MC, "
            "two control variates, and their three antithetic counterparts. "
            "Print NPV and estimated standard error (se), and save a CSV."
        ),
    )
    parser.add_argument(
        "--n-paths", type=int, default=2**20,
        help=(
            "total simulated paths per method as an integer (default: %(default)s); "
            "antithetic methods use floor(n_paths/2) independent pairs"
        ),
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "PyTorch device, e.g. cpu, cuda, cuda:0; "
            "default: CUDA when available, otherwise CPU (currently: %(default)s)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="random seed for reproducible simulation (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("csv"),
        help=(
            "CSV directory (default: %(default)s); create it if needed; "
            "overwrite mc_with_{n_paths}_paths.csv"
        ),
    )
    return parser


def _export_p2_results(rows, n_paths, output_dir):
    
    columns = ("Method", "NPV", "se")
    formatted_rows = [
        [str(n_steps), *(f"{value:.5f}" for value in values)]
        for n_steps, *values in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"mc_with_{str(n_paths)}_paths.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(formatted_rows)

    print(f"\n{'Method':<32} {'NPV':>12} {'se':>12}")
    for method, npv, se in rows:
        print(f"{method:<32} {npv:>12.4f} {se:>12.4f}")


def p2(args):
    from mc_asian import MCPricer, bsm_european_call

    mkt_cond = {
        'stock_spot': 100,
        'dividend': 0.08,
        'vol': 0.35,
        'rate': 0.05
    }
    contract = {
        'dtm': 125,
        'strike': 100
    }
    mc_spec = {
        'seed': args.seed,
        'device': torch.device(args.device),
        'n_samples': args.n_paths
    }

    print(
        f"Simulation info: total paths={args.n_paths:,}, "
        f"seed={args.seed}, device={mc_spec['device']}."
    )

    pricer = MCPricer(**mkt_cond, **mc_spec)
    rows = []

    print("Calculating Plain MC...", flush=True)
    npv, se = pricer.run_mc(**contract)
    rows.append(("Plain MC", npv, se))

    print("Calculating Control variate (stock_prices)...", flush=True)
    npv, se = pricer.run_mc(**contract, ctrl_var='stock_prices')
    rows.append(("Control variate (stock_prices)", npv, se))

    print("Calculating Control variate (geom_asian)...", flush=True)
    npv, se = pricer.run_mc(**contract, ctrl_var='geom_asian')
    rows.append(("Control variate (geom_asian)", npv, se))

    pricer.n_samples = int(mc_spec['n_samples'] / 2)
    print("Calculating Antithetic...", flush=True)
    npv, se = pricer.run_mc(**contract, antithetic=True)
    rows.append(("Antithetic", npv, se))

    print("Calculating Antithetic + CV (stock_prices)...", flush=True)
    npv, se = pricer.run_mc(**contract, antithetic=True, ctrl_var='stock_prices')
    rows.append(("Antithetic + CV (stock_prices)", npv, se))

    print("Calculating Antithetic + CV (geom_asian)...", flush=True)
    npv, se = pricer.run_mc(**contract, antithetic=True, ctrl_var='geom_asian')
    rows.append(("Antithetic + CV (geom_asian)", npv, se))

    vanilla_price = bsm_european_call(
        **mkt_cond, 
        strike=contract['strike'], 
        ttm=contract['dtm']/pricer.day_count
        )


    _export_p2_results(
        rows=rows, n_paths=args.n_paths, output_dir=args.output_dir 
    )
    print(
        "\nThe vanilla call with the same strike and maturity "
        f"has the price {vanilla_price}."
    )