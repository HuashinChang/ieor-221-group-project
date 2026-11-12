from argparse import ArgumentParser
import torch
from pathlib import Path
import csv


def _get_p1_parser():
    parser = ArgumentParser(
        description="Calculate price and Greeks at doubling tree step counts; save CSV.",
        add_help=False
    )
    parser.add_argument(
        "--start-n-steps", type=int, default=2**8,
        help="starting step count (default: 256); multiply by 2 up to max number of step",
    )
    parser.add_argument("--rate-bump-size", type=float, default=1e-4)
    parser.add_argument("--vol-bump-size", type=float, default=1e-4)
    parser.add_argument("--method", choices=["CRR", "BBSR", 'CRRS'], default="CRR")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("plots")
    )
    parser.add_argument(
        "--max-n-steps", type=int, default=32768,
        help="maximun number of step count(default: 32768)"
        )
    return parser


def _export_p1_result(rows, method, output_dir):
    columns = ("N steps", "NPV", "Delta", "Gamma", "Theta", "Vega", "Rho")
    formatted_rows = [
        [str(n_steps), *(f"{value:.5f}" for value in values)]
        for n_steps, *values in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{method.lower()}_convergence.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(formatted_rows)

    print(" ".join(f"{column:>12}" for column in columns))
    for row in formatted_rows:
        print(" ".join(f"{value:>12}" for value in row))


def _p1(parser: ArgumentParser):
    from tree_american import select_pricer, cent_fdm, bsm_call_price

    args = parser.parse_args()
    
    max_n_steps = args.max_n_steps
    if not 4 <= args.start_n_steps <= max_n_steps:
        parser.error(f"--start-n-steps must be between 4 and {max_n_steps}")

    step_counts = []
    n_steps = args.start_n_steps
    while n_steps < max_n_steps:
        step_counts.append(n_steps)
        n_steps *= 2
    step_counts.append(max_n_steps)

    inputs = {
        "stock_spot": 100.0,
        "vol": 0.35,
        "ttm": 5,
        "rate": 0.05,
        "strike": 100.0,
        "dividend": 0.08,
    }

    pricer = select_pricer(args.method)
    rows = []
    for n_steps in step_counts:
        print(f"Calculating {args.method} at N={n_steps:,}...", flush=True)
        vega = cent_fdm(
            **inputs, n_steps=n_steps, bump_size=args.vol_bump_size,
            bump_param="vol", tree_method=args.method,
        )
        rho = cent_fdm(
            **inputs, n_steps=n_steps, bump_size=args.rate_bump_size,
            bump_param="rate", tree_method=args.method,
        )
        if args.method == "CRRS":
            inputs["smoothing"] = "CRRS"
        price, delta, gamma, theta = pricer(**inputs, n_steps=n_steps)
        if args.method == "CRRS":
            inputs.pop("smoothing")
        rows.append((n_steps, price, delta, gamma, theta, vega, rho))
        
    print("")
    _export_p1_result(rows, args.method, args.output_dir)
    vanilla_price = bsm_call_price(**inputs)
    print(f"\nCorresponding European call price: {vanilla_price:.5f}")
    

def _get_p2_parser():
    parser = ArgumentParser(
        add_help=False,
        description="Run MC simulation in every implemented method."
    )
    parser.add_argument("--n-paths", type=int, default=2**20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
            "--output-dir", type=Path, default=Path(__file__).with_name("plots")
        )
    return parser


def _export_p2_results(rows, n_paths, output_dir):
    
    columns = ("Method", "NPV", "MC std")
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

    print(f"\n{'Method':<32} {'NPV':>12} {'MC std':>12}")
    for method, npv, std in rows:
        print(f"{method:<32} {npv:>12.4f} {std:>12.4f}")


def _p2(parser: ArgumentParser):
    from mc_asian import MCPricer, bsm_european_call

    args = parser.parse_args()
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
    print("Adjust with --n-paths, --seed, --device ('cpu'/'cuda'); see --help.\n")

    pricer = MCPricer(**mkt_cond, **mc_spec)
    rows = []

    print("Calculating Plain MC...", flush=True)
    npv, std = pricer.run_mc(**contract)
    rows.append(("Plain MC", npv, std))

    print("Calculating Control variate (stock_prices)...", flush=True)
    npv, std = pricer.run_mc(**contract, ctrl_var='stock_prices')
    rows.append(("Control variate (stock_prices)", npv, std))

    print("Calculating Control variate (geom_asian)...", flush=True)
    npv, std = pricer.run_mc(**contract, ctrl_var='geom_asian')
    rows.append(("Control variate (geom_asian)", npv, std))

    pricer.n_samples = int(mc_spec['n_samples'] / 2)
    print("Calculating Antithetic...", flush=True)
    npv, std = pricer.run_mc(**contract, antithetic=True)
    rows.append(("Antithetic", npv, std))

    print("Calculating Antithetic + CV (stock_prices)...", flush=True)
    npv, std = pricer.run_mc(**contract, antithetic=True, ctrl_var='stock_prices')
    rows.append(("Antithetic + CV (stock_prices)", npv, std))

    print("Calculating Antithetic + CV (geom_asian)...", flush=True)
    npv, std = pricer.run_mc(**contract, antithetic=True, ctrl_var='geom_asian')
    rows.append(("Antithetic + CV (geom_asian)", npv, std))

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


def main():
    parser = ArgumentParser()
    subparser = parser.add_subparsers(
        required=False,
        dest='command',
        metavar="{p1, p2}"
    )

    p1_parser = subparser.add_parser("p1", parents=[_get_p1_parser()])
    p1_parser.set_defaults(func=_p1)

    p2_parser = subparser.add_parser("p2", parents=[_get_p2_parser()])
    p2_parser.set_defaults(func=_p2)

    args = parser.parse_args()

    if args.command is None:
        print("Problem 1:")
        _p1(_get_p1_parser())
        print("\nProblem 2:")
        _p2(_get_p2_parser())
    else:
        args.func(parser)

if __name__ == '__main__':
    main()