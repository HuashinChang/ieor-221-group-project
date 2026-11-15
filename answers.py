from argparse import ArgumentParser, RawDescriptionHelpFormatter
import sys
import torch
from pathlib import Path
import csv


def _get_p1_parser():
    parser = ArgumentParser(
        description=(
            "Calculate American call prices and Greeks; print tables and save CSVs. "
            "By default, run CRR and BBSR at even step counts N=2**k."
        ),
        add_help=False
    )
    parser.add_argument(
        "--start-exp", type=int, default=11,
        help=(
            "first exponent k (default: %(default)s); even N=2**k, odd N=2**k-1; "
            "minimum 2 for CRR, 3 when BBS or BBSR runs"
        ),
    )
    parser.add_argument(
        "--rate-bump-size", type=float, default=1e-4,
        help="absolute rate bump for central-difference Rho (default: %(default)s)",
    )
    parser.add_argument(
        "--vol-bump-size", type=float, default=1e-4,
        help=(
            "absolute volatility bump for central-difference Vega "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--method", choices=["CRR", "BBSR", "BBS"],
        help=(
            "run one method instead of both CRR and BBSR; BBS uses smoothing, "
            "BBSR adds Richardson extrapolation; --odd overrides this with CRR"
        ),
    )
    parser.add_argument(
        "--odd", choices=["include", "only"],
        help=(
            "run CRR only: include outputs both even and odd results, "
            "only outputs odd results; omit for even results only"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("csv"),
        help=(
            "CSV directory (default: %(default)s); create it if needed; "
            "overwrite {method}_convergence_even.csv or _odd.csv for each result"
        ),
    )
    parser.add_argument(
        "--max-exp", type=int, default=15,
        help=(
            "last exponent k, inclusive (default: %(default)s); "
            "increase k by 1, so step counts approximately double"
        ),
    )
    return parser


def _export_p1_result(rows, method, output_dir, *, odd=False):
    columns = ("N steps", "NPV", "Delta", "Gamma", "Theta", "Vega", "Rho")
    formatted_rows = [
        [str(n_steps), *(f"{value:.4f}" for value in values)]
        for n_steps, *values in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_odd" if odd else "_even"
    csv_path = output_dir / f"{method.lower()}_convergence{suffix}.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(formatted_rows)

    print(" ".join(f"{column:>12}" for column in columns))
    for row in formatted_rows:
        print(" ".join(f"{value:>12}" for value in row))


def _p1(args, parser: ArgumentParser):
    from tree_american import select_pricer, cent_fdm, bsm_call_price

    if args.odd:
        methods = ["CRR"]
    elif args.method:
        methods = [args.method]
    else:
        methods = ["CRR", "BBSR"]
    parities = (("Even", 0),)
    if args.odd == "include":
        parities = (("Even", 0), ("Odd", 1))
    elif args.odd == "only":
        parities = (("Odd", 1),)

    min_exp = 2 if methods == ["CRR"] else 3
    if args.start_exp < min_exp:
        parser.error(f"--start-exp must be at least {min_exp} for {', '.join(methods)}")
    if args.max_exp < args.start_exp:
        parser.error("--max-exp must be at least --start-exp")

    inputs = {
        "stock_spot": 100.0,
        "vol": 0.35,
        "ttm": 5,
        "rate": 0.05,
        "strike": 100.0,
        "dividend": 0.08,
    }

    for method in methods:
        pricer = select_pricer(method)
        for parity, offset in parities:
            print(f"\n{method}, {parity.lower()} steps:")
            rows = []
            for exponent in range(args.start_exp, args.max_exp + 1):
                n_steps = 2**exponent - offset
                print(f"Calculating {method} at N={n_steps:,}...", flush=True)
                vega = cent_fdm(
                    **inputs, n_steps=n_steps, bump_size=args.vol_bump_size,
                    bump_param="vol", tree_method=method,
                )
                rho = cent_fdm(
                    **inputs, n_steps=n_steps, bump_size=args.rate_bump_size,
                    bump_param="rate", tree_method=method,
                )
                if method == "BBS":
                    inputs["smoothing"] = True
                price, delta, gamma, theta = pricer(**inputs, n_steps=n_steps)
                if method == "BBS":
                    inputs.pop("smoothing")
                rows.append((n_steps, price, delta, gamma, theta, vega, rho))

            print("")
            _export_p1_result(rows, method, args.output_dir, odd=offset == 1)
    vanilla_price = bsm_call_price(**inputs)
    print(f"\nCorresponding European call price: {vanilla_price:.5f}")
    

def _get_p2_parser():
    parser = ArgumentParser(
        add_help=False,
        description=(
            "Price the arithmetic Asian call with six MC configurations: plain MC, "
            "two control variates, and their three antithetic counterparts. "
            "Print NPV and estimated standard error (MC std), and save a CSV."
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


def _p2(args):
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
    parser = ArgumentParser(
        description=(
            "Reproduce answer tables and save CSVs. With no command, run p1: "
            "CRR and BBSR at even step counts. Run p2 explicitly for Monte Carlo."
        ),
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""Examples:
  python answers.py                         # same as python answers.py p1
  python answers.py p1 --method BBSR         # even results for BBSR only
  python answers.py p1 --odd include         # CRR even and odd results
  python answers.py p1 --odd only            # CRR odd results only
  python answers.py --odd only               # p1 may be omitted
  python answers.py p1 --start-exp 3 --max-exp 5 --output-dir /tmp/p1
  python answers.py p2 --n-paths 4096 --device cpu --seed 42

Use p1 --help or p2 --help for all options and defaults.
Pass integer counts: --n-paths 1048576, not --n-paths 2**20.
Relative output directories are resolved from the current working directory.""",
    )
    subparser = parser.add_subparsers(
        required=False,
        dest='command',
        metavar="{p1, p2}"
    )

    p1_options = _get_p1_parser()
    p1_parser = subparser.add_parser(
        "p1", parents=[p1_options], description=p1_options.description,
        help="American call tree prices and Greeks (default command)",
    )

    p2_options = _get_p2_parser()
    subparser.add_parser(
        "p2", parents=[p2_options], description=p2_options.description,
        help="Asian call Monte Carlo comparison with six configurations",
    )

    argv = sys.argv[1:]
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["p1", *argv]
    args = parser.parse_args(argv)

    if args.command == "p1":
        print("Problem 1:")
        _p1(args, p1_parser)
    else:
        _p2(args)

if __name__ == '__main__':
    main()
