from argparse import ArgumentParser
from pathlib import Path
import csv


def get_p1_parser():
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


def p1(args, parser: ArgumentParser):
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
