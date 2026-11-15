from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from _p1 import p1, get_p1_parser
from _p2 import p2, get_p2_parser


def main():
    parser = ArgumentParser(
        parents=[get_p1_parser(), get_p2_parser()],
        conflict_handler="resolve",
        description=(
            "Reproduce answer tables and save CSVs. With no command, run p1 then p2.\n"
            "By default, p1 runs CRR and BBSR at even step counts.\n"
            "Tree options (--method, --odd, exponents and bumps) affect p1 only;\n"
            "MC options (--n-paths, --device and --seed) affect p2 only.\n"
            "Without a command, --output-dir applies to both problems."
        ),
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""Examples:
  python answers.py                         # run p1 then p2 with defaults
  python answers.py p1                      # CRR and BBSR, even results only
  python answers.py p1 --method BBSR         # even results for BBSR only
  python answers.py p1 --odd include         # CRR even and odd results
  python answers.py p1 --odd only            # CRR odd results only
  python answers.py --odd only               # CRR odd results, then unchanged p2
  python answers.py p1 --start-exp 3 --max-exp 5 --output-dir /tmp/p1
  python answers.py p2 --n-paths 4096 --device cpu --seed 42
  python answers.py --start-exp 3 --max-exp 5 --n-paths 4096 --device cpu

Use --help for all options and defaults, including which problem each option affects.
Pass integer counts: --n-paths 1048576, not --n-paths 2**20.
Relative output directories are resolved from the current working directory.""",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=parser.get_default("output_dir"),
        help=(
            "CSV directory for all selected problems (default: %(default)s); "
            "create it if needed; overwrite CSVs with the same filenames"
        ),
    )
    parser.add_argument(
        "command", nargs="?", choices=("p1", "p2"),
        help="run only the selected problem; omit to run p1 then p2",
    )

    args = parser.parse_args()

    if args.command in (None, "p1"):
        print("Problem 1:")
        p1(args, parser)
    if args.command in (None, "p2"):
        print("\nProblem 2:")
        p2(args)


if __name__ == '__main__':
    main()
