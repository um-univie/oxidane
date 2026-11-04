"""Command line for `process_file`: `oxidane-normalize` or `python -m oxidane`."""

import argparse
import sys
import time

from ._files import COLUMNS, process_file

EXAMPLES = """\
examples:
  oxidane-normalize molecules.smi molecules.tsv
  oxidane-normalize data.csv data_out.csv --smiles-column smiles
  oxidane-normalize CID-SMILES.gz keys.tsv.gz --smiles-column 1 --columns tautomer_key
  sqlite3 -tabs db.sqlite "SELECT id, smiles FROM compounds" | oxidane-normalize - out.tsv -s 1
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="oxidane-normalize",
        description="Append canonical SMILES, tautomer keys and canonical tautomers "
                    "to every row of a SMILES file.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="input file; .csv, .tsv, .smi, optionally .gz; - for stdin")
    parser.add_argument("output", help="output file, same formats; - for stdout")
    parser.add_argument("-s", "--smiles-column", default="0",
                        help="0-based index, or a header name (implies --header); default 0")
    parser.add_argument("-c", "--columns", default=",".join(COLUMNS),
                        help="comma-separated results to append; default: all of "
                             + ", ".join(COLUMNS))
    parser.add_argument("--header", action=argparse.BooleanOptionalAction, default=None,
                        help="first row is a header; default: only with a column name")
    parser.add_argument("-d", "--delimiter",
                        help="field delimiter for both files; default: from the extension")
    parser.add_argument("--carbon-endpoints", action="store_true",
                        help="also merge keto and enol forms")
    parser.add_argument("--ionizable-endpoints", action="store_true",
                        help="also merge zwitterions with their neutral forms")
    parser.add_argument("-j", "--threads", type=int, help="threads to use; default: every core")
    parser.add_argument("--errors", choices=("keep", "none", "raise"), default="keep",
                        help="unreadable SMILES: keep the input (default), leave empty, or stop")
    args = parser.parse_args(argv)

    column = args.smiles_column
    start = time.perf_counter()
    try:
        counts = process_file(
            args.input, args.output,
            columns=[name.strip() for name in args.columns.split(",") if name.strip()],
            smiles_column=int(column) if column.isdigit() else column,
            header=args.header,
            delimiter=args.delimiter,
            carbon_endpoints=args.carbon_endpoints,
            ionizable_endpoints=args.ionizable_endpoints,
            threads=args.threads,
            errors=args.errors,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"oxidane-normalize: {error}\n")
    seconds = time.perf_counter() - start
    sys.stderr.write(f"oxidane-normalize: {counts['rows']} rows, {counts['unparsed']} unreadable, "
                     f"{seconds:.1f} s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
