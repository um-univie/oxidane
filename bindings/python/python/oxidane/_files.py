"""Canonical SMILES, tautomer keys and canonical tautomers for whole files."""

import csv
import gzip
import io
import sys
from contextlib import nullcontext
from itertools import islice
from os import PathLike, fspath
from typing import Dict, Optional, Sequence, Union

from ._oxidane import canonical_smiles, canonical_tautomer, tautomer_key

COLUMNS = ("canonical_smiles", "tautomer_key", "canonical_tautomer")
REPORTED = 10  # unreadable lines named on stderr per file; the rest are counted


def _open(path, mode):
    if fspath(path) == "-":
        return nullcontext(sys.stdin if mode == "r" else sys.stdout)
    if fspath(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def _delimiter(path):
    name = fspath(path).lower()
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".csv"):
        return ","
    if name.endswith(".tsv"):
        return "\t"
    return None  # .smi and anything else: whitespace-separated


def _rows(handle, delimiter):
    """Non-empty rows, each with the line it ends on."""
    if delimiter is None:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, line.split()
    else:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if row:
                yield reader.line_num, row


def process_file(
    input: Union[str, PathLike],
    output: Union[str, PathLike],
    *,
    columns: Sequence[str] = COLUMNS,
    smiles_column: Union[int, str] = 0,
    header: Optional[bool] = None,
    delimiter: Optional[str] = None,
    carbon_endpoints: bool = False,
    ionizable_endpoints: bool = False,
    threads: Optional[int] = None,
    errors: str = "keep",
    check_valence: bool = False,
    chunk_size: int = 100_000,
) -> Dict[str, int]:
    """Append canonical SMILES, tautomer keys and/or canonical tautomers to every
    row of a file.

    Each output row is the input row followed by one field per entry of
    `columns`. Rows are read `chunk_size` at a time and each chunk is computed on
    every core, so files of any size stream through in bounded memory.

    The delimiter follows the extension unless given: `,` for .csv, tab for .tsv,
    and whitespace for .smi and anything else (written back as tabs). A `.gz`
    suffix on either path compresses that side, and `-` reads standard input or
    writes standard output. `smiles_column` is a 0-based
    index, or a column name, which implies a header row. With a header, the
    output header gains the names in `columns`.

    A SMILES that does not parse is handled as `errors` says: "keep" (default)
    writes the input SMILES unchanged in each result field and reports the line
    on stderr (the first ten, then a count), "none" leaves the fields empty
    silently, "raise" stops with ValueError. With `check_valence`, a SMILES with
    an atom bonded past its element's valences counts as not parsing. Returns
    counts of rows written and SMILES that did not parse.
    """
    unknown = [name for name in columns if name not in COLUMNS]
    if unknown:
        raise ValueError(f"unknown columns {unknown}; choose from {list(COLUMNS)}")
    if errors not in ("keep", "none", "raise"):
        raise ValueError('errors must be "keep", "none" or "raise"')
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if header is None:
        header = isinstance(smiles_column, str)
    rules = {"carbon_endpoints": carbon_endpoints,
             "ionizable_endpoints": ionizable_endpoints}
    functions = {
        "canonical_smiles": lambda batch: canonical_smiles(batch, threads=threads, errors="none",
                                                          check_valence=check_valence),
        "tautomer_key": lambda batch: tautomer_key(batch, threads=threads, errors="none",
                                                  check_valence=check_valence, **rules),
        "canonical_tautomer": lambda batch: canonical_tautomer(
            batch, threads=threads, errors="none", check_valence=check_valence, **rules),
    }
    read_delimiter = delimiter if delimiter is not None else _delimiter(input)
    write_delimiter = delimiter if delimiter is not None else (_delimiter(output) or "\t")

    rows_written = unparsed = 0
    with _open(input, "r") as source, _open(output, "w") as sink:
        rows = _rows(source, read_delimiter)
        writer = csv.writer(sink, delimiter=write_delimiter, lineterminator="\n")
        if header:
            _, names = next(rows, (0, None))
            if names is None:
                return {"rows": 0, "unparsed": 0}
            if isinstance(smiles_column, str):
                if smiles_column not in names:
                    raise ValueError(f"no column named {smiles_column!r} in the header {names}")
                smiles_column = names.index(smiles_column)
            writer.writerow(list(names) + list(columns))
        while True:
            chunk = list(islice(rows, chunk_size))
            if not chunk:
                break
            batch = [row[smiles_column] if smiles_column < len(row) else "" for _, row in chunk]
            results = [functions[name](batch) for name in columns]
            report = io.StringIO()
            for index, (line, row) in enumerate(chunk):
                answers = [result[index] for result in results]
                if columns and answers[0] is None:
                    unparsed += 1
                    message = f"cannot read SMILES {batch[index]!r} on line {line}"
                    if errors == "raise":
                        raise ValueError(f"{fspath(input)}: {message}")
                    if errors == "keep":
                        if unparsed <= REPORTED:
                            report.write(
                                f"oxidane: {fspath(input)}: {message}, kept as it was\n")
                        answers = [batch[index]] * len(answers)
                    else:
                        answers = [""] * len(answers)
                writer.writerow(list(row) + answers)
            rows_written += len(chunk)
            if report.tell():
                sys.stderr.write(report.getvalue())
    if errors == "keep" and unparsed > REPORTED:
        sys.stderr.write(f"oxidane: {fspath(input)}: {unparsed - REPORTED} more unreadable "
                         "SMILES kept as they were\n")
    return {"rows": rows_written, "unparsed": unparsed}
