"""How long each pipeline takes to turn a SMILES into a relaxed conformer.

Both sides are timed around the same job -- parse, embed, relax -- and
neither is charged for process startup: `embed_timed` times itself inside
one process for the whole corpus, and RDKit is timed in this one.

    ours      `embed_3d`, which is distance geometry, MMFF94 relaxation and
              the torsion search, at `--trials` starts (its default is five).
    RDKit     `EmbedMolecule` with ETKDGv3 then `MMFFOptimizeMolecule`, one
              start, which is what their documentation has a user write.

That is not the same amount of work, and the point of reporting it beside
`tests/embed_budget_benchmark.py` is that the two have to be read together:
this says what each costs, that says what each buys. `--trials 1` puts the
two on the same number of starts.

Times are milliseconds per molecule. The summary reports the median rather
than the mean, because a handful of large flexible molecules dominate a mean
and the median is what a user embedding a library of them will see; the
totals are there for the mean anyway.

Usage:
    cargo build --release --example embed_timed
    python3 tests/embed_speed.py [--corpus FILE] [--trials N] [--limit N]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).parent))
from embed_benchmark import CORPUS, read_corpus  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_TIMED_BIN = REPO_ROOT / "target" / "release" / "examples" / "embed_timed"


def ours(corpus, trials, search=True):
    """`{smiles: (atoms, milliseconds)}` from one run over the whole corpus."""
    request = "\n".join(smiles for _, _, smiles in corpus) + "\n"
    command = [str(EMBED_TIMED_BIN), str(trials or 5)]
    if not search:
        command.append("nosearch")
    done = subprocess.run(command, input=request, capture_output=True, text=True, check=False)
    timings = {}
    for line in done.stdout.splitlines():
        smiles, atoms, elapsed = line.rsplit("\t", 2)
        timings[smiles] = (int(atoms), None if elapsed == "ERROR" else float(elapsed))
    return timings


def theirs(smiles, seed):
    """Milliseconds for ETKDGv3 plus an MMFF94 minimisation, or `None`."""
    start = time.perf_counter()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        return None
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=5000)
    return (time.perf_counter() - start) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="file of 'group name SMILES' lines")
    parser.add_argument("--trials", type=int, help="our starts per molecule")
    parser.add_argument("--limit", type=int, help="only the first N molecules")
    parser.add_argument("--no-search", action="store_true",
                        help="switch our torsion search off, to price it")
    parser.add_argument("--seed", type=int, default=0xF00D, help="RDKit embedding seed")
    arguments = parser.parse_args()

    if not EMBED_TIMED_BIN.exists():
        sys.exit(f"{EMBED_TIMED_BIN} not built - "
                 "run: cargo build --release --example embed_timed")

    corpus = read_corpus(arguments.corpus) if arguments.corpus else CORPUS
    if arguments.limit:
        corpus = corpus[: arguments.limit]

    timings = ours(corpus, arguments.trials, not arguments.no_search)

    header = f"{'molecule':<22}{'atoms':>6}{'ours':>10}{'RDKit':>10}{'ratio':>8}"
    print(f"ours: embed_3d at {arguments.trials or 5} starts; "
          f"RDKit: ETKDGv3 + MMFF94, one start. Milliseconds.\n")
    print(header)
    print("-" * len(header))

    rows = []
    for _group, name, smiles in corpus:
        atoms, mine = timings.get(smiles, (0, None))
        their = theirs(smiles, arguments.seed)
        if mine is None or their is None:
            print(f"{name:<22}{atoms:>6}{'failed':>10}")
            continue
        rows.append((atoms, mine, their))
        if len(corpus) <= 80:
            print(f"{name:<22}{atoms:>6}{mine:>10.1f}{their:>10.1f}{mine / their:>8.1f}")

    if not rows:
        sys.exit("nothing was timed")
    rows.sort()
    median = rows[len(rows) // 2]
    ours_total = sum(row[1] for row in rows)
    their_total = sum(row[2] for row in rows)
    ratios = sorted(row[1] / row[2] for row in rows)

    print(f"\n{len(rows)} molecules, {rows[0][0]} to {rows[-1][0]} atoms")
    print(f"{'median molecule':<22}{median[0]:>6}{median[1]:>10.1f}{median[2]:>10.1f}"
          f"{median[1] / median[2]:>8.1f}")
    print(f"{'median ratio':<22}{'':>6}{'':>10}{'':>10}{ratios[len(ratios) // 2]:>8.1f}")
    print(f"{'total':<22}{'':>6}{ours_total:>10.0f}{their_total:>10.0f}"
          f"{ours_total / their_total:>8.1f}")
    # Where the time goes: the slowest tenth against the rest.
    by_cost = sorted(rows, key=lambda row: -row[1])
    tenth = max(1, len(rows) // 10)
    print(f"the slowest {tenth} of ours take "
          f"{sum(row[1] for row in by_cost[:tenth]) / ours_total:.0%} of our total")


if __name__ == "__main__":
    main()
