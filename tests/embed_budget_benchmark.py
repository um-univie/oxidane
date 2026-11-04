"""Conformer search at an equal trial budget: ours against RDKit ETKDGv3.

`tests/embed_benchmark.py` runs each pipeline once and compares the structure
it returns. That answers "what do I get by default", but it cannot separate
two different things: whether a pipeline puts a conformer where it belongs,
and how many conformers it looked at. Both sides here get `--trials` starts,
minimise every one, and keep the lowest MMFF94 energy, so the difference that
remains is the search, not the budget.

    ours    `embed_budget` at `max_attempts = 1`, called once per seed. One
            call with `max_attempts = N` would also produce N trials, but the
            selection among them would be ours rather than the lowest-MMFF
            rule applied to the other side, and the comparison is of searches.
    theirs  ETKDGv3 at one seed per trial, each minimised with RDKit MMFF94
            -- their documented pipeline, `EmbedMolecule` then
            `MMFFOptimizeMolecule`.

Reported per molecule: the best MMFF94 energy each side reached, the closest
non-bonded contact in that structure, and the difference as `ours - theirs`
both in MMFF94 and, with `--xtb`, in GFN2-xTB. Negative is ours winning.
`spread` is the best-to-worst range over our own trials: a small spread next
to a losing `dMMFF` means the trials are landing in one basin rather than
sampling, which is a different problem than landing in the wrong one.

Absolute MMFF energies are comparable only between the two columns of one row.

Usage:
    cargo build --release --example embed_budget
    python3 tests/embed_budget_benchmark.py [--trials N] [--xtb] [--corpus FILE]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

sys.path.insert(0, str(Path(__file__).parent))
from embed_benchmark import (  # noqa: E402
    CORPUS,
    HARTREE_KCAL,
    align_hydrogens,
    closest_nonbonded,
    rdkit_etkdg,
    read_corpus,
    xtb_energy,
)

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_BUDGET_BIN = REPO_ROOT / "target" / "release" / "examples" / "embed_budget"

# Ours is deterministic in its seed, so a trial is a seed. RDKit's is too.
OUR_SEED = 0xE7D65EED
THEIR_SEED = 0xF00D


def ours(smiles, reference, attempts, seed, search=True):
    """`reference` re-posed onto what `embed_budget` prints, or a complaint.

    Same contract as `embed_benchmark.ours`: the XYZ carries no bonds, so
    RDKit supplies connectivity from the same SMILES and only the positions
    come from us, which holds exactly as long as the two agree on atom order.
    """
    done = subprocess.run(
        [str(EMBED_BUDGET_BIN), smiles, str(attempts), str(seed)]
        + ([] if search else ["nosearch"]),
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()
        return None, f"embed failed: {detail[-1][:60] if detail else 'no output'}"
    rows = [line.split() for line in done.stdout.strip().split("\n")[2:] if line.strip()]
    if len(rows) != reference.GetNumAtoms():
        return None, f"{len(rows)} atoms, RDKit builds {reference.GetNumAtoms()}"
    coordinates = np.array([[float(x), float(y), float(z)] for _, x, y, z in rows])
    order, problem = align_hydrogens(reference, rows, coordinates)
    if problem:
        return None, problem

    molecule = Chem.Mol(reference)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, source in enumerate(order):
        conformer.SetAtomPosition(index, Point3D(*coordinates[source]))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    # Matching elements do not prove matching atom order -- every hydrogen is
    # an "H". Under RDKit's connectivity, every bond in our coordinates has to
    # be a plausible length; a scrambled structure fails that at once.
    positions = np.array(molecule.GetConformer().GetPositions())
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = float(np.linalg.norm(positions[i] - positions[j]))
        if not 0.85 <= length <= 2.1:
            return None, f"atom order differs: bond {i}-{j} is {length:.2f} A"
    return molecule, None


def mmff_energy(molecule, minimise=False):
    """MMFF94 energy, minimising in place first if asked, or `None`."""
    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    if properties is None:
        return None
    field = AllChem.MMFFGetMoleculeForceField(molecule, properties)
    if minimise:
        field.Minimize(maxIts=5000)
    return field.CalcEnergy()


def best_of(candidates):
    """The lowest-energy `(energy, molecule)` pair, or `(inf, None)`."""
    best = (float("inf"), None)
    for energy, molecule in candidates:
        if energy is not None and energy < best[0]:
            best = (energy, molecule)
    return best


def our_trials(smiles, reference, trials, search=True):
    """One single-attempt embedding per seed, already relaxed by us."""
    for trial in range(trials):
        molecule, problem = ours(smiles, reference, 1, OUR_SEED + trial, search)
        if problem:
            continue
        yield mmff_energy(molecule), molecule


def their_trials(reference, trials):
    """One ETKDGv3 embedding per seed, each minimised by RDKit's MMFF94."""
    for trial in range(trials):
        molecule, problem = rdkit_etkdg(reference, THEIR_SEED + trial)
        if problem:
            continue
        yield mmff_energy(molecule, minimise=True), molecule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8, help="starts per side")
    parser.add_argument("--corpus", type=Path, help="file of 'group name SMILES' lines")
    parser.add_argument("--xtb", action="store_true", help="add a GFN2-xTB comparison")
    parser.add_argument("--no-search", action="store_true",
                        help="switch our torsion search off, for a like-for-like comparison")
    arguments = parser.parse_args()

    if not EMBED_BUDGET_BIN.exists():
        sys.exit(f"{EMBED_BUDGET_BIN} not built - "
                 "run: cargo build --release --example embed_budget")

    corpus = read_corpus(arguments.corpus) if arguments.corpus else CORPUS

    xtb_column = f"{'dE(xTB)':>10}" if arguments.xtb else ""
    header = (f"{'':<10}{'molecule':<18}{'n':>4} | {'MMFF':>9}{'clash':>7}{'spread':>8} | "
              f"{'MMFF':>9}{'clash':>7} | {'dMMFF':>8}{xtb_column}")
    print(f"best of {arguments.trials} trials per side, lowest MMFF94 kept\n")
    print(f"{'':<32} | {'ours':^26} | {'RDKit ETKDGv3+MMFF':^18} | {'ours - theirs':^18}")
    print(header)
    print("-" * len(header))

    wins = {"mmff": 0, "xtb": 0}
    sums = {"mmff": 0.0, "xtb": 0.0}
    counted = {"mmff": 0, "xtb": 0}
    for group, name, smiles in corpus:
        reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
        label = f"{group:<10}{name:<18}{reference.GetNumAtoms():>4}"

        mine = list(our_trials(smiles, reference, arguments.trials, not arguments.no_search))
        theirs = list(their_trials(reference, arguments.trials))
        my_energy, my_best = best_of(mine)
        their_energy, their_best = best_of(theirs)
        if my_best is None or their_best is None:
            missing = "ours" if my_best is None else "RDKit"
            print(f"{label} | {missing} produced no scorable conformer")
            continue

        finite = [energy for energy, _ in mine if energy is not None]
        spread = max(finite) - min(finite)
        difference = my_energy - their_energy
        counted["mmff"] += 1
        sums["mmff"] += difference
        wins["mmff"] += difference <= 0.1

        cell = ""
        if arguments.xtb:
            mine_xtb, theirs_xtb = xtb_energy(my_best), xtb_energy(their_best)
            if mine_xtb is None or theirs_xtb is None:
                cell = f"{'n/a':>10}"
            else:
                delta = (mine_xtb - theirs_xtb) * HARTREE_KCAL
                counted["xtb"] += 1
                sums["xtb"] += delta
                wins["xtb"] += delta <= 0.1
                cell = f"{delta:>10.1f}"

        print(f"{label} | {my_energy:>9.1f}{closest_nonbonded(my_best):>7.2f}{spread:>8.1f} | "
              f"{their_energy:>9.1f}{closest_nonbonded(their_best):>7.2f} | "
              f"{difference:>8.1f}{cell}", flush=True)

    if not counted["mmff"]:
        sys.exit("nothing was comparable")
    summary = f"\nours at least as good on MMFF: {wins['mmff']}/{counted['mmff']}"
    if counted["xtb"]:
        summary += f"   on GFN2-xTB: {wins['xtb']}/{counted['xtb']}"
    print(summary)
    means = f"mean dMMFF {sums['mmff'] / counted['mmff']:+.2f}"
    if counted["xtb"]:
        means += f"   mean dE(xTB) {sums['xtb'] / counted['xtb']:+.2f}"
    print(f"{means} kcal/mol")


if __name__ == "__main__":
    main()
