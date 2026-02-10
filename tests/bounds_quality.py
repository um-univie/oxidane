"""Are the distance bounds true, and are they tight?

A bound is what the embedder knows about a molecule before it has a single
coordinate, and it is the one thing an end-to-end conformer energy cannot
measure: the refinement and the force field between them absorb most of any
change to one. So this asks the two questions a bound can be wrong about,
against real geometry rather than against another energy.

    violated   the fraction of pairs whose distance in a real conformer falls
               outside [lower, upper]. A violated bound is a lie the
               embedder is told, and the refinement spends its iterations
               trying to satisfy it.
    slack      the mean width of the bounds that are not violated, by how
               many bonds apart the pair is. Narrower is better *only* while
               nothing is violated: a bound that excludes the truth is worse
               than no bound at all.

The reference geometry is RDKit's own: several ETKDGv3 conformers of the same
molecule, each relaxed with MMFF94, so that a pair's range over the set
approximates what the molecule can actually do rather than one arbitrary
pose. A bound has to contain every conformer to count as true.

Usage:
    cargo build --release --example bounds_dump
    python3 tests/bounds_quality.py [--corpus FILE] [--limit N] [--conformers N]
"""

import argparse
import collections
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdmolops

sys.path.insert(0, str(Path(__file__).parent))
from embed_benchmark import CORPUS, read_corpus  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
BOUNDS_DUMP_BIN = REPO_ROOT / "target" / "release" / "examples" / "bounds_dump"


# How far above the lowest a reference conformer may sit and still describe
# "what the molecule can do", in kcal/mol. A tangled structure is a real
# local minimum and MMFF reports it as converged, but it is not a pose this
# molecule adopts, and one of them poisons the min/max range the bounds are
# judged against.
REFERENCE_ENERGY_WINDOW = 50.0

# And how far a bond in one may stray from the sum of its covalent radii.
REFERENCE_BOND_TOLERANCE = 1.25


def sane_conformers(molecule, energies):
    """The conformer ids worth believing, as physical geometry.

    Found the hard way: a glycoside's fourth conformer came back "converged"
    at 858 kcal/mol against its siblings' 380, with a C-O bond stretched to
    1.82 A, and it alone produced the worst "violations" in the whole
    natural-product set -- a 1-3 distance of 3.24 A that two bonds cannot
    reach at any angle. A reference has to be checked like anything else.
    """
    table = Chem.GetPeriodicTable()
    floor = min(energies) if energies else None
    usable = []
    for index, conformer in enumerate(molecule.GetConformers()):
        if energies is not None and energies[index] > floor + REFERENCE_ENERGY_WINDOW:
            continue
        positions = conformer.GetPositions()
        if all(
            np.linalg.norm(positions[bond.GetBeginAtomIdx()] - positions[bond.GetEndAtomIdx()])
            <= REFERENCE_BOND_TOLERANCE
            * (table.GetRcovalent(bond.GetBeginAtom().GetAtomicNum())
               + table.GetRcovalent(bond.GetEndAtom().GetAtomicNum()))
            for bond in molecule.GetBonds()
        ):
            usable.append(conformer.GetId())
    return usable


def our_bounds(smiles):
    """`(symbols, {(i, j): (lower, upper)})`, or `None`."""
    done = subprocess.run([str(BOUNDS_DUMP_BIN), smiles], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    lines = done.stdout.strip().split("\n")
    symbols = lines[2].split()
    bounds = {}
    for line in lines[3:]:
        i, j, lower, upper = line.split("\t")
        bounds[(int(i), int(j))] = (float(lower), float(upper))
    return symbols, bounds


def reference_distances(smiles, conformers, seed):
    """Per-pair `(min, max)` over several relaxed RDKit conformers."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    ids = AllChem.EmbedMultipleConfs(molecule, numConfs=conformers, params=parameters)
    if not len(ids):
        return None, None
    energies = None
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        energies = [energy for _converged, energy
                    in AllChem.MMFFOptimizeMoleculeConfs(molecule, maxIters=2000)]
    usable = sane_conformers(molecule, energies)
    if not usable:
        return None, None
    stack = np.array([molecule.GetConformer(index).GetPositions() for index in usable])
    # pairwise distances per conformer, then the range over conformers
    deltas = stack[:, :, None, :] - stack[:, None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=-1))
    return molecule, (distances.min(axis=0), distances.max(axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="file of 'group name SMILES' lines")
    parser.add_argument("--limit", type=int, help="only the first N molecules")
    parser.add_argument("--conformers", type=int, default=4, help="reference conformers per molecule")
    parser.add_argument("--seed", type=int, default=0xF00D)
    parser.add_argument("--worst", type=int, default=6, help="how many offenders to name")
    arguments = parser.parse_args()

    if not BOUNDS_DUMP_BIN.exists():
        sys.exit(f"{BOUNDS_DUMP_BIN} not built - run: cargo build --release --example bounds_dump")

    corpus = read_corpus(arguments.corpus) if arguments.corpus else CORPUS
    if arguments.limit:
        corpus = corpus[: arguments.limit]

    counted = collections.Counter()
    violated = collections.Counter()
    slack = collections.defaultdict(float)
    excess = collections.defaultdict(float)
    worst = []
    molecules = 0

    for _group, _name, smiles in corpus:
        ours = our_bounds(smiles)
        molecule, span = reference_distances(smiles, arguments.conformers, arguments.seed)
        if ours is None or molecule is None:
            continue
        symbols, bounds = ours
        if len(symbols) != molecule.GetNumAtoms():
            continue
        # Heavy atoms only. Both sides parsed the same SMILES so their heavy
        # atoms are in the same order -- checked, not assumed -- while the
        # hydrogens are appended differently and would need the geometric
        # matching the other harnesses do, which there are no coordinates
        # here to do.
        heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
        if any(symbols[index] != molecule.GetAtomWithIdx(index).GetSymbol() for index in heavy):
            continue
        heavy_set = set(heavy)
        shortest = rdmolops.GetDistanceMatrix(molecule)
        low, high = span
        molecules += 1
        for (i, j), (lower, upper) in bounds.items():
            if i not in heavy_set or j not in heavy_set:
                continue
            path = int(shortest[i][j])
            key = f"1-{min(path, 5) + 1}" if path <= 4 else "1-6+"
            counted[key] += 1
            # the bound has to hold for every reference conformer
            if low[i][j] < lower - 1e-6 or high[i][j] > upper + 1e-6:
                violated[key] += 1
                by = max(lower - low[i][j], high[i][j] - upper)
                excess[key] += by
                worst.append((by, key, f"{molecule.GetAtomWithIdx(i).GetSymbol()}{i}",
                              f"{molecule.GetAtomWithIdx(j).GetSymbol()}{j}",
                              round(lower, 2), round(upper, 2),
                              round(low[i][j], 2), round(high[i][j], 2), smiles[:40]))
            else:
                slack[key] += upper - lower

    print(f"{molecules} molecules, {arguments.conformers} reference conformers each\n")
    print(f"{'pairs':>8}{'count':>10}{'violated':>11}{'mean excess':>13}{'mean slack':>12}")
    for key in ("1-2", "1-3", "1-4", "1-5", "1-6+"):
        if not counted[key]:
            continue
        kept = counted[key] - violated[key]
        print(f"{key:>8}{counted[key]:>10}{violated[key]:>7} ({100 * violated[key] / counted[key]:4.1f}%)"
              f"{(excess[key] / violated[key] if violated[key] else 0.0):>13.3f}"
              f"{(slack[key] / kept if kept else 0.0):>12.2f}")
    total = sum(counted.values())
    print(f"{'all':>8}{total:>10}{sum(violated.values()):>7} "
          f"({100 * sum(violated.values()) / max(total, 1):4.1f}%)")
    if worst:
        print("\nworst violations (bound, then what the conformers actually do):")
        for by, key, a, b, lower, upper, low, high, smiles in sorted(worst, reverse=True)[: arguments.worst]:
            print(f"  {key}  {a}-{b}  bound [{lower}, {upper}]  real [{low}, {high}]  off by {by:.2f}  {smiles}")


if __name__ == "__main__":
    main()
