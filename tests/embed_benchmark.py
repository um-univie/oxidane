"""How good are the 3D coordinates `Embed3D` produces, next to RDKit's ETKDG?

Oxidane's default output includes MMFF94 relaxation where supported. RDKit's
ETKDGv3 output here does not include its optional force-field cleanup, so E0
compares different pipeline stages and is a strain diagnostic, not an
independent accuracy test. Emin applies RDKit MMFF94 relaxation to both.
The optional GFN2-xTB column supplies an independent energy model.

Four numbers per molecule, starting from the structure each pipeline returns:

    E0      MMFF energy of the raw coordinates. Lower is better.
    Emin    MMFF energy after minimisation. Says which basin the embedder
            landed in, rather than how tidy it left it.
    shift   RMSD the atoms move during that minimisation. How far the
            structure sat from a real minimum.
    clash   Closest approach between two atoms at least four bonds apart.
            Higher is better; below about 2.0 A is a steric clash.

With `--xtb`, a fifth column reports GFN2-xTB's opinion of the same two raw
structures, as `ours - theirs` in kcal/mol: positive means our geometry is the
strained one. It is a difference rather than two energies because xTB returns
a total electronic energy, comparable only between conformers of one molecule.
Worth having because MMFF94 is a fitted force field with its own preferences,
and it would be circular-ish to tune a geometry generator against a single
scorer; a semiempirical QM method agreeing is a much stronger statement than
MMFF alone. Needs `tblite` (`uv pip install tblite`).

CREST is deliberately not here. It samples conformational space by
metadynamics -- minutes to hours per molecule, and answering a different
question than "how good is the one structure this embedder just produced".

`clash` must be read before minimisation, which is why it is computed first:
`ff.Minimize()` rewrites the conformer in place, and measuring afterwards
silently reports the minimised structure's clashes for both sides.

Absolute MMFF energies are not comparable between molecules -- only the two
columns for one molecule are comparable with each other.

Usage:
    cargo build --release --example embed_xyz
    python3 tests/embed_benchmark.py [--corpus FILE] [--seed N]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_XYZ_BIN = REPO_ROOT / "target" / "release" / "examples" / "embed_xyz"

# The corpus lives in `tests/embed_corpus.tsv` rather than here: both
# benchmarks read it, so do the diagnostics, and a molecule worth adding is
# worth adding once. `--corpus` overrides it.
CORPUS_FILE = REPO_ROOT / "tests" / "embed_corpus.tsv"


def read_corpus(path):
    """`[(group, name, smiles)]` from a `group name SMILES` file."""
    return [
        tuple(line.split(None, 2))
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


CORPUS = read_corpus(CORPUS_FILE)


def ours(smiles, reference):
    """`reference`, re-posed onto the coordinates `embed_xyz` prints.

    The XYZ carries no bonds, so RDKit supplies connectivity from the same
    SMILES and only the positions come from us -- which is sound exactly as
    long as the two agree on atom order. `Chem.AddHs` appends hydrogens after
    the heavy atoms and so does `add_hydrogens`, and the element sequences are
    compared below rather than assumed: a mismatch would quietly score a
    scrambled molecule instead of ours.
    """
    result = subprocess.run(
        [str(EMBED_XYZ_BIN), smiles], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return None, f"embed failed: {result.stderr.strip().splitlines()[-1][:60]}"

    rows = [line.split() for line in result.stdout.strip().split("\n")[2:] if line.strip()]
    if len(rows) != reference.GetNumAtoms():
        return None, f"{len(rows)} atoms, RDKit builds {reference.GetNumAtoms()}"
    expected = [atom.GetSymbol() for atom in reference.GetAtoms()]
    if [row[0] for row in rows] != expected:
        return None, "element sequence differs from RDKit's"

    coordinates = np.array([[float(x), float(y), float(z)] for _, x, y, z in rows])
    order, problem = align_hydrogens(reference, rows, coordinates)
    if problem:
        return None, problem

    molecule = Chem.Mol(reference)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, source in enumerate(order):
        x, y, z = coordinates[source]
        conformer.SetAtomPosition(index, Point3D(x, y, z))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)

    # Matching element sequences are not enough to prove the two sides agree
    # on atom order: every hydrogen is an "H", so a molecule whose hydrogens
    # were attached to different parents passes that check and then scores as
    # a scrambled structure. Indole did exactly that, reporting an MMFF
    # energy of 254473 and a 1.03 A "clash" that was really its N-H bond
    # measured as a non-bonded contact. Bond lengths catch it: under RDKit's
    # connectivity, every bond in our coordinates has to be a plausible one.
    #
    # Scaled by the two atoms' covalent radii rather than a fixed window: a
    # 2.1 A ceiling is right for carbon and wrong for the rest of the table,
    # and it was quietly reporting valid structures as scrambled ones. On
    # the verification corpus it rejected fourteen molecules of ours, every
    # one of them for a real bond -- C-I at 2.10 to 2.14, Si-P at 2.25,
    # silyl ethers -- and counted them as our failures.
    positions = np.array(molecule.GetConformer().GetPositions())
    table = Chem.GetPeriodicTable()
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = float(np.linalg.norm(positions[i] - positions[j]))
        nominal = (table.GetRcovalent(molecule.GetAtomWithIdx(i).GetAtomicNum())
                   + table.GetRcovalent(molecule.GetAtomWithIdx(j).GetAtomicNum()))
        if not 0.6 * nominal <= length <= 1.4 * nominal:
            return None, (
                f"atom order differs: bond {i}-{j} "
                f"({molecule.GetAtomWithIdx(i).GetSymbol()}-"
                f"{molecule.GetAtomWithIdx(j).GetSymbol()}) is {length:.2f} A"
            )
    return molecule, None


BOHR_PER_ANGSTROM = 1.8897261254535
HARTREE_KCAL = 627.5094740631

def xtb_energy(molecule):
    """GFN2-xTB single-point energy, in Hartree, or `None` if it will not run."""
    from tblite.interface import Calculator

    numbers = np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    positions = np.array(molecule.GetConformer().GetPositions()) * BOHR_PER_ANGSTROM
    try:
        calculator = Calculator(
            "GFN2-xTB", numbers, positions,
            charge=float(Chem.GetFormalCharge(molecule)),
            uhf=sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()),
        )
        calculator.set("verbosity", 0)
        return float(calculator.singlepoint().get("energy"))
    except Exception:
        return None  # SCF failures are a property of the geometry, not a crash


def align_hydrogens(reference, rows, coordinates):
    """Our row index for each of RDKit's atoms.

    Both sides put the heavy atoms first and in the same order, but they do
    not agree on the order of the hydrogens hung off them -- and since every
    hydrogen is an "H", comparing element sequences cannot tell. Indole
    slipped through that check and scored as a scrambled molecule: MMFF
    energy 254473, and a 1.03 A "clash" that was really its N-H bond being
    measured as a non-bonded contact.

    So hydrogens are matched by which heavy atom they hang off, found from
    our own coordinates by proximity. Which hydrogen of a CH3 goes where is
    left arbitrary: they are interchangeable, and permuting them leaves the
    molecule and its MMFF energy unchanged.
    """
    if len(rows) != reference.GetNumAtoms():
        return None, "atom count differs from RDKit's"
    if [row[0] for row in rows] != [atom.GetSymbol() for atom in reference.GetAtoms()]:
        return None, "element sequence differs from RDKit's"
    heavy = [index for index, row in enumerate(rows) if row[0] != "H"]
    if not heavy:
        return list(range(len(rows))), None
    table = Chem.GetPeriodicTable()
    # A fixed 1.4 A cutoff rejects valid P-H bonds. Use covalent radii for
    # both parent assignment and its plausibility check.
    radii = np.array([table.GetRcovalent(rows[h][0]) + table.GetRcovalent(1)
                      for h in heavy])
    ours_by_parent = {}
    for index, row in enumerate(rows):
        if row[0] == "H":
            distances = [float(np.linalg.norm(coordinates[index] - coordinates[h])) for h in heavy]
            scaled = np.array(distances) / radii
            nearest = int(np.argmin(scaled))
            if scaled[nearest] > 1.35:
                return None, f"hydrogen {index} has no plausible parent (nearest scaled distance {scaled[nearest]:.2f})"
            ours_by_parent.setdefault(heavy[nearest], []).append(index)

    order = []
    for atom in reference.GetAtoms():
        if atom.GetSymbol() != "H":
            order.append(atom.GetIdx())
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            return None, f"hydrogen {atom.GetIdx()} has {len(neighbors)} bonds in RDKit"
        parent = neighbors[0].GetIdx()
        available = ours_by_parent.get(parent)
        if not available:
            symbol = reference.GetAtomWithIdx(parent).GetSymbol()
            return None, f"no hydrogen of ours is bonded to {symbol}{parent}"
        order.append(available.pop())
    return order, None


def rdkit_etkdg(reference, seed):
    molecule = Chem.Mol(reference)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, params) != 0:
        return None, "ETKDGv3 failed to embed"
    return molecule, None


def closest_nonbonded(molecule):
    """Closest approach between atoms at least four bonds apart."""
    positions = np.array(molecule.GetConformer().GetPositions())
    topological = Chem.GetDistanceMatrix(molecule)
    closest = float("inf")
    for i in range(molecule.GetNumAtoms()):
        for j in range(i + 1, molecule.GetNumAtoms()):
            if topological[i][j] >= 4:
                closest = min(closest, float(np.linalg.norm(positions[i] - positions[j])))
    return closest


def score(molecule):
    """`(E0, Emin, shift, clash)`, or `None` if MMFF has no parameters."""
    clash = closest_nonbonded(molecule)  # before Minimize rewrites the conformer
    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    if properties is None:
        return None
    field = AllChem.MMFFGetMoleculeForceField(molecule, properties)
    before = np.array(molecule.GetConformer().GetPositions())
    raw = field.CalcEnergy()
    field.Minimize(maxIts=2000)
    minimised = field.CalcEnergy()
    after = np.array(molecule.GetConformer().GetPositions())
    shift = float(np.sqrt(((after - before) ** 2).sum(axis=1).mean()))
    return raw, minimised, shift, clash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0xF00D, help="RDKit embedding seed")
    parser.add_argument("--corpus", type=Path, help="file of 'group name SMILES' lines")
    parser.add_argument(
        "--xtb", action="store_true", help="add a GFN2-xTB single-point comparison"
    )
    arguments = parser.parse_args()

    if not EMBED_XYZ_BIN.exists():
        sys.exit(f"{EMBED_XYZ_BIN} not built - run: cargo build --release --example embed_xyz")

    corpus = read_corpus(arguments.corpus) if arguments.corpus else CORPUS

    xtb_column = f"{'dE(xTB)':>10}" if arguments.xtb else ""
    header = f"{'':<10}{'molecule':<18}{'n':>4} | {'E0':>9}{'Emin':>9}{'shift':>7}{'clash':>7} | {'E0':>9}{'Emin':>9}{'shift':>7}{'clash':>7}{xtb_column}"
    print(f"{'':<32} | {'ours':^32} | {'RDKit ETKDGv3':^32}")
    print(header)
    print("-" * len(header))

    wins = {"E0": 0, "clash": 0}
    scored = {}
    total = 0
    for group, name, smiles in corpus:
        reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
        mine, problem = ours(smiles, reference)
        if problem:
            print(f"{group:<10}{name:<18}{reference.GetNumAtoms():>4} | {problem}")
            continue
        theirs, problem = rdkit_etkdg(reference, arguments.seed)
        if problem:
            print(f"{group:<10}{name:<18}{reference.GetNumAtoms():>4} | {problem}")
            continue

        a, b = score(mine), score(theirs)
        if a is None or b is None:
            print(f"{group:<10}{name:<18}{reference.GetNumAtoms():>4} | no MMFF parameters")
            continue

        total += 1
        wins["E0"] += a[0] <= b[0]
        wins["clash"] += a[3] >= b[3]

        difference = ""
        if arguments.xtb:
            # Both `score` calls minimised in place, so the raw geometries
            # have to be rebuilt before xTB sees them.
            mine_raw, _ = ours(smiles, reference)
            theirs_raw, _ = rdkit_etkdg(reference, arguments.seed)
            mine_energy = xtb_energy(mine_raw) if mine_raw else None
            theirs_energy = xtb_energy(theirs_raw) if theirs_raw else None
            if mine_energy is None or theirs_energy is None:
                difference = f"{'n/a':>10}"
            else:
                delta = (mine_energy - theirs_energy) * HARTREE_KCAL
                wins["xtb"] = wins.get("xtb", 0) + (delta <= 0)
                scored["xtb"] = scored.get("xtb", 0) + 1
                difference = f"{delta:>10.1f}"

        print(
            f"{group:<10}{name:<18}{reference.GetNumAtoms():>4} |"
            f"{a[0]:>9.1f}{a[1]:>9.1f}{a[2]:>7.2f}{a[3]:>7.2f} |"
            f"{b[0]:>9.1f}{b[1]:>9.1f}{b[2]:>7.2f}{b[3]:>7.2f}{difference}"
        )

    if total:
        summary = (
            f"\nours at least as good on E0: {wins['E0']}/{total}"
            f"   on clash: {wins['clash']}/{total}"
        )
        if scored.get("xtb"):
            summary += f"   on GFN2-xTB: {wins.get('xtb', 0)}/{scored['xtb']}"
        print(summary)


if __name__ == "__main__":
    main()
