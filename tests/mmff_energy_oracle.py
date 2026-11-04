"""Compare this crate's MMFF94 energy against RDKit's, one term at a time.

Usage:  uv run --with rdkit --with numpy python tests/mmff_energy_oracle.py [SMILES ...]
        uv run --with rdkit --with numpy python tests/mmff_energy_oracle.py --file tests/smiles.txt

`examples/mmff_energy.rs` embeds each molecule, prints its own MMFF energy
broken down by term, and prints the coordinates. This hands those same
coordinates to RDKit and asks for each term on its own, by switching the
other six off through `MMFFMolProperties.SetMMFF*Term`.

A total that agrees is not evidence: two errors cancel as easily as one is
absent. A term-by-term comparison says which parameter table or which
expression is wrong, which is the difference between a morning and a week.
"""

import subprocess
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from embed_benchmark import align_hydrogens  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Absolute, plus a relative allowance: the coordinates travel through six
# decimal places of text, and MMFF's quartic bond term at a strained geometry
# turns that last digit into a few thousandths of a kcal/mol.
TOLERANCE = 2e-3
RELATIVE_TOLERANCE = 1e-5


def agrees(ours, theirs):
    return abs(ours - theirs) <= TOLERANCE + RELATIVE_TOLERANCE * abs(theirs)

# Each term, with the setter that leaves only it switched on.
TERMS = [
    ("BOND", "SetMMFFBondTerm"),
    ("ANGLE", "SetMMFFAngleTerm"),
    ("STRETCH_BEND", "SetMMFFStretchBendTerm"),
    ("OOP", "SetMMFFOopTerm"),
    ("TORSION", "SetMMFFTorsionTerm"),
    ("VDW", "SetMMFFVdWTerm"),
    ("ELE", "SetMMFFEleTerm"),
]

DEFAULT = [
    "CCO", "CC(C)C", "C=CC", "c1ccccc1", "CC(=O)NC", "OC(=O)C", "C1CCCCC1",
    "c1ccc2ccccc2c1", "c1cc[nH]c1", "c1ccsc1", "c1ccncc1", "CS(=O)(=O)C",
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C1CC1", "C1CCC1",
    "CC(=O)[O-]", "C[NH3+]", "CC#N", "COC", "CSC", "CCCC", "c1ccc(cc1)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "C1=CC=C(C=C1)S(=O)(=O)N", "OP(=O)(O)O",
]


def ours(smiles):
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "mmff_energy", smiles],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if out and out[0].startswith("SKIPPED"):
        return None, out[0], None
    energies = {}
    for line in out[:8]:
        name, value = line.split()
        energies[name] = float(value)
    rows = [line.split() for line in out[9:]]
    coordinates = np.array([[float(v) for v in row[1:4]] for row in rows])
    return energies, rows, coordinates


def theirs(smiles, rows, coordinates):
    reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
    order, problem = align_hydrogens(reference, rows, coordinates)
    if order is None:
        return None, problem
    conformer = Chem.Conformer(reference.GetNumAtoms())
    for their_index, our_index in enumerate(order):
        conformer.SetAtomPosition(their_index, coordinates[our_index].tolist())
    reference.AddConformer(conformer, assignId=True)

    energies = {}
    for name, _ in TERMS:
        properties = AllChem.MMFFGetMoleculeProperties(reference)
        if properties is None:
            return None, "RDKit declined to type the molecule"
        for other, setter in TERMS:
            getattr(properties, setter)(other == name)
        field = AllChem.MMFFGetMoleculeForceField(reference, properties)
        if field is None:
            return None, "RDKit declined to build the force field"
        energies[name] = field.CalcEnergy()
    return energies, None


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ["--file"]:
        names = [line.split()[0] for line in open(arguments[1]) if line.strip()]
    else:
        names = arguments or DEFAULT
    subprocess.run(["cargo", "build", "--release", "--quiet", "--example", "mmff_energy"], check=True)

    header = f"{'SMILES':34}" + "".join(f"{name:>14}" for name, _ in TERMS)
    print(header)
    worst = {name: 0.0 for name, _ in TERMS}
    disagreed = 0
    for smiles in names:
        our_energies, rows, coordinates = ours(smiles)
        if our_energies is None:
            print(f"{smiles:34} {rows}")
            continue
        their_energies, problem = theirs(smiles, rows, coordinates)
        if their_energies is None:
            print(f"{smiles:34} {problem}")
            disagreed += 1
            continue
        differences = {name: our_energies[name] - their_energies[name] for name, _ in TERMS}
        line = f"{smiles:34}" + "".join(f"{differences[name]:>14.5f}" for name, _ in TERMS)
        print(line)
        for name, _ in TERMS:
            worst[name] = max(worst[name], abs(differences[name]))
        if any(not agrees(our_energies[name], their_energies[name]) for name, _ in TERMS):
            disagreed += 1

    print("\nworst difference, kcal/mol")
    for name, _ in TERMS:
        print(f"  {name:14} {worst[name]:.6f}")
    print(f"\n{len(names) - disagreed}/{len(names)} molecules agree with RDKit on every term")
    return 1 if disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
