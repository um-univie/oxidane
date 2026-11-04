"""Compare this crate's UFF energy against RDKit's, on identical coordinates.

Usage:  uv run --with rdkit --with numpy python tests/uff_oracle.py [SMILES ...]

`examples/uff_energy.rs` embeds each molecule, prints its own UFF energy and
the XYZ; this hands those same coordinates to RDKit's UFF and prints both.

The two are the same published expressions with the same published
parameters, so on a molecule of sp3 atoms, plain alkenes and aromatic
rings they agree to the last digit, and a difference there is a bug in one
of them. On a molecule with a conjugated sp2 carbon, nitrogen, oxygen or
sulfur they differ on purpose, twice over: RDKit types every such atom
with the aromatic `R` types -- a carbonyl oxygen gets furan's bond radius
-- and it leaves the paper's 1.41 amide bond order unwired, where this
crate applies both. The difference this prints there is the size of those
two choices, not a regression; the module comment on
src/molecule/forcefield/uff/mod.rs records them.
"""

import subprocess
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from embed_benchmark import align_hydrogens  # noqa: E402

DEFAULT = [
    "CCO", "CC(=O)NC", "c1ccccc1", "c1ccc2ccccc2c1", "C1CCCCC1", "OC(=O)C",
    "c1ccc(cc1)C(=O)N", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1cc[nH]c1", "c1ccsc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C1=CC=C(C=C1)S(=O)(=O)N", "CC(=O)Oc1ccccc1C(=O)O",
]


def ours(smiles):
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "uff_energy", smiles],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    energy = float(out[1].split()[1])
    rows = [line.split() for line in out[2:]]
    coordinates = np.array([[float(v) for v in row[1:4]] for row in rows])
    return energy, rows, coordinates


def theirs(smiles, rows, coordinates):
    reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
    order, problem = align_hydrogens(reference, rows, coordinates)
    if order is None:
        return None, problem
    conformer = Chem.Conformer(reference.GetNumAtoms())
    for their_index, our_index in enumerate(order):
        conformer.SetAtomPosition(their_index, coordinates[our_index].tolist())
    reference.AddConformer(conformer, assignId=True)
    field = AllChem.UFFGetMoleculeForceField(reference)
    if field is None:
        return None, "RDKit declined to build a UFF field"
    return field.CalcEnergy(), None


def main():
    names = sys.argv[1:] or DEFAULT
    print(f"{'SMILES':44} {'ours':>12} {'rdkit':>12} {'diff':>10}")
    worst = 0.0
    for smiles in names:
        energy, rows, coordinates = ours(smiles)
        other, problem = theirs(smiles, rows, coordinates)
        if other is None:
            print(f"{smiles:44} {energy:12.4f} {'-':>12}  {problem}")
            continue
        worst = max(worst, abs(energy - other))
        print(f"{smiles:44} {energy:12.4f} {other:12.4f} {energy - other:10.4f}")
    print(f"\nlargest disagreement: {worst:.4f} kcal/mol")


if __name__ == "__main__":
    main()
