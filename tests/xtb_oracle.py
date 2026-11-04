"""Which force field produces the best structures, judged by GFN2-xTB.

Usage:  uv run --with rdkit --with numpy --with tblite python tests/xtb_oracle.py [SMILES ...]

Relaxes each molecule four ways -- this crate's UFF and MMFF94, RDKit's
MMFF94, and GFN-FF -- from the same distance-geometry start, then scores
all four with GFN2-xTB. The MMFF94 comparison also checks whether our
minimizer reaches the same quality of minima as the independent implementation.

GFN-FF needs the `xtb` binary (tblite carries GFN1/GFN2-xTB but not GFN-FF).
Set `XTB_BIN`, or install it where this looks:
    ~/.local/share/xtb/xtb-dist/bin/xtb
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).parent))
from embed_benchmark import BOHR_PER_ANGSTROM, HARTREE_KCAL, align_hydrogens, xtb_energy  # noqa: E402

DEFAULT_XTB = Path.home() / ".local/share/xtb/xtb-dist/bin/xtb"
DEFAULT = [
    "c1ccccc1", "c1ccc2ccccc2c1", "c1ccoc1", "c1ccsc1", "c1ccc2[nH]ccc2c1",
    "c1nc2[nH]cnc2cn1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
]


def xtb_binary():
    found = os.environ.get("XTB_BIN") or shutil.which("xtb") or (
        str(DEFAULT_XTB) if DEFAULT_XTB.exists() else None
    )
    return found


def ours(smiles, scheme):
    """The shared unrelaxed start and requested relaxed geometry as RDKit mols."""
    result = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "uff_relax", smiles, scheme],
        capture_output=True, text=True,
    )
    if result.returncode:
        return None, None, result.stderr.strip().splitlines()[-1]
    blocks = result.stdout.split("END\n")
    start_rows = [line.split() for line in blocks[0].splitlines()[1:]]
    end_rows = [line.split() for line in blocks[1].splitlines()]
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    coordinates = np.array([[float(v) for v in row[1:4]] for row in start_rows])
    order, problem = align_hydrogens(molecule, start_rows, coordinates)
    if order is None:
        return None, None, problem
    results = []
    for rows in (start_rows, end_rows):
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        for theirs, mine in enumerate(order):
            conformer.SetAtomPosition(theirs, [float(v) for v in rows[mine][1:4]])
        geometry = Chem.Mol(molecule)
        geometry.AddConformer(conformer, assignId=True)
        results.append(geometry)
    return *results, None


def gfnff_relaxed(binary, molecule):
    """`molecule`'s geometry after a GFN-FF optimisation, or `None`."""
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "in.xyz"
        conformer = molecule.GetConformer()
        with source.open("w") as handle:
            handle.write(f"{molecule.GetNumAtoms()}\n\n")
            for atom in molecule.GetAtoms():
                p = conformer.GetAtomPosition(atom.GetIdx())
                handle.write(f"{atom.GetSymbol()} {p.x:.8f} {p.y:.8f} {p.z:.8f}\n")
        env = dict(os.environ)
        env.setdefault("XTBPATH", str(Path(binary).parent.parent / "share/xtb"))
        done = subprocess.run(
            [binary, source.name, "--chrg", str(Chem.GetFormalCharge(molecule)), "--gfnff", "--opt", "--silent"],
            cwd=work, capture_output=True, text=True, env=env, timeout=120,
        )
        optimised = Path(work) / "xtbopt.xyz"
        if done.returncode != 0 or not optimised.exists() or not (Path(work) / ".xtboptok").exists():
            return None
        lines = optimised.read_text().splitlines()
        relaxed = Chem.Mol(molecule)
        target = relaxed.GetConformer()
        for index, line in enumerate(lines[2 : 2 + molecule.GetNumAtoms()]):
            _, x, y, z = line.split()[:4]
            target.SetAtomPosition(index, (float(x), float(y), float(z)))
        return relaxed


def main():
    names = sys.argv[1:] or DEFAULT
    binary = xtb_binary()
    if binary is None:
        sys.exit("xtb not found - set XTB_BIN or install it (see this file's docstring)")
    print(f"xtb: {binary}")
    print("GFN2-xTB energies in kcal/mol relative to the best available result\n")
    methods = ["Rust UFF", "Rust MMFF94", "RDKit MMFF94", "GFN-FF"]
    print(f"{'molecule':44}" + "".join(f"{name:>14}" for name in methods))
    totals = dict.fromkeys(methods, 0.0)
    counted = 0
    for smiles in names:
        molecule, uff, problem = ours(smiles, "uff")
        if molecule is None:
            print(f"{smiles:44} {problem}", flush=True)
            continue
        _, mmff_ours, _ = ours(smiles, "mmff")
        scores = {"Rust UFF": xtb_energy(uff),
                  "Rust MMFF94": xtb_energy(mmff_ours) if mmff_ours is not None else None}

        mmff = Chem.Mol(molecule)
        converged = (AllChem.MMFFHasAllMoleculeParams(mmff)
                     and AllChem.MMFFOptimizeMolecule(mmff, maxIters=5000) == 0)
        scores["RDKit MMFF94"] = xtb_energy(mmff) if converged else None

        try:
            relaxed = gfnff_relaxed(binary, molecule)
        except subprocess.TimeoutExpired:
            relaxed = None
        scores["GFN-FF"] = xtb_energy(relaxed) if relaxed is not None else None

        available = [value for value in scores.values() if value is not None]
        if not available:
            print(f"{smiles:44} (no method converged)", flush=True)
            continue
        best = min(available)
        gaps = {key: (value - best) * HARTREE_KCAL if value is not None else None
                for key, value in scores.items()}
        # Means use only complete rows so different coverage cannot bias them.
        if all(value is not None for value in gaps.values()):
            for key in totals:
                totals[key] += gaps[key]
            counted += 1
        print(f"{smiles:44}" + "".join(
            f"{gaps[key]:>14.2f}" if gaps[key] is not None else f"{'n/a':>14}"
            for key in methods), flush=True)
    if counted:
        print(f"\n{'mean gap (complete rows)':44}" +
              "".join(f"{totals[key] / counted:>14.2f}" for key in methods))
        print(f"{counted} complete comparisons")



if __name__ == "__main__":
    main()
