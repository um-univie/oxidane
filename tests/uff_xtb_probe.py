"""Score this crate's UFF-relaxed geometries with GFN2-xTB (single variant).

Usage:  uv run --with rdkit --with numpy python tests/uff_xtb_probe.py [SMILES ...]

`examples/uff_relax.rs` embeds each molecule without relaxing, relaxes with
this crate's UFF and prints both geometries. This script optimises the same
start with GFN2-xTB (chained twice -- one round from a rough geometry can
stick in a spurious basin) and with RDKit's MMFF94, takes the best energy
reached as the reference, then scores the UFF geometry with a GFN2 single
point and prints the geometric parameters that the changes under test act
on, against the same parameters of the reference geometry.
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from embed_benchmark import align_hydrogens

RDLogger.DisableLog("rdApp.*")

XTB = os.environ.get("XTB_BIN", "/scr/kaldra/muhlir/micromamba/envs/qcxms2_memtest/bin/xtb")
XTBPATH = os.environ.get("XTBPATH", str(Path(XTB).parent.parent / "share/xtb"))
HARTREE_KCAL = 627.5094740631

# (SMILES, name, centre index, centre symbol, neighbour cutoff). The centre
# is the atom the changes act on: amide/aniline nitrogens, thiophene sulfur,
# furan oxygen, the phosphorus of the phosphines and the phosphonium.
MOLECULES = [
    ("CCO", "ethanol (control)", None, None, None),
    ("CC(=O)N", "acetamide", 3, "N", 1.6),
    ("CC(=O)NC", "N-methylacetamide", 3, "N", 1.6),
    ("NC(=O)N", "urea", 0, "N", 1.6),
    ("Nc1ccccc1", "aniline", 0, "N", 1.6),
    ("c1ccsc1", "thiophene", 3, "S", 1.9),
    ("c1ccoc1", "furan", 3, "O", 1.6),
    ("CP(C)C", "trimethylphosphine", 1, "P", 2.0),
    ("C1CCPC1", "phospholane (5-ring)", 3, "P", 2.0),
    ("C1CPC1", "phosphetane (4-ring)", 2, "P", 2.0),
    ("C1PC1", "phosphirane (3-ring)", 1, "P", 2.0),
    ("C[P+](C)(C)C", "tetramethylphosphonium", 1, "P", 2.0),
    ("P(c1ccccc1)(c1ccccc1)c1ccccc1", "triphenylphosphine", 0, "P", 2.0),
]


def run_example(smiles):
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "uff_relax", smiles],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    start, end, mode = [], [], None
    for line in out:
        if line == "START":
            mode = start
            continue
        if line == "END":
            mode = end
            continue
        if mode is not None:
            parts = line.split()
            mode.append((parts[0], np.array([float(v) for v in parts[1:4]])))
    return start, end


def xyz_text(rows):
    return f"{len(rows)}\n\n" + "\n".join(
        f"{symbol} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for symbol, p in rows
    ) + "\n"


def xtb(rows, args, charge=0):
    with tempfile.TemporaryDirectory() as work:
        with open(os.path.join(work, "in.xyz"), "w") as handle:
            handle.write(xyz_text(rows))
        env = dict(os.environ)
        env["XTBPATH"] = XTBPATH
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        try:
            done = subprocess.run(
                [XTB, "in.xyz", "--chrg", str(charge)] + args, cwd=work,
                capture_output=True, text=True, env=env, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return None, None
        if done.returncode != 0 or ("--opt" in args and
                (not os.path.exists(os.path.join(work, ".xtboptok"))
                 or os.path.exists(os.path.join(work, "NOT_CONVERGED")))):
            return None, None
        energy = None
        for line in done.stdout.splitlines():
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[3])
        optimised = None
        optfile = os.path.join(work, "xtbopt.xyz")
        if "--opt" in args and os.path.exists(optfile):
            lines = open(optfile).read().splitlines()
            optimised = [
                (l.split()[0], np.array([float(v) for v in l.split()[1:4]]))
                for l in lines[2:2 + len(rows)]
            ]
        return energy, optimised


def sp_energy(rows, charge=0):
    return xtb(rows, ["--gfn", "2"], charge)[0]


def opt_chained(rows, charge=0, rounds=2):
    current = rows
    for _ in range(rounds):
        _, optimised = xtb(current, ["--gfn", "2", "--opt"], charge)
        if optimised is None:
            return None, None
        current = optimised
    return sp_energy(current, charge), current


def mmff_relaxed(smiles, start):
    """RDKit MMFF94 from the same start, returned in Oxidane atom order.

    Hydrogens must be mapped to their bonded parents before minimising;
    matching element sequences alone can silently scramble the molecule.
    """
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        return None
    order, problem = align_hydrogens(molecule, start, np.array([p for _, p in start]))
    if problem:
        raise ValueError(f"{smiles}: {problem}")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, source in enumerate(order):
        conformer.SetAtomPosition(index, start[source][1].tolist())
    molecule.AddConformer(conformer, assignId=True)
    if AllChem.MMFFOptimizeMolecule(molecule, maxIters=2000) != 0:
        return None
    result = [None] * len(start)
    positions = molecule.GetConformer().GetPositions()
    for index, source in enumerate(order):
        result[source] = (start[source][0], positions[index])
    return result


def neighbours(rows, i, cutoff):
    return [j for j in range(len(rows))
            if j != i and np.linalg.norm(rows[j][1] - rows[i][1]) < cutoff]


def out_of_plane(rows, i, cutoff):
    nbrs = neighbours(rows, i, cutoff)
    if len(nbrs) != 3:
        return None
    a, b, c = (rows[n][1] - rows[i][1] for n in nbrs)
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    return abs(float(np.dot(normal, c)))


def angle_sum(rows, i, cutoff):
    nbrs = neighbours(rows, i, cutoff)
    if len(nbrs) != 3:
        return None
    def angle(j, k):
        v1, v2 = rows[j][1] - rows[i][1], rows[k][1] - rows[i][1]
        return math.degrees(math.acos(
            np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
    return sum(angle(nbrs[a], nbrs[b])
               for a in range(3) for b in range(a + 1, 3))


def mean_angle(rows, i, cutoff):
    nbrs = neighbours(rows, i, cutoff)
    if len(nbrs) < 2:
        return None
    def angle(j, k):
        v1, v2 = rows[j][1] - rows[i][1], rows[k][1] - rows[i][1]
        return math.degrees(math.acos(
            np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
    angles = [angle(nbrs[a], nbrs[b])
              for a in range(len(nbrs)) for b in range(a + 1, len(nbrs))]
    return sum(angles) / len(angles)


def angle(rows, i, j, k):
    v1, v2 = rows[i][1] - rows[j][1], rows[k][1] - rows[j][1]
    return math.degrees(math.acos(
        np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))


def dist(rows, i, j):
    return float(np.linalg.norm(rows[i][1] - rows[j][1]))


def measures(smiles, rows):
    if smiles in ("CC(=O)N", "CC(=O)NC"):
        return [("N oop A", out_of_plane(rows, 3, 1.6)),
                ("N ang sum", angle_sum(rows, 3, 1.6)),
                ("C=O A", dist(rows, 1, 2)),
                ("C-N A", dist(rows, 1, 3))]
    if smiles == "NC(=O)N":
        return [("C-N A", (dist(rows, 1, 0) + dist(rows, 1, 3)) / 2),
                ("C=O A", dist(rows, 1, 2))]
    if smiles == "Nc1ccccc1":
        return [("N oop A", out_of_plane(rows, 0, 1.6)),
                ("N ang sum", angle_sum(rows, 0, 1.6))]
    if smiles == "c1ccsc1":
        return [("C-S-C deg", angle(rows, 2, 3, 4)),
                ("C-S A", (dist(rows, 2, 3) + dist(rows, 3, 4)) / 2)]
    if smiles == "c1ccoc1":
        return [("C-O-C deg", angle(rows, 2, 3, 4)),
                ("C-O A", (dist(rows, 2, 3) + dist(rows, 3, 4)) / 2)]
    if smiles == "C[P+](C)(C)C":
        # four-coordinate: the mean C-P-C angle says tetrahedral (109.5)
        # against pyramidal (93.8)
        return [("mean C-P-C deg", mean_angle(rows, 1, 2.0)),
                ("mean C-P A", sum(dist(rows, 1, c) for c in (0, 2, 3, 4)) / 4)]
    if smiles in ("CP(C)C", "C1CCPC1", "C1CPC1", "C1PC1"):
        centre = {"CP(C)C": 1, "C1CCPC1": 3, "C1CPC1": 2, "C1PC1": 1}[smiles]
        return [("P oop A", out_of_plane(rows, centre, 2.0)),
                ("P ang sum", angle_sum(rows, centre, 2.0))]
    if smiles == "P(c1ccccc1)(c1ccccc1)c1ccccc1":
        return [("P oop A", out_of_plane(rows, 0, 2.0)),
                ("P ang sum", angle_sum(rows, 0, 2.0))]
    return []


def formal_charge(smiles):
    return Chem.GetFormalCharge(Chem.AddHs(Chem.MolFromSmiles(smiles)))


COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05}


def expected_bonds(smiles, rows):
    """SMILES connectivity in the coordinate atom order, including hydrogens."""
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    order, problem = align_hydrogens(molecule, rows, np.array([p for _, p in rows]))
    if problem:
        raise ValueError(f"{smiles}: {problem}")
    return {tuple(sorted((order[b.GetBeginAtomIdx()], order[b.GetEndAtomIdx()])))
            for b in molecule.GetBonds()}


def bond_set(rows):
    """The set of bonded atom pairs, by distance. A reference route whose
    optimisation broke or made a bond has relaxed into a different molecule
    and must not be allowed to define the reference."""
    bonds = set()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ri = COVALENT_RADII.get(rows[i][0], 0.8)
            rj = COVALENT_RADII.get(rows[j][0], 0.8)
            if np.linalg.norm(rows[i][1] - rows[j][1]) < 1.25 * (ri + rj):
                bonds.add((i, j))
    return bonds


def main():
    print(f"{'molecule':28}{'gap':>8}   parameters (UFF -> xtb reference)")
    molecules = MOLECULES if not sys.argv[1:] else [
        next((entry for entry in MOLECULES if entry[0] == smiles),
             (smiles, smiles, None, None, None)) for smiles in sys.argv[1:]
    ]
    for smiles, name, centre, symbol, cutoff in molecules:
        start, end = run_example(smiles)
        if centre is not None:
            assert start[centre][0] == symbol, (smiles, start[centre][0])
        charge = formal_charge(smiles)

        start_bonds = expected_bonds(smiles, start)
        routes = {"uff": end}
        mmff = mmff_relaxed(smiles, start)
        if mmff is not None:
            routes["mmff"] = mmff
        candidates = {}
        for route, geometry in routes.items():
            energy, optimised = opt_chained(geometry, charge)
            if energy is None or optimised is None or bond_set(optimised) != start_bonds:
                continue  # relaxed into a different molecule; not a reference
            candidates[route] = (energy, optimised)
        if not candidates:
            print(f"{name:28}     n/a   no converged reference retaining connectivity", flush=True)
            continue
        best_route = min(candidates, key=lambda route: candidates[route][0])
        reference_energy, reference_geometry = candidates[best_route]

        energy = sp_energy(end, charge)
        if energy is None:
            print(f"{name:28}     n/a   UFF single point failed", flush=True)
            continue
        gap = (energy - reference_energy) * HARTREE_KCAL
        line = f"{name:28}{gap:8.2f}"
        for measure, _ in measures(smiles, end):
            ours = dict(measures(smiles, end))[measure]
            theirs = dict(measures(smiles, reference_geometry))[measure]
            if ours is None or theirs is None:
                continue
            line += f"   {measure} {ours:.3f} -> {theirs:.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
