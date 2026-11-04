"""Does the structure itself make sense, rather than does its energy number?

Every other check here reports one scalar. `embed_benchmark.py` compares
MMFF energies, `embed_budget_benchmark.py` compares the best of several
starts, and both can be satisfied by a structure that is wrong in a way
MMFF happens not to charge for. This one looks at the geometry:

    bonds        every bond length inside a chemically possible range, and
                 -- the stronger test -- unmoved when RDKit minimises our
                 own structure against its own MMFF94. A converged
                 structure does not move; one that is merely low does.
    aromatics    the largest distance of any aromatic ring atom from its
                 ring's best-fit plane.
    stereo       every centre and every double bond the *input SMILES
                 specifies*, read back out of the coordinates by RDKit's
                 `AssignStereochemistryFrom3D`. A centre the SMILES leaves
                 open is one the embedder may decide, so it is not
                 compared -- counting those as failures is what an earlier
                 version of this script did, and it reported nine
                 disagreements that were not.
    contacts     the closest approach between two atoms at least four bonds
                 apart, and which two elements they are. A short contact is
                 not automatically a clash: at 1.6 to 2.0 Angstroms an
                 H...O or H...N is a hydrogen bond and belongs there, while
                 an H...H at the same distance wants explaining.

`--control` runs the same audit over RDKit's ETKDGv3 plus MMFF94 structures
for the same molecules, which is the only way to read the contact numbers:
a tight H...H that RDKit also has, or that sits in a conformer several
kcal/mol below RDKit's, is the molecule's and not the embedder's.

Usage:
    cargo build --release --example embed_xyz
    python3 tests/geometry_audit.py [--corpus FILE] [--limit N] [--control]
"""

import argparse
import collections
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdmolops
from rdkit.Geometry import Point3D

sys.path.insert(0, str(Path(__file__).parent))
from embed_benchmark import CORPUS, align_hydrogens, read_corpus  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_XYZ_BIN = REPO_ROOT / "target" / "release" / "examples" / "embed_xyz"

# Below this, a non-bonded contact is worth reporting.
CONTACT_THRESHOLD = 2.0
# An aromatic ring flatter than this is flat.
PLANARITY_THRESHOLD = 0.1
# A bond outside this range is not a bond.
BOND_RANGE = (0.85, 2.4)
# How far a bond may move when RDKit minimises our structure.
CONVERGENCE_THRESHOLD = 0.05


def ours(smiles, reference):
    """`reference` re-posed onto what `embed_xyz` prints, or `None`."""
    done = subprocess.run([str(EMBED_XYZ_BIN), smiles], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    rows = [line.split() for line in done.stdout.strip().split("\n")[2:] if line.strip()]
    if len(rows) != reference.GetNumAtoms():
        return None
    coordinates = np.array([[float(x), float(y), float(z)] for _, x, y, z in rows])
    order, problem = align_hydrogens(reference, rows, coordinates)
    if problem:
        return None
    molecule = Chem.Mol(reference)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, source in enumerate(order):
        conformer.SetAtomPosition(index, Point3D(*coordinates[source]))
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def rdkit_conformer(reference, seed):
    """ETKDGv3 then MMFF94, which is RDKit's documented pipeline."""
    molecule = Chem.Mol(reference)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        return None
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=5000)
    return molecule


def bond_lengths(molecule):
    conformer = molecule.GetConformer()
    return np.array(
        [
            np.linalg.norm(
                np.array(conformer.GetAtomPosition(bond.GetBeginAtomIdx()))
                - np.array(conformer.GetAtomPosition(bond.GetEndAtomIdx()))
            )
            for bond in molecule.GetBonds()
        ]
    )


def planarity(molecule, positions):
    """The worst aromatic ring atom's distance from its ring's plane."""
    worst = 0.0
    for ring in molecule.GetRingInfo().AtomRings():
        if not all(molecule.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring):
            continue
        points = positions[list(ring)]
        centred = points - points.mean(axis=0)
        normal = np.linalg.svd(centred)[2][2]
        worst = max(worst, float(np.abs(centred @ normal).max()))
    return worst


def stereo_mismatches(molecule, smiles):
    """`(mismatches, specified)` over what the SMILES actually pins down."""
    read = Chem.Mol(molecule)
    Chem.AssignStereochemistryFrom3D(read)
    Chem.AssignStereochemistry(read, cleanIt=True, force=True)
    written = Chem.MolFromSmiles(smiles)
    if written is None:
        return 0, 0
    written = Chem.AddHs(written)
    Chem.AssignStereochemistry(written, cleanIt=True, force=True)

    mismatches = specified = 0
    for atom in written.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        specified += 1
        other = read.GetAtomWithIdx(atom.GetIdx())
        if not other.HasProp("_CIPCode") or other.GetProp("_CIPCode") != atom.GetProp("_CIPCode"):
            mismatches += 1
    for bond in written.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        specified += 1
        other = read.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if other is None or other.GetStereo() != bond.GetStereo():
            mismatches += 1
    return mismatches, specified


def closest_contact(molecule, positions):
    """The nearest pair at least four bonds apart, as `(distance, elements)`."""
    separations = rdmolops.GetDistanceMatrix(molecule)
    count = molecule.GetNumAtoms()
    best = (float("inf"), None)
    for i in range(count):
        for j in range(i + 1, count):
            if separations[i][j] < 4:
                continue
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance < best[0]:
                best = (
                    distance,
                    (
                        molecule.GetAtomWithIdx(i).GetSymbol(),
                        molecule.GetAtomWithIdx(j).GetSymbol(),
                    ),
                )
    return best


def is_hydrogen_bond(elements, distance):
    """A short H to N/O/F/S contact is a hydrogen bond, not a clash."""
    if elements is None or distance <= 1.4:
        return False
    first, second = sorted(elements)
    return first == "H" and second in ("N", "O", "F", "S")


def audit(molecule, smiles):
    """Everything that can be wrong with one conformer."""
    positions = np.array(molecule.GetConformer().GetPositions())
    if not np.isfinite(positions).all():
        return {"nonfinite": True}

    lengths = bond_lengths(molecule)
    result = {
        "bond_min": float(lengths.min()),
        "bond_max": float(lengths.max()),
        "planarity": planarity(molecule, positions),
    }
    result["stereo_mismatches"], result["stereo_specified"] = stereo_mismatches(molecule, smiles)
    result["contact"], result["contact_elements"] = closest_contact(molecule, positions)

    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    if properties is not None:
        energy = AllChem.MMFFGetMoleculeForceField(molecule, properties).CalcEnergy()
        relaxed = Chem.Mol(molecule)
        field = AllChem.MMFFGetMoleculeForceField(
            relaxed, AllChem.MMFFGetMoleculeProperties(relaxed)
        )
        field.Minimize(maxIts=200)
        result["bond_movement"] = float(np.abs(bond_lengths(relaxed) - lengths).max())
        result["energy_above_minimum"] = float(energy - field.CalcEnergy())
    return result


def report(label, results, contacts):
    counts = collections.Counter()
    movements, above = [], []
    for result in results:
        counts["audited"] += 1
        if result.get("nonfinite"):
            counts["non-finite coordinates"] += 1
            continue
        if not BOND_RANGE[0] <= result["bond_min"] or result["bond_max"] > BOND_RANGE[1]:
            counts["bond length out of range"] += 1
        if result.get("bond_movement", 0.0) > CONVERGENCE_THRESHOLD:
            counts["bond moves when RDKit minimises it"] += 1
        if result["stereo_mismatches"]:
            counts["stereo differs from the SMILES"] += 1
        if result["planarity"] > PLANARITY_THRESHOLD:
            counts["aromatic ring not flat"] += 1
        if result["contact"] < CONTACT_THRESHOLD:
            if is_hydrogen_bond(result["contact_elements"], result["contact"]):
                counts["closest contact is a hydrogen bond"] += 1
            else:
                counts["closest contact is short and not a hydrogen bond"] += 1
        movements.append(result.get("bond_movement", 0.0))
        above.append(result.get("energy_above_minimum", 0.0))

    print(f"\n{label}: {counts['audited']} structures")
    for key, value in counts.most_common():
        if key != "audited":
            print(f"  {value:>5}  {key}")
    if above:
        print(
            f"  MMFF energy above RDKit's own minimisation of the same structure:"
            f" median {np.median(above):.3f}, worst {max(above):.2f} kcal/mol"
        )
        print(f"  worst bond movement: {max(movements):.4f} A")
    if contacts:
        print("  short contacts that are not hydrogen bonds:")
        for distance, elements, smiles in sorted(contacts)[:10]:
            print(f"    {distance:.2f} A  {'-'.join(elements)}  {smiles[:60]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="file of 'group name SMILES' lines")
    parser.add_argument("--limit", type=int, help="only the first N molecules")
    parser.add_argument("--control", action="store_true", help="audit RDKit's structures too")
    parser.add_argument("--seed", type=int, default=0xF00D, help="RDKit embedding seed")
    arguments = parser.parse_args()

    if not EMBED_XYZ_BIN.exists():
        sys.exit(f"{EMBED_XYZ_BIN} not built - run: cargo build --release --example embed_xyz")

    corpus = read_corpus(arguments.corpus) if arguments.corpus else CORPUS
    if arguments.limit:
        corpus = corpus[: arguments.limit]

    mine, control = [], []
    my_contacts, control_contacts = [], []
    missing = 0
    for _group, _name, smiles in corpus:
        written = Chem.MolFromSmiles(smiles)
        if written is None:
            continue
        reference = Chem.AddHs(written)
        molecule = ours(smiles, reference)
        if molecule is None:
            missing += 1
            continue
        result = audit(molecule, smiles)
        mine.append(result)
        if result.get("contact", 9.0) < CONTACT_THRESHOLD and not is_hydrogen_bond(
            result.get("contact_elements"), result["contact"]
        ):
            my_contacts.append((result["contact"], result["contact_elements"], smiles))

        if arguments.control:
            other = rdkit_conformer(reference, arguments.seed)
            if other is None:
                continue
            result = audit(other, smiles)
            control.append(result)
            if result.get("contact", 9.0) < CONTACT_THRESHOLD and not is_hydrogen_bond(
                result.get("contact_elements"), result["contact"]
            ):
                control_contacts.append((result["contact"], result["contact_elements"], smiles))

    if missing:
        print(f"{missing} molecules produced no structure")
    report("ours", mine, my_contacts)
    if arguments.control:
        report("RDKit ETKDGv3 + MMFF94", control, control_contacts)


if __name__ == "__main__":
    main()
