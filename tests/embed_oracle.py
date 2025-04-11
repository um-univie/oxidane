#!/usr/bin/env python3
"""RDKit cross-check for the `Embed3D` (ETKDG-lite) distance-geometry
embedder in src/molecule/embed/.

For each SMILES: runs the `embed_xyz` Rust example to get our coordinates,
attaches them to an RDKit molecule built from the *same* SMILES (RDKit
supplies bond connectivity/order; we supply only positions, since this crate
has no SDF/molfile writer), then asks RDKit to independently re-derive
stereochemistry from that 3D structure (`AssignStereochemistryFrom3D`) and
checks it against the stereochemistry RDKit reads directly from the input
SMILES. This validates our embedding against a trusted, independent
implementation rather than this crate's own (self-consistent by
construction) oracle.

Usage:
    cargo build --example embed_xyz
    python3 tests/embed_oracle.py
"""

import subprocess
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_XYZ_BIN = REPO_ROOT / "target" / "debug" / "examples" / "embed_xyz"

BOND_LENGTH_MIN = 0.6
BOND_LENGTH_MAX = 2.2

STEREO_CORPUS = [
    "[C@](F)(Cl)(Br)I",
    "[C@@](F)(Cl)(Br)I",
    "N[C@@H](C)C(=O)O",
    "N[C@H](C)C(=O)O",
    "OC(=O)[C@H](O)[C@H](O)C(=O)O",
    "OC(=O)[C@H](O)[C@@H](O)C(=O)O",
    "F/C=C/F",
    "F/C=C\\F",
    "C/C=C/C(=O)O",
    "C/C=C\\C(=O)O",
]

# Square-planar centres have their own trans-pairing angle check, not the
# R/S/E-Z comparison `check_molecule` does - see `check_square_planar`.
# Allene (AL) centres are implemented but deliberately NOT included here:
# RDKit 2025.09.1's `AssignStereochemistryFrom3D` /
# `AssignAtomChiralTagsFromStructure` do not perceive allene stereo from
# either 3D coordinates or `/`,`\`-decorated cumulated-diene SMILES
# (confirmed by direct experimentation during development), so there is no
# RDKit ground truth to check against for AL - see the "unverified" doc
# comment on `embed::chirality::AllenCenter`.
SQUARE_PLANAR_CORPUS = [
    "[Pt@SP1](F)(Cl)(Br)I",
    "[Pt@SP2](F)(Cl)(Br)I",
    "[Pt@SP3](F)(Cl)(Br)I",
]

GENERAL_CORPUS = [
    "c1ccccc1",
    "C1CCCCC1",
    "c1ccc2ccccc2c1",
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "c1ccc(cc1)C(=O)O",  # benzoic acid
    "C1CCC2(CC1)CCCCC2",  # spiro
    "NCC(=O)NCC(=O)O",  # peptide fragment
    "CCO",
    "c1ccncc1",  # pyridine
]


def run_embed_xyz(smiles: str) -> tuple[list[str], list[tuple[float, float, float]]] | None:
    result = subprocess.run([str(EMBED_XYZ_BIN), smiles], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  embed_xyz failed: {result.stderr.strip()}")
        return None
    lines = result.stdout.splitlines()
    n = int(lines[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + n]:
        sym, x, y, z = line.split()
        symbols.append(sym)
        coords.append((float(x), float(y), float(z)))
    return symbols, coords


def check_bond_lengths(mol_h: Chem.Mol, conf_id: int) -> list[str]:
    # Heavy-heavy bonds only: heavy-atom indices are reliably 1:1 between
    # RDKit's mol (built straight from the SMILES) and ours (both preserve
    # SMILES-parse heavy-atom order), but the *hydrogens* RDKit's AddHs
    # appends are not guaranteed to land at the same indices as the ones our
    # add_hydrogens() appends - e.g. for a bracket atom like `[C@@H]`, RDKit
    # appends every new H in strict heavy-atom order, while this crate's
    # parser materializes the bracket-specified H separately from the later
    # implicit-H pass, giving it an earlier index than some other atom's
    # implicit H even though that atom has a lower heavy-atom index. Checking
    # X-H bonds here would silently compare unrelated atom pairs; those bond
    # lengths are already validated (correctly, index-ambiguity-free) by the
    # crate's own bounds-based Rust tests.
    problems = []
    conf = mol_h.GetConformer(conf_id)
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol_h.GetAtomWithIdx(i).GetSymbol() == "H" or mol_h.GetAtomWithIdx(j).GetSymbol() == "H":
            continue
        length = rdMolTransforms.GetBondLength(conf, i, j)
        if not (BOND_LENGTH_MIN <= length <= BOND_LENGTH_MAX):
            a, b = mol_h.GetAtomWithIdx(i).GetSymbol(), mol_h.GetAtomWithIdx(j).GetSymbol()
            problems.append(f"bond {a}{i}-{b}{j} length {length:.3f} A out of [{BOND_LENGTH_MIN},{BOND_LENGTH_MAX}]")
    return problems


def stereo_labels(mol: Chem.Mol) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    bonds = {}
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo != Chem.BondStereo.STEREONONE:
            bonds[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = str(stereo)
    return centers, bonds


def check_molecule(smiles: str, expect_stereo: bool) -> bool:
    print(f"* {smiles}")
    xyz = run_embed_xyz(smiles)
    if xyz is None:
        return False
    symbols, coords = xyz

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("  RDKit failed to parse SMILES, skipping")
        return False
    expected_centers, expected_bonds = stereo_labels(mol)

    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() != len(symbols):
        print(f"  FAIL: atom count mismatch (ours={len(symbols)}, rdkit AddHs={mol_h.GetNumAtoms()})")
        return False
    for i, atom in enumerate(mol_h.GetAtoms()):
        if atom.GetSymbol() != symbols[i]:
            print(f"  FAIL: atom order mismatch at index {i} (ours={symbols[i]}, rdkit={atom.GetSymbol()})")
            return False

    conf = Chem.Conformer(mol_h.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    conf_id = mol_h.AddConformer(conf, assignId=True)

    bond_problems = check_bond_lengths(mol_h, conf_id)
    for p in bond_problems:
        print(f"  FAIL: {p}")

    Chem.AssignStereochemistryFrom3D(mol_h, confId=conf_id)
    actual_centers, actual_bonds = stereo_labels(mol_h)

    stereo_ok = True
    for idx, expected_label in expected_centers.items():
        if expected_label in ("?",):
            continue
        actual_label = actual_centers.get(idx)
        if actual_label != expected_label:
            print(f"  FAIL: chiral centre atom {idx} expected {expected_label}, embedding+RDKit gave {actual_label}")
            stereo_ok = False
    for (i, j), expected_label in expected_bonds.items():
        actual_label = actual_bonds.get((i, j)) or actual_bonds.get((j, i))
        if actual_label != expected_label:
            print(f"  FAIL: bond stereo {i}-{j} expected {expected_label}, embedding+RDKit gave {actual_label}")
            stereo_ok = False

    if expect_stereo and not expected_centers and not expected_bonds:
        print("  WARNING: expected this SMILES to carry stereo tags but RDKit found none")

    ok = stereo_ok and not bond_problems
    print("  ok" if ok else "  (see failures above)")
    return ok


def check_square_planar(smiles: str) -> bool:
    """Square-planar geometry has no separate handedness once the trans-
    pairing is fixed (see `embed::bounds::square_planar_trans_pairs`), so
    round-tripping the `_chiralPermutation` (1/2/3, i.e. SP1/SP2/SP3) is the
    right check here, not an R/S-style label comparison."""
    print(f"* {smiles}")
    xyz = run_embed_xyz(smiles)
    if xyz is None:
        return False
    symbols, coords = xyz

    mol = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(mol)
    if mol_h.GetNumAtoms() != len(symbols):
        print(f"  FAIL: atom count mismatch (ours={len(symbols)}, rdkit AddHs={mol_h.GetNumAtoms()})")
        return False

    conf = Chem.Conformer(mol_h.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    conf_id = mol_h.AddConformer(conf, assignId=True)

    expected = mol.GetAtomWithIdx(0).GetPropsAsDict().get("_chiralPermutation")
    Chem.AssignStereochemistryFrom3D(mol_h, confId=conf_id)
    actual = mol_h.GetAtomWithIdx(0).GetPropsAsDict().get("_chiralPermutation")
    ok = expected is not None and expected == actual
    if not ok:
        print(f"  FAIL: expected SP{expected}, embedding+RDKit gave SP{actual}")
    else:
        print("  ok")
    return ok


def main() -> int:
    if not EMBED_XYZ_BIN.exists():
        print(f"missing {EMBED_XYZ_BIN} - run `cargo build --example embed_xyz` first", file=sys.stderr)
        return 2

    total = 0
    passed = 0
    print("--- stereo corpus (chirality/cis-trans round-trip vs RDKit) ---")
    for smiles in STEREO_CORPUS:
        total += 1
        passed += check_molecule(smiles, expect_stereo=True)

    print("\n--- square-planar corpus (trans-pairing round-trip vs RDKit) ---")
    for smiles in SQUARE_PLANAR_CORPUS:
        total += 1
        passed += check_square_planar(smiles)

    print("\n--- general corpus (bond-length sanity only) ---")
    for smiles in GENERAL_CORPUS:
        total += 1
        passed += check_molecule(smiles, expect_stereo=False)

    print("\n--- aromatization_tests.smi fuzz sample (bond-length sanity only) ---")
    fuzz_total = 0
    fuzz_passed = 0
    for smiles in fuzz_sample_from_aromatization_corpus():
        total += 1
        fuzz_total += 1
        ok = check_molecule(smiles, expect_stereo=False)
        passed += ok
        fuzz_passed += ok
    print(f"  ({fuzz_passed}/{fuzz_total} fuzz-sampled molecules passed)")

    print(f"\n{passed}/{total} molecules passed")
    return 0 if passed == total else 1


def fuzz_sample_from_aromatization_corpus(limit: int = 25, max_atoms: int = 40) -> list[str]:
    """Mirrors the Rust-side `embeds_aromatization_corpus_without_panicking`
    fuzz test: a bounded sample of real SMILES from the existing
    aromaticity-test corpus, kept small enough (`max_atoms`, `limit`) that
    this script stays fast to run by hand."""
    path = REPO_ROOT / "tests" / "aromatization_tests.smi"
    if not path.exists():
        return []
    sample = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smiles = line.split(",")[0].strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > max_atoms:
            continue
        sample.append(smiles)
        if len(sample) >= limit:
            break
    return sample


if __name__ == "__main__":
    raise SystemExit(main())
