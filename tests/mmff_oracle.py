"""Compare this crate's MMFF94 atom types and charges against RDKit's.

Usage:  uv run --with rdkit python tests/mmff_oracle.py [SMILES ...]
        uv run --with rdkit python tests/mmff_oracle.py --file tests/smiles.txt

MMFF94 has ~99 symbolic atom types, and a mistyped atom does not raise: it
quietly scores the molecule with somebody else's parameters. So the typing is
checked on its own, before any energy term exists, against the only
independent implementation to hand.

`examples/mmff_types.rs` prints our type and partial charge for each atom;
RDKit's come from `AllChem.MMFFGetMoleculeProperties`. Heavy atoms are
compared by index -- both sides keep the SMILES order -- and hydrogens by the
multiset of type and charge hanging off each heavy atom, since neither side
promises the same order for the hydrogens of a methyl, and they are
interchangeable anyway.

Charges are compared to 1e-4, which is the width our side prints. They are
sums of tabulated increments over shared formal charges, so agreement is
exact until something is wrong -- a disagreement of 0.01 is as much a bug as
one of 1.0.
"""

import collections
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

TOLERANCE = 1e-4

# A spread of the type families that are easy to get wrong: five-membered
# heteroaromatics (alpha vs beta), fused and charged aromatics, amide against
# aniline against enamine nitrogen, the sulfur and phosphorus oxidation
# states, and the small-ring carbons.
DEFAULT = [
    "CCO", "CC(=O)NC", "c1ccccc1", "c1ccc2ccccc2c1", "C1CCCCC1", "OC(=O)C",
    "CC(=O)[O-]", "C(=O)([O-])[O-]", "c1cc[nH]c1", "c1ccsc1", "c1ccoc1",
    "c1cnc[nH]1", "c1cn[nH]c1", "c1cnoc1", "c1cnsc1", "c1ccncc1", "c1cnccn1",
    "c1cc[n+](C)cc1", "[O-][n+]1ccccc1", "c1c[nH+]c[nH]1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C1CC1", "C1CCC1", "C1=CC1", "C1=CCC1", "CC(=N)N", "NC(N)=[NH2+]",
    "CC(=O)N", "CC(=S)N", "Nc1ccccc1", "C=CN", "N#CN", "CS(=O)(=O)N",
    "CS(=O)(=O)C", "CS(=O)C", "CSC", "CSSC", "CS", "C=S", "CC(=S)[S-]",
    "OP(=O)(O)O", "COP(=O)(OC)OC", "CP(C)C", "C[N+](C)(C)C", "C[NH3+]",
    "CC#N", "C[N+]#[C-]", "N=[N+]=[N-]", "CN=O", "C[N+](=O)[O-]", "CON=O",
    "[O-]Cl(=O)(=O)=O", "O", "[OH3+]", "[Na+]", "[Cl-]", "CO", "COC", "c1ccc(cc1)O",
    "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)[nH]c1ccccc12", "c1ccc2[nH]ccc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "C1=CC=C(C=C1)S(=O)(=O)N", "FC(F)(F)Br",
    "CCI", "c1ccc2ncccc2c1", "O=c1cccc[nH]1", "O=C1C=CC(=O)C=C1",
]


def ours(names):
    """Our types for every molecule at once, keyed by SMILES.

    One process for the whole corpus: the typing itself is microseconds, so
    a `cargo run` per molecule would be measuring process startup.
    """
    out = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "mmff_types"],
        input="\n".join(names), capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    blocks = {}
    current = None
    remaining = 0
    for line in out:
        if line.startswith("SMILES "):
            current = line[len("SMILES "):]
            blocks[current] = []
            remaining = 0
        elif line.startswith(("UNTYPED", "UNPARSED")):
            blocks[current] = line
        elif remaining == 0:
            remaining = int(line)
        else:
            blocks[current].append(line.split())
            remaining -= 1
    return blocks


def theirs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "RDKit could not parse the SMILES"
    mol = Chem.AddHs(mol)
    properties = AllChem.MMFFGetMoleculeProperties(mol)
    if properties is None:
        return None, "RDKit declined to type the molecule"
    return mol, properties


def compare(smiles, rows):
    """Every disagreement on one molecule, as printable lines."""
    mol, properties = theirs(smiles)
    if isinstance(rows, str):
        # Refusing to type is an answer, and the right one for a molecule
        # MMFF94 has no types for -- carbonate, say. It only counts against
        # us if RDKit managed where we did not.
        return [] if mol is None else [f"  ours: {rows}"]
    if mol is None:
        return [f"  theirs: {properties}"]
    if mol.GetNumAtoms() != len(rows):
        return [f"  {len(rows)} atoms here, {mol.GetNumAtoms()} in RDKit"]

    their_types = [properties.GetMMFFAtomType(i) for i in range(mol.GetNumAtoms())]
    their_charges = [properties.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
    complaints = []

    for row in rows:
        atom, element, number, symbol = int(row[0]), row[1], int(row[2]), row[3]
        charge = float(row[5])
        if element == "H":
            continue
        if mol.GetAtomWithIdx(atom).GetSymbol() != element:
            return [f"  atom {atom} is {element} here and "
                    f"{mol.GetAtomWithIdx(atom).GetSymbol()} in RDKit -- orders differ"]
        if number != their_types[atom]:
            complaints.append(
                f"  atom {atom} {element}: type {number} ({symbol}) vs RDKit {their_types[atom]}")
        elif abs(charge - their_charges[atom]) > TOLERANCE:
            complaints.append(
                f"  atom {atom} {element} ({symbol}): charge {charge:+.4f} "
                f"vs RDKit {their_charges[atom]:+.4f}")

    # Hydrogens by parent, as multisets: neither side promises the same order
    # for the three hydrogens of a methyl, and they are interchangeable.
    our_hydrogens = collections.defaultdict(collections.Counter)
    for row in rows:
        if row[1] == "H":
            our_hydrogens[int(row[4])][(int(row[2]), round(float(row[5]), 4))] += 1
    their_hydrogens = collections.defaultdict(collections.Counter)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            parent = atom.GetNeighbors()[0].GetIdx()
            index = atom.GetIdx()
            their_hydrogens[parent][(their_types[index], round(their_charges[index], 4))] += 1
    for parent in sorted(set(our_hydrogens) | set(their_hydrogens)):
        if our_hydrogens[parent] != their_hydrogens[parent]:
            complaints.append(
                f"  hydrogens on atom {parent}: {sorted(our_hydrogens[parent].elements())} "
                f"vs RDKit {sorted(their_hydrogens[parent].elements())}"
            )

    return complaints


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ["--file"]:
        # Tab-separated files hold several writings of one molecule per line;
        # the first is enough, and the rest only re-test the parser.
        names = [line.split()[0] for line in open(arguments[1]) if line.strip()]
    else:
        names = arguments or DEFAULT

    # Each fragment of a salt is typed on its own: MMFF typing never reaches
    # across a fragment boundary, and this crate's `from_smiles_single` reads
    # one fragment where RDKit reads them all.
    names = [fragment for smiles in names for fragment in smiles.split(".")]
    names = list(dict.fromkeys(names))

    subprocess.run(["cargo", "build", "--release", "--quiet", "--example", "mmff_types"], check=True)
    blocks = ours(names)

    disagreed = 0
    for smiles in names:
        rows = blocks.get(smiles)
        if rows is None:
            print(smiles)
            print("  ours: no output")
            disagreed += 1
            continue
        complaints = compare(smiles, rows)
        if complaints:
            disagreed += 1
            print(smiles)
            print("\n".join(complaints))

    print(f"\n{len(names) - disagreed}/{len(names)} molecules typed and charged exactly as RDKit does")
    return 1 if disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
