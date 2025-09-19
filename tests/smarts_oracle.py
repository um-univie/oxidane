"""SMARTS match counts from RDKit, for calibrating the matcher against.

Writes `smarts<TAB>target<TAB>count` for every pattern against every molecule
read from stdin. Hydrogens are made explicit so that both sides count the same
graph, and matches are not uniquified, since a mapping is what this crate's
`find_cxsmarts_matches` returns.
"""
import sys
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PATTERNS = [
    "c1ccccc1", "[CX3](=O)[OX2H1]", "[#6]=[#6]", "[N;!H0]", "[O;!H0]",
    "*@*!@*@*", "*@*(@*)@*", "*!@*@*!@*", "*!@[#7]@*", "*@*!@[#7]",
    "[#6]=;@[#6](@*)@*", "c:n", "[!#6;!#1;!H0]", "[!#6;!#1]~[!#6;!#1]",
    "[$(*~[CH2]~[CH2]~*)]", "[$([!#6;!#1;!H0]~*~*~[CH2]~*)]",
    "[$(C=O)]", "[$([NH2][CX4])]", "[R]", "[R2]", "[r5]", "[r6]",
    "[CH3]~*~[CH3]", "*1~*~*~*~*~*~1", "[#7]~[#8]", "[F,Cl,Br,I]",
    "[#16]-[#16]", "[$([OX2H])]", "[nX3;H1]", "[a]", "[!c;!n;a]",
    "C(=O)N", "[CX4H3]", "[#7;R]", "[$([#8]=[#6]([#7])[#7])]",
    "[Cv4]", "[Cv3]", "[CX4]", "[CX3]", "[OX1]", "[OX2]", "[13C]", "[A]",
    "[D1]", "[D2]", "[D3]", "[!R]", "[#6;R]", "[#6;!R]", "[O-]", "[N+]",
    "[se]", "[$([CX3]=[OX1])]", "[C;!$(C=O)]", "[#6]-[#6]", "[#6]:[#6]",
    "[#6]#[#7]", "[#6]~[#7]", "*!@*", "*@*", "[R0]", "[R1]", "[R3]",
    "[#7X3][CX3]=[OX1]", "[#8;H1,H0]", "[c;!$(c-[OX2])]", "[Cl,Br]",
    "[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]1", "[!#1;!#6;!#7;!#8]",
]

targets = []
for line in sys.stdin:
    smiles = line.split("\t")[0].strip()
    if not smiles:
        continue
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        continue
    targets.append((smiles, Chem.AddHs(molecule)))

for smarts in PATTERNS:
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        print(f"# {smarts} does not parse", file=sys.stderr)
        continue
    for smiles, molecule in targets:
        count = len(molecule.GetSubstructMatches(pattern, uniquify=False, maxMatches=100000))
        print(f"{smarts}\t{smiles}\t{count}")
