"""Bemis-Murcko scaffolds from RDKit, for comparison against
`examples/scaffold_diff.rs`.

Reads SMILES on stdin, one per line, and writes
`input<TAB>scaffold<TAB>framework` with the same canonicalisation the Rust
side uses -- RDKit's canonical SMILES of the scaffold, and of the generic
(all-carbon, all-single-bond) framework.
"""

import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "PARSE_ERROR", "PARSE_ERROR"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "", ""
    try:
        framework = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        framework_key = Chem.MolToSmiles(framework)
    except Exception:
        framework_key = "GENERIC_ERROR"
    return Chem.MolToSmiles(scaffold), framework_key


def main():
    out = sys.stdout
    for line in sys.stdin:
        smiles = line.strip()
        if not smiles:
            continue
        scaffold, framework = keys(smiles)
        out.write(f"{smiles}\t{scaffold}\t{framework}\n")


if __name__ == "__main__":
    main()
