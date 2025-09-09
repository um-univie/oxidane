"""RDKit's hybridization per atom, for calibrating ours against.

Reads SMILES on stdin, writes `smiles<TAB>label,label,...` sorted, so that the
two sides need not agree on atom order. Hydrogens are made explicit, as they
are in this crate, and left out of the list, since RDKit gives them no state.
"""
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog("rdApp.*")

NAMES = {
    rdchem.HybridizationType.S: "S",
    rdchem.HybridizationType.SP: "SP",
    rdchem.HybridizationType.SP2: "SP2",
    rdchem.HybridizationType.SP3: "SP3",
    rdchem.HybridizationType.SP3D: "SP3D",
    rdchem.HybridizationType.SP3D2: "SP3D2",
    rdchem.HybridizationType.UNSPECIFIED: "?",
}

for line in sys.stdin:
    smiles = line.split("\t")[0].strip()
    if not smiles:
        continue
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        continue
    molecule = Chem.AddHs(molecule)
    labels = sorted(
        NAMES.get(atom.GetHybridization(), "?")
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 1
    )
    print(f"{smiles}\t{','.join(labels)}")
