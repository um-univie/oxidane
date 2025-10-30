"""RDKit's MACCS keys, for calibrating ours against.

Reads SMILES on stdin, writes `smiles<TAB>bit,bit,...` with the set bits in
order.
"""
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as descriptors

RDLogger.DisableLog("rdApp.*")

for line in sys.stdin:
    smiles = line.split("\t")[0].strip()
    if not smiles:
        continue
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        continue
    keys = descriptors.GetMACCSKeysFingerprint(molecule)
    aromatic = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
    print(f"{smiles}\t{','.join(str(bit) for bit in keys.GetOnBits())}\t{aromatic}")
