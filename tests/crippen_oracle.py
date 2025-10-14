"""Per-atom Wildman-Crippen contributions from RDKit, for calibrating ours.

RDKit will not hand back the atom type labels through its Python wrapper, so
they are recovered from the (log P, MR) pair, which the shipped parameter file
makes very nearly unique. Labels are sorted before printing: the two sides then
need not agree on atom order for their type counts to be compared.

Reads SMILES on stdin, writes `smiles<TAB>labels<TAB>logp<TAB>mr`.
"""
import os
import sys
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import rdMolDescriptors as descriptors

RDLogger.DisableLog("rdApp.*")

BY_VALUE = {}
with open(os.path.join(RDConfig.RDDataDir, "Crippen.txt")) as handle:
    for row in handle:
        if row.startswith("#") or not row.strip():
            continue
        fields = row.split("\t")
        if len(fields) < 4:
            continue
        name, logp, refractivity = fields[0], fields[2].strip(), fields[3].strip()
        key = (round(float(logp), 4), round(float(refractivity or 0.0), 4))
        if key in BY_VALUE and BY_VALUE[key] != name:
            print(f"# {key} is both {BY_VALUE[key]} and {name}", file=sys.stderr)
        BY_VALUE[key] = name

for line in sys.stdin:
    smiles = line.split("\t")[0].strip()
    if not smiles:
        continue
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        continue
    molecule = Chem.AddHs(molecule)
    contributions = descriptors._CalcCrippenContribs(molecule)
    labels = [
        BY_VALUE.get((round(logp, 4), round(refractivity, 4)), f"?{logp}/{refractivity}")
        for logp, refractivity in contributions
    ]
    logp = sum(pair[0] for pair in contributions)
    refractivity = sum(pair[1] for pair in contributions)
    aromatic = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
    print(
        f"{smiles}\t{','.join(sorted(labels))}\t{logp:.4f}\t{refractivity:.4f}\t{aromatic}"
    )
