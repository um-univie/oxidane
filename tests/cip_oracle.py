"""Build a stereochemistry-rich corpus from drug-like skeletons.

Reads SMILES on stdin, enumerates stereoisomers of each (capped), and for
every one emits `smiles<TAB>idx:label;idx:label` using RDKit's *modern* CIP
labeller (`rdCIPLabeler`, the port of Hanson/Mayfield `centres`), which is
the implementation validated against Hanson's reference suite.

R/S live on atoms; E/Z live on *bonds* in RDKit, but on both endpoint atoms
in our representation, so a bond's descriptor is emitted against both of its
atom indices.
"""

import sys
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

opts = StereoEnumerationOptions(maxIsomers=32, onlyUnassigned=True, unique=True)


def labels_of(mol):
    rdCIPLabeler.AssignCIPLabels(mol)
    out = {}
    for a in mol.GetAtoms():
        code = a.GetPropsAsDict().get("_CIPCode")
        if code:
            out[a.GetIdx()] = code
    for b in mol.GetBonds():
        code = b.GetPropsAsDict().get("_CIPCode")
        if code:
            out[b.GetBeginAtomIdx()] = code
            out[b.GetEndAtomIdx()] = code
    return out


seen = set()
for line in sys.stdin:
    smi = line.strip().split()[0] if line.strip() else ""
    if not smi:
        continue
    base = Chem.MolFromSmiles(smi)
    if base is None:
        continue
    try:
        isomers = list(EnumerateStereoisomers(base, options=opts))
    except Exception:
        continue
    for iso in isomers:
        out = Chem.MolToSmiles(iso, isomericSmiles=True)
        if out in seen:
            continue
        seen.add(out)
        mol = Chem.MolFromSmiles(out)
        if mol is None:
            continue
        labels = labels_of(mol)
        print(f"{out}\t" + ";".join(f"{i}:{c}" for i, c in sorted(labels.items())))
