"""Classify invariance failures: chirality bug vs base-canonization nondeterminism.

For each skeleton, feed N random writings to the Rust canonizer. If the outputs
disagree, re-parse each distinct output with RDKit and compare RDKit canonical
SMILES:
  - all RDKit-equal  -> BASE nondeterminism (same molecule, different valid order)
  - RDKit differ     -> CHIRALITY bug (stereo actually changed)
Stereo on sp2 `@` tags is invisible to RDKit, so cis/trans is checked with sp3
awareness noted separately.
"""
import subprocess, random
from rdkit import Chem
random.seed(20260709)
BIN = "target/release/examples/canon_stdin"

def rust_canon(smiles_list):
    p = subprocess.run([BIN], input="\n".join(smiles_list) + "\n",
                       capture_output=True, text=True, timeout=600)
    out = p.stdout.split("\n")
    while out and out[-1] == "":
        out.pop()
    return out

def writings(smiles, n):
    m = Chem.MolFromSmiles(smiles)
    outs = set()
    t = 0
    while len(outs) < n and t < n*8:
        t += 1
        outs.add(Chem.MolToSmiles(m, doRandom=True, canonical=False))
    return list(outs)

def rdkit_key(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else f"UNPARSEABLE({s})"

SKELETONS = [
    ("alanine","N[C@@H](C)C(=O)O"),("cysteine","C([C@@H](C(=O)O)N)S"),
    ("threonine","C[C@@H](O)[C@@H](N)C(=O)O"),("isoleucine","CC[C@H](C)[C@@H](N)C(=O)O"),
    ("tartaric","OC(=O)[C@H](O)[C@@H](O)C(=O)O"),("glucose","OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("menthol","CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"),("camphor","CC1(C)[C@@H]2CC[C@]1(C)C(=O)C2"),
    ("2-butanol","C[C@H](O)CC"),("chlorofluorobromo","F[C@H](Cl)Br"),
    ("difluoroethene_E","F/C=C/F"),("difluoroethene_Z","F/C=C\\F"),("butene_2E","C/C=C/C"),
    ("allene","CC(Br)=[C@]=C(O)C"),("tetrahedral_quat","C[C@](N)(O)F"),
    ("bromochloro_ring","Cl[C@H]1CC[C@@H](Br)CC1"),("epoxide","C[C@H]1O[C@@H]1C"),
    ("proline","OC(=O)[C@@H]1CCCN1"),("two_centers_far","N[C@@H](C)CCCCCC[C@@H](O)C"),
    ("sugar_like","OC[C@@H](O)[C@@H](O)[C@H](O)CO"),("bicyclic","C[C@H]1CC[C@@H]2CCCC[C@@H]2C1"),
    ("phenyl_chiral","c1ccccc1[C@H](N)C(=O)O"),("multi_ring_stereo","O[C@H]1CC[C@]2(C)CCCC[C@@H]2C1"),
]

chir, base, ok = [], [], []
for name, smi in SKELETONS:
    ws = writings(smi, 25)
    if not ws:
        continue
    outs = rust_canon(ws)
    uniq = sorted(set(outs))
    if len(uniq) == 1 and not outs[0].startswith(("ERROR","PANIC")):
        ok.append(name); continue
    keys = sorted(set(rdkit_key(u) for u in uniq))
    if len(keys) == 1:
        base.append((name, smi, uniq, keys))
    else:
        chir.append((name, smi, uniq, keys))

print(f"STABLE: {len(ok)}/{len(SKELETONS)}  ->", ", ".join(ok))
print(f"\nCHIRALITY BUGS ({len(chir)}): our outputs are genuinely different stereoisomers")
for name, smi, uniq, keys in chir:
    print(f"  {name}  ({smi})")
    for u in uniq: print(f"      {u}")
print(f"\nBASE NONDETERMINISM ({len(base)}): same molecule, different valid atom order")
for name, smi, uniq, keys in base:
    print(f"  {name}  ({smi})  RDKit-> {keys[0]}")
    for u in uniq: print(f"      {u}")
