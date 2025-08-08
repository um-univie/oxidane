"""Can a toolkit tell stereoisomers apart?

For each stereo class SMILES defines, this enumerates the *complete* set of
distinct stereoisomers of one centre and asks each toolkit to canonicalize
them. A toolkit that maps two distinct isomers onto one string has collapsed
them: downstream, the two compounds become one entry and nobody is told.

The isomer counts are not arbitrary. With k distinct ligands on a centre whose
rotation group has order r, there are k!/r distinct arrangements:

    tetrahedral   4!/12  =  2     (@, @@)
    allene        4!/12  =  2
    square planar 4!/8   =  3     (@SP1..3)
    trigonal bip. 5!/6   = 20     (@TB1..20)
    octahedral    6!/24  = 30     (@OH1..30)

which is exactly the range each SMILES descriptor index runs over. So writing
the same ligand sequence with every descriptor index enumerates every isomer
exactly once, and the number of distinct canonical strings a toolkit produces
should equal the number of isomers.

Two failure directions are reported separately, because they are different
bugs: *collapsed* means distinct isomers share a canonical form (false
identity); *split* means one isomer canonicalizes two ways depending on how it
was written (false difference), which is checked by re-canonicalizing the
canonical form.

Usage:
    python3 tests/stereo_benchmark.py [path-to-canon_stdin]
"""

import subprocess
import sys
from collections import OrderedDict

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

OXIDANE = sys.argv[1] if len(sys.argv) > 1 else "target/release/examples/canon_stdin"


def isomer_set(name):
    """Every distinct stereoisomer of one centre, as SMILES."""
    if name == "tetrahedral":
        return [f"[C{tag}](F)(Cl)(Br)I" for tag in ("@", "@@")]
    if name == "allene":
        return [f"FC(Cl)=[C{tag}]=C(Br)I" for tag in ("@", "@@")]
    if name == "cis/trans":
        return ["F/C=C/F", "F/C=C\\F"]
    if name == "square planar":
        return [f"[Pt@SP{k}](F)(Cl)(Br)I" for k in range(1, 4)]
    if name == "trigonal bipyramidal":
        return [f"[P@TB{k}](F)(Cl)(Br)I" for k in range(1, 21)]
    if name == "octahedral":
        return [
            f"[Co@OH{k}]([SH])([Br])([I])([CH3])([F])[Cl]" for k in range(1, 31)
        ]
    raise ValueError(name)


CLASSES = [
    "tetrahedral",
    "allene",
    "cis/trans",
    "square planar",
    "trigonal bipyramidal",
    "octahedral",
]


def rdkit_smiles(smiles_list):
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToSmiles(m) if m is not None else None)
    return out


def rdkit_inchi(smiles_list):
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            out.append(None)
            continue
        try:
            inchi = Chem.MolToInchi(m)
        except Exception:
            inchi = None
        out.append(inchi or None)
    return out


def oxidane_smiles(smiles_list):
    payload = "\n".join(smiles_list) + "\n"
    result = subprocess.run(
        [OXIDANE], input=payload, capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    if len(lines) != len(smiles_list):
        return [None] * len(smiles_list)
    return [None if l.startswith(("ERROR", "PANIC")) else l for l in lines]


TOOLKITS = OrderedDict(
    [
        ("RDKit SMILES", rdkit_smiles),
        ("RDKit InChI", rdkit_inchi),
        ("oxidane", oxidane_smiles),
    ]
)


def main():
    print(f"{'stereo class':22} {'isomers':>8}", end="")
    for name in TOOLKITS:
        print(f" {name:>16}", end="")
    print()
    print("-" * (22 + 8 + 17 * len(TOOLKITS)))

    totals = {name: [0, 0] for name in TOOLKITS}  # [distinguished, total]

    for cls in CLASSES:
        isomers = isomer_set(cls)
        print(f"{cls:22} {len(isomers):>8}", end="")
        for name, fn in TOOLKITS.items():
            canonical = fn(isomers)
            usable = [c for c in canonical if c is not None]
            distinct = len(set(usable))
            if len(usable) < len(isomers):
                cell = f"{distinct}/{len(isomers)} (+{len(isomers)-len(usable)} err)"
            else:
                cell = f"{distinct}/{len(isomers)}"
            totals[name][0] += distinct
            totals[name][1] += len(isomers)
            print(f" {cell:>16}", end="")
        print()

    print("-" * (22 + 8 + 17 * len(TOOLKITS)))
    print(f"{'total':22} {sum(len(isomer_set(c)) for c in CLASSES):>8}", end="")
    for name in TOOLKITS:
        distinguished, total = totals[name]
        print(f" {f'{distinguished}/{total}':>16}", end="")
    print()

    # Idempotence: a canonical form must canonicalize to itself.
    print("\nre-canonicalizing each canonical form (idempotence):")
    for cls in CLASSES:
        isomers = isomer_set(cls)
        for name, fn in TOOLKITS.items():
            if name == "RDKit InChI":
                continue  # InChI is not SMILES; nothing to feed back
            once = [c for c in fn(isomers) if c is not None]
            twice = fn(once)
            unstable = sum(
                1 for a, b in zip(once, twice) if b is not None and a != b
            )
            if unstable:
                print(f"  {cls:22} {name:14} {unstable} of {len(once)} not stable")
    print("  (nothing listed above means every canonical form was stable)")


if __name__ == "__main__":
    main()
