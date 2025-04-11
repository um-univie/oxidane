"""Derive TB (@TB1..20) and SP (@SP1..3) coset tables from RDKit ground truth,
mirroring the octahedral derivation: a vertex-assignment table per class plus the
descriptor's symmetry group as vertex permutations, so a runtime remap is a coset
lookup.
"""
from rdkit import Chem
from itertools import permutations

def derive(tag, n_ligands, n_classes, ligands):
    def smi(order, cls):
        l = [ligands[i] for i in order]
        inner = "".join(f"({x})" for x in l[1:-1])
        return f"{l[0]}[{CENTER}@{tag}{cls}]{inner}{l[-1]}"
    def canon(s):
        m = Chem.MolFromSmiles(s)
        assert m is not None, s
        return Chem.MolToSmiles(m)

    ref = canon(smi(tuple(range(n_ligands)), 1))

    # vertex table: for each class, the ligand permutation P (in SMILES position
    # order) that reproduces the class-1 reference; store P as the vertex map.
    tables = {}
    for cls in range(1, n_classes + 1):
        found = None
        for P in permutations(range(n_ligands)):
            if canon(smi(P, cls)) == ref:
                found = list(P)
                break
        assert found, (tag, cls)
        tables[cls] = found

    # symmetry group: position permutations that preserve class 1.
    group = []
    for P in permutations(range(n_ligands)):
        if canon(smi(P, 1)) == ref:
            group.append(list(P))

    # Convert the vertex tables + group into the (vertex-table, rotations) form
    # used by the octahedral coset lookup. Here "vertex" = reference position.
    # tables[cls][pos] = vertex occupied by the ligand at SMILES position `pos`.
    # The group elements are position perms; express them as vertex perms via the
    # identity frame (class 1 = identity vertex map).
    rotations = group  # already vertex perms since class 1 is identity

    # sanity: cosets distinct
    def coset_key(vmap):
        best = None
        for r in rotations:
            cand = tuple(r[v] for v in vmap)
            best = cand if best is None or cand < best else best
        return best
    keys = {}
    for cls, vm in tables.items():
        k = coset_key(vm)
        assert k not in keys, (tag, cls, keys[k])
        keys[k] = cls
    print(f"// {tag}: {n_classes} classes, group order {len(rotations)}, all cosets distinct")

    # runtime remap validation
    def remap(config, old, new):
        vm = tables[config]
        q = [vm[old.index(a)] for a in new]
        for cls, vmc in tables.items():
            for r in rotations:
                if all(r[q[j]] == vmc[j] for j in range(n_ligands)):
                    return cls
        raise RuntimeError("no match")

    import random
    random.seed(1)
    for _ in range(300):
        c = random.randint(1, n_classes)
        old = list(range(n_ligands))
        new = list(range(n_ligands)); random.shuffle(new)
        m = remap(c, old, new)
        assert canon(smi(tuple(old), c)) == canon(smi(tuple(new), m)), (tag, c, new, m)

    # emit Rust
    print(f"const {tag}_VERTEX_TABLE: [[u8; {n_ligands}]; {n_classes}] = [")
    for cls in range(1, n_classes + 1):
        print(f"    {tables[cls]}, // {tag}{cls}")
    print("];")
    print(f"const {tag}_ROTATIONS: [[u8; {n_ligands}]; {len(rotations)}] = [")
    for r in rotations:
        print(f"    {r},")
    print("];")
    print(f"validated {tag} on 300 random remaps vs RDKit\n")

CENTER = "As"
derive("TB", 5, 20, ["F", "Cl", "Br", "I", "N"])
CENTER = "Pt"
derive("SP", 4, 3, ["F", "Cl", "Br", "I"])
