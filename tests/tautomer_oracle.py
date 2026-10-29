#!/usr/bin/env python3
"""Compares this crate's tautomer keys against RDKit's tautomer hashes.

Both sides answer the same question -- which structures are the same compound
once a proton is allowed to move -- so the thing to compare is the *partition*
each induces over a corpus, not any one string. Two structures RDKit gives one
hash must get one key from the crate, and two it separates must stay separated.

RDKit's two hashes line up with the crate's two rule sets:

    HetAtomTautomer     heteroatom endpoints only     TautomerRules::new()
    HetAtomTautomerv2   carbon endpoints as well      with_carbon_endpoints()

Despite the shared name they are not the same relation: v2 merges acetone with
its enol and v1 does not.

The corpus alone holds few structures that are tautomers of anything, so
RDKit's enumerator is used to populate the classes -- it decides which
structures get compared, and the hashes decide what the answer should be.

There is a second question the partitions cannot answer: given that two
structures are the same compound, which one gets *stored*. `--representative`
compares `normalize`'s answer with RDKit's canonical tautomer. Disagreement
there is not a correctness failure -- both are conventions, and uniqueness is
what this crate promises -- but it is worth watching, because the ways they
differ are informative.

    python3 tests/tautomer_oracle.py                     # heteroatom endpoints
    python3 tests/tautomer_oracle.py --limit 500
    python3 tests/tautomer_oracle.py --carbon-endpoints  # keto/enol merged too
    python3 tests/tautomer_oracle.py --representative    # which form gets stored
"""

import argparse
import json
import pathlib
import subprocess
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolHash
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def structures(corpus, cap):
    """The corpus plus every tautomer RDKit generates from it."""
    found = set()
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(cap)
    for smiles in corpus:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        found.add(Chem.MolToSmiles(molecule))
        try:
            for tautomer in enumerator.Enumerate(molecule):
                found.add(Chem.MolToSmiles(tautomer))
        except Exception:
            continue
    return sorted(found)


def hashes(smiles_list, carbon_endpoints):
    """RDKit's tautomer hash for each structure, by SMILES."""
    function = (
        rdMolHash.HashFunction.HetAtomTautomerv2
        if carbon_endpoints
        else rdMolHash.HashFunction.HetAtomTautomer
    )
    result = {}
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        try:
            result[smiles] = rdMolHash.MolHash(molecule, function)
        except Exception:
            continue
    return result


def keys(smiles_list, carbon_endpoints):
    """The crate's tautomer key for each structure, by SMILES."""
    command = ["cargo", "run", "--release", "--quiet", "--example", "tautomer_keys"]
    if carbon_endpoints:
        command.extend(["--", "--carbon-endpoints"])
    result = subprocess.run(
        command,
        cwd=ROOT,
        input="\n".join(smiles_list),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"the crate failed:\n{result.stderr}")
    return dict(
        line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line
    )


def representatives(smiles_list):
    """The crate's canonical tautomer for each structure, by SMILES."""
    result = subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "normalize_keys"],
        cwd=ROOT,
        input="\n".join(smiles_list),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"the crate failed:\n{result.stderr}")
    return dict(
        line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line
    )


def compare_representatives(smiles_list, show):
    """Reports how often the two canonical tautomers agree."""
    canonicalizer = rdMolStandardize.TautomerEnumerator()
    ours = representatives(smiles_list)
    agree, differ, examples = 0, 0, []
    for smiles in smiles_list:
        if smiles not in ours:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        mine = Chem.MolFromSmiles(ours[smiles])
        if molecule is None or mine is None:
            differ += 1
            continue
        try:
            theirs = Chem.MolToSmiles(canonicalizer.Canonicalize(molecule))
        except Exception:
            continue
        if Chem.MolToSmiles(mine) == theirs:
            agree += 1
        else:
            differ += 1
            if len(examples) < show:
                examples.append((smiles, Chem.MolToSmiles(mine), theirs))
    print(f"{agree + differ} compared: {agree} agree, {differ} differ")
    for smiles, mine, theirs in examples:
        print(f"  {smiles}\n    ours  {mine}\n    rdkit {theirs}")
    return 0


def partition(labels):
    """Members grouped by label."""
    groups = {}
    for member, label in labels.items():
        groups.setdefault(label, []).append(member)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tests/smiles.txt")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--max-tautomers", type=int, default=16)
    parser.add_argument("--carbon-endpoints", action="store_true")
    parser.add_argument(
        "--representative",
        action="store_true",
        help="compare which tautomer gets stored, rather than the partitions",
    )
    parser.add_argument("--show", type=int, default=6)
    parser.add_argument("--report", default=None, help="write findings as JSON")
    arguments = parser.parse_args()

    corpus = (ROOT / arguments.corpus).read_text().split()
    corpus = [smiles for smiles in corpus if smiles][: arguments.limit]

    if arguments.representative:
        parsed = [
            Chem.MolToSmiles(molecule)
            for molecule in (Chem.MolFromSmiles(smiles) for smiles in corpus)
            if molecule is not None
        ]
        return compare_representatives(sorted(set(parsed)), arguments.show)

    everything = structures(corpus, arguments.max_tautomers)
    reference = hashes(everything, arguments.carbon_endpoints)
    ours = keys(everything, arguments.carbon_endpoints)
    shared = sorted(set(reference) & set(ours))
    reference = {smiles: reference[smiles] for smiles in shared}
    ours = {smiles: ours[smiles] for smiles in shared}

    # A class RDKit merges that the crate splits: structures RDKit calls the
    # same compound coming back with more than one key.
    split = []
    for members in partition(reference).values():
        distinct = partition({member: ours[member] for member in members})
        if len(distinct) > 1:
            split.append(distinct)

    # The other direction: one key covering structures RDKit keeps apart.
    merged = []
    for members in partition(ours).values():
        distinct = partition({member: reference[member] for member in members})
        if len(distinct) > 1:
            merged.append(distinct)

    classes = len(partition(reference))
    print(f"{len(shared)} structures, {classes} RDKit classes")
    print(f"split by the crate:  {len(split)}")
    print(f"merged by the crate: {len(merged)}")

    for title, findings in (("split", split), ("merged", merged)):
        for finding in findings[: arguments.show]:
            print(f"  {title}:")
            for label, members in sorted(finding.items()):
                print(f"    {label}")
                for member in members[:4]:
                    print(f"      {member}")
    if arguments.report:
        pathlib.Path(arguments.report).write_text(
            json.dumps({"split": split, "merged": merged}, indent=2)
        )
    return 1 if split or merged else 0


if __name__ == "__main__":
    sys.exit(main())
