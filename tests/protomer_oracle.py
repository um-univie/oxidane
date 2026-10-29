#!/usr/bin/env python3
"""Compares this crate's ionizable-endpoint keys against standard InChI.

`TautomerRules::with_ionizable_endpoints` says that two structures differing
only in which of a molecule's acidic or basic sites holds a proton are one
compound. Standard InChI says the same -- its mobile-H layer moves protons
within a tautomeric system, and its normalization moves them between one and a
charged site -- so InChI is the reference the wider relation can be measured
against, where RDKit's tautomer hash is the one for the narrower.

The corpus is built from the SMILES given plus, for each, the forms RDKit's
tautomer enumerator generates and the neutral form its `Uncharger` produces.
The last of those is what puts zwitterion/neutral pairs in front of the
comparison at all: a corpus of deposited structures holds few of them.

    python3 tests/protomer_oracle.py --corpus tests/smiles.txt
    python3 tests/protomer_oracle.py --corpus <path> --limit 2000 --show 5
"""

import argparse
import pathlib
import subprocess
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def structures(corpus, cap):
    """The corpus, its tautomers, and the neutral form of each."""
    found = set()
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(cap)
    uncharger = rdMolStandardize.Uncharger()
    for smiles in corpus:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        forms = [molecule]
        try:
            forms.extend(enumerator.Enumerate(molecule))
        except Exception:
            pass
        for form in list(forms):
            try:
                forms.append(uncharger.uncharge(Chem.Mol(form)))
            except Exception:
                continue
        for form in forms:
            try:
                found.add(Chem.MolToSmiles(form))
            except Exception:
                continue
    return sorted(found)


def inchis(smiles_list):
    """Standard InChI for each structure, by SMILES."""
    result = {}
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        try:
            identifier = Chem.MolToInchi(molecule)
        except Exception:
            continue
        if identifier:
            result[smiles] = identifier
    return result


def keys(smiles_list, ionizable):
    """The crate's key for each structure, by SMILES."""
    command = ["cargo", "run", "--release", "--quiet", "--example", "tautomer_keys"]
    if ionizable:
        command.extend(["--", "--ionizable-endpoints"])
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


def partition(labels):
    groups = {}
    for member, label in labels.items():
        groups.setdefault(label, []).append(member)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tests/smiles.txt")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--max-tautomers", type=int, default=16)
    parser.add_argument("--show", type=int, default=4)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="use the default rules, which do not merge protomers, so that what"
        " admitting every heteroatom buys can be read off the difference",
    )
    arguments = parser.parse_args()

    corpus = (ROOT / arguments.corpus).read_text().split()
    corpus = [smiles for smiles in corpus if smiles][: arguments.limit]

    everything = structures(corpus, arguments.max_tautomers)
    reference = inchis(everything)
    ours = keys(everything, not arguments.plain)
    shared = sorted(set(reference) & set(ours))
    reference = {smiles: reference[smiles] for smiles in shared}
    ours = {smiles: ours[smiles] for smiles in shared}

    split = [
        distinct
        for members in partition(reference).values()
        for distinct in [partition({member: ours[member] for member in members})]
        if len(distinct) > 1
    ]
    merged = [
        distinct
        for members in partition(ours).values()
        for distinct in [partition({member: reference[member] for member in members})]
        if len(distinct) > 1
    ]

    print(f"{len(shared)} structures, {len(partition(reference))} InChI classes")
    print(f"split by the crate:  {len(split)}")
    print(f"merged by the crate: {len(merged)}")
    for title, findings in (("split", split), ("merged", merged)):
        for finding in findings[: arguments.show]:
            print(f"  {title}:")
            for label, members in sorted(finding.items()):
                print(f"    {label}")
                for member in members[:3]:
                    print(f"      {member}")
    return 1 if split or merged else 0


if __name__ == "__main__":
    sys.exit(main())
