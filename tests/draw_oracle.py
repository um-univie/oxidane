#!/usr/bin/env python3
"""Independent RDKit check of 2D coordinates and wedge directions.

Run: cargo test --lib rdkit_depiction_oracle -- --ignored --nocapture
Requires RDKit in python3. MOL blocks arrive on stdin from the Rust test;
the first line of each block is the original SMILES.
"""
import sys
from rdkit import Chem

failures = []
blocks = [block for block in sys.stdin.read().split("$$$$\n") if block.strip()]
for block in blocks:
    smiles = block.splitlines()[0]
    expected = Chem.MolFromSmiles(smiles)
    actual = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
    want = Chem.MolToSmiles(expected, isomericSmiles=True)
    got = Chem.MolToSmiles(actual, isomericSmiles=True) if actual else "INVALID"
    if want != got:
        failures.append(f"{smiles}: expected {want}; drawing gives {got}")
if failures:
    sys.exit("\n".join(failures))
print(f"RDKit agrees with all {len(blocks)} stereo drawings")
