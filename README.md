# Molecule

A library to work with molecules.

Features:

Efficient implementation of subgraph matching to find a mapping from Molecule A into Molecule B using the VF2 algorithm.
Can parse xyz and smiles into the internal molecular graph representation.
Determines connectivity automatically based on the molecular coordinates when parsing an xyz file.

To-Do:
Implement parsing of planned additional formats:
    InChi
    mol
    sdf
    pdb
Remove Openbabel in favor of a pure Rust canonicalisation algorithm.


    


