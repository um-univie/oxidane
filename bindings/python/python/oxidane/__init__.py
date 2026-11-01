"""Oxidane molecules, SMARTS queries, reactions, and notebook display."""

from ._oxidane import (
    MolParser,
    PyMolecule,
    PyMolecule3D,
    ReactionSmarts,
    Smarts,
    canonical_smiles,
    canonical_tautomer,
    tautomer_key,
)
from ._display import Reaction

Molecule = PyMolecule
Molecule3D = PyMolecule3D
__version__ = "0.2.0"
__all__ = [
    "Molecule", "Molecule3D", "PyMolecule", "PyMolecule3D", "MolParser",
    "Smarts", "Reaction", "ReactionSmarts",
    "canonical_smiles", "canonical_tautomer", "tautomer_key",
]
