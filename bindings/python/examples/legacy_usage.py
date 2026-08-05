"""Original molecules-py usage example, with fixture paths made location-independent."""
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests"

from molecules import MolParser, PyMolecule

mol = PyMolecule("CCO")
print(mol.to_smiles())
mol2 = mol.canonize_mol()
print(mol2.to_smiles())
# You can also canonize the molecule in place for better performance (although negligible for small molecules)
mol.canonize()
print(mol.to_smiles())


mol = PyMolecule("C1=CC=CC=C1")
# You can also canonize the molecule in place for better performance (although negligible for small molecules)
mol.canonize()
print(mol.to_smiles())


mol_parser = MolParser()
# XYZ files can have multiple molecules so we can't just construct a single molecule from it
mols = mol_parser.parse_file(FIXTURES / "cyclohexane.xyz")
for mol in mols:
    mol.canonize()
    print(mol.to_smiles())

# SDF files can have multiple molecules so we can't just construct a single molecule from it
mols = mol_parser.parse_file(FIXTURES / "pentene.sdf")
for mol in mols:
    mol.canonize()
    print(mol.to_smiles())
