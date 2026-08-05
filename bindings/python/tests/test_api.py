from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import oxidane as ox
import molecules

FIXTURES = Path(__file__).parent


def test_original_api_and_import_names():
    assert molecules.PyMolecule is ox.Molecule is ox.PyMolecule
    assert molecules.PyMolecule3D is ox.Molecule3D
    assert molecules.MolParser is ox.MolParser
    mol = molecules.PyMolecule("OCC")
    original = mol.to_smiles()
    canonical = mol.canonize_mol()
    assert isinstance(canonical, ox.Molecule)
    assert mol.to_smiles() == original
    assert canonical.to_smiles() == mol.to_canonical_smiles()
    mol.canonize()
    assert mol.to_smiles() == canonical.to_smiles()
    mol.canonize()
    assert mol.to_smiles() == canonical.to_smiles()


@pytest.mark.parametrize("filename", ["pentene.sdf", "cyclohexane.xyz"])
def test_legacy_file_parsers_and_3d_methods(filename):
    mols = ox.MolParser().parse_file(FIXTURES / filename)
    assert mols
    if filename.endswith("xyz"):
        assert len(mols) == 2  # Upstream's blank-separated, headerless second frame.
    for mol in mols:
        assert isinstance(mol, ox.Molecule3D)
        before = mol.to_smiles()
        clone = mol.canonize_mol()
        assert isinstance(clone, ox.Molecule3D)
        assert mol.to_smiles() == before
        assert mol.contains_smarts("C")
        assert mol.contains(ox.Smarts("C"))
        assert not mol.contains("N")
        assert mol.to_graph().contains("C")
        ET.fromstring(mol.to_svg())
        assert mol.to_smiles() == before
        mol.canonize()
        assert mol.to_smiles() == clone.to_smiles()


def test_multi_frame_xyz_and_file_errors(tmp_path):
    path = tmp_path / "frames.XYZ"
    path.write_text("1\nhelium\nHe 0 0 0\n1\nneon\nNe 0 0 0\n")
    mols = ox.MolParser().parse_file(path)
    assert len(mols) == 2
    assert mols[0].contains("[He]")
    assert not mols[0].contains("[Ne]")
    assert mols[1].contains("[Ne]")
    for suffix in ["xyz", "sdf"]:
        with pytest.raises(OSError):
            ox.MolParser().parse_file(tmp_path / f"missing.{suffix}")
    with pytest.raises(ValueError):
        ox.MolParser().parse_file("file.unknown")
    for invalid in ["1\nmissing atom\n", "two\n", "1\n\nXxx 0 0 0\n",
                    "1\n\nC nan 0 0\n", "1\n\nC bad 0 0\n"]:
        path.write_text(invalid)
        with pytest.raises(ValueError):
            ox.MolParser().parse_file(path)


def test_contains_semantics_and_invalid_queries():
    ethanol = ox.Molecule("CCO")
    alcohol = ox.Smarts("[OX2H1][CX4]")
    assert ethanol.contains(alcohol)
    assert ethanol.contains_smarts("[OX2H1]")
    assert not ox.Molecule("CC(=O)C").contains(alcohol)
    assert ox.Molecule("[NH4+].[Cl-]").contains("[N+].[Cl-]")
    assert ox.Molecule("c1ccccc1").contains("c1ccccc1")
    assert ox.Molecule("CC(=O)O").contains("[$([CX3](=O)[OX2H1])]")
    isotope = ox.Molecule("[13CH3][C@H](F)O")
    assert isotope.contains("[13C][C@H](F)O")
    assert not isotope.contains("[13C][C@@H](F)O")
    assert not isotope.contains("[14C]")
    spatial = ox.MolParser().parse_file(FIXTURES / "pentene.sdf")[0]
    for source in ["", "[", "C1CC", "C(C", "C="]:
        with pytest.raises(ValueError):
            ox.Smarts(source)
        for mol in [ethanol, spatial]:
            with pytest.raises(ValueError):
                mol.contains_smarts(source)
            with pytest.raises(ValueError):
                mol.contains(source)
    with pytest.raises(TypeError):
        ethanol.contains(42)
    with pytest.raises(ValueError):
        ox.Molecule("not a smiles")


def test_reaction_rewriting_and_diagrams():
    mol = ox.Molecule("CCO")
    rule = ox.ReactionSmarts("[C:1]-[O:2]>>[C:1]=[O:2]")
    products = rule.apply(mol)
    assert any(product.contains("C=O") for product in products)
    assert mol.contains("[OX2H1]")
    assert len(rule.apply_to([mol])) == len(products)
    assert len(ET.fromstring(rule.to_svg(reactants=[mol])).findall("{*}svg")) == len(products)
    assert "No matching" in rule.to_svg(reactants=[ox.Molecule("N")])
    query_text = "".join(ET.fromstring(rule.to_svg()).itertext())
    assert "[C:1]-[O:2]" in query_text
    reaction = ox.Reaction(["CCO", "O"], products, name='<script>&" example')
    root = ET.fromstring(reaction.to_svg())
    assert root.find("{*}title").text == '<script>&" example'
    assert root.find(".//{*}script") is None
    assert len(root.findall("{*}svg")) == 2 + len(products)
    assert "∅" in ox.Reaction([], ["O"]).to_svg()
    assert "∅" in ox.Reaction(["O"], []).to_svg()
    with pytest.raises(TypeError):
        ox.Reaction("CCO", ["CC=O"])
    with pytest.raises(ValueError):
        ox.ReactionSmarts("not a reaction")


@pytest.mark.parametrize("obj", [ox.Molecule("CCO"), ox.Reaction(["CCO"], ["CC=O"]),
                                 ox.ReactionSmarts("[C:1]-[O:2]>>[C:1]=[O:2]")])
def test_svg_dimensions(obj):
    root = ET.fromstring(obj.to_svg(width=600, height=200))
    assert root.get("width") == "600"
    assert root.get("height") == "200"
    with pytest.raises(ValueError):
        obj.to_svg(width=0)
    with pytest.raises(OverflowError):
        obj.to_svg(width=-1)
