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


def test_tautomer_keys_and_canonical_tautomers():
    # 2-pyridone and 2-hydroxypyridine are one compound.
    assert ox.tautomer_key("O=c1cccc[nH]1") == ox.tautomer_key("Oc1ccccn1")
    assert ox.canonical_tautomer("O=c1cccc[nH]1") == ox.canonical_tautomer("Oc1ccccn1")
    assert ox.canonical_tautomer("CC(O)=N") == "O=C(N)C"
    # Nothing mobile: the key is the canonical SMILES, with no marker.
    assert ox.tautomer_key("c1ccccc1") == "c1ccccc1"
    # Keto and enol only merge with carbon endpoints, which the key says.
    assert ox.tautomer_key("CC(C)=O") != ox.tautomer_key("CC(O)=C")
    carbon = ox.tautomer_key(["CC(C)=O", "CC(O)=C"], carbon_endpoints=True)
    assert carbon[0] == carbon[1] and carbon[0].startswith("c/")
    # The default key moves protons between ionizable sites; the narrower key
    # without them keeps a zwitterion apart and says so.
    assert ox.tautomer_key("[NH3+]CC(=O)[O-]") == ox.tautomer_key("NCC(=O)O")
    narrow = ox.tautomer_key(["[NH3+]CC(=O)[O-]", "NCC(=O)O"], ionizable_endpoints=False)
    assert narrow[0] != narrow[1] and narrow[0].startswith("h/")
    mol = ox.Molecule("Oc1ccccn1")
    assert mol.tautomer_key() == ox.tautomer_key("Oc1ccccn1")
    assert mol.canonical_tautomer() == ox.canonical_tautomer("Oc1ccccn1")


def test_batches_match_single_calls_in_order():
    smiles = ["OCC", "[Na+].[Cl-]", "O=c1cccc[nH]1", "CC(O)=N", "c1ccccc1O"] * 40
    for function in (ox.canonical_smiles, ox.tautomer_key, ox.canonical_tautomer):
        expected = [function(s) for s in smiles]
        assert function(smiles) == expected
        assert function(tuple(smiles), threads=1) == expected
        assert function(iter(smiles), threads=3) == expected
    ethanol, salt = ox.canonical_smiles(["CCO", "[Na+].[Cl-]"])
    assert ethanol == ox.canonical_smiles("OCC")
    assert salt == ox.canonical_smiles("[Cl-].[Na+]")
    assert ox.canonical_smiles([]) == []


def test_batch_errors(capsys):
    # By default a bad SMILES comes back unchanged and is reported on stderr.
    ethanol = ox.canonical_smiles("OCC")
    assert ox.canonical_smiles(["OCC", "C1CC"]) == [ethanol, "C1CC"]
    assert 'cannot read SMILES "C1CC" at index 1' in capsys.readouterr().err
    assert ox.tautomer_key("C1CC") == "C1CC"
    assert 'cannot read SMILES "C1CC"' in capsys.readouterr().err
    with pytest.raises(ValueError, match="index 1"):
        ox.canonical_smiles(["CCO", "C1CC"], errors="raise")
    with pytest.raises(ValueError):
        ox.tautomer_key("C1CC", errors="raise")
    answers = ox.canonical_tautomer(["CCO", "C1CC"], errors="none")
    assert answers == [ox.canonical_tautomer("CCO"), None]
    assert ox.canonical_smiles("C1CC", errors="none") is None
    assert capsys.readouterr().err == ""
    with pytest.raises(ValueError):
        ox.canonical_smiles(["CCO"], errors="ignore")
    with pytest.raises(ValueError):
        ox.canonical_smiles(["CCO"], threads=0)
    with pytest.raises(TypeError):
        ox.canonical_smiles(["CCO", 3])
    with pytest.raises(TypeError):
        ox.canonical_smiles(42)


def test_process_smi_file(tmp_path, capsys):
    source = tmp_path / "in.smi"
    source.write_text("Oc1ccccn1 hydroxypyridine\nO=c1cccc[nH]1\tpyridone\n\nC1CC broken\n")
    counts = ox.process_file(source, tmp_path / "out.tsv")
    assert counts == {"rows": 3, "unparsed": 1}
    assert "line 4" in capsys.readouterr().err
    rows = [line.split("\t") for line in (tmp_path / "out.tsv").read_text().splitlines()]
    assert rows[0][:2] == ["Oc1ccccn1", "hydroxypyridine"]
    assert rows[0][2:] == [ox.canonical_smiles("Oc1ccccn1"), ox.tautomer_key("Oc1ccccn1"),
                           ox.canonical_tautomer("Oc1ccccn1")]
    assert rows[0][3:] == rows[1][3:]  # one compound, one key and one tautomer
    assert rows[2] == ["C1CC", "broken", "C1CC", "C1CC", "C1CC"]


def test_process_csv_with_header_and_gzip(tmp_path):
    import csv
    import gzip
    source = tmp_path / "in.csv"
    source.write_text('id,name,smiles\n1,"acetic acid, glacial",CC(=O)O\n2,imidic,CC(O)=N\n')
    output = tmp_path / "out.csv.gz"
    counts = ox.process_file(source, output, smiles_column="smiles",
                             columns=["tautomer_key"], threads=2, chunk_size=1)
    assert counts == {"rows": 2, "unparsed": 0}
    with gzip.open(output, "rt", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["id", "name", "smiles", "tautomer_key"]
    assert rows[1] == ["1", "acetic acid, glacial", "CC(=O)O", ox.tautomer_key("CC(=O)O")]
    assert rows[2][3] == ox.tautomer_key("CC(=O)N")
    with pytest.raises(ValueError, match="no column named"):
        ox.process_file(source, tmp_path / "x.csv", smiles_column="SMILES")
    with pytest.raises(ValueError, match="unknown columns"):
        ox.process_file(source, tmp_path / "x.csv", columns=["inchi"])
    with pytest.raises(ValueError, match="line 1"):
        ox.process_file(source, tmp_path / "x.csv", errors="raise")  # "id" read as SMILES


def test_unreadable_smiles_are_counted_after_ten(tmp_path, capsys):
    assert ox.canonical_smiles(["C1CC"] * 25 + ["CCO"])[-1] == ox.canonical_smiles("CCO")
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 11 and err[-1] == (
        "oxidane: 15 more SMILES could not be read and were kept as they were")
    source = tmp_path / "in.smi"
    source.write_text("C1CC\n" * 25)
    assert ox.process_file(source, tmp_path / "out.tsv", chunk_size=4)["unparsed"] == 25
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 11 and "line 10" in err[9] and "15 more" in err[-1]


def test_command_line(tmp_path, capsys):
    from oxidane.__main__ import main
    source = tmp_path / "in.csv"
    source.write_text("id,smiles\n1,Oc1ccccn1\n2,\n3,C1CC\n")
    assert main([str(source), str(tmp_path / "out.csv"), "-s", "smiles", "--errors", "none",
                 "-c", "tautomer_key,canonical_tautomer"]) == 0
    assert "3 rows, 1 unreadable" in capsys.readouterr().err
    import csv
    rows = list(csv.reader((tmp_path / "out.csv").open(newline="")))
    assert rows[0] == ["id", "smiles", "tautomer_key", "canonical_tautomer"]
    assert rows[1] == ["1", "Oc1ccccn1", ox.tautomer_key("Oc1ccccn1"),  # key has commas, quoted
                       ox.canonical_tautomer("Oc1ccccn1")]
    assert rows[2:] == [["2", "", "", ""], ["3", "C1CC", "", ""]]
    with pytest.raises(SystemExit):
        main([str(source), str(tmp_path / "x.csv"), "-s", "SMILES"])
    assert "no column named" in capsys.readouterr().err


def test_command_line_pipes():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "oxidane", "-", "-", "-c", "canonical_smiles"],
                            input="OCC 1\nC(=O)O 2\n", capture_output=True, text=True, check=True)
    ethanol, formic = ox.canonical_smiles(["OCC", "C(=O)O"])
    assert result.stdout == f"OCC\t1\t{ethanol}\nC(=O)O\t2\t{formic}\n"


def test_sqlite_example(tmp_path):
    import runpy
    import sqlite3
    example = runpy.run_path(str(FIXTURES.parent / "examples" / "normalize_sqlite.py"))
    path = tmp_path / "compounds.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE compounds (id INTEGER, smiles TEXT)")
        connection.executemany("INSERT INTO compounds VALUES (?, ?)",
                               [(1, "Oc1ccccn1"), (2, None), (3, "C1CC"), (4, "O=c1cccc[nH]1")])
    assert example["normalize_table"](path, "compounds", batch_size=3) == 4
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT * FROM compounds ORDER BY id").fetchall()
    assert rows[0][2:] == (ox.canonical_smiles("Oc1ccccn1"), ox.tautomer_key("Oc1ccccn1"),
                           ox.canonical_tautomer("Oc1ccccn1"))
    assert rows[1][2:] == rows[2][2:] == (None, None, None)
    assert rows[3][3:] == rows[0][3:]


def test_missing_values_and_whitespace():
    nan = float("nan")
    for function in (ox.canonical_smiles, ox.tautomer_key, ox.canonical_tautomer):
        ethanol = function("OCC")
        assert function(["OCC", None, nan, " OCC\n"]) == [ethanol, None, None, ethanol]
        assert function(None) is None
        assert function(nan) is None
        assert function("\tOCC ") == ethanol
    assert ox.Molecule(" OCC ").to_smiles() == ox.Molecule("OCC").to_smiles()


def test_pandas_columns_with_gaps():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame({"smiles": ["Oc1ccccn1", None, "O=c1cccc[nH]1"]})
    frame["tautomer"] = ox.canonical_tautomer(frame["smiles"])
    assert frame["tautomer"].isna().tolist() == [False, True, False]
    assert frame["tautomer"][0] == frame["tautomer"][2]
    typed = pandas.Series(["OCC", pandas.NA], dtype="string")
    assert ox.canonical_smiles(typed) == [ox.canonical_smiles("OCC"), None]
    assert frame["smiles"].map(ox.canonical_smiles).isna().tolist() == [False, True, False]


def test_molecule_methods_agree_with_the_functions():
    for smiles in ["OCC", "[Na+].[Cl-]", "CC(=O)[O-].[NH4+]", "Oc1ccccn1.O"]:
        molecule = ox.Molecule(smiles)
        assert molecule.to_canonical_smiles() == ox.canonical_smiles(smiles)
        assert molecule.tautomer_key() == ox.tautomer_key(smiles)
        assert molecule.canonical_tautomer() == ox.canonical_tautomer(smiles)


def test_threads_reuse_their_pool():
    smiles = ["OCC", "c1ccccc1O", "CC(O)=N"] * 10
    expected = ox.canonical_tautomer(smiles)
    for _ in range(50):
        assert ox.canonical_tautomer(smiles, threads=3) == expected


# Each row is one compound written several ways, in different atom orders,
# aromatic and Kekulé, and as different tautomers.
WRITINGS = [
    ["CC(=O)Nc1ccc(O)cc1", "Oc1ccc(NC(C)=O)cc1", "CC(O)=Nc1ccc(O)cc1", "C1=CC(=CC=C1NC(=O)C)O"],
    ["O=c1cccc[nH]1", "Oc1ccccn1", "C1=CC(=O)NC=C1", "n1c(O)cccc1"],
    ["Nc1nc2[nH]cnc2c(=O)[nH]1", "Nc1nc2nc[nH]c2c(O)n1", "NC1=NC(=O)C2=C(N1)N=CN2"],
    ["OC(=O)c1ccccc1O", "Oc1ccccc1C(O)=O", "O=C(O)C1=CC=CC=C1O"],
    ["[O-][N+](=O)c1ccc(N)cc1", "Nc1ccc([N+]([O-])=O)cc1", "NC1=CC=C(C=C1)[N+](=O)[O-]"],
    ["N#Cc1ccc(O)cc1", "Oc1ccc(C#N)cc1", "C1=CC(=CC=C1C#N)O"],
    ["C[C@H](N)C(=O)O", "OC(=O)[C@H](C)N", "N[C@@H](C)C(O)=O"],
    ["CC(=O)[O-].[Na+]", "[Na+].CC([O-])=O", "[Na+].[O-]C(C)=O"],
]


@pytest.mark.parametrize("forms", WRITINGS)
def test_every_writing_of_a_compound_agrees(forms):
    assert len(set(ox.tautomer_key(forms))) == 1
    assert len(set(ox.canonical_tautomer(forms))) == 1
    tautomer = ox.canonical_tautomer(forms[0])
    assert ox.canonical_tautomer(tautomer) == tautomer
    assert ox.tautomer_key(tautomer) == ox.tautomer_key(forms[0])
    assert ox.canonical_smiles(ox.canonical_smiles(forms[0])) == ox.canonical_smiles(forms[0])


@pytest.mark.parametrize("smiles", [
    "CC(C)(C)CN(C)S(=O)(=O)c1ccc(NN)c([N+](=O)[O-])c1",  # nitro beside NH donors
    "CCCc1nn(C)c(N2CCOCC2C(=O)O)c1[N+](=O)[O-]",  # nitro beside a carboxylic acid
    "NC=CC#N",  # nitrile, not ketenimine
    "O=C(CCS)Oc1ccc2[nH]c(-c3ccccc3)[nH+]c2c1",  # cation on nitrogen, not the ester
    "Cn1c(-c2ccc(O)cc2)cccc1=O",  # phenol beside a pyridone stays neutral
    "C[C@@H]1OC(=O)c2c(O)cccc21",  # phenol beside a lactone stays aromatic
])
def test_preferred_forms_are_kept(smiles):
    assert ox.canonical_tautomer(smiles) == ox.canonical_smiles(smiles)


def test_valence_check_is_opt_in(tmp_path, capsys):
    assert ox.canonical_smiles("FF=F", errors="raise")  # off by default
    assert ox.canonical_smiles(["FF=F", "CCO"], check_valence=True, errors="none") == [
        None, ox.canonical_smiles("CCO")]
    with pytest.raises(ValueError, match="more than the 4"):
        ox.canonical_tautomer("C(C)(C)(C)(C)C", check_valence=True, errors="raise")
    assert ox.tautomer_key("[O-]Cl(=O)(=O)=O", check_valence=True, errors="raise")
    with pytest.raises(ValueError, match="Valence"):
        ox.Molecule("C[N](C)(C)C", check_valence=True)
    assert ox.Molecule("C[N+](C)(C)C", check_valence=True).to_smiles()
    from oxidane.__main__ import main
    source = tmp_path / "in.smi"
    source.write_text("CCO\nFF=F\n")
    assert main([str(source), str(tmp_path / "out.tsv"), "--check-valence", "--errors", "none"]) == 0
    assert "2 rows, 1 unreadable" in capsys.readouterr().err


def test_charge_reduced_key():
    keys = ox.tautomer_key(["CCN.Cl", "CC[NH3+].[Cl-]", "NCC(=O)O", "[NH3+]CC(=O)[O-]"],
                           charge_reduced=True)
    assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0].startswith("p/")
    assert ox.tautomer_key("CC(=O)O") != ox.tautomer_key("CC(=O)[O-]")
    assert ox.Molecule("CC(=O)[O-]").tautomer_key(charge_reduced=True) == ox.tautomer_key(
        "CC(=O)O", charge_reduced=True)


def test_the_default_key_moves_protons_but_never_adds_them():
    key = ox.tautomer_key
    assert key("CN.Cl") == key("C[NH3+].[Cl-]")
    assert key("NCC(=O)O") == key("[NH3+]CC(=O)[O-]")
    assert key("CC(=O)O") != key("CC(=O)[O-]")
    assert key("[CH3-].[Ti]") != key("C.[Ti]")
    assert key("CN.Cl", ionizable_endpoints=False).startswith("h/")
    assert key("CN.Cl", ionizable_endpoints=False) != key("C[NH3+].[Cl-]", ionizable_endpoints=False)


def test_parent_compound_merges_salts_metals_and_protomers():
    same = [
        ["CCN", "CCN.Cl", "CC[NH3+].[Cl-]"],
        ["CC(=O)O", "CC(=O)[O-].[Na+]", "CC(=O)O[Na]"],
        ["NCC(=O)O", "[NH3+]CC([O-])=O"],
        ["C[N+](C)(C)CC(=O)[O-]", "C[N+](C)(C)CC(=O)O"],
        ["CS(C)=O", "C[S+](C)[O-]"],
    ]
    for forms in same:
        parents = ox.parent_compound(forms)
        assert len(set(parents)) == 1, (forms, parents)
    # A quaternary ammonium with nothing to balance it stays a cation.
    assert "+" in ox.parent_compound("OCC[N+](C)(C)C.[Cl-]")
    # Each step can be switched off.
    assert "." in ox.parent_compound("CCN.Cl", strip_salts=False)
    assert "+" in ox.parent_compound("CC[NH3+]", neutralize=False)
    assert "Na" in ox.parent_compound("CC(=O)O[Na]", disconnect_metals=False)
    assert "+" in ox.parent_compound("C[S+](C)[O-]", normalize_groups=False)
    # Same error handling and missing values as the other functions.
    assert ox.parent_compound(["C1CC", None, "CCO"], errors="none") == [
        None, None, ox.parent_compound("CCO")]


def test_parent_is_opt_in_for_files_and_the_command_line(tmp_path, capsys):
    from oxidane.__main__ import main
    source = tmp_path / "in.smi"
    source.write_text("CC[NH3+].[Cl-] salt\nCCN base\n")
    ox.process_file(source, tmp_path / "plain.tsv", columns=["canonical_tautomer"])
    plain = [line.split("\t") for line in (tmp_path / "plain.tsv").read_text().splitlines()]
    assert plain[0][2] != plain[1][2]  # default: the salt is kept as written
    assert main([str(source), str(tmp_path / "parent.tsv"), "--parent",
                 "-c", "parent,canonical_tautomer"]) == 0
    capsys.readouterr()
    rows = [line.split("\t") for line in (tmp_path / "parent.tsv").read_text().splitlines()]
    assert rows[0][2:] == rows[1][2:]
    assert rows[0][2] == ox.parent_compound("CCN")
