from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import oxidane as ox


def objects():
    return [ox.Molecule("CCO"), ox.Reaction(["CCO"], ["CC=O"]),
            ox.ReactionSmarts("[C:1]-[O:2]>>[C:1]=[O:2]"),
            ox.MolParser().parse_file(Path(__file__).with_name("pentene.sdf"))[0]]


def test_svg_export_without_ipython():
    result = subprocess.run([sys.executable, "-c", '''
import sys
sys.modules['IPython'] = None
from oxidane import Molecule, Reaction, ReactionSmarts
for obj in [Molecule('CCO'), Reaction(['CCO'], ['CC=O']), ReactionSmarts('[C:1]-[O:2]>>[C:1]=[O:2]')]:
    assert obj.to_svg().startswith('<svg')
    assert obj._repr_svg_().startswith('<svg')
    try:
        obj.plot()
    except ImportError as exc:
        assert 'oxidane[notebook]' in str(exc)
    else:
        raise AssertionError('plot should require IPython')
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_display_formatter_and_plot_in_loops():
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    try:
        for obj in objects():
            data, _ = shell.display_formatter.format(obj)
            ET.fromstring(data["image/svg+xml"])
        with capture_output(display=True) as output:
            for obj in objects():
                assert obj.plot(width=640, height=320) is None
        assert len(output.outputs) == 4
        for item in output.outputs:
            root = ET.fromstring(item.data["image/svg+xml"])
            assert root.get("width") == "640"
            assert root.get("height") == "320"
    finally:
        InteractiveShell.clear_instance()


def test_real_jupyter_kernel():
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    sources = [
        '''from oxidane import Molecule, Reaction, ReactionSmarts
mol = Molecule('CCO')
reaction = Reaction(['CCO'], ['CC=O'])
rule = ReactionSmarts('[C:1]-[O:2]>>[C:1]=[O:2]')''',
        "mol", "reaction", "rule",
        "mol.plot()\nreaction.plot()\nrule.plot(reactants=[mol])",
        "for obj in [mol, reaction, rule]:\n    obj.plot()",
    ]
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code) for code in sources])
    client = NotebookClient(nb, timeout=60, kernel_name="python3")
    client.create_kernel_manager().kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    nb = client.execute()
    for cell, expected in zip(nb.cells, [0, 1, 1, 1, 3, 3]):
        assert len(cell.outputs) == expected
        for output in cell.outputs:
            assert output.output_type in {"display_data", "execute_result"}
            ET.fromstring(output.data["image/svg+xml"])
