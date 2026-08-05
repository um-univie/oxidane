# Oxidane Python bindings

The standalone [`um-univie/molecules-py`](https://github.com/um-univie/molecules-py)
bindings now live here and compile against the Rust library two directories up.
The starting source and SDF/XYZ fixtures come from commit
`0f1eaed5e592d00b131da387600146d733628262` (2026-02-06). No separate checkout of
`molecules` or `stochsim` is needed. The old repository is not deleted or archived
by this migration.

## Install from this checkout

From the repository root, with Rust and `uv` available:

```sh
uv venv bindings/python/.venv
uv pip install --python bindings/python/.venv/bin/python './bindings/python[notebook]'
```

The package is named `oxidane`. This command builds the local source; no PyPI
publication is required. Omit `[notebook]` if only using the chemistry API or
exporting SVG strings. Select this environment in Jupyter and restart an existing
kernel after rebuilding the native extension.

For an editable development install, run inside `bindings/python`:

```sh
uv sync --extra notebook
uv run --with maturin maturin develop --uv
uv run pytest
```

## Molecules and SMARTS

```python
from oxidane import Molecule, Smarts

ethanol = Molecule("CCO")
assert ethanol.contains_smarts("[OX2H1]")
alcohol = Smarts("[OX2H1][CX4]")
assert ethanol.contains(alcohol)
assert not Molecule("CC(=O)C").contains(alcohol)
```

`contains` accepts either a SMARTS string or a reusable `Smarts` object. Invalid
SMARTS raises `ValueError`; a valid absent pattern returns `False`. Dot-separated
SMILES retain all components, including counterions.

The old imports remain supported:

```python
from molecules import PyMolecule, PyMolecule3D, MolParser

mol = PyMolecule("CCO")
canonical_copy = mol.canonize_mol()
mol.canonize()
print(mol.to_canonical_smiles())
```

`Molecule` and `Molecule3D` are aliases for `PyMolecule` and `PyMolecule3D`.
`MolParser().parse_file(path)` still reads SDF and XYZ files and returns a list
of `Molecule3D` objects. Paths may be strings or `pathlib.Path` objects. File I/O
errors raise `OSError`; malformed counted XYZ frames raise `ValueError` rather
than being silently skipped. Multiple XYZ frames are kept separate, including
the original binding's blank-separated atom blocks after the first XYZ frame.

## Jupyter display

```python
from oxidane import Molecule, Reaction, ReactionSmarts

mol = Molecule("CC(=O)Oc1ccccc1C(=O)O")
mol.plot()  # Displays immediately, including inside loops
mol         # Displays automatically as the last expression of a cell

reaction = Reaction(["CCO"], ["CC=O"], name="Oxidation")
reaction.plot(width=800, height=260)

rule = ReactionSmarts("[C:1]-[O:2]>>[C:1]=[O:2]")
rule.plot()                            # Exact SMARTS query text
rule.plot(reactants=[Molecule("CCO")])  # Draw every concrete outcome
products = rule.apply(Molecule("CCO"))
```

`Reaction` is a depiction of supplied participants; it does not assert mass
balance or chemical validity. `ReactionSmarts` executes the Rust rewrite engine
and requires explicit atom maps. Each returned product graph represents one
match and may contain multiple disconnected components. Symmetric matches can
produce duplicate products, matching the Rust API.

Molecules, 3D molecules, reactions, and rules support `to_svg()`, `plot()` and
IPython's [`_repr_svg_` protocol](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html).
`plot()` displays once and returns `None`; `to_svg()` returns text and needs no
IPython installation. Width and height are positive integer pixel dimensions;
for rule outcomes, height applies to each outcome row. The stored topology of a
3D molecule is drawn in 2D; plotting does not alter its coordinates or infer new
bonds. Repeated queries on a 3D molecule can use `to_graph()` once to avoid copies.

See [the runnable notebook](examples/notebook_display.ipynb).

## Build and verification

The binding has its own Cargo manifest and lockfile, so ordinary root Rust
builds do not acquire Python dependencies. A wheel can be built from the root:

```sh
uv run --project bindings/python --with maturin maturin build --manifest-path bindings/python/Cargo.toml
```

Tests cover the legacy imports and canonicalization methods, file parsing,
SMARTS errors and matching semantics, SVG export, immediate and automatic
notebook display, and execution through a real Jupyter kernel.
