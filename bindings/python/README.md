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

## Canonical SMILES and tautomer keys

```python
import oxidane as ox

ox.canonical_smiles("OCC")             # one canonical SMILES
ox.tautomer_key("O=c1cccc[nH]1") == ox.tautomer_key("Oc1ccccn1")  # True
ox.canonical_tautomer("CC(O)=N")       # 'O=C(N)C', the form stored for the class

smiles = [line.split()[0] for line in open("molecules.smi")]
canonical = ox.canonical_smiles(smiles)                # list, input order
keys = ox.tautomer_key(smiles, errors="none")          # None where a SMILES does not parse
tautomers = ox.canonical_tautomer(smiles, threads=8)   # cap the threads used
```

All three take a single SMILES string, which returns a string, or any iterable
of strings (list, tuple, generator, pandas Series), which returns a list in
input order. Lists are processed on every core by default, with the GIL
released; `threads=n` limits that, and `RAYON_NUM_THREADS` sets the default.
A SMILES that does not parse raises `ValueError` naming its index, or gives
`None` with `errors="none"`. Dot-separated components are handled one at a
time and sorted, so a salt written either way round gives the same string.
On a 24-core machine, 100,000 PubChem records take about 0.5 s for canonical
SMILES, 0.7 s for tautomer keys and 1.9 s for canonical tautomers.

- `canonical_smiles` is invariant under atom order.
- `tautomer_key` is equal for every tautomer of a compound: the canonical SMILES
  of the skeleton with its mobile hydrogens and bond orders erased, followed
  by `/`-separated fields saying where they can go. A molecule with nothing
  mobile keys to its canonical SMILES. Treat the key as an identifier; it is
  not a structure to read back.
- `canonical_tautomer` is a real structure: the one tautomer every member of
  the class normalizes to, as canonical SMILES. It has the same key as the
  input.

By default only heteroatoms (N, O, S, Se, Te) exchange hydrogens, as in InChI's
mobile-H layer. `carbon_endpoints=True` also merges keto and enol forms;
`ionizable_endpoints=True` also merges zwitterions with their neutral forms.
Keys made that way start with `c/`, `i/` or `ci/`, so they cannot be confused
with default keys. `Molecule.tautomer_key()` and `Molecule.canonical_tautomer()`
give the same answers for a molecule already built.

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

Tests cover the legacy imports and canonicalization methods, batch canonical
SMILES and tautomer keys, file parsing,
SMARTS errors and matching semantics, SVG export, immediate and automatic
notebook display, and execution through a real Jupyter kernel.
