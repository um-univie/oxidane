# Oxidane Python bindings

Canonical SMILES, tautomer keys and canonical tautomers for millions of
molecules, plus molecules, SMARTS queries, reactions and SVG display in
Jupyter. The package is called `oxidane`; it needs nothing but Python 3.9 or
newer.

## Install

With the wheel file, which runs on any x86-64 Linux with glibc 2.17 or newer
and any Python from 3.9 on:

```sh
uv venv && source .venv/bin/activate
uv pip install oxidane-0.2.0-cp39-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

Wheels for every platform (Linux x86-64 and ARM64 with glibc or musl, macOS
Intel and Apple silicon, Windows x64 and ARM64) are built by CI: on the
repository's Actions tab, open "Build the Python wheels", start it on `main`
with "Run workflow" (or open its latest run), download `oxidane-python-wheels`,
and run `uv pip install --no-index --find-links <unpacked folder> oxidane`.

From the repository, on any platform, which builds the native module and so
needs Rust (`curl https://sh.rustup.rs -sSf | sh`; compiling takes a few minutes)
and, for the first line, read access to the repository on GitHub:

```sh
uv pip install "oxidane @ git+https://github.com/um-univie/oxidane-dev#subdirectory=bindings/python"
uv pip install ./bindings/python             # or from a clone, at its root
uv tool install ./bindings/python            # only the oxidane-normalize command, on PATH
```

`./bindings/python[notebook]` adds IPython and ipykernel for the Jupyter
display. To build a wheel to hand on, see [Build and verification](#build-and-verification).

## Normalize a SMILES file or database

The `oxidane-normalize` command (also `python -m oxidane`) appends three
columns to every row of a CSV, TSV or SMILES file, on every core, streaming, so
memory stays small however large the file is:

```sh
oxidane-normalize molecules.smi molecules.tsv                 # SMILES in the first column
oxidane-normalize data.csv data_out.csv --smiles-column smiles   # by header name
oxidane-normalize CID-SMILES.gz keys.tsv.gz -s 1 -c tautomer_key # 2nd column, gzip, one result
sqlite3 -tabs compounds.db "SELECT id, smiles FROM compounds" | oxidane-normalize - out.tsv -s 1
oxidane-normalize --help
```

The three results are:

| column | what it is | use it for |
|---|---|---|
| `canonical_smiles` | the input as canonical SMILES | exact duplicates, one spelling per structure |
| `tautomer_key` | an identifier shared by every tautomer of a compound | grouping and joining; not a structure |
| `canonical_tautomer` | the one tautomer every member of the class normalizes to, as canonical SMILES | the structure to store or compute on |

Nothing else is changed: counterions, charges that no tautomer removes,
isotopes and stereochemistry stay as written, and components of a salt are
sorted, so `[Na+].CC([O-])=O` and `CC(=O)[O-].[Na+]` come out the same. Strip
salts or neutralize first if the database should hold parent compounds.

A SMILES that does not parse is written back unchanged in the result columns and
reported on stderr (the first ten, then a count); `--errors none` leaves the
fields empty instead, `--errors raise` stops. The reader takes any valence a
SMILES writes, so a typo such as `FF=F` goes through; `--check-valence` treats
an atom with more bonds than its element allows as unreadable, refusing the
same strings RDKit does. On 24 cores a file goes at about
25,000 rows per second with all three columns; a single column is faster.

In Python the same three functions take one SMILES or a whole column:

```python
import oxidane as ox
import pandas

df = pandas.read_parquet("compounds.parquet")      # or read_csv, read_sql, ...
df["canonical_smiles"] = ox.canonical_smiles(df["smiles"])
df["tautomer_key"] = ox.tautomer_key(df["smiles"])
df["canonical_tautomer"] = ox.canonical_tautomer(df["smiles"], errors="none")
duplicates = df[df.duplicated("tautomer_key", keep=False)]  # same compound, other tautomer
```

With polars, give the column to the function inside `map_batches`, which also
works in lazy queries:

```python
import polars as pl

df = pl.read_parquet("compounds.parquet").with_columns(
    pl.col("smiles").map_batches(lambda s: pl.Series(ox.canonical_tautomer(s)),
                                 return_dtype=pl.String).alias("canonical_tautomer"))
```

Each call spreads the whole column over every core. Calling once per row,
with pandas' `apply` or polars' `map_elements`, runs on one core and is much
slower. Missing values come back as `None` (pandas) or `null` (polars).
`threads=8` caps the cores used. The column is copied into memory first,
which is fine for millions of rows; for larger tables, stream them through
`oxidane-normalize`. For 100,000 PubChem SMILES, pandas and polars alike:

| function | one core | 24 cores |
|---|---|---|
| `canonical_smiles` | 7.7 s (77 µs each) | 0.55 s (180,000 a second) |
| `tautomer_key` | 9.8 s (98 µs each) | 0.7 s (140,000 a second) |
| `canonical_tautomer` | 26 s (0.26 ms each) | 2.1 s (47,000 a second) |

For a table in a database, read it in batches of row ids and SMILES, compute,
and write back; [examples/normalize_sqlite.py](examples/normalize_sqlite.py)
does this for SQLite in 40 lines (`python normalize_sqlite.py compounds.db
compounds smiles`), and the loop is the same with any other driver.

## Canonical SMILES and tautomer keys

```python
import oxidane as ox
import pandas

ox.canonical_smiles("OCC")             # one canonical SMILES
ox.tautomer_key("O=c1cccc[nH]1") == ox.tautomer_key("Oc1ccccn1")  # True
ox.canonical_tautomer("CC(O)=N")       # 'O=C(N)C', the form stored for the class

df = pandas.read_csv("data.csv")
smiles = df["smiles"]                                  # any iterable of strings
df["canonical"] = ox.canonical_smiles(smiles)          # list, input order; gaps stay None
df["key"] = ox.tautomer_key(smiles)
df["tautomer"] = ox.canonical_tautomer(smiles, threads=8)   # cap the threads used

# Whole files, streamed: every row gets the three columns appended.
ox.process_file("molecules.smi", "molecules.tsv")
ox.process_file("CID-SMILES.gz", "keys.tsv.gz", smiles_column=1, columns=["tautomer_key"])
ox.process_file("data.csv", "data_out.csv", smiles_column="smiles")  # header row
```

All three take a single SMILES string, which returns a string, or any iterable
of strings (list, tuple, generator, pandas Series), which returns a list in
input order. A missing entry (`None`, NaN or `pandas.NA`) gives `None`, so a
column with gaps goes through as it is, and whitespace around a SMILES is
ignored. Lists are processed on every core by default, with the GIL
released; `threads=n` limits that, and `RAYON_NUM_THREADS` sets the default.
A SMILES that does not parse is given back unchanged and reported on stderr
with its index (the first ten of a call, then a count); `errors="none"` gives `None` in its place instead, and
`errors="raise"` raises `ValueError`. `check_valence=True` counts an atom bonded
past its element's valences (`C(C)(C)(C)(C)C`, `FF=F`) as not parsing; the
same flag is on `ox.Molecule`. Dot-separated components are handled one at a
time and sorted, so a salt written either way round gives the same string.
Timings are in the table above; on one core, a canonical tautomer takes
about 0.25 ms, against about 20 ms for RDKit's
`TautomerEnumerator.Canonicalize`.

`process_file(input, output)` writes each input row followed by one field per
entry of `columns` (default: `canonical_smiles`, `tautomer_key`,
`canonical_tautomer`). It reads `chunk_size` rows (100,000) at a time and
computes each chunk on every core, so memory stays bounded however large the
file is; about 25,000 rows per second with all three columns on 24 cores. The
delimiter follows the extension: `,` for `.csv`, tab for `.tsv`, whitespace for
`.smi` and anything else (written back as tabs); `delimiter=` overrides it.
CSV quoting is honoured, a `.gz` suffix on either path compresses that
side, and `-` stands for standard input or output. `smiles_column` is a 0-based index or, with a header row, a column name;
`header=True` keeps a header with an index too. The same `errors` modes apply,
reported with the input line number, and `carbon_endpoints`,
`ionizable_endpoints`, `threads` and `check_valence` are passed through. It returns
`{"rows": ..., "unparsed": ...}`.

- `canonical_smiles` is invariant under atom order.
- `tautomer_key` is equal for every tautomer of a compound: the canonical SMILES
  of the skeleton with its mobile hydrogens and bond orders erased, followed
  by `/`-separated fields saying where they can go. A molecule with nothing
  mobile keys to its canonical SMILES. Treat the key as an identifier; it is
  not a structure to read back.
- `canonical_tautomer` is a real structure: the one tautomer every member of
  the class normalizes to, as canonical SMILES. It has the same key as the
  input. Of the forms in a class it takes the one with the fewest charges,
  then the most aromatic bonds, then the carbonyl over the enol; nitro groups
  and nitriles stay nitro groups and nitriles, and a cation stays on nitrogen
  rather than moving to an ether or ester oxygen.

By default only heteroatoms (N, O, S, Se, Te) exchange hydrogens, as in InChI's
mobile-H layer. `ionizable_endpoints=True` also merges zwitterions with their
neutral forms. `carbon_endpoints=True` also merges keto and enol forms, but is
experimental: it lets every carbon exchange hydrogens, including those of
alkyl groups, so it merges compounds that are not tautomers, and about a third
of a PubChem sample comes back with an aromatic ring dearomatized. Use it for
keys at most, not for structures to store.
`tautomer_key(..., charge_reduced=True)` goes one step further and gives every
protonation state of a compound one key: acetic acid and acetate, an amine and
its hydrochloride, a zwitterion and its neutral form. It keeps only what a
proton cannot change, so a quaternary ammonium, a counterion such as `[Na+]`
and the skeleton itself still count. Keys made that way start with `c/`, `i/`,
`p/` or `ci/`, so they cannot be confused with default keys. `Molecule.tautomer_key()` and `Molecule.canonical_tautomer()`
give the same answers for a molecule already built, as
`Molecule.to_canonical_smiles()` does for `canonical_smiles`.

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

The bindings started as the standalone
[`um-univie/molecules-py`](https://github.com/um-univie/molecules-py) (commit
`0f1eaed5e592d00b131da387600146d733628262`, 2026-02-06, with its SDF/XYZ
fixtures) and now compile against the Rust library two directories up. The old
imports remain supported:

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
builds do not acquire Python dependencies. For an editable development install,
run inside `bindings/python`:

```sh
uv sync --extra notebook
uv run --with maturin maturin develop --uv --release
uv run pytest
```

A wheel uses Python's stable ABI, so one file serves every Python from 3.9 on.
From the repository root, this builds one that runs on any x86-64 Linux with
glibc 2.17 or newer (zig stands in for an old toolchain):

```sh
uv run --no-project --with 'maturin>=1.9,<2' --with ziglang maturin build --release \
    --zig --compatibility manylinux2014 --manifest-path bindings/python/Cargo.toml --out dist
```

Without `--zig --compatibility manylinux2014` the wheel is for the building
machine's platform (macOS and Windows included).
[`.github/workflows/wheels.yml`](../../.github/workflows/wheels.yml) builds and
smoke-tests the wheels of every platform when started by hand, on a tag, or on
a pull request labelled `full-ci`; `scripts/ci-local.sh wheels` cross-builds
them locally (see [docs/CHECKS.md](../../docs/CHECKS.md)). Restart a Jupyter kernel
after reinstalling the native module.

Tests cover the legacy imports and canonicalization methods, batch canonical
SMILES, tautomer keys and file processing, file parsing,
SMARTS errors and matching semantics, SVG export, immediate and automatic
notebook display, and execution through a real Jupyter kernel.
