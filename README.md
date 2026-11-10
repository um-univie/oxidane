<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="oxidane" width="320">
  </picture>
</p>

# oxidane

Python users: the [Python bindings and notebook examples](bindings/python/README.md)
live in this repository. They support `import oxidane` and the legacy
`from molecules import PyMolecule, MolParser` API.

A cheminformatics library in pure Rust. It reads SMILES, SMARTS and SDF;
perceives aromaticity, hydrogens and stereochemistry; gives canonical SMILES
and tautomer keys; searches substructures and maximum common subgraphs;
applies reactions; computes fingerprints and descriptors; and builds 3D
conformers and relaxes them with MMFF94 or UFF.

Oxidane is the IUPAC name for water.

## Install

Oxidane is not on crates.io. Add the checkout to another Rust project's
`Cargo.toml` (adjust the path):

```toml
[dependencies]
oxidane = { path = "../oxidane" }
```

Enable the optional neural-network modules with `features = ["ml"]`. Parsing,
descriptors, featurization and dataset splitting do not need it. From the
checkout, check the library and run its documentation examples with:

```sh
cargo check --locked --all-features
cargo test --locked --doc
```

There is no `oxidane` command-line binary. See [RELEASING.md](RELEASING.md) for
source releases.

## Five minutes

```rust
use oxidane::prelude::*;

// Parse. A SMILES can hold several disconnected molecules, so you get a Vec.
let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();

assert_eq!(aspirin.molecular_formula().to_string(), "C9H8O4");

// Substructure search. This is a carboxylic acid.
assert!(matches_cxsmarts(&aspirin, "[CX3](=O)[OX2H1]").unwrap());

// Two writings of the same molecule canonicalise to one string.
use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;

assert_eq!(
    canonicalize_and_sort_smiles("CC(=O)Oc1ccccc1C(=O)O", false).unwrap(),
    canonicalize_and_sort_smiles("O=C(C)Oc1ccccc1C(O)=O", false).unwrap(),
);
```

## What it does

**Reads and writes.** SMILES, SDF and MOL (V2000), XYZ, GML, DOT, and SVG for
a quick look. The SDF reader interprets the parts of the format that are codes
rather than numbers -- the charge column, the isotope column as a mass
difference, the valence column -- and honours the `M CHG`, `M RAD` and `M ISO`
property lines that supersede the atom block. Wedge and hash bonds become
stereochemistry, so a 2D molfile from PubChem or a vendor catalogue arrives
with its stereocentres intact.

**Perceives what the file left out.** Aromaticity, implicit hydrogens, ring
systems, and stereochemistry in every class the SMILES specification has:
tetrahedral, cis/trans, allene, square-planar, trigonal-bipyramidal and
octahedral. CIP descriptors (R/S, E/Z) come from a port of RDKit's CIPLabeler,
rules 1a through 6 with auxiliary descriptors.

**Canonizes, including stereochemistry and tautomers.** Every way of writing a
molecule gives one canonical SMILES, and two stereoisomers give two. A
tautomer key is shared by every tautomer of a compound, so 2-pyridone and
2-hydroxypyridine deduplicate to one row, and a canonical tautomer picks one
structure to stand for them all.

**Finds the parent compound.** For mass spectrometry, the neutral molecule M a
spectral library records: sodium or potassium drawn as a bond broken off, each
component's canonical tautomer, protons moved until the net charge is as low as
the fixed charges allow, and counterions dropped, so a hydrochloride, a sodium
salt and a zwitterion each come back as the compound they are a form of
(`oxidane::molecule::mass_spec`, `ox.parent_compound`, `oxidane-normalize --parent`).

**Matches.** CxSMARTS patterns with the full atom expression language, ring
and degree constraints, recursion, and chirality as part of the query. The
search is monomorphic rather than induced, so a three-atom chain matches
inside cyclopropane and benzene matches inside naphthalene, which is what a
chemist means by "contains".

**Compares.** Maximum common edge subgraph two ways: a clique reduction over
the modular product, and McSplit, which finishes on pairs of real drug
molecules where the clique search does not.

**Transforms.** Reaction SMARTS, applied through the atom map, with hydrogen
counts re-derived for every atom whose bonding changed. Reactions are
double-pushout rules underneath, so they can also be read from MØD's GML,
composed, and overlapped into critical pairs.

**Fingerprints and descriptors.** Circular (Morgan/ECFP), atom-pair and
topological-torsion fingerprints, the 166 MACCS keys, and the minhashed MHFP
and MAP4, kept as sparse counts and folded to bits on request. Crippen log P
and molar refractivity, TPSA, hydrogen-bond donors and acceptors, rotatable
bonds, Bemis-Murcko scaffolds, and for a conformer USR, USRCAT and the moments
of inertia.

**Embeds.** Distance geometry in the style of ETKDG: bounds smoothing,
four-dimensional embedding, RDKit's experimental torsion preferences, and
handedness terms for tetrahedral, cis/trans, allene and square-planar centres,
since a bounds matrix alone cannot tell a stereocentre from its mirror image.
Each conformer is then minimised with MMFF94 or UFF.

**Force fields.** MMFF94, with its own atom typing and partial charges, and UFF
for molecules MMFF94 cannot type. Both give energies and gradients and
minimise with L-BFGS; the energy terms are vectorised through `pulp` and give
the same bits on AVX2, NEON and scalar hardware.

**Learns.** Per-atom and per-bond feature matrices, scaffold splits and
batching for message-passing networks, and behind the `ml` feature an encoder
with graph, atom and bond heads built on `candle`.

## Two representations, one set of traits

`MolecularGraph` is a molecule without coordinates: what SMILES gives you, and
what matching, canonization and formula work on. `Molecule3D` is the same
molecule with positions, from an XYZ file, a 3D SDF, or the embedder.

Neither type owns any behaviour. `MoleculeCore` is the atoms, the bonds and
the adjacency; everything else is a trait with a blanket implementation on top
of it, and `Molecule` is their sum. A function that only walks bonds takes
`impl MoleculeCore` and does not drag ring perception in with it; a new
representation gets the entire library by writing `impl Molecule for`.

```rust
use oxidane::prelude::*;
use oxidane::molecule::descriptors::MolecularDescriptors;

let caffeine =
    MolecularGraph::from_smiles_single("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();

assert_eq!(caffeine.molecular_formula().to_string(), "C8H10N4O2");
assert!((caffeine.molecular_weight() - 194.19).abs() < 0.01);
```

## Reading a real file

```rust
use oxidane::prelude::*;
use oxidane::io::SdfParseOptions;

// Hydrogens, stereochemistry and aromaticity are all perceived by default.
let molecules = Molecule3D::from_sdf("tests/pentene.sdf").unwrap();
assert_eq!(molecules.len(), 1);

// Or ask for exactly the atoms the file listed, and nothing else.
let literal = Molecule3D::from_sdf_with(
    "tests/pentene.sdf",
    SdfParseOptions { hydrogens: false, stereochemistry: false, aromaticity: None },
).unwrap();
assert!(literal[0].atomic_numbers().len() <= molecules[0].atomic_numbers().len());
```

## Substructures and reactions

```rust
use oxidane::prelude::*;
use oxidane::molecule::reaction::ReactionSmarts;

let ester = MolecularGraph::from_smiles_single("CC(=O)OCC").unwrap();

// Every match, as atom indices into the molecule.
let matches = find_cxsmarts_matches(&ester, "[CX3](=O)[OX2][CX4]").unwrap();
assert_eq!(matches.len(), 1);

// Hydrolysis, written as a mapped transformation.
let hydrolysis =
    ReactionSmarts::parse("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[O:3].[C:4]")
        .unwrap();
let products = hydrolysis.apply(&ester);
assert!(!products.is_empty());
```

## Three dimensions

```rust
use oxidane::prelude::*;

let molecule = MolecularGraph::from_smiles_single("C[C@H](N)C(=O)O").unwrap();
let embedded = molecule.embed_3d().unwrap();

// Coordinates for every atom, with the stereocentre the SMILES asked for.
assert_eq!(embedded.positions.len(), embedded.atomic_numbers().len());

// Score or relax a structure with MMFF94, or UFF where MMFF94 has no types.
use oxidane::molecule::forcefield::Field;

let field = Field::best_for(&embedded).unwrap();
let mut positions = embedded.positions.clone();
let energy = field.minimize(&mut positions, 500);
assert_eq!(field.name(), "MMFF94");
assert!(energy <= field.energy(&embedded.positions) + 1e-9);
```

## Related

[`chemistry_consts`](https://crates.io/crates/chemistry_consts) holds the
periodic table this is built on: masses, isotopes, radii, valences and
electronegativities as plain `const` arrays indexed by atomic number.

## Logo

The mark is a water molecule, since oxidane is water's IUPAC name: the "o" is
the oxygen and its hydrogens sit at the H-O-H angle `embed_3d` gives `[H]O[H]`
(103.98 degrees, from MMFF94). Regenerate the SVGs (light, dark, square mark)
with:

```sh
cargo run --example logo
```

## Licence

MIT.
