# oxidane

A cheminformatics library in Rust. It reads the formats molecules actually
arrive in, works out what the file left implicit, and answers the questions
that follow: what is this, does it contain that, are these two the same, what
do they have in common, and what does it look like in three dimensions.

Oxidane is the IUPAC name for water.

```toml
[dependencies]
oxidane = "0.2"
```

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
rules 1a through 6 with auxiliary descriptors, rather than the
highest-atomic-number approximation that gets most cases right and the
interesting ones wrong.

**Canonizes, including stereochemistry.** Every way of writing a molecule
gives one canonical SMILES, and two stereoisomers give two -- which is the
half that is easy to get wrong silently.

**Matches.** CxSMARTS patterns with the full atom expression language, ring
and degree constraints, recursion, and chirality as part of the query. The
search is monomorphic rather than induced, so a three-atom chain matches
inside cyclopropane and benzene matches inside naphthalene, which is what a
chemist means by "contains".

**Compares.** Maximum common edge subgraph two ways: a clique reduction over
the modular product, and McSplit, which finishes on pairs of real drug
molecules where the clique search does not.

**Transforms.** Reaction SMARTS, applied through the atom map, with hydrogen
counts re-derived for every atom whose bonding changed.

**Embeds.** Distance geometry from a connection table to 3D coordinates, with
explicit handedness terms -- a bounds matrix alone cannot tell a stereocentre
from its mirror image, since every distance is identical in both.

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
```

## How it was checked

Nearly every claim above was established by differential testing against
RDKit, and the harnesses are in the repository rather than in someone's shell
history: a Python script builds a corpus with the expected answer, a Rust
example in `examples/` runs our side and reports only the disagreements.

- **SDF reading.** RDKit writes a molblock, we read it and write SMILES, and
  both go through RDKit's canonicaliser. Over 1501 molecules the formula and
  skeleton agree 1501 times and the full stereochemical SMILES 1466. The 35
  remaining are bridgehead nitrogen, where the lone pair is the fourth
  substituent.
- **CIP labelling.** 381 of 383 stereocentres agree with `rdCIPLabeler`.
- **Canonization.** 86 curated molecules relabelled 20 ways each, asserting
  invariance, idempotence and formula preservation, plus a stereochemistry
  suite that asserts discrimination as well as invariance. A larger replay
  corpus of 100,000 molecules with ~85 writings apiece -- 8.5M comparisons --
  has two known failures, both an aromatic ring containing an explicit triple
  bond, where RDKit's own ground truth calls the same atoms both SP-hybridized
  and aromatic.
- **3D embedding.** Read back through RDKit's `AssignStereochemistryFrom3D`.
  That comparison found a sign inversion in our own tetrahedral perception,
  which had been mislabelling every molecule loaded from XYZ or a 3D SDF.

331 tests and 98 doctests, all of which assert something -- including every
example on this page, which is compiled and run as part of the suite.

## What it is not

Not a force field, not a conformer generator with a torsion library, not a
descriptor suite for QSAR, and not a substitute for RDKit's breadth. There is
no InChI, no PDB, no fingerprints, and the V3000 molfile format is unsupported.

Where behaviour differs from RDKit it is documented and deliberate: charge code
4 in a molfile atom block is a doublet radical and is kept rather than dropped,
and elements with no stable isotope use the IUPAC conventional nuclide.

## Related

[`chemistry_consts`](https://crates.io/crates/chemistry_consts) holds the
periodic table this is built on: masses, isotopes, radii, valences and
electronegativities as plain `const` arrays indexed by atomic number.

## Licence

MIT.
