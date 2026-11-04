# Oxidane Cheminformatics Cookbook

A collection of practical, copy-pasteable recipes for common cheminformatics
and computational drug discovery tasks in Rust.

These examples are compiled and run by `cargo test --doc`. They use the default
features; the optional `ml` feature is needed for neural-network models, not for
the data preparation recipes below.

---

## Table of Contents

1. [Virtual Screening and Drug-Likeness (Lipinski & Veber)](#1-virtual-screening-and-drug-likeness)
2. [Pharmacophore Matching and Structural Filtering (CxSMARTS)](#2-pharmacophore-matching-and-structural-filtering)
3. [Canonicalization and Tautomer-Invariant Deduplication](#3-canonicalization-and-tautomer-invariant-deduplication)
4. [Chemical Similarity and Fingerprinting (ECFP & MACCS)](#4-chemical-similarity-and-fingerprinting)
5. [Scaffold and Series Analysis (Bemis-Murcko)](#5-scaffold-and-series-analysis)
6. [3D Conformer Generation and Force Field Relaxation (MMFF94)](#6-3d-conformer-generation-and-force-field-relaxation)
7. [Reaction Informatics (Reaction SMARTS)](#7-reaction-informatics)
8. [2D Depiction (SVG)](#8-2d-depiction)
9. [Machine Learning Preparation: Featurization and Scaffold Splitting](#9-machine-learning-preparation)

---

## 1. Virtual Screening and Drug-Likeness

Lipinski's Rule of Five and Veber's criteria are standard filters used to assess
molecular properties in early-stage small-molecule discovery. They are heuristics,
not predictions of efficacy, safety, or oral bioavailability for an individual compound.

Acceptor counts use N/O/S functional-group rules corresponding to
[RDKit's NumHBA definition](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Lipinski.cpp).
Paracetamol has two acceptors; its amide nitrogen is excluded. Donor counts use
N/O atoms bearing explicit hydrogen and do not include sulfur donors.

Mass descriptors honor explicit isotope labels. If a written nuclide has no
mass in the constants table, the molecular mass methods return NaN; check
`is_finite()` before using these values. These methods report neutral atomic
mass sums, not charge-adjusted ion masses.

- **Molecular Weight (MW)**: <= 500 Da
- **Calculated LogP (Wildman-Crippen)**: <= 5.0
- **Hydrogen Bond Donors (HBD)**: <= 5
- **Hydrogen Bond Acceptors (HBA)**: <= 10
- **Rotatable Bonds**: <= 10
- **Topological Polar Surface Area (TPSA)**: <= 140 A^2

```rust
use oxidane::prelude::*;

struct DrugLikeness {
    mw: f64,
    log_p: f64,
    hbd: usize,
    hba: usize,
    rotatable_bonds: usize,
    tpsa: f64,
    fsp3: f64,
}

fn evaluate_drug_likeness(smiles: &str) -> Option<DrugLikeness> {
    let mol = MolecularGraph::from_smiles_single(smiles).ok()?;
    Some(DrugLikeness {
        mw: mol.molecular_weight(),
        log_p: mol.crippen_log_p(),
        hbd: mol.number_of_hydrogen_bond_donors(),
        hba: mol.number_of_hydrogen_bond_acceptors(),
        rotatable_bonds: mol.rotatable_bond_count(),
        tpsa: mol.topological_polar_surface_area(),
        fsp3: mol.fraction_saturated_carbons(),
    })
}

fn passes_approximate_property_filter(props: &DrugLikeness) -> bool {
    props.mw.is_finite()
        && props.mw <= 500.0
        && props.log_p <= 5.0
        && props.hbd <= 5
        && props.hba <= 10
        && props.rotatable_bonds <= 10
        && props.tpsa <= 140.0
}

fn main() {
    // Paracetamol (acetaminophen)
    let paracetamol_smiles = "CC(=O)Nc1ccc(O)cc1";
    let props = evaluate_drug_likeness(paracetamol_smiles).unwrap();

    assert!(passes_approximate_property_filter(&props));
    assert!((props.mw - 151.16).abs() < 0.05);
    assert!(props.log_p < 2.0);
    assert_eq!(props.hbd, 2); // Phenol OH + amide NH
    assert_eq!(props.hba, 2); // Phenol O and carbonyl O
    assert_eq!(props.rotatable_bonds, 1); // N-aryl bond; the carbonyl C-N bond is excluded
}
```

---

## 2. Pharmacophore Matching and Structural Filtering

Oxidane supports extended CxSMARTS expressions. Substructure search is
monomorphic (chemist's "contains"), allowing you to detect functional groups,
pharmacophoric features, or toxicophores across chemical libraries.

For a quick check use `molecule.contains_smarts("...")?`; for a parsed pattern use
`molecule.contains(&pattern)`. These methods also work on `Molecule3D` using its
stored topology and stereo tags. For many queries on one molecule, use
`MoleculeMatchContext` to reuse ring and aromaticity analysis. Convert a 3D
molecule with `to_graph()` once before that loop.

```rust
use oxidane::prelude::*;
use oxidane::io::{CxSMARTSPattern, MoleculeMatchContext, parse_cxsmarts};

fn main() {
    // Penicillin G — featuring the famously strained beta-lactam core
    let penicillin_g =
        MolecularGraph::from_smiles_single("CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O")
            .unwrap();

    // One-off query:
    assert!(penicillin_g.contains_smarts("[C&r4](=O)[N&r4]").unwrap());

    // Context computes ring membership and aromaticity once:
    let context = MoleculeMatchContext::new(&penicillin_g);

    // Beta-lactam core: 4-membered ring amide [C&r4](=O)[N&r4]
    let beta_lactam = parse_cxsmarts("[C&r4](=O)[N&r4]").unwrap();
    assert!(penicillin_g.contains(&beta_lactam));
    assert!(beta_lactam.matches_with(&penicillin_g, &context));

    // Carboxylic acid: [CX3](=O)[OX2H1]
    let carboxylic_acid = parse_cxsmarts("[CX3](=O)[OX2H1]").unwrap();
    assert!(carboxylic_acid.matches_with(&penicillin_g, &context));

    // Phenyl ring: c1ccccc1
    let phenyl = parse_cxsmarts("c1ccccc1").unwrap();
    assert!(phenyl.matches_with(&penicillin_g, &context));

    // Retrieve indices for the C(=O)-N motif, not all four ring atoms:
    let matches = beta_lactam.find_matches_with(&penicillin_g, &context);
    assert_eq!(matches.len(), 1);
}
```

---

## 3. Canonicalization and Tautomer-Invariant Deduplication

In compound registration and database deduplication, identical chemical entities
frequently arrive with different atom numbering, different Kekulé structures, or
as alternate tautomeric forms.

### Standard Canonical SMILES
```rust
use oxidane::prelude::*;
use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;

fn main() {
    // Aspirin drawn starting from the acetyl group vs carboxylic acid:
    let smiles_a = "CC(=O)Oc1ccccc1C(=O)O";
    let smiles_b = "O=C(C)Oc1ccccc1C(O)=O";

    let canon_a = canonicalize_and_sort_smiles(smiles_a, false).unwrap();
    let canon_b = canonicalize_and_sort_smiles(smiles_b, false).unwrap();

    assert_eq!(canon_a, canon_b);
}
```

### Tautomer-Invariant Keys
Tautomers such as 2-pyridone and 2-hydroxypyridine may be grouped together for
registration or deduplication. This is a choice of identity rules; different
applications may need to retain the individual forms.

Oxidane provides an erasure-based tautomer key that removes selected bond-order
and proton-location distinctions while retaining conserved totals. The result is
an opaque, versioned key, not a valid SMILES or a prediction of the dominant
tautomer. Compare keys produced with the same rules and implementation version.
Default rules cover heteroatom endpoints; carbon-endpoint rules are opt-in.

```rust
use oxidane::prelude::*;
use oxidane::molecule::tautomer::{TautomerRules, tautomer_key};

fn main() {
    // 2-Pyridone (keto form) and 2-Hydroxypyridine (enol form)
    let keto = MolecularGraph::from_smiles_single("O=c1[nH]cccc1").unwrap();
    let enol = MolecularGraph::from_smiles_single("Oc1ncccc1").unwrap();

    let rules = TautomerRules::new();
    let key_keto = tautomer_key(&keto, &rules);
    let key_enol = tautomer_key(&enol, &rules);

    // Both tautomers generate the exact same canonical tautomer key:
    assert_eq!(key_keto, key_enol);
}
```

---

## 4. Chemical Similarity and Fingerprinting

Oxidane offers multiple fingerprinting schemes:
- **Circular Fingerprints (Morgan-style)**: Environment hashing around each atom.
- **MACCS Keys**: 166 pre-defined structural pattern queries.
- **Atom Pairs and Topological Torsions**: Path and distance-based correlations.
- **MinHash (MHFP)**: Fast probabilistic sketching.

Oxidane uses its own hashing scheme. Generate both sides of a comparison with the
same scheme, parameters, and version; ECFP-like environments do not imply bitwise
compatibility with another toolkit. Similarity thresholds are specific to the
representation and application.

The morphine and codeine stereostructures below follow
[PubChem morphine](https://pubchem.ncbi.nlm.nih.gov/compound/5288826) and
[PubChem codeine](https://pubchem.ncbi.nlm.nih.gov/compound/5284371).

```rust
use oxidane::prelude::*;
use oxidane::molecule::fingerprint::{circular_fingerprint, maccs_keys};

fn main() {
    // Morphine and Codeine: Codeine differs only by an O-methylation
    let morphine = MolecularGraph::from_smiles_single(
        "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O"
    ).unwrap();
    let codeine = MolecularGraph::from_smiles_single(
        "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O"
    ).unwrap();
    // Diazepam (Valium) — structurally unrelated benzodiazepine
    let diazepam = MolecularGraph::from_smiles_single(
        "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13"
    ).unwrap();

    // 1. Circular fingerprint (radius = 2, diameter = 4, 2048-bit fold)
    let ecfp_morphine = circular_fingerprint(&morphine, 2).fingerprint.fold(2048);
    let ecfp_codeine = circular_fingerprint(&codeine, 2).fingerprint.fold(2048);
    let ecfp_diazepam = circular_fingerprint(&diazepam, 2).fingerprint.fold(2048);

    let sim_analog = ecfp_morphine.tanimoto(&ecfp_codeine);
    let sim_distant = ecfp_morphine.tanimoto(&ecfp_diazepam);

    assert!(sim_analog > 0.70);  // High structural similarity between close analogs
    assert!(sim_distant < 0.20); // Very low similarity across chemotypes

    // 2. MACCS 166 Structural Keys
    let maccs_morphine = maccs_keys(&morphine);
    let maccs_codeine = maccs_keys(&codeine);
    assert!(maccs_morphine.tanimoto(&maccs_codeine) > 0.85);
}
```

---

## 5. Scaffold and Series Analysis

Bemis-Murcko scaffold extraction strips away exocyclic single-bonded chains while
preserving rings, linkers between rings, and doubly-bonded exocyclic atoms (such
as carbonyls).

This is critical for dataset curation and series grouping to prevent identical
scaffolds from appearing in both training and testing sets.

```rust
use oxidane::prelude::*;
use oxidane::molecule::scaffold::{murcko_framework, murcko_scaffold, scaffold_key};

fn main() {
    // Imatinib (Gleevec) — BCR-ABL tyrosine kinase inhibitor
    let imatinib = MolecularGraph::from_smiles_single(
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5"
    ).unwrap();

    // 1. Scaffold SMILES key for group-by operations:
    let key = scaffold_key(&imatinib);
    assert!(!key.is_empty());

    // 2. Extracted scaffold as a MolecularGraph:
    let scaffold = murcko_scaffold(&imatinib);
    assert!(scaffold.len() < imatinib.len());

    // 3. Carbon framework: converts all heteroatoms and bond orders to single carbon bonds
    let framework = murcko_framework(&imatinib);
    assert!(framework.atomic_numbers().iter().all(|&z| z == 6 || z == 1));
}
```

---

## 6. 3D Conformer Generation and Force Field Relaxation

Distance geometry converts a 2D connection table into a reasonable 3D conformer
by calculating upper and lower interatomic distance bounds, performing
multidimensional scaling (MDS), and refining atomic positions against chiral
volume constraints.

With the default configuration, candidates are relaxed using MMFF94 when its
parameters cover the molecule, with UFF as a fallback. `relax: false` disables
this step. Embedding returns a conformer, not a guarantee of the global minimum
or the experimentally populated structure.

```rust
use oxidane::prelude::*;

fn main() {
    // Caffeine: planar bicyclic xanthine core
    let caffeine = MolecularGraph::from_smiles_single("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();

    // Default embedding: 5 trials + torsion search + MMFF94 relaxation
    let conformer_3d = caffeine.embed_3d().unwrap();
    assert_eq!(conformer_3d.positions.len(), conformer_3d.atomic_numbers().len());

    // Fast embedding preset: single trial, suitable for large chemical libraries
    let fast_config = EmbedConfig::fast();
    let fast_conformer = caffeine.embed_3d_with_config(&fast_config).unwrap();
    assert_eq!(fast_conformer.positions.len(), fast_conformer.atomic_numbers().len());
}
```

---

## 7. Reaction Informatics

Reaction SMARTS allows mapped chemical transformations where changes in bonds,
formal charges, and elements are applied programmatically. Hydrogen counts are
recomputed at modified atoms. These are graph rewrites, not predictions of
reaction feasibility, conditions, or yield. Every template atom needs a unique
map number; unmapped reagents and byproducts are not inferred. A result from
`apply_to` contains the whole rewritten graph, including disconnected components.

```rust
use oxidane::prelude::*;
use oxidane::molecule::reaction::ReactionSmarts;

fn main() {
    // Amide bond formation: Carboxylic Acid + Amine -> Amide + Water
    let amide_coupling = ReactionSmarts::parse(
        "[C:1](=[O:2])[O;H1:3].[N;H2:4]>>[C:1](=[O:2])[N:4].[O:3]"
    ).unwrap();

    let benzoic_acid = MolecularGraph::from_smiles_single("c1ccccc1C(=O)O").unwrap();
    let methylamine = MolecularGraph::from_smiles_single("CN").unwrap();

    assert!(amide_coupling.check_mass_conservation().is_ok());
    let products = amide_coupling.apply_to(&[&benzoic_acid, &methylamine]);
    assert!(!products.is_empty());

    use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
    let expected = canonicalize_and_sort_smiles("c1ccccc1C(=O)NC.O", false).unwrap();
    for product in products {
        // N-methylbenzamide and water are disconnected parts of one result.
        assert_eq!(canonicalize_and_sort_smiles(&product.to_smiles(), false).unwrap(), expected);
    }
}
```

---

## 8. 2D Depiction

Oxidane renders molecular graphs directly into crisp SVG without external GUI
libraries. Layout uses deterministic ring templates and tetrahedral wedges.
Crowded bridged rings and macrocycles use a best-effort projection, and extended
stereo classes such as allene or square-planar tags are not depicted. Inspect
complex structures before using a drawing in a publication.

```rust
use oxidane::prelude::*;

fn main() {
    // Acetylsalicylic acid (Aspirin)
    let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();

    // 1. Default standard SVG:
    let svg = aspirin.to_svg();
    assert!(svg.starts_with("<svg"));
    assert!(svg.trim_end().ends_with("</svg>"));

    // 2. Custom styling:
    let custom_options = DrawOptions {
        width: 800,
        height: 600,
        bond_length: 42.0,
        padding: 30.0,
        font_size: 18.0,
        stroke_width: 2.0,
        atom_colors: true,
        cleanup: false,
    };
    let styled_svg = aspirin.to_svg_with_options(&custom_options);
    assert!(styled_svg.contains("width=\"800\""));

    // 3. For bridged cages (e.g., Camphor or Norbornane), enable 2D cleanup
    let camphor = MolecularGraph::from_smiles_single("CC1(C)C2CCC1(C)C(=O)C2").unwrap();
    let cleaned_options = DrawOptions { cleanup: true, ..Default::default() };
    let cleaned_svg = camphor.to_svg_with_options(&cleaned_options);
    assert!(cleaned_svg.starts_with("<svg"));
}
```

---

## 9. Machine Learning Preparation

Oxidane includes built-in graph featurizers and dataset splitters designed for
machine learning models (like Graph Neural Networks).

### Scaffold-Based Splitting
```rust
use oxidane::prelude::*;
use oxidane::ml::split::{Fractions, scaffold_split};
use oxidane::molecule::scaffold::scaffold_key;

fn main() {
    let smiles_list = [
        "CC(=O)Nc1ccc(O)cc1",             // Paracetamol (benzene scaffold)
        "CC(=O)Oc1ccccc1C(=O)O",          // Aspirin (benzene scaffold)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     // Ibuprofen (benzene scaffold)
        "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O", // Morphine (morphinan)
        "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O", // Codeine (morphinan)
    ];

    let mols: Vec<MolecularGraph> = smiles_list
        .iter()
        .map(|s| MolecularGraph::from_smiles_single(s).unwrap())
        .collect();

    let scaffold_keys: Vec<String> = mols.iter().map(|m| scaffold_key(m)).collect();

    // Request 80% train, 0% validation, 20% test:
    let fractions = Fractions::new(0.8, 0.0, 0.2);
    let split = scaffold_split(&scaffold_keys, fractions);

    // Whole scaffold groups stay together; exact fractions may be impossible.
    assert_eq!(split.len(), mols.len());
    for i in 0..mols.len() {
        for j in 0..i {
            if scaffold_keys[i] == scaffold_keys[j] {
                assert_eq!(split.side_of(i), split.side_of(j));
            }
        }
    }
}
```

### Atom and Bond Featurization Matrices
```rust
use oxidane::prelude::*;
use oxidane::molecule::featurize::graph_features;

fn main() {
    let caffeine = MolecularGraph::from_smiles_single("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();
    let features = graph_features(&caffeine);

    // Atom rows include explicit hydrogens; columns include element, charge and aromaticity
    assert_eq!(features.atom_count, caffeine.len());
    let atom_0_features = features.atom_row(0);
    assert!(!atom_0_features.is_empty());

    // Directed bond features: order one-hot, conjugation, ring membership, stereochemistry
    assert!(features.edge_count > 0);
    let edge_0_features = features.bond_row(0);
    assert!(!edge_0_features.is_empty());
}
```
