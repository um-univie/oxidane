//! The data layer: scaffolds, batching, splitting, normalization, datasets.
//!
//! The scaffold cases came from RDKit's `MurckoScaffold.GetScaffoldForMol`,
//! checked on 4996 molecules by `examples/scaffold_diff.rs` against
//! `tests/scaffold_oracle.py`. The three that differ there are hydrogen counts
//! the writer places differently on hypervalent phosphorus and on a charged
//! aromatic nitrogen, not scaffolds; the cases below are ones where the two
//! agree exactly.

use crate::graph_algorithms::mol_canonization::canonize_mol;
use crate::io::ToSMILES;
use crate::ml::batch::GraphBatch;
use crate::ml::dataset::Dataset;
use crate::ml::normalize::FeatureStats;
use crate::ml::split::{Fractions, Side, grouped_split, hashed_side, random_split, scaffold_split};
use crate::molecule::base::MoleculeCore;
use crate::molecule::featurize::{
    ATOM_CONTINUOUS_COLUMNS, ATOM_FEATURE_WIDTH, FEATURE_SCHEMA_VERSION, graph_features,
};
use crate::molecule::scaffold::{
    framework_key, murcko_scaffold, murcko_scaffold_atoms, scaffold_key,
};
use crate::prelude::*;

fn parse(smiles: &str) -> MolecularGraph {
    MolecularGraph::from_smiles_single(smiles).unwrap()
}

/// `(input, scaffold, framework)`, every expectation from RDKit.
const SCAFFOLDS: &[(&str, &str, &str)] = &[
    // No rings, no scaffold.
    ("CCCC", "", ""),
    ("O", "", ""),
    ("C(=O)(O)C", "", ""),
    // The ring itself.
    ("c1ccccc1", "c1ccccc1", "C1CCCCC1"),
    ("c1ccc2ccccc2c1", "c1ccc2ccccc2c1", "C1CCC2CCCCC2C1"),
    ("C1CC2CCC1CC2", "C1CC2CCC1CC2", "C1CC2CCC1CC2"),
    // Side chains come off, however long.
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccccc1", "C1CCCCC1"),
    ("CCOC(=O)c1ccc(N)cc1", "c1ccccc1", "C1CCCCC1"),
    ("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "C1CCCCC1"),
    ("N#Cc1ccccc1", "c1ccccc1", "C1CCCCC1"),
    ("C1CC1C(=O)NC", "C1CC1", "C1CC1"),
    // An exocyclic multiple bond holds its atom on.
    ("O=C1CCCN1C", "O=C1CCCN1", "CC1CCCC1"),
    ("C1CCC(=O)CC1", "O=C1CCCCC1", "CC1CCCCC1"),
    ("C1CCCCC1=NC", "N=C1CCCCC1", "CC1CCCCC1"),
    ("C1CCCCC1=CC(=O)O", "C=C1CCCCC1", "CC1CCCCC1"),
    // ...but only what the bond itself holds. The acetyl on the ring nitrogen
    // goes entirely; the ring's own carbonyl stays.
    ("CC(=O)N1CCCC1=O", "O=C1CCCN1", "CC1CCCC1"),
    // Linkers between two ring systems are the scaffold too.
    (
        "c1ccc(CCc2ccncc2)cc1",
        "c1ccc(CCc2ccncc2)cc1",
        "C1CCC(CCC2CCCCC2)CC1",
    ),
    (
        "c1ccccc1C(=O)Nc1ccccc1",
        "O=C(Nc1ccccc1)c1ccccc1",
        "CC(CC1CCCCC1)C1CCCCC1",
    ),
];

#[test]
fn scaffolds_match_rdkit() {
    for &(smiles, expected_scaffold, expected_framework) in SCAFFOLDS {
        let molecule = parse(smiles);
        let scaffold = scaffold_key(&molecule);
        let expected = if expected_scaffold.is_empty() {
            String::new()
        } else {
            canonize_mol(&parse(expected_scaffold)).to_smiles()
        };
        assert_eq!(scaffold, expected, "scaffold of {smiles}");

        let framework = framework_key(&molecule);
        let expected = if expected_framework.is_empty() {
            String::new()
        } else {
            canonize_mol(&parse(expected_framework)).to_smiles()
        };
        assert_eq!(framework, expected, "framework of {smiles}");
    }
}

#[test]
fn a_substituted_aromatic_nitrogen_gets_its_hydrogen_back() {
    // Cutting the aromatic form leaves an `n` with two ring bonds and no
    // hydrogen, which is not a molecule and which RDKit refuses to read. The
    // cut is made on a Kekule copy for exactly this.
    for smiles in [
        "c1ccc2[nH]ccc2c1",
        "c1ccc2[nH]c(C)cc2c1",
        "CCC(Cn1ccc2c1cc(CNCCC)cc2)CC",
    ] {
        assert_eq!(
            scaffold_key(&parse(smiles)),
            canonize_mol(&parse("c1ccc2[nH]ccc2c1")).to_smiles(),
            "indole scaffold of {smiles}"
        );
    }
}

#[test]
fn a_scaffold_is_a_whole_molecule() {
    // Every atom has the hydrogens its bonds leave room for, rather than the
    // ones the parent gave it.
    let scaffold = murcko_scaffold(&parse("CC(C)Cc1ccc(cc1)C(C)C(=O)O"));
    assert_eq!(scaffold.molecular_formula().to_string(), "C6H6");
}

#[test]
fn a_scaffold_does_not_depend_on_how_the_molecule_was_written() {
    let writings = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(C)Oc1ccccc1C(O)=O",
        "OC(=O)c1ccccc1OC(C)=O",
    ];
    let keys: Vec<String> = writings.iter().map(|s| scaffold_key(&parse(s))).collect();
    assert!(keys.windows(2).all(|pair| pair[0] == pair[1]), "{keys:?}");
}

#[test]
fn scaffold_atoms_are_ascending_and_within_range() {
    let molecule = parse("CC(=O)Nc1ccc(O)cc1");
    let atoms = murcko_scaffold_atoms(&molecule);
    assert!(atoms.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(atoms.iter().all(|&atom| atom < molecule.len()));
    assert!(
        atoms
            .iter()
            .all(|&atom| molecule.get_atomic_number(atom) != 1)
    );
}

#[test]
fn collating_lays_graphs_side_by_side() {
    let ethanol = graph_features(&parse("CCO"));
    let benzene = graph_features(&parse("c1ccccc1"));
    let batch = GraphBatch::collate(&[ethanol.clone(), benzene.clone()]);

    assert_eq!(batch.graph_count, 2);
    assert_eq!(batch.atom_count, ethanol.atom_count + benzene.atom_count);
    assert_eq!(batch.edge_count, ethanol.edge_count + benzene.edge_count);
    assert_eq!(batch.atom_offsets, vec![0, 3, 9]);
    assert_eq!(batch.batch, vec![0, 0, 0, 1, 1, 1, 1, 1, 1]);

    // Every feature row survives unchanged, and in order.
    assert_eq!(batch.atom_row(0), ethanol.atom_row(0));
    assert_eq!(batch.atom_row(3), benzene.atom_row(0));

    // No edge crosses between the two molecules.
    for edge in 0..batch.edge_count {
        let (from, to) = batch.edge(edge);
        assert_eq!(batch.batch[from], batch.batch[to], "edge {edge} crosses");
    }
}

#[test]
fn collating_nothing_gives_an_empty_batch() {
    let batch = GraphBatch::collate(&[]);
    assert_eq!(batch.graph_count, 0);
    assert_eq!(batch.atom_count, 0);
    assert_eq!(batch.atom_offsets, vec![0]);
    assert!(batch.segment_sum(&[], 4).is_empty());
}

#[test]
fn pooling_keeps_molecules_apart() {
    let batch = GraphBatch::collate(&[
        graph_features(&parse("CCO")),
        graph_features(&parse("c1ccccc1")),
    ]);

    // One value per atom: its position in the batch.
    let rows: Vec<f32> = (0..batch.atom_count).map(|row| row as f32).collect();
    assert_eq!(batch.segment_sum(&rows, 1), vec![0.0 + 1.0 + 2.0, 3.0 + 8.0 + 4.0 + 5.0 + 6.0 + 7.0]);
    assert_eq!(batch.segment_max(&rows, 1), vec![2.0, 8.0]);
    let means = batch.segment_mean(&rows, 1);
    assert!((means[0] - 1.0).abs() < 1e-6);
    assert!((means[1] - 5.5).abs() < 1e-6);
}

#[test]
fn pooling_a_molecule_with_no_heavy_atoms_gives_zeros() {
    // Molecular hydrogen has no rows: featurization leaves hydrogens out.
    let batch = GraphBatch::collate(&[
        graph_features(&parse("[HH]")),
        graph_features(&parse("CCO")),
    ]);
    assert_eq!(batch.graph_atom_count(0), 0);
    let rows: Vec<f32> = (0..batch.atom_count).map(|_| 1.0).collect();
    assert_eq!(batch.segment_mean(&rows, 1)[0], 0.0);
    assert_eq!(batch.segment_max(&rows, 1)[0], 0.0);
}

#[test]
fn scattering_edges_reaches_every_atom_of_a_connected_molecule() {
    let batch = GraphBatch::collate(&[graph_features(&parse("c1ccccc1C(=O)O"))]);
    let messages = vec![1.0f32; batch.edge_count];
    let gathered = batch.scatter_edges_to_atoms(&messages, 1);

    let degrees = batch.in_degrees();
    assert_eq!(gathered.len(), batch.atom_count);
    for atom in 0..batch.atom_count {
        assert_eq!(gathered[atom], degrees[atom] as f32);
        assert!(degrees[atom] > 0, "atom {atom} receives nothing");
    }
    // Both directions of every bond are present, so in-degree sums to the edge count.
    assert_eq!(degrees.iter().sum::<u32>() as usize, batch.edge_count);
}

#[test]
fn normalization_scales_the_quantities_and_leaves_the_indicators_alone() {
    let batch = GraphBatch::collate(&[
        graph_features(&parse("CCO")),
        graph_features(&parse("c1ccccc1Br")),
        graph_features(&parse("CI")),
    ]);
    let stats = FeatureStats::fit(&batch);
    assert_eq!(stats.schema_version, FEATURE_SCHEMA_VERSION);

    let mut scaled = batch.atom_features.clone();
    stats.apply(&mut scaled);

    for (row, (before, after)) in batch
        .atom_features
        .chunks_exact(ATOM_FEATURE_WIDTH)
        .zip(scaled.chunks_exact(ATOM_FEATURE_WIDTH))
        .enumerate()
    {
        for column in 0..ATOM_FEATURE_WIDTH {
            if ATOM_CONTINUOUS_COLUMNS.contains(&column) {
                continue;
            }
            assert_eq!(before[column], after[column], "row {row} column {column}");
        }
    }

    // The mass column now has mean zero and unit deviation.
    let mass = ATOM_CONTINUOUS_COLUMNS[0];
    let masses: Vec<f32> = scaled
        .chunks_exact(ATOM_FEATURE_WIDTH)
        .map(|row| row[mass])
        .collect();
    let mean = masses.iter().sum::<f32>() / masses.len() as f32;
    let variance = masses.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / masses.len() as f32;
    assert!(mean.abs() < 1e-4, "mean {mean}");
    assert!((variance - 1.0).abs() < 1e-3, "variance {variance}");
}

#[test]
fn normalization_inverts() {
    let batch = GraphBatch::collate(&[graph_features(&parse("c1ccccc1Br"))]);
    let stats = FeatureStats::fit(&batch);
    let mut rows = batch.atom_features.clone();
    stats.apply(&mut rows);
    stats.invert(&mut rows);
    for (before, after) in batch.atom_features.iter().zip(rows.iter()) {
        assert!((before - after).abs() < 1e-3, "{before} != {after}");
    }
}

#[test]
fn normalization_of_a_constant_column_does_not_divide_by_zero() {
    // One molecule of one element: the mass column does not vary at all.
    let batch = GraphBatch::collate(&[graph_features(&parse("CCCC"))]);
    let stats = FeatureStats::fit(&batch);
    let mut rows = batch.atom_features.clone();
    stats.apply(&mut rows);
    assert!(rows.iter().all(|value| value.is_finite()));
}

#[test]
fn splits_are_disjoint_and_complete() {
    for count in [0, 1, 2, 7, 100, 1001] {
        let split = random_split(count, Fractions::default(), 7);
        assert_eq!(split.len(), count, "count {count}");
        let mut all: Vec<usize> = split
            .train
            .iter()
            .chain(&split.valid)
            .chain(&split.test)
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..count).collect::<Vec<_>>(), "count {count}");
        assert!(split.train.windows(2).all(|pair| pair[0] < pair[1]));
    }
}

#[test]
fn a_random_split_is_the_same_split_every_time() {
    assert_eq!(
        random_split(500, Fractions::default(), 42),
        random_split(500, Fractions::default(), 42)
    );
    assert_ne!(
        random_split(500, Fractions::default(), 42),
        random_split(500, Fractions::default(), 43)
    );
}

#[test]
fn a_random_split_hits_its_fractions() {
    let split = random_split(1000, Fractions::new(0.8, 0.1, 0.1), 1);
    assert_eq!(split.train.len(), 800);
    assert_eq!(split.valid.len(), 100);
    assert_eq!(split.test.len(), 100);
}

#[test]
fn a_scaffold_split_never_separates_a_scaffold() {
    let smiles = [
        "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC", // benzene
        "c1ccncc1C", "c1ccncc1CC", "c1ccncc1CCC", // pyridine
        "C1CCCCC1C", "C1CCCCC1CC", // cyclohexane
        "CCCC", "CCCCC", // no scaffold at all
    ];
    let keys: Vec<String> = smiles.iter().map(|s| scaffold_key(&parse(s))).collect();
    let split = scaffold_split(&keys, Fractions::new(0.6, 0.2, 0.2));

    assert_eq!(split.len(), smiles.len());
    for (a, key_a) in keys.iter().enumerate() {
        if key_a.is_empty() {
            continue;
        }
        for (b, key_b) in keys.iter().enumerate() {
            if key_a == key_b {
                assert_eq!(
                    split.side_of(a),
                    split.side_of(b),
                    "{} and {} share a scaffold",
                    smiles[a],
                    smiles[b]
                );
            }
        }
    }
}

#[test]
fn a_scaffold_split_does_not_put_everything_in_train() {
    // Filling train to its quota first is the obvious rule and the wrong one:
    // the large groups all land in train and the test set is left as whatever
    // singletons are over.
    let smiles: Vec<String> = (0..12)
        .flat_map(|ring| {
            let core = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCCC1"][ring % 4];
            (0..4).map(move |tail| format!("{}{}", core, "C".repeat(tail + 1)))
        })
        .collect();
    let keys: Vec<String> = smiles.iter().map(|s| scaffold_key(&parse(s))).collect();
    let split = scaffold_split(&keys, Fractions::new(0.5, 0.25, 0.25));

    assert!(!split.valid.is_empty(), "{split:?}");
    assert!(!split.test.is_empty(), "{split:?}");
}

#[test]
fn a_scaffold_split_is_the_same_split_every_time() {
    let smiles = [
        "c1ccccc1C", "c1ccncc1C", "C1CCCCC1C", "c1ccccc1CC", "CCCC", "c1ccc2ccccc2c1",
    ];
    let keys: Vec<String> = smiles.iter().map(|s| scaffold_key(&parse(s))).collect();
    let first = scaffold_split(&keys, Fractions::default());
    for _ in 0..8 {
        assert_eq!(scaffold_split(&keys, Fractions::default()), first);
    }
}

#[test]
fn acyclic_molecules_do_not_all_land_on_one_side() {
    // They all share the empty key, so treating that as a group would put
    // every one of them wherever the group went.
    let keys: Vec<String> = (0..100).map(|_| String::new()).collect();
    let split = scaffold_split(&keys, Fractions::new(0.5, 0.25, 0.25));
    assert!(!split.train.is_empty());
    assert!(!split.valid.is_empty());
    assert!(!split.test.is_empty());
}

#[test]
fn grouped_splits_keep_their_groups() {
    let groups = vec![vec![0, 1, 2, 3], vec![4, 5], vec![6], vec![7, 8, 9]];
    let split = grouped_split(&groups, 10, Fractions::new(0.5, 0.25, 0.25));
    assert_eq!(split.len(), 10);
    for group in &groups {
        let side = split.side_of(group[0]);
        assert!(group.iter().all(|&index| split.side_of(index) == side));
    }
}

#[test]
fn a_hashed_side_does_not_depend_on_the_dataset() {
    let key = scaffold_key(&parse("c1ccccc1C"));
    let side = hashed_side(&key, Fractions::default());
    for _ in 0..8 {
        assert_eq!(hashed_side(&key, Fractions::default()), side);
    }

    // And over many keys it lands roughly on the fractions asked for.
    let sides: Vec<Side> = (0..4000)
        .map(|index| hashed_side(&format!("scaffold-{index}"), Fractions::new(0.8, 0.1, 0.1)))
        .collect();
    let train = sides.iter().filter(|&&s| s == Side::Train).count();
    assert!((2800..3600).contains(&train), "train {train} of 4000");
}

#[test]
fn a_dataset_hands_out_every_molecule_once_per_epoch() {
    let mut dataset = Dataset::new(2);
    for index in 0..17 {
        let molecule = parse(&format!("C{}O", "C".repeat(index % 5)));
        dataset.push(&molecule, &[Some(index as f64), None]);
    }

    let mut seen: Vec<usize> = dataset
        .epoch(4, 11)
        .flat_map(|batch| batch.indices)
        .collect();
    assert_eq!(seen.len(), 17);
    seen.sort_unstable();
    assert_eq!(seen, (0..17).collect::<Vec<_>>());
}

#[test]
fn a_dataset_batch_carries_the_targets_of_the_molecules_in_it() {
    let mut dataset = Dataset::new(2);
    for index in 0..6 {
        dataset.push(&parse("CCO"), &[Some(index as f64), None]);
    }

    for batch in dataset.epoch(4, 3) {
        assert_eq!(batch.targets.len(), batch.graph.graph_count * 2);
        assert_eq!(batch.measured_count(), batch.graph.graph_count);
        for (position, &index) in batch.indices.iter().enumerate() {
            let (targets, mask) = batch.targets_of(position);
            assert_eq!(targets[0], index as f32);
            assert_eq!(mask, [true, false]);
        }
    }
}

#[test]
fn an_unshuffled_epoch_keeps_the_dataset_order() {
    let mut dataset = Dataset::new(1);
    for _ in 0..5 {
        dataset.push(&parse("CCO"), &[Some(0.0)]);
    }
    let seen: Vec<usize> = dataset.epoch(2, 0).flat_map(|batch| batch.indices).collect();
    assert_eq!(seen, vec![0, 1, 2, 3, 4]);
}

#[test]
fn a_dataset_iterates_only_the_side_it_was_given() {
    let mut dataset = Dataset::new(1);
    for _ in 0..20 {
        dataset.push(&parse("CCO"), &[Some(1.0)]);
    }
    let split = random_split(20, Fractions::new(0.5, 0.25, 0.25), 5);
    let seen: Vec<usize> = dataset
        .epoch_side(&split, Side::Test, 3, 0)
        .flat_map(|batch| batch.indices)
        .collect();
    assert_eq!(seen, split.test);
}

#[test]
#[should_panic(expected = "targets in a dataset of")]
fn a_ragged_target_row_is_refused() {
    let mut dataset = Dataset::new(3);
    dataset.push(&parse("CCO"), &[Some(1.0)]);
}
