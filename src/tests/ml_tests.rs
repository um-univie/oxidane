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
use crate::ml::normalize::{BondFeatureStats, ClassMeans, FeatureStats, GlobalStats, TargetStats};
use crate::ml::split::{
    Fractions, Side, grouped_split, hashed_side, random_split, scaffold_split,
    scaffold_split_balanced,
};
use crate::ml::tokenize::{
    END, PADDING, RESERVED, START, SmilesVocabulary, TokenBatch, UNKNOWN, tokenize,
};
use crate::molecule::base::MoleculeCore;
use crate::molecule::descriptors::MolecularDescriptors;
use crate::molecule::featurize::{
    ATOM_CONTINUOUS_COLUMNS, ATOM_FEATURE_WIDTH, BOND_CONTINUOUS_COLUMNS, BOND_FEATURE_WIDTH,
    FEATURE_SCHEMA_VERSION, GraphFeatures, MOLECULE_FEATURE_WIDTH, atom_class, bond_class,
    graph_features, graph_features_with_geometry, molecule_features,
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
    // Ethanol: 3 heavy + 6 H = 9. Benzene: 6 heavy + 6 H = 12.
    assert_eq!(batch.atom_offsets, vec![0, 9, 21]);
    assert_eq!(batch.batch, [[0u32; 9].as_slice(), &[1; 12]].concat());

    // Every feature row survives unchanged, and in order.
    assert_eq!(batch.atom_row(0), ethanol.atom_row(0));
    assert_eq!(batch.atom_row(9), benzene.atom_row(0));

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

    // One value per atom: its position in the batch. Ethanol takes rows
    // 0-8, benzene 9-20.
    let rows: Vec<f32> = (0..batch.atom_count).map(|row| row as f32).collect();
    assert_eq!(
        batch.segment_sum(&rows, 1),
        vec![(0..9).sum::<i32>() as f32, (9..21).sum::<i32>() as f32]
    );
    assert_eq!(batch.segment_max(&rows, 1), vec![8.0, 20.0]);
    let means = batch.segment_mean(&rows, 1);
    assert!((means[0] - 4.0).abs() < 1e-6);
    assert!((means[1] - 14.5).abs() < 1e-6);
}

#[test]
fn pooling_a_molecule_with_no_atom_rows_gives_zeros() {
    // Every real molecule has at least one row now that hydrogens get one
    // too, so this exercises the defensive zero-atom-count path in
    // `segment_mean`/`segment_max` directly, on a `GraphFeatures` built by
    // hand rather than on a molecule that can no longer produce one.
    let empty = GraphFeatures {
        atom_count: 0,
        atom_features: Vec::new(),
        edge_count: 0,
        bond_features: Vec::new(),
        edge_index: Vec::new(),
        atoms: Vec::new(),
    };
    let batch = GraphBatch::collate(&[empty, graph_features(&parse("CCO"))]);
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
        "c1ccccc1C",
        "c1ccccc1CC",
        "c1ccccc1CCC",
        "c1ccccc1CCCC", // benzene
        "c1ccncc1C",
        "c1ccncc1CC",
        "c1ccncc1CCC", // pyridine
        "C1CCCCC1C",
        "C1CCCCC1CC", // cyclohexane
        "CCCC",
        "CCCCC", // no scaffold at all
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
        "c1ccccc1C",
        "c1ccncc1C",
        "C1CCCCC1C",
        "c1ccccc1CC",
        "CCCC",
        "c1ccc2ccccc2c1",
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
    let seen: Vec<usize> = dataset
        .epoch(2, 0)
        .flat_map(|batch| batch.indices)
        .collect();
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

#[test]
fn atom_labels_land_on_the_row_of_the_atom_they_name() {
    // CCO: atoms 0-2 are the heavy atoms, in their original order, since
    // rows follow atom index one to one now that hydrogens get rows too --
    // atoms 3-8 are its six hydrogens, all unlabelled here.
    let features = graph_features(&parse("CCO"));
    let labels = std::collections::HashMap::from([(0, 1.5), (2, -2.5)]);
    assert_eq!(
        features.atom_targets_from(&labels),
        [
            Some(1.5),
            None,
            Some(-2.5),
            None,
            None,
            None,
            None,
            None,
            None
        ]
    );
}

#[test]
fn bond_labels_land_on_both_directed_edges_of_the_bond_they_name() {
    let features = graph_features(&parse("CCO"));
    // The C-C bond, the C-O bond, and the six C-H/O-H bonds.
    assert_eq!(features.bonds().len(), 8, "{:?}", features.bonds());

    let labels = std::collections::HashMap::from([((0, 1), 88.9)]);
    let per_edge = features.bond_targets_from(&labels);
    assert_eq!(per_edge.len(), features.edge_count);
    // Both directions of the C-C bond carry the label; every other bond
    // (including every C-H/O-H one) has none.
    assert_eq!(per_edge.iter().filter(|v| **v == Some(88.9)).count(), 2);
    assert_eq!(per_edge.iter().filter(|v| v.is_none()).count(), 14);
}

#[test]
fn a_dataset_carries_atom_level_targets_row_for_row() {
    let mut dataset = Dataset::new(0).with_atom_tasks(1);
    let features = graph_features(&parse("CCO"));
    let labels = std::collections::HashMap::from([(0, 1.5), (2, -2.5)]);
    let atom_targets = features.atom_targets_from(&labels);
    dataset.push_features(features, &[], &atom_targets, &[]);

    let (values, mask) = dataset.atom_targets(0);
    assert_eq!(values, [1.5, 0.0, -2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        mask,
        [true, false, true, false, false, false, false, false, false]
    );

    let batch = dataset.batch(&[0]);
    assert_eq!(batch.atom_targets, values);
    assert_eq!(batch.atom_mask, mask);
    assert_eq!(batch.atom_task_count, 1);
    assert_eq!(batch.atom_targets.len(), batch.graph.atom_count);
}

#[test]
fn a_dataset_carries_bond_level_targets_edge_for_edge() {
    let mut dataset = Dataset::new(0).with_bond_tasks(1);
    let features = graph_features(&parse("CCO"));
    let labels = std::collections::HashMap::from([((0, 1), 88.9)]);
    let bond_targets = features.bond_targets_from(&labels);
    dataset.push_features(features, &[], &[], &bond_targets);

    let (values, mask) = dataset.bond_targets(0);
    assert_eq!(mask.iter().filter(|&&m| m).count(), 2);
    assert_eq!(values.iter().filter(|&&v| v == 88.9).count(), 2);

    let batch = dataset.batch(&[0]);
    assert_eq!(batch.bond_targets, values);
    assert_eq!(batch.bond_mask, mask);
    assert_eq!(batch.bond_task_count, 1);
    assert_eq!(batch.bond_targets.len(), batch.graph.edge_count);
}

#[test]
fn atom_and_bond_targets_of_several_molecules_stay_in_their_own_rows() {
    // CCO has 9 atom rows (3 heavy + 6 H), CC has 8 (2 heavy + 6 H) --
    // offsets that differ from a fixed `task_count * index` stride would
    // misalign the second molecule's rows against the first's leftovers.
    let mut dataset = Dataset::new(0).with_atom_tasks(1);
    for (smiles, label) in [("CCO", 1.0), ("CC", 2.0)] {
        let features = graph_features(&parse(smiles));
        let labels: std::collections::HashMap<usize, f64> =
            (0..features.atom_count).map(|atom| (atom, label)).collect();
        let atom_targets = features.atom_targets_from(&labels);
        dataset.push_features(features, &[], &atom_targets, &[]);
    }
    let (first, _) = dataset.atom_targets(0);
    let (second, _) = dataset.atom_targets(1);
    assert_eq!(first, [1.0; 9]);
    assert_eq!(second, [2.0; 8]);

    let batch = dataset.batch(&[0, 1]);
    assert_eq!(
        batch.atom_targets,
        [[1.0; 9].as_slice(), [2.0; 8].as_slice()].concat()
    );
}

#[test]
fn atom_class_groups_chemically_similar_atoms_and_splits_different_ones() {
    let features = graph_features(&parse("CCO"));
    // The two carbons are both sp3, non-aromatic -- the same class despite
    // different substitution (one is a terminal methyl, the other bonded to
    // oxygen); the oxygen is a different element entirely.
    let carbon_a = atom_class(features.atom_row(0));
    let carbon_b = atom_class(features.atom_row(1));
    let oxygen = atom_class(features.atom_row(2));
    assert_eq!(carbon_a, carbon_b);
    assert_ne!(carbon_a, oxygen);

    // An aromatic carbon is a different class from an sp3 one, even though
    // both are carbon.
    let toluene = graph_features(&parse("Cc1ccccc1"));
    let methyl = atom_class(toluene.atom_row(0));
    let ring = atom_class(toluene.atom_row(1));
    assert_ne!(methyl, ring);
}

#[test]
fn bond_class_is_order_independent_and_distinguishes_element_pairs() {
    let features = graph_features(&parse("CCO"));
    let edge_for = |a: usize, b: usize| -> usize {
        (0..features.edge_count)
            .find(|&edge| features.edge(edge) == (a, b))
            .unwrap()
    };

    // Reading the C-C bond in either direction gives the same class.
    let forward = bond_class(
        features.atom_row(0),
        features.atom_row(1),
        features.bond_row(edge_for(0, 1)),
    );
    let backward = bond_class(
        features.atom_row(1),
        features.atom_row(0),
        features.bond_row(edge_for(1, 0)),
    );
    assert_eq!(
        forward, backward,
        "reading a bond the other way changed its class"
    );

    // A C-O bond is a different class from a C-C bond.
    let carbon_oxygen = bond_class(
        features.atom_row(1),
        features.atom_row(2),
        features.bond_row(edge_for(1, 2)),
    );
    assert_ne!(forward, carbon_oxygen);
}

#[test]
fn atom_and_bond_classes_are_computed_automatically_and_stay_aligned() {
    let mut dataset = Dataset::new(0).with_atom_tasks(1).with_bond_tasks(1);
    for smiles in ["CCO", "CC"] {
        let features = graph_features(&parse(smiles));
        let atom_targets = vec![Some(0.0); features.atom_count];
        let bond_targets = vec![Some(0.0); features.edge_count];
        dataset.push_features(features, &[], &atom_targets, &bond_targets);
    }
    assert_eq!(dataset.atom_classes(0).len(), 9); // CCO: 3 heavy + 6 H
    assert_eq!(dataset.atom_classes(1).len(), 8); // CC: 2 heavy + 6 H
    // The two carbons of CCO share a class -- see the dedicated class test.
    assert_eq!(dataset.atom_classes(0)[0], dataset.atom_classes(0)[1]);

    let batch = dataset.batch(&[0, 1]);
    assert_eq!(batch.atom_class.len(), batch.graph.atom_count);
    assert_eq!(batch.bond_class.len(), batch.graph.edge_count);
    assert_eq!(&batch.atom_class[0..9], dataset.atom_classes(0));
    assert_eq!(&batch.atom_class[9..17], dataset.atom_classes(1));
}

#[test]
fn class_means_averages_per_class_and_falls_back_for_an_unseen_class() {
    let mut dataset = Dataset::new(0).with_atom_tasks(1);
    // Ethanol: both carbons are sp3, non-aromatic -- one class. 10.0 and
    // 20.0 average to 15.0. Its oxygen and six hydrogens are left unmeasured.
    let ethanol = graph_features(&parse("CCO"));
    let ethanol_targets = [
        Some(10.0),
        Some(20.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    dataset.push_features(ethanol, &[], &ethanol_targets, &[]);
    // Ammonia: a different element entirely, a different class. 5.0 for the
    // nitrogen; its three hydrogens are left unmeasured.
    let ammonia = graph_features(&parse("N"));
    dataset.push_features(ammonia, &[], &[Some(5.0), None, None, None], &[]);
    // Toluene's aromatic ring carbons are yet another class (sp2, aromatic)
    // -- left entirely unmeasured, so the fit never sees it.
    let toluene = graph_features(&parse("Cc1ccccc1"));
    let toluene_targets = vec![None; toluene.atom_count];
    dataset.push_features(toluene, &[], &toluene_targets, &[]);

    let means = ClassMeans::fit_atom(&dataset, &[0, 1, 2]);
    let overall_mean = (10.0 + 20.0 + 5.0) / 3.0;

    let carbon_class = dataset.atom_classes(0)[0];
    let mut residual = [10.0f32, 20.0];
    means.subtract(&[carbon_class, carbon_class], &mut residual);
    assert!((residual[0] - (10.0 - 15.0)).abs() < 1e-3, "{residual:?}");
    assert!((residual[1] - (20.0 - 15.0)).abs() < 1e-3, "{residual:?}");

    // The aromatic ring carbon's class was never measured -- it falls back
    // to the overall mean, not zero and not the sp3-carbon class's mean.
    let ring_class = dataset.atom_classes(2)[1];
    assert_ne!(ring_class, carbon_class);
    let mut value = [0.0f32];
    means.subtract(&[ring_class], &mut value);
    assert!(
        (value[0] - -(overall_mean as f32)).abs() < 1e-3,
        "{value:?} vs fallback {overall_mean}"
    );

    let mut restored = residual;
    means.add_back(&[carbon_class, carbon_class], &mut restored);
    assert!((restored[0] - 10.0).abs() < 1e-3);
    assert!((restored[1] - 20.0).abs() < 1e-3);
}

#[test]
fn a_balanced_scaffold_split_still_keeps_its_groups_together() {
    let smiles: Vec<String> = (0..40)
        .map(|index| {
            let core = [
                "c1ccccc1",
                "c1ccncc1",
                "C1CCCCC1",
                "C1CCCC1",
                "c1ccc2ccccc2c1",
            ][index % 5];
            format!("{}{}", core, "C".repeat(index / 5 + 1))
        })
        .collect();
    let keys: Vec<String> = smiles.iter().map(|s| scaffold_key(&parse(s))).collect();

    for seed in 0..4 {
        let split = scaffold_split_balanced(&keys, Fractions::default(), seed);
        assert_eq!(split.len(), keys.len(), "seed {seed}");
        for (a, key_a) in keys.iter().enumerate() {
            for (b, key_b) in keys.iter().enumerate() {
                if key_a == key_b && !key_a.is_empty() {
                    assert_eq!(split.side_of(a), split.side_of(b), "seed {seed}");
                }
            }
        }
    }
}

#[test]
fn a_balanced_scaffold_split_moves_with_its_seed_and_not_otherwise() {
    let smiles: Vec<String> = (0..200)
        .map(|index| {
            let core = [
                "c1ccccc1",
                "c1ccncc1",
                "C1CCCCC1",
                "C1CCCC1",
                "c1ccc2ccccc2c1",
                "C1CC1",
                "c1cscc1",
                "c1ccoc1",
                "C1CCNCC1",
                "c1cnc2ccccc2c1",
            ][index % 10];
            format!("{}{}", core, "C".repeat(index / 10 + 1))
        })
        .collect();
    let keys: Vec<String> = smiles.iter().map(|s| scaffold_key(&parse(s))).collect();

    let first = scaffold_split_balanced(&keys, Fractions::default(), 0);
    assert_eq!(
        first,
        scaffold_split_balanced(&keys, Fractions::default(), 0)
    );

    let moved =
        (1..6).any(|seed| scaffold_split_balanced(&keys, Fractions::default(), seed) != first);
    assert!(moved, "every seed gave the same split");
}

#[test]
fn molecule_features_are_the_descriptors_the_crate_already_computes() {
    let caffeine = parse("Cn1cnc2c1c(=O)n(C)c(=O)n2C");
    let row = molecule_features(&caffeine);
    assert_eq!(row.len(), MOLECULE_FEATURE_WIDTH);
    assert_eq!(row[0], caffeine.heavy_atom_count() as f32);
    assert!((row[1] - caffeine.molecular_weight() as f32).abs() < 1e-3);
    assert!((row[2] - caffeine.crippen_log_p() as f32).abs() < 1e-4);
    assert!((row[4] - caffeine.topological_polar_surface_area() as f32).abs() < 1e-3);
    assert_eq!(row[8], caffeine.ring_count() as f32);
    assert!(row.iter().all(|value| value.is_finite()));
}

#[test]
fn molecule_features_do_not_depend_on_how_the_molecule_was_written() {
    let writings = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(C)Oc1ccccc1C(O)=O",
        "OC(=O)c1ccccc1OC(C)=O",
    ];
    let rows: Vec<Vec<f32>> = writings
        .iter()
        .map(|s| molecule_features(&parse(s)))
        .collect();
    for other in &rows[1..] {
        for (a, b) in rows[0].iter().zip(other) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }
}

#[test]
fn a_dataset_carries_descriptors_through_to_the_batch() {
    let mut dataset = Dataset::new(1);
    for smiles in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"] {
        dataset.push_with_globals(&parse(smiles), &[Some(1.0)]);
    }
    assert_eq!(dataset.global_width(), MOLECULE_FEATURE_WIDTH);

    let batch = dataset.batch(&[2, 0]);
    assert_eq!(batch.graph.global_width, MOLECULE_FEATURE_WIDTH);
    assert_eq!(
        batch.graph.global_features.len(),
        2 * MOLECULE_FEATURE_WIDTH
    );
    // In the order asked for, not the order stored.
    assert_eq!(
        &batch.graph.global_features[..MOLECULE_FEATURE_WIDTH],
        dataset.globals(2)
    );
}

#[test]
fn a_dataset_without_descriptors_has_none() {
    let mut dataset = Dataset::new(1);
    dataset.push(&parse("CCO"), &[Some(1.0)]);
    assert_eq!(dataset.global_width(), 0);
    assert!(dataset.globals(0).is_empty());
    assert!(dataset.batch(&[0]).graph.global_features.is_empty());
}

#[test]
fn descriptor_scaling_centres_every_column() {
    let mut dataset = Dataset::new(1);
    for smiles in [
        "CCO",
        "c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CCCCCCCCCC",
        "c1ccc2ccccc2c1O",
        "CN(C)CCOC(c1ccccc1)c1ccccc1",
    ] {
        dataset.push_with_globals(&parse(smiles), &[Some(0.0)]);
    }
    let batch = dataset.batch(&(0..dataset.len()).collect::<Vec<_>>());
    let stats = GlobalStats::fit(&batch.graph);
    assert!(!stats.is_empty());

    let mut scaled = batch.graph.global_features.clone();
    stats.apply(&mut scaled);
    for column in 0..MOLECULE_FEATURE_WIDTH {
        let values: Vec<f32> = scaled
            .chunks_exact(MOLECULE_FEATURE_WIDTH)
            .map(|row| row[column])
            .collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!(mean.abs() < 1e-3, "column {column} mean {mean}");
        assert!(values.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn descriptor_scaling_of_an_empty_batch_does_nothing() {
    let mut dataset = Dataset::new(1);
    dataset.push(&parse("CCO"), &[Some(0.0)]);
    let stats = GlobalStats::fit(&dataset.batch(&[0]).graph);
    assert!(stats.is_empty());
    let mut nothing: Vec<f32> = Vec::new();
    stats.apply(&mut nothing);
    assert!(nothing.is_empty());
}

#[test]
fn target_scaling_centres_each_task_over_the_rows_it_was_given() {
    let mut dataset = Dataset::new(2);
    for (index, value) in [1.0, 3.0, 5.0, 7.0, 9.0].iter().enumerate() {
        dataset.push(&parse("CCO"), &[Some(*value), Some(value * 10.0)]);
        let _ = index;
    }
    let stats = TargetStats::fit(&dataset, &[0, 1, 2, 3, 4]);
    assert!((stats.mean[0] - 5.0).abs() < 1e-5, "{:?}", stats.mean);
    assert!((stats.mean[1] - 50.0).abs() < 1e-4, "{:?}", stats.mean);
    // The second task is ten times the first, so its spread is too.
    assert!((stats.deviation[1] / stats.deviation[0] - 10.0).abs() < 1e-3);

    let mut rows: Vec<f32> = vec![1.0, 10.0, 9.0, 90.0];
    stats.apply(&mut rows);
    assert!(rows.iter().all(|value| value.abs() < 3.0), "{rows:?}");
    stats.invert(&mut rows);
    for (found, expected) in rows.iter().zip([1.0, 10.0, 9.0, 90.0]) {
        assert!((found - expected).abs() < 1e-3, "{found} != {expected}");
    }
}

#[test]
fn target_scaling_skips_the_measurements_that_were_never_made() {
    // Counting a missing measurement as zero would drag both statistics
    // toward the origin in proportion to how sparse the task is.
    let mut dataset = Dataset::new(1);
    for value in [10.0, 12.0, 14.0] {
        dataset.push(&parse("CCO"), &[Some(value)]);
    }
    for _ in 0..20 {
        dataset.push(&parse("CCO"), &[None]);
    }
    let stats = TargetStats::fit(&dataset, &(0..dataset.len()).collect::<Vec<_>>());
    assert!((stats.mean[0] - 12.0).abs() < 1e-4, "{:?}", stats.mean);
}

#[test]
fn target_scaling_of_a_constant_task_does_not_divide_by_zero() {
    let mut dataset = Dataset::new(1);
    for _ in 0..4 {
        dataset.push(&parse("CCO"), &[Some(7.0)]);
    }
    let stats = TargetStats::fit(&dataset, &[0, 1, 2, 3]);
    let mut rows = vec![7.0f32, 7.0];
    stats.apply(&mut rows);
    assert!(rows.iter().all(|value| value.is_finite()), "{rows:?}");
}

#[test]
fn target_scaling_is_fitted_on_the_rows_it_is_given_and_no_others() {
    let mut dataset = Dataset::new(1);
    for value in [0.0, 0.0, 0.0, 100.0, 100.0, 100.0] {
        dataset.push(&parse("CCO"), &[Some(value)]);
    }
    // The training side alone, so the held-out rows cannot reach the model
    // through the scaling.
    let stats = TargetStats::fit(&dataset, &[0, 1, 2]);
    assert!((stats.mean[0]).abs() < 1e-6, "{:?}", stats.mean);
}

#[test]
fn atom_target_scaling_pools_every_labelled_atom_across_every_molecule() {
    let mut dataset = Dataset::new(0).with_atom_tasks(1);
    // Molecule one: atoms 0 and 2 labelled 0.0 and 10.0, atom 1 unmeasured.
    let features = graph_features(&parse("CCO"));
    let labels = std::collections::HashMap::from([(0, 0.0), (2, 10.0)]);
    let atom_targets = features.atom_targets_from(&labels);
    dataset.push_features(features, &[], &atom_targets, &[]);
    // Molecule two: its one atom labelled 20.0.
    let features = graph_features(&parse("C"));
    let labels = std::collections::HashMap::from([(0, 20.0)]);
    let atom_targets = features.atom_targets_from(&labels);
    dataset.push_features(features, &[], &atom_targets, &[]);

    let stats = TargetStats::fit_atom(&dataset, &[0, 1]);
    // Three measurements, the unmeasured atom skipped rather than read as 0.
    assert!((stats.mean[0] - 10.0).abs() < 1e-4, "{:?}", stats.mean);
}

#[test]
fn bond_target_scaling_pools_every_labelled_bond_across_every_molecule() {
    let mut dataset = Dataset::new(0).with_bond_tasks(1);
    let features = graph_features(&parse("CCO"));
    let labels = std::collections::HashMap::from([((0, 1), 88.0), ((1, 2), 92.0)]);
    let bond_targets = features.bond_targets_from(&labels);
    dataset.push_features(features, &[], &[], &bond_targets);

    let stats = TargetStats::fit_bond(&dataset, &[0]);
    // Each bond is measured on both its directed edges, but that duplication
    // is symmetric and leaves the mean the two labels' mean, not skewed
    // toward either.
    assert!((stats.mean[0] - 90.0).abs() < 1e-4, "{:?}", stats.mean);
}

#[test]
fn geometry_free_features_leave_the_new_columns_at_zero() {
    let features = graph_features(&parse("CCO"));
    let angle_column = ATOM_FEATURE_WIDTH - 1;
    let distance_column = BOND_FEATURE_WIDTH - 1;
    for row in 0..features.atom_count {
        assert_eq!(features.atom_row(row)[angle_column], 0.0, "atom row {row}");
    }
    for edge in 0..features.edge_count {
        assert_eq!(features.bond_row(edge)[distance_column], 0.0, "edge {edge}");
    }
}

#[test]
fn geometry_features_carry_the_embedded_bond_length_and_a_real_angle() {
    let molecule = parse("CCO");
    let embedded = molecule.embed_3d().unwrap();
    let features = graph_features_with_geometry(&molecule, &embedded.positions);
    let distance_column = BOND_FEATURE_WIDTH - 1;
    let angle_column = ATOM_FEATURE_WIDTH - 1;

    // Every bond's stored length is the actual distance between its two
    // embedded endpoints -- a wrong column offset would show up here.
    for edge in 0..features.edge_count {
        let (source, target) = features.edge(edge);
        let expected = embedded.positions[source].distance(&embedded.positions[target]) as f32;
        let stored = features.bond_row(edge)[distance_column];
        assert!(
            (stored - expected).abs() < 1e-4,
            "edge {edge}: {stored} != {expected}"
        );
    }

    // The middle carbon has two heavy-atom neighbours (plus hydrogens), so it
    // has a real local angle -- not the 0.0 a fewer-than-two-neighbour atom
    // (every hydrogen here) falls back to.
    let has_a_nonzero_angle =
        (0..features.atom_count).any(|atom| features.atom_row(atom)[angle_column] != 0.0);
    assert!(
        has_a_nonzero_angle,
        "no atom carried a nonzero mean local angle"
    );
}

#[test]
fn bond_feature_scaling_centres_the_distance_column_and_leaves_indicators_alone() {
    let molecules = ["CCO", "CI", "c1ccccc1Br"];
    let features: Vec<GraphFeatures> = molecules
        .iter()
        .map(|smiles| {
            let molecule = parse(smiles);
            let embedded = molecule.embed_3d().unwrap();
            graph_features_with_geometry(&molecule, &embedded.positions)
        })
        .collect();
    let batch = GraphBatch::collate(&features);
    let stats = BondFeatureStats::fit(&batch);
    assert_eq!(stats.schema_version, FEATURE_SCHEMA_VERSION);

    let mut scaled = batch.bond_features.clone();
    stats.apply(&mut scaled);
    for (edge, (before, after)) in batch
        .bond_features
        .chunks_exact(BOND_FEATURE_WIDTH)
        .zip(scaled.chunks_exact(BOND_FEATURE_WIDTH))
        .enumerate()
    {
        for column in 0..BOND_FEATURE_WIDTH {
            if BOND_CONTINUOUS_COLUMNS.contains(&column) {
                continue;
            }
            assert_eq!(before[column], after[column], "edge {edge} column {column}");
        }
    }

    let distance = BOND_CONTINUOUS_COLUMNS[0];
    let distances: Vec<f32> = scaled
        .chunks_exact(BOND_FEATURE_WIDTH)
        .map(|row| row[distance])
        .collect();
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    let variance =
        distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
    assert!(mean.abs() < 1e-3, "mean {mean}");
    assert!((variance - 1.0).abs() < 1e-3, "variance {variance}");
}

#[test]
fn bond_feature_scaling_inverts() {
    let molecule = parse("c1ccccc1Br");
    let embedded = molecule.embed_3d().unwrap();
    let features = graph_features_with_geometry(&molecule, &embedded.positions);
    let batch = GraphBatch::collate(&[features]);
    let stats = BondFeatureStats::fit(&batch);
    let mut rows = batch.bond_features.clone();
    stats.apply(&mut rows);
    stats.invert(&mut rows);
    for (before, after) in batch.bond_features.iter().zip(rows.iter()) {
        assert!((before - after).abs() < 1e-3, "{before} != {after}");
    }
}

// --------------------------------------------------------------- tokenize --

#[test]
fn a_two_letter_element_is_one_token() {
    assert_eq!(tokenize("CC(=O)Cl"), ["C", "C", "(", "=", "O", ")", "Cl"]);
    assert_eq!(tokenize("BrCCBr"), ["Br", "C", "C", "Br"]);
    // A lone B and a lone C are elements in their own right, and the letter
    // after them decides nothing on its own.
    assert_eq!(tokenize("BC"), ["B", "C"]);
}

#[test]
fn a_bracket_atom_is_one_token() {
    assert_eq!(
        tokenize("c1cc[nH]c1"),
        ["c", "1", "c", "c", "[nH]", "c", "1"]
    );
    assert_eq!(tokenize("[13CH3][Fe+3]"), ["[13CH3]", "[Fe+3]"]);
    // An unclosed bracket is malformed rather than a reason to lose bytes.
    assert_eq!(tokenize("C[nH"), ["C", "[nH"]);
}

#[test]
fn a_wide_ring_closure_is_one_token() {
    assert_eq!(tokenize("C%10CCC%10"), ["C", "%10", "C", "C", "C", "%10"]);
    // A percent with nothing usable after it stays a character of its own.
    assert_eq!(tokenize("C%1"), ["C", "%", "1"]);
}

#[test]
fn tokenizing_loses_nothing() {
    for smiles in [
        "CC(=O)Oc1ccccc1C(=O)O",
        "C[C@H](N)C(=O)O",
        "c1cc2c(cc1)[nH]c1ccccc12",
        "[Na+].[Cl-]",
        "C%10CC%10Br",
    ] {
        assert_eq!(tokenize(smiles).concat(), smiles);
    }
}

#[test]
fn a_vocabulary_keeps_what_the_corpus_repeats() {
    let vocabulary = SmilesVocabulary::from_corpus(["CCO", "CCC", "[Fe]"], 2);
    assert!(vocabulary.contains("C"));
    assert!(!vocabulary.contains("[Fe]"));
    assert!(!vocabulary.contains("O"));

    // The reserved ids are counted, since they are columns of a decoder's
    // output like any other.
    assert_eq!(vocabulary.len(), RESERVED + vocabulary.tokens().len());
}

#[test]
fn a_vocabulary_is_ordered_by_the_corpus_and_not_by_the_reading_order() {
    let forwards = SmilesVocabulary::from_corpus(["CCO", "c1ccccc1", "CCN"], 1);
    let backwards = SmilesVocabulary::from_corpus(["CCN", "c1ccccc1", "CCO"], 1);
    assert_eq!(forwards, backwards);
}

#[test]
fn a_vocabulary_round_trips_a_molecule_and_a_file() {
    let smiles = "CC(=O)Oc1ccccc1C(=O)O";
    let vocabulary = SmilesVocabulary::from_corpus([smiles], 1);

    let ids = vocabulary.encode_known(smiles).unwrap();
    assert_eq!(vocabulary.decode(&ids), smiles);

    // A token the corpus never had cannot be spelled, and saying so is the
    // point: such a molecule belongs out of the training set.
    assert!(vocabulary.encode_known("CC[Fe]").is_none());
    assert!(vocabulary.encode("CC[Fe]").contains(&UNKNOWN));

    let reloaded = SmilesVocabulary::from_lines(&vocabulary.to_lines());
    assert_eq!(reloaded, vocabulary);
    assert_eq!(reloaded.encode_known(smiles).unwrap(), ids);
}

#[test]
fn a_token_batch_pads_to_the_longest_and_masks_the_padding() {
    let vocabulary = SmilesVocabulary::from_corpus(["CCO", "CO"], 1);
    let sequences = vec![vocabulary.encode("CCO"), vocabulary.encode("CO")];
    let batch = TokenBatch::collate(&sequences);

    assert_eq!((batch.rows, batch.width), (2, 4));
    // Every row is read from the start marker and written to the end one.
    assert_eq!(batch.inputs[0], START);
    assert_eq!(batch.inputs[batch.width], START);
    assert_eq!(batch.targets[3], END);
    assert_eq!(batch.targets[batch.width + 2], END);

    // The short row's fourth step is padding, and is not learned from.
    assert_eq!(batch.targets[batch.width + 3], PADDING);
    assert_eq!(batch.mask[batch.width + 3], 0.0);
    assert_eq!(batch.mask.iter().sum::<f32>(), 4.0 + 3.0);

    // What went in comes back out, end marker excluded.
    assert_eq!(batch.sequence(0), sequences[0].as_slice());
    assert_eq!(batch.sequence(1), sequences[1].as_slice());
}

#[test]
fn a_token_batch_of_nothing_still_has_a_step() {
    let batch = TokenBatch::collate(&[Vec::new()]);
    assert_eq!((batch.rows, batch.width), (1, 1));
    assert_eq!(batch.targets[0], END);
    assert_eq!(batch.mask[0], 1.0);
    assert!(batch.sequence(0).is_empty());
}

/// The stereo columns of every bond in `smiles`, as `(atom, atom, slot)` with
/// the atoms in ascending order and only bonds that carry one.
fn stereo_bonds(smiles: &str) -> Vec<(usize, usize, usize)> {
    let features = graph_features(&parse(smiles));
    let stereo = BOND_FEATURE_WIDTH - 5..BOND_FEATURE_WIDTH - 1;
    let mut found: Vec<(usize, usize, usize)> = (0..features.edge_count)
        .filter_map(|edge| {
            let (a, b) = features.edge(edge);
            let slot = features.bond_row(edge)[stereo.clone()]
                .iter()
                .position(|&value| value > 0.5)?;
            (slot > 0 && a < b).then_some((a, b, slot))
        })
        .collect();
    found.sort_unstable();
    found
}

/// A double bond's geometry is read against its highest-ranked substituents,
/// so cis and trans come out as such however the SMILES leans its bonds, and
/// the single bonds beside it carry nothing. The stored class used to be read
/// straight into the column -- onto every bond touching either end, and
/// relative to whichever neighbour the string happened to list first.
#[test]
fn only_the_double_bond_carries_its_geometry() {
    const CIS: usize = 1;
    const TRANS: usize = 2;
    assert_eq!(stereo_bonds("C/C=C/C"), [(1, 2, TRANS)]);
    assert_eq!(stereo_bonds("C/C=C\\C"), [(1, 2, CIS)]);
    assert_eq!(stereo_bonds("C(/C)=C/C"), [(0, 2, CIS)]);
    // Bromine outranks fluorine, so it is the reference on its end: trans to
    // chlorine in the second spelling, cis in the first.
    assert_eq!(stereo_bonds("F/C(Br)=C/Cl"), [(1, 3, CIS)]);
    assert_eq!(stereo_bonds("Br/C(F)=C/Cl"), [(1, 3, TRANS)]);
    assert!(stereo_bonds("CC=CC").is_empty());
}

/// The same molecule spelled from different starting atoms gets the same
/// bond rows, as a multiset.
#[test]
fn bond_features_do_not_depend_on_the_spelling() {
    let sorted_rows = |smiles: &str| {
        let features = graph_features(&parse(smiles));
        let mut rows: Vec<Vec<u32>> = (0..features.edge_count)
            .map(|edge| {
                features
                    .bond_row(edge)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        rows.sort_unstable();
        rows
    };
    for writings in [
        &["C/C=C(\\C)[Si]1CCCCO1", "C(=C\\C)(/[Si]1OCCCC1)C"][..],
        &["C=CN=C/C=C(\\C)C(=O)I", "C(=C)N=C/C=C(/C(I)=O)C"],
        &["CC(/C=C(\\N)C1CCCC1)N", "C1CCC(/C(=C/C(N)C)N)C1"],
    ] {
        let first = sorted_rows(writings[0]);
        for other in &writings[1..] {
            assert_eq!(sorted_rows(other), first, "{} against {other}", writings[0]);
        }
    }
}


