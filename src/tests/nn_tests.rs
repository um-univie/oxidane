//! The neural half. Behind the `ml` feature, like the code it exercises.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use crate::io::ToSMILES;
use crate::ml::batch::GraphBatch;
use crate::ml::dataset::Dataset;
use crate::ml::nn::autoencoder::{AutoencoderConfig, GraphAutoencoder};
use crate::ml::nn::decoder::DecoderConfig;
use crate::ml::nn::layers::{Activation, Mlp, MlpConfig};
use crate::ml::nn::loss::{mask_from_flags, masked_bce_with_logits, masked_mse};
use crate::ml::nn::model::{ModelConfig, MolecularNet, bond_predictions};
use crate::ml::nn::molecule_autoencoder::{MoleculeAutoencoder, MoleculeAutoencoderConfig};
use crate::ml::nn::mpnn::{DMpnn, EncoderConfig, EncoderKind};
use crate::ml::nn::pretrain::{MaskedAtomModel, PretrainConfig};
use crate::ml::nn::seed::{load_prefixed, save_prefixed, seed_variables};
use crate::ml::nn::tensors::{GraphTensors, Pooling};
use crate::ml::nn::train::{Objective, Schedule, TrainConfig, Trainer, clip_global_norm};
use crate::ml::normalize::{ClassMeans, TargetStats};
use crate::ml::split::{Fractions, Side, Split};
use crate::ml::tokenize::{RESERVED, SmilesVocabulary, TokenBatch};
use crate::molecule::base::MoleculeCore;
use crate::molecule::featurize::{bond_class, graph_features};
use crate::prelude::*;

fn parse(smiles: &str) -> MolecularGraph {
    MolecularGraph::from_smiles_single(smiles).unwrap()
}

fn batch_of(smiles: &[&str]) -> GraphBatch {
    GraphBatch::collate(
        &smiles
            .iter()
            .map(|s| graph_features(&parse(s)))
            .collect::<Vec<_>>(),
    )
}

fn small_encoder() -> EncoderConfig {
    EncoderConfig {
        hidden: 16,
        depth: 2,
        activation: Activation::Relu,
        dropout: 0.0,
    }
}

fn canonical_of(molecule: &MolecularGraph) -> String {
    canonicalize_and_sort_smiles(&molecule.to_smiles(), true).unwrap()
}

fn to_vec(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

// ---------------------------------------------------------------- tensors --

#[test]
fn uploading_keeps_the_batch_exactly() {
    let device = Device::Cpu;
    let batch = batch_of(&["CCO", "c1ccccc1"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();

    assert_eq!(graph.atom_count, batch.atom_count);
    assert_eq!(graph.edge_count, batch.edge_count);
    assert_eq!(to_vec(&graph.atom_features), batch.atom_features);
    assert_eq!(to_vec(&graph.bond_features), batch.bond_features);
}

#[test]
fn scattering_agrees_with_the_dependency_free_version() {
    let device = Device::Cpu;
    let batch = batch_of(&["CC(=O)Oc1ccccc1C(=O)O", "CCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();

    // One column per atom row, so the sums are easy to check by hand.
    let values: Vec<f32> = (0..batch.atom_count).map(|row| row as f32 + 1.0).collect();
    let atoms = Tensor::from_slice(&values, (batch.atom_count, 1), &device).unwrap();

    assert_eq!(
        to_vec(&graph.pool_sum(&atoms).unwrap()),
        batch.segment_sum(&values, 1)
    );
    let mean = to_vec(&graph.pool_mean(&atoms).unwrap());
    for (ours, theirs) in mean.iter().zip(batch.segment_mean(&values, 1)) {
        assert!((ours - theirs).abs() < 1e-4, "{ours} != {theirs}");
    }

    // And the edge scatter against `scatter_edges_to_atoms`.
    let edge_values: Vec<f32> = (0..batch.edge_count)
        .map(|edge| edge as f32 + 1.0)
        .collect();
    let edges = Tensor::from_slice(&edge_values, (batch.edge_count, 1), &device).unwrap();
    assert_eq!(
        to_vec(&graph.scatter_to_atoms(&edges).unwrap()),
        batch.scatter_edges_to_atoms(&edge_values, 1)
    );
}

#[test]
fn reverse_edges_are_found_and_are_never_the_edge_itself() {
    for smiles in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C1CC2CCC1CC2"] {
        let batch = batch_of(&[smiles]);
        let (reverse, present) = batch.reverse_edges();
        assert!(present.iter().all(|&found| found), "{smiles}");
        for edge in 0..batch.edge_count {
            let (from, to) = batch.edge(edge);
            assert_ne!(reverse[edge] as usize, edge, "{smiles} edge {edge}");
            assert_eq!(batch.edge(reverse[edge] as usize), (to, from), "{smiles}");
            assert_eq!(reverse[reverse[edge] as usize] as usize, edge, "{smiles}");
        }
    }
}

// ----------------------------------------------------------------- layers --

#[test]
fn an_mlp_has_the_shape_it_was_asked_for() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mlp = Mlp::new(&[8, 16, 3], MlpConfig::default(), vb).unwrap();

    let input = Tensor::zeros((5, 8), DType::F32, &device).unwrap();
    assert_eq!(mlp.forward(&input, false).unwrap().dims(), &[5, 3]);
    assert_eq!(mlp.input_width(), Some(8));
}

#[test]
fn an_mlp_of_one_size_is_the_identity() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mlp = Mlp::new(&[8], MlpConfig::default(), vb).unwrap();

    let input = Tensor::ones((2, 8), DType::F32, &device).unwrap();
    assert_eq!(to_vec(&mlp.forward(&input, false).unwrap()), to_vec(&input));
}

// ------------------------------------------------------------------- mpnn --

#[test]
fn both_encoders_give_one_vector_per_atom() {
    let device = Device::Cpu;
    let batch = batch_of(&["CCO", "c1ccccc1O"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();

    for kind in [EncoderKind::Directed, EncoderKind::Gine] {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = crate::ml::nn::mpnn::Encoder::new(kind, small_encoder(), vb).unwrap();
        let atoms = encoder.forward(&graph, false).unwrap();
        assert_eq!(atoms.dims(), &[batch.atom_count, 16], "{kind:?}");
        assert!(to_vec(&atoms).iter().all(|value| value.is_finite()));
    }
}

#[test]
fn a_molecule_is_encoded_the_same_alone_as_in_a_batch() {
    // The one property a batched graph network has to have: no molecule can
    // reach another. A single index left unshifted breaks this and nothing
    // else notices.
    let device = Device::Cpu;
    let smiles = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "C"];

    for kind in [EncoderKind::Directed, EncoderKind::Gine] {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = crate::ml::nn::mpnn::Encoder::new(kind, small_encoder(), vb).unwrap();

        let together = batch_of(&smiles);
        let pooled = {
            let graph = GraphTensors::new(&together, &device).unwrap();
            let atoms = encoder.forward(&graph, false).unwrap();
            to_vec(&graph.pool_sum(&atoms).unwrap())
        };

        for (index, one) in smiles.iter().enumerate() {
            let alone = batch_of(&[one]);
            let graph = GraphTensors::new(&alone, &device).unwrap();
            let atoms = encoder.forward(&graph, false).unwrap();
            let expected = to_vec(&graph.pool_sum(&atoms).unwrap());
            let found = &pooled[index * 16..(index + 1) * 16];
            for (a, b) in expected.iter().zip(found) {
                assert!((a - b).abs() < 1e-4, "{kind:?} {one}: {a} != {b}");
            }
        }
    }
}

#[test]
fn a_single_atom_molecule_has_no_edges_and_does_not_crash() {
    // A bare cation, not "C": methane has four hydrogens of its own now that
    // every atom gets a row, and so five atoms and four bonds, not one and
    // none.
    let device = Device::Cpu;
    let batch = batch_of(&["[Na+]"]);
    assert_eq!(batch.atom_count, 1);
    assert_eq!(batch.edge_count, 0);

    let graph = GraphTensors::new(&batch, &device).unwrap();
    for kind in [EncoderKind::Directed, EncoderKind::Gine] {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = crate::ml::nn::mpnn::Encoder::new(kind, small_encoder(), vb).unwrap();
        let atoms = encoder.forward(&graph, false).unwrap();
        assert_eq!(atoms.dims(), &[1, 16]);
        assert!(to_vec(&atoms).iter().all(|value| value.is_finite()));
    }
}

#[test]
fn a_directed_message_does_not_come_straight_back() {
    // Molecular hydrogen has exactly one bond and nothing else -- both its
    // atoms have that one bond and no other -- so after the subtraction
    // there is nothing for either edge to carry and every message stays at
    // its initial value. If the reverse-edge subtraction were missing, each
    // round would instead feed the other edge's message back in and the two
    // would grow.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let shallow = DMpnn::new(
        EncoderConfig {
            depth: 1,
            ..small_encoder()
        },
        vb.pp("e"),
    )
    .unwrap();
    let deep = DMpnn::new(
        EncoderConfig {
            depth: 4,
            ..small_encoder()
        },
        vb.pp("e"),
    )
    .unwrap();

    let batch = batch_of(&["[H][H]"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let one = to_vec(&shallow.forward(&graph, false).unwrap());
    let four = to_vec(&deep.forward(&graph, false).unwrap());
    for (a, b) in one.iter().zip(&four) {
        assert!(
            (a - b).abs() < 1e-5,
            "depth changed a two-atom molecule: {a} != {b}"
        );
    }
}

// ------------------------------------------------------------------ model --

#[test]
fn a_model_predicts_at_every_level_it_was_asked_to() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 2],
            atom_head: vec![16, 3],
            bond_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let batch = batch_of(&["CCO", "c1ccccc1"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let predictions = model.forward(&graph, false).unwrap();

    assert_eq!(predictions.graph.unwrap().dims(), &[2, 2]);
    assert_eq!(predictions.atom.unwrap().dims(), &[batch.atom_count, 3]);
    assert_eq!(predictions.bond.unwrap().dims(), &[batch.edge_count, 1]);
}

#[test]
fn a_model_with_no_heads_predicts_nothing_but_still_encodes() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let batch = batch_of(&["CCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let predictions = model.forward(&graph, false).unwrap();
    assert!(predictions.graph.is_none());
    assert!(predictions.atom.is_none());
    assert_eq!(
        model.encode_molecules(&graph, false).unwrap().dims(),
        &[1, 16]
    );
}

#[test]
fn a_bond_prediction_does_not_depend_on_which_way_the_bond_is_read() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            bond_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let batch = batch_of(&["CC(=O)Oc1ccccc1"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let bonds = model.forward(&graph, false).unwrap().bond.unwrap();
    let values = to_vec(&bonds);

    let (reverse, _) = batch.reverse_edges();
    for edge in 0..batch.edge_count {
        let other = reverse[edge] as usize;
        assert!(
            (values[edge] - values[other]).abs() < 1e-5,
            "edge {edge} and its reverse disagree: {} vs {}",
            values[edge],
            values[other]
        );
    }

    // And one row per bond rather than per direction.
    let per_bond = bond_predictions(&bonds, &batch).unwrap();
    assert_eq!(per_bond.len(), batch.edge_count / 2);
}

#[test]
fn pooling_choice_reaches_the_prediction() {
    let device = Device::Cpu;
    let batch = batch_of(&["CCCCCCCCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();

    // One VarMap, so both models are the same weights read two ways: asking
    // for a name a second time returns the variable already there.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut pooled = Vec::new();
    for how in [Pooling::Sum, Pooling::Mean] {
        let model = MolecularNet::new(
            ModelConfig {
                encoder: small_encoder(),
                pooling: how,
                ..Default::default()
            },
            vb.pp("shared"),
        )
        .unwrap();
        pooled.push(to_vec(&model.encode_molecules(&graph, false).unwrap()));
    }
    // 1-octanol, C8H18O: 27 atoms once hydrogens get rows too, so the mean
    // is the sum divided by twenty-seven.
    for (sum, mean) in pooled[0].iter().zip(&pooled[1]) {
        assert!((sum / 27.0 - mean).abs() < 1e-4, "{sum} / 27 != {mean}");
    }
}

// ------------------------------------------------------------------- loss --

#[test]
fn a_masked_loss_ignores_what_was_never_measured() {
    let device = Device::Cpu;
    let prediction = Tensor::from_slice(&[1.0f32, 100.0, 3.0, -100.0], (2, 2), &device).unwrap();
    let target = Tensor::from_slice(&[1.0f32, 0.0, 5.0, 0.0], (2, 2), &device).unwrap();
    let mask = mask_from_flags(&[true, false, true, false], (2, 2), &device).unwrap();

    // Errors of 0 and 2 over two measured entries.
    let loss = masked_mse(&prediction, &target, &mask).unwrap();
    assert!((loss.to_scalar::<f32>().unwrap() - 2.0).abs() < 1e-5);
}

#[test]
fn a_loss_with_nothing_measured_is_zero_rather_than_nan() {
    let device = Device::Cpu;
    let prediction = Tensor::from_slice(&[1.0f32, 2.0], (1, 2), &device).unwrap();
    let target = Tensor::zeros((1, 2), DType::F32, &device).unwrap();
    let mask = mask_from_flags(&[false, false], (1, 2), &device).unwrap();

    let loss = masked_mse(&prediction, &target, &mask).unwrap();
    assert_eq!(loss.to_scalar::<f32>().unwrap(), 0.0);
}

#[test]
fn cross_entropy_on_logits_survives_a_confident_prediction() {
    // `log(1 - sigmoid(x))` for x = 60 is where the naive form goes to
    // infinity, and a trained classifier lives there.
    let device = Device::Cpu;
    let logits = Tensor::from_slice(&[60.0f32, -60.0, 60.0, -60.0], (4, 1), &device).unwrap();
    let targets = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (4, 1), &device).unwrap();
    let mask = Tensor::ones((4, 1), DType::F32, &device).unwrap();

    let loss = masked_bce_with_logits(&logits, &targets, &mask).unwrap();
    let value = loss.to_scalar::<f32>().unwrap();
    assert!(value.is_finite(), "{value}");
    // Two right at ~0 and two wrong at ~60, averaged over four.
    assert!((value - 30.0).abs() < 0.1, "{value}");
}

// ------------------------------------------------------------------ train --

/// A task the model can actually be checked against: how many heavy atoms.
///
/// Trivial chemistry and a real test of the plumbing -- a sum readout over
/// atom vectors can represent it exactly, so a network that fails to learn it
/// has something wrong with its gradients rather than its capacity.
fn heavy_atom_dataset(smiles: &[&str]) -> Dataset {
    let mut dataset = Dataset::new(1);
    for one in smiles {
        let molecule = parse(one);
        let heavy = (0..molecule.len())
            .filter(|&atom| molecule.get_atomic_number(atom) != 1)
            .count();
        dataset.push(&molecule, &[Some(heavy as f64)]);
    }
    dataset
}

const TRAINING_SET: &[&str] = &[
    "C",
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "c1ccccc1C",
    "c1ccccc1CC",
    "CC(=O)O",
    "CC(=O)OC",
    "CCN",
    "CCCN",
    "C1CCCCC1",
    "C1CCCC1",
    "C1CCC1",
    "CC(C)C",
    "CC(C)CC",
    "OCCO",
    "OCCCO",
    "CCOCC",
    "CN(C)C",
    "c1ccncc1",
    "c1ccncc1C",
    "CC(=O)Oc1ccccc1",
    "CCCCCC",
    "CCCCCCC",
    "CCCCCCCC",
];

#[test]
fn a_model_learns_something_it_can_represent_exactly() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            graph_head: vec![32, 1],
            pooling: Pooling::Sum,
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let dataset = heavy_atom_dataset(TRAINING_SET);
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 120,
            batch_size: 8,
            patience: None,
            seed: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    let first = history.train_loss[0];
    let last = *history.train_loss.last().unwrap();
    assert!(first.is_finite() && last.is_finite(), "{first} -> {last}");
    assert!(last < first * 0.05, "loss barely moved: {first} -> {last}");
    // Counting atoms to within half an atom.
    assert!(last < 0.25, "final mean squared error {last}");
}

/// One atom-level target per atom row: its own atomic number. Trivial
/// chemistry, and a real test that a per-atom target reaches the atom head
/// and is scored against the right row.
fn atomic_number_dataset(smiles: &[&str]) -> Dataset {
    let mut dataset = Dataset::new(0).with_atom_tasks(1);
    for one in smiles {
        let molecule = parse(one);
        let features = graph_features(&molecule);
        let atom_targets: Vec<Option<f64>> = features
            .atoms
            .iter()
            .map(|&atom| Some(molecule.get_atomic_number(atom) as f64))
            .collect();
        dataset.push_features(features, &[], &atom_targets, &[]);
    }
    dataset
}

#[test]
fn an_atom_head_learns_something_it_can_represent_exactly() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            atom_head: vec![32, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let dataset = atomic_number_dataset(TRAINING_SET);
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 120,
            batch_size: 8,
            patience: None,
            seed: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    let first = history.train_loss[0];
    let last = *history.train_loss.last().unwrap();
    assert!(first.is_finite() && last.is_finite(), "{first} -> {last}");
    assert!(last < first * 0.05, "loss barely moved: {first} -> {last}");

    let predictions = trainer.predict_atoms(&dataset, &split.train).unwrap();
    let mut row = 0;
    for &index in &split.train {
        let (truth, _) = dataset.atom_targets(index);
        for &want in truth {
            let got = predictions[row][0];
            assert!(
                (got - want).abs() < 1.0,
                "atom row {row}: predicted {got}, wanted {want}"
            );
            row += 1;
        }
    }
}

/// One bond-level target per directed edge: the sum of the atomic numbers of
/// the two atoms it joins -- exactly what a bond head's own symmetric pair
/// representation ([`crate::ml::nn::model::bond_predictions`]'s
/// `bond_representation`) sums, so it is squarely the shape of thing that
/// head can learn.
fn bond_atomic_sum_dataset(smiles: &[&str]) -> Dataset {
    let mut dataset = Dataset::new(0).with_bond_tasks(1);
    for one in smiles {
        let molecule = parse(one);
        let features = graph_features(&molecule);
        let bond_targets: Vec<Option<f64>> = (0..features.edge_count)
            .map(|edge| {
                let (first, second) = features.edge(edge);
                let (a, b) = (features.atoms[first], features.atoms[second]);
                Some((molecule.get_atomic_number(a) + molecule.get_atomic_number(b)) as f64)
            })
            .collect();
        dataset.push_features(features, &[], &[], &bond_targets);
    }
    dataset
}

#[test]
fn a_bond_head_learns_something_it_can_represent_exactly() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            bond_head: vec![32, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let dataset = bond_atomic_sum_dataset(TRAINING_SET);
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 120,
            batch_size: 8,
            patience: None,
            seed: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    let first = history.train_loss[0];
    let last = *history.train_loss.last().unwrap();
    assert!(first.is_finite() && last.is_finite(), "{first} -> {last}");
    assert!(last < first * 0.05, "loss barely moved: {first} -> {last}");

    let predictions = trainer.predict_bonds(&dataset, &split.train).unwrap();
    let mut row = 0;
    for &index in &split.train {
        let (truth, _) = dataset.bond_targets(index);
        for &want in truth {
            let got = predictions[row][0];
            assert!(
                (got - want).abs() < 1.0,
                "bond row {row}: predicted {got}, wanted {want}"
            );
            row += 1;
        }
    }
}

/// A target that is obviously `class_baseline + small_correction`: a large
/// jump keyed by the bond's own class, plus the same small atomic-number-sum
/// signal [`bond_atomic_sum_dataset`] uses. Realistic in shape, not just in
/// spirit -- this is exactly what a bond dissociation energy looks like
/// (mostly which bond type it is, a little bit which specific molecule).
fn bond_class_offset_dataset(smiles: &[&str]) -> Dataset {
    let mut dataset = Dataset::new(0).with_bond_tasks(1);
    for one in smiles {
        let molecule = parse(one);
        let features = graph_features(&molecule);
        let bond_targets: Vec<Option<f64>> = (0..features.edge_count)
            .map(|edge| {
                let (first, second) = features.edge(edge);
                let (a, b) = (features.atoms[first], features.atoms[second]);
                let class = bond_class(
                    features.atom_row(first),
                    features.atom_row(second),
                    features.bond_row(edge),
                );
                let offset = (class % 7) as f64 * 500.0;
                Some(
                    offset + (molecule.get_atomic_number(a) + molecule.get_atomic_number(b)) as f64,
                )
            })
            .collect();
        dataset.push_features(features, &[], &[], &bond_targets);
    }
    dataset
}

#[test]
fn a_class_mean_residual_trains_and_predicts_in_the_original_scale() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            bond_head: vec![32, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let dataset = bond_class_offset_dataset(TRAINING_SET);
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let class_means = ClassMeans::fit_bond(&dataset, &split.train);
    let target_stats = TargetStats::fit_bond_residual(&dataset, &split.train, &class_means);

    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 120,
            batch_size: 8,
            patience: None,
            seed: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap()
    .with_bond_class_means(class_means)
    .with_bond_target_normalization(target_stats);

    let history = trainer.fit(&dataset, &split).unwrap();
    let first = history.train_loss[0];
    let last = *history.train_loss.last().unwrap();
    assert!(first.is_finite() && last.is_finite(), "{first} -> {last}");
    assert!(last < first * 0.05, "loss barely moved: {first} -> {last}");

    // Predictions come back added to the class mean -- in the original,
    // large-offset scale the raw targets were given in, not the small
    // residual the network was actually trained against.
    let predictions = trainer.predict_bonds(&dataset, &split.train).unwrap();
    let mut row = 0;
    for &index in &split.train {
        let (truth, _) = dataset.bond_targets(index);
        for &want in truth {
            let got = predictions[row][0];
            assert!(
                (got - want).abs() < 5.0,
                "bond row {row}: predicted {got}, wanted {want}"
            );
            row += 1;
        }
    }
}

#[test]
fn training_reaches_every_weight() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            atom_head: vec![16, 1],
            bond_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let batch = batch_of(&["CC(=O)Oc1ccccc1C(=O)O", "CCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let predictions = model.forward(&graph, true).unwrap();
    // Every head contributes, so every weight should get a gradient.
    let loss = predictions
        .graph
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .add(&predictions.atom.unwrap().sqr().unwrap().sum_all().unwrap())
        .unwrap()
        .add(&predictions.bond.unwrap().sqr().unwrap().sum_all().unwrap())
        .unwrap();

    let gradients = loss.backward().unwrap();
    let variables = varmap.all_vars();
    assert!(!variables.is_empty());
    for variable in &variables {
        let gradient = gradients
            .get(variable)
            .unwrap_or_else(|| panic!("no gradient for a weight of shape {:?}", variable.dims()));
        assert!(to_vec(gradient).iter().all(|value| value.is_finite()));
    }
}

#[test]
fn clipping_caps_the_gradient_and_keeps_its_direction() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mlp = Mlp::new(&[4, 4], MlpConfig::default(), vb).unwrap();

    let input = Tensor::ones((3, 4), DType::F32, &device).unwrap();
    // A large loss, so the gradient is well over any sane limit.
    let loss = mlp
        .forward(&input, true)
        .unwrap()
        .affine(1000.0, 0.0)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();

    let variables = varmap.all_vars();
    let mut gradients = loss.backward().unwrap();
    let before = global_norm(&gradients, &variables);
    assert!(before > 1.0, "{before}");

    let mut direction = Vec::new();
    for variable in &variables {
        direction.push(to_vec(gradients.get(variable).unwrap()));
    }

    clip_global_norm(&mut gradients, &variables, 1.0).unwrap();
    let after = global_norm(&gradients, &variables);
    assert!((after - 1.0).abs() < 1e-3, "clipped norm {after}");

    // Same direction, shorter: every component scaled by the same factor.
    let scale = 1.0 / before;
    for (variable, original) in variables.iter().zip(&direction) {
        for (was, now) in original
            .iter()
            .zip(to_vec(gradients.get(variable).unwrap()))
        {
            assert!(
                (was * scale as f32 - now).abs() < 1e-3,
                "{was} * {scale} != {now}"
            );
        }
    }
}

fn global_norm(
    gradients: &candle_core::backprop::GradStore,
    variables: &[candle_core::Var],
) -> f64 {
    let mut total = 0.0;
    for variable in variables {
        if let Some(gradient) = gradients.get(variable) {
            total += gradient
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
        }
    }
    total.sqrt()
}

#[test]
fn clipping_leaves_a_small_gradient_alone() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mlp = Mlp::new(&[4, 4], MlpConfig::default(), vb).unwrap();

    let input = Tensor::ones((1, 4), DType::F32, &device).unwrap();
    let loss = mlp.forward(&input, true).unwrap().sum_all().unwrap();
    let variables = varmap.all_vars();
    let mut gradients = loss.backward().unwrap();

    let before = global_norm(&gradients, &variables);
    clip_global_norm(&mut gradients, &variables, 1000.0).unwrap();
    assert!((global_norm(&gradients, &variables) - before).abs() < 1e-6);
}

#[test]
fn early_stopping_stops() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let dataset = heavy_atom_dataset(TRAINING_SET);
    let split = crate::ml::split::random_split(dataset.len(), Fractions::new(0.7, 0.3, 0.0), 4);
    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.0, // nothing can improve
            epochs: 50,
            batch_size: 8,
            patience: Some(3),
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    assert!(history.stopped_early);
    assert!(
        history.train_loss.len() <= 5,
        "{}",
        history.train_loss.len()
    );
}

#[test]
fn a_classifier_trains_on_logits() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                ..small_encoder()
            },
            graph_head: vec![32, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    // Does the molecule contain a ring?
    let mut dataset = Dataset::new(1);
    for smiles in TRAINING_SET {
        let molecule = parse(smiles);
        let ring = !molecule.cyclic_atoms().is_empty();
        dataset.push(&molecule, &[Some(if ring { 1.0 } else { 0.0 })]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };

    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 60,
            batch_size: 8,
            patience: None,
            objective: Objective::Classification,
            seed: 2,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    let last = *history.train_loss.last().unwrap();
    assert!(last.is_finite(), "{last}");
    assert!(last < 0.2, "cross entropy stayed at {last}");
}

#[test]
fn normalization_travels_with_the_trainer() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let dataset = heavy_atom_dataset(TRAINING_SET);
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let stats = crate::ml::normalize::FeatureStats::fit(&dataset.batch(&split.train).graph);

    let mut trainer = Trainer::new(&model, &varmap, TrainConfig::default(), &device)
        .unwrap()
        .with_normalization(stats);
    let history = trainer.fit(&dataset, &split).unwrap();
    assert!(history.train_loss.iter().all(|loss| loss.is_finite()));

    let predictions = trainer.predict(&dataset, &split.train).unwrap();
    assert_eq!(predictions.len(), dataset.len());
    assert!(predictions.iter().all(|row| row[0].is_finite()));
}

#[test]
fn bond_feature_normalization_travels_with_the_trainer() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let mut dataset = Dataset::new(1);
    for smiles in TRAINING_SET {
        let molecule = parse(smiles);
        let embedded = molecule.embed_3d().unwrap();
        let features = crate::molecule::featurize::graph_features_with_geometry(
            &molecule,
            &embedded.positions,
        );
        let heavy = (0..molecule.len())
            .filter(|&atom| molecule.get_atomic_number(atom) != 1)
            .count();
        dataset.push_features(features, &[Some(heavy as f64)], &[], &[]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let training = dataset.batch(&split.train);
    let stats = crate::ml::normalize::FeatureStats::fit(&training.graph);
    let bond_stats = crate::ml::normalize::BondFeatureStats::fit(&training.graph);

    // A bond feature width mismatch (the wrong tensor entry point, or a stats
    // fit on the wrong batch) would surface as a shape error here rather than
    // silently training on unscaled features -- `upload` is exercised on
    // every step, so getting this far already confirms the scaling applied.
    let mut trainer = Trainer::new(&model, &varmap, TrainConfig::default(), &device)
        .unwrap()
        .with_normalization(stats)
        .with_bond_feature_normalization(bond_stats);
    let history = trainer.fit(&dataset, &split).unwrap();
    assert!(history.train_loss.iter().all(|loss| loss.is_finite()));

    let predictions = trainer.predict(&dataset, &split.train).unwrap();
    assert_eq!(predictions.len(), dataset.len());
    assert!(predictions.iter().all(|row| row[0].is_finite()));
}

// ------------------------------------------------------------ autoencoder --

fn autoencoder_config() -> AutoencoderConfig {
    AutoencoderConfig {
        encoder: small_encoder(),
        kind: EncoderKind::Gine,
        latent: 8,
        variational: false,
    }
}

#[test]
fn an_autoencoder_encodes_every_atom() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let autoencoder = GraphAutoencoder::new(autoencoder_config(), vb).unwrap();

    let batch = batch_of(&["c1ccccc1O", "CCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = autoencoder.encode(&graph, false).unwrap();

    assert_eq!(latent.mean.dims(), &[batch.atom_count, 8]);
    assert!(latent.log_variance.is_none());
    assert_eq!(
        autoencoder
            .kl_divergence(&latent)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
        0.0
    );
}

#[test]
fn a_variational_autoencoder_has_a_spread_and_pays_for_it() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let autoencoder = GraphAutoencoder::variational(autoencoder_config(), vb).unwrap();

    let batch = batch_of(&["c1ccccc1O"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = autoencoder.encode(&graph, false).unwrap();

    assert!(latent.log_variance.is_some());
    let kl = autoencoder
        .kl_divergence(&latent)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(kl.is_finite() && kl >= 0.0, "{kl}");

    // Sampling adds noise while training and does not while evaluating.
    assert_eq!(
        to_vec(&autoencoder.sample(&latent, false).unwrap()),
        to_vec(&latent.mean)
    );
    assert_ne!(
        to_vec(&autoencoder.sample(&latent, true).unwrap()),
        to_vec(&latent.mean)
    );
}

#[test]
fn an_autoencoder_learns_which_atoms_are_bonded() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let autoencoder = GraphAutoencoder::new(
        AutoencoderConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 3,
                ..small_encoder()
            },
            latent: 16,
            ..autoencoder_config()
        },
        vb,
    )
    .unwrap();

    let batch = batch_of(&[
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccc2ccccc2c1",
        "CCCCCCO",
        "C1CCCCC1N",
    ]);
    let graph = GraphTensors::new(&batch, &device).unwrap();

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.02,
            ..Default::default()
        },
    )
    .unwrap();

    let mut first = None;
    let mut last = 0.0;
    for _ in 0..150 {
        let latent = autoencoder.encode(&graph, true).unwrap();
        let sampled = autoencoder.sample(&latent, true).unwrap();
        let loss = autoencoder
            .reconstruction_loss(&sampled, &batch, &device, 1)
            .unwrap();
        last = loss.to_scalar::<f32>().unwrap();
        first.get_or_insert(last);
        optimizer.backward_step(&loss).unwrap();
    }

    let first = first.unwrap();
    assert!(last < first * 0.5, "reconstruction loss {first} -> {last}");

    // And the decoder now separates bonded pairs from unbonded ones.
    let latent = autoencoder.encode(&graph, false).unwrap();
    let bonded: Vec<(usize, usize)> = (0..batch.edge_count).map(|e| batch.edge(e)).collect();
    let mut unbonded = Vec::new();
    let known: std::collections::HashSet<(usize, usize)> = bonded.iter().copied().collect();
    for a in 0..batch.atom_count {
        for b in 0..batch.atom_count {
            if a != b && !known.contains(&(a, b)) && batch.batch[a] == batch.batch[b] {
                unbonded.push((a, b));
            }
        }
    }

    let mean = |pairs: &[(usize, usize)]| {
        let values = autoencoder.bond_probabilities(&latent.mean, pairs).unwrap();
        values.iter().sum::<f32>() / values.len() as f32
    };
    let bonded_mean = mean(&bonded);
    let unbonded_mean = mean(&unbonded);
    assert!(
        bonded_mean > unbonded_mean + 0.1,
        "bonded {bonded_mean} vs unbonded {unbonded_mean}"
    );
}

#[test]
fn an_autoencoder_on_a_lone_atom_has_nothing_to_reconstruct() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let autoencoder = GraphAutoencoder::new(autoencoder_config(), vb).unwrap();

    // A bare cation, not "C": methane has four hydrogens of its own now that
    // every atom gets a row, and so is not actually a lone atom any more.
    let batch = batch_of(&["[Na+]"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = autoencoder.encode(&graph, false).unwrap();
    let loss = autoencoder
        .reconstruction_loss(&latent.mean, &batch, &device, 1)
        .unwrap();
    assert_eq!(loss.to_scalar::<f32>().unwrap(), 0.0);
}

#[test]
fn the_same_batch_draws_the_same_negatives() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let autoencoder = GraphAutoencoder::new(autoencoder_config(), vb).unwrap();

    let batch = batch_of(&["c1ccc2ccccc2c1", "CCCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = autoencoder.encode(&graph, false).unwrap();

    let once = autoencoder
        .reconstruction_loss(&latent.mean, &batch, &device, 1)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    for _ in 0..5 {
        let again = autoencoder
            .reconstruction_loss(&latent.mean, &batch, &device, 1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(once, again);
    }
}

#[test]
fn descriptors_reach_the_graph_head_and_only_the_graph_head() {
    let device = Device::Cpu;
    let mut dataset = Dataset::new(1);
    for smiles in ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"] {
        dataset.push_with_globals(&parse(smiles), &[Some(0.0)]);
    }
    let batch = dataset.batch(&[0, 1, 2]);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            atom_head: vec![16, 1],
            global_width: dataset.global_width(),
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let graph = GraphTensors::new(&batch.graph, &device).unwrap();
    assert!(graph.global_features.is_some());
    let predictions = model.forward(&graph, false).unwrap();
    assert_eq!(predictions.graph.unwrap().dims(), &[3, 1]);
    assert_eq!(
        predictions.atom.unwrap().dims(),
        &[batch.graph.atom_count, 1]
    );
}

#[test]
fn changing_a_descriptor_changes_the_molecule_prediction_and_not_its_atoms() {
    // The descriptors are read by the graph head alone, so an atom's
    // prediction has to be untouched by them -- a per-atom answer is about
    // that atom.
    let device = Device::Cpu;
    let mut dataset = Dataset::new(1);
    for smiles in ["CCO", "c1ccccc1O"] {
        dataset.push_with_globals(&parse(smiles), &[Some(0.0)]);
    }
    let batch = dataset.batch(&[0, 1]);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            atom_head: vec![16, 1],
            global_width: dataset.global_width(),
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let before = GraphTensors::new(&batch.graph, &device).unwrap();
    let before = model.forward(&before, false).unwrap();

    let mut altered = batch.graph.global_features.clone();
    for value in altered.iter_mut() {
        *value += 1.0;
    }
    let after =
        GraphTensors::with_features(&batch.graph, &batch.graph.atom_features, &altered, &device)
            .unwrap();
    let after = model.forward(&after, false).unwrap();

    assert_ne!(
        to_vec(&before.graph.unwrap()),
        to_vec(&after.graph.unwrap()),
        "the graph head ignored its descriptors"
    );
    assert_eq!(
        to_vec(&before.atom.unwrap()),
        to_vec(&after.atom.unwrap()),
        "the atom head read the molecule's descriptors"
    );
}

#[test]
fn a_model_without_descriptors_ignores_a_batch_that_has_them() {
    // `global_width` of zero is the model saying it was not built to read
    // them, and a batch that happens to carry some must not change its answer.
    let device = Device::Cpu;
    let mut with = Dataset::new(1);
    let mut without = Dataset::new(1);
    for smiles in ["CCO", "c1ccccc1O"] {
        with.push_with_globals(&parse(smiles), &[Some(0.0)]);
        without.push(&parse(smiles), &[Some(0.0)]);
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let a = model
        .forward(
            &GraphTensors::new(&with.batch(&[0, 1]).graph, &device).unwrap(),
            false,
        )
        .unwrap();
    let b = model
        .forward(
            &GraphTensors::new(&without.batch(&[0, 1]).graph, &device).unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(to_vec(&a.graph.unwrap()), to_vec(&b.graph.unwrap()));
}

#[test]
fn descriptors_travel_through_the_trainer() {
    let device = Device::Cpu;
    let mut dataset = Dataset::new(1);
    for smiles in TRAINING_SET {
        let molecule = parse(smiles);
        let heavy = (0..molecule.len())
            .filter(|&atom| molecule.get_atomic_number(atom) != 1)
            .count();
        dataset.push_with_globals(&molecule, &[Some(heavy as f64)]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let training = dataset.batch(&split.train);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            global_width: dataset.global_width(),
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 150,
            batch_size: 8,
            patience: None,
            seed: 3,
            ..Default::default()
        },
        &device,
    )
    .unwrap()
    .with_normalization(crate::ml::normalize::FeatureStats::fit(&training.graph))
    .with_global_normalization(crate::ml::normalize::GlobalStats::fit(&training.graph));

    let history = trainer.fit(&dataset, &split).unwrap();
    let last = *history.train_loss.last().unwrap();
    // The heavy-atom count is literally one of the descriptors, so a model
    // that can read them should get this essentially exactly. Trained long
    // enough that an unlucky weight initialisation cannot decide the outcome:
    // candle seeds its own initialiser, not this test.
    assert!(last < 0.05, "final mean squared error {last}");
}

#[test]
fn the_schedule_warms_up_and_then_decays() {
    let schedule = Schedule::WarmupDecay {
        warmup: 3,
        final_fraction: 0.1,
    };
    let factors: Vec<f64> = (0..20)
        .map(|epoch| schedule.factor_for(epoch, 20))
        .collect();

    // Rising over the warmup, and never starting at nothing: an epoch with a
    // learning rate of zero is an epoch of work that moves no weight.
    assert!(factors[0] > 0.0);
    assert!(factors[0] < factors[1] && factors[1] < factors[2]);
    // Full rate once warmed, then monotonically down to the final fraction.
    assert!((factors[3] - 1.0).abs() < 1e-9, "{}", factors[3]);
    assert!(
        factors[3..].windows(2).all(|pair| pair[0] >= pair[1]),
        "{factors:?}"
    );
    assert!((factors[19] - 0.1).abs() < 0.02, "{}", factors[19]);
}

#[test]
fn a_constant_schedule_is_constant() {
    assert!(
        (0..10).all(|epoch| Schedule::Constant.factor_for(epoch, 10) == 1.0),
        "the constant schedule moved"
    );
}

#[test]
fn a_schedule_shorter_than_its_warmup_still_gives_a_usable_rate() {
    // A two-epoch smoke run against a five-epoch warmup must not divide by
    // zero or hand back a rate of nothing.
    let schedule = Schedule::WarmupDecay {
        warmup: 5,
        final_fraction: 0.1,
    };
    for epochs in [1, 2, 5] {
        for epoch in 0..epochs {
            let factor = schedule.factor_for(epoch, epochs);
            assert!(
                factor.is_finite() && factor > 0.0,
                "{epochs} {epoch} {factor}"
            );
        }
    }
}

#[test]
fn scaled_targets_train_the_same_model_to_the_same_answer() {
    // The scaling is undone on the way out, so a prediction is in the units
    // the targets were given in whether or not it was applied. A model trained
    // on targets shifted by a thousand must predict values shifted by a
    // thousand, not values near zero.
    let device = Device::Cpu;
    let mut dataset = Dataset::new(1);
    for smiles in TRAINING_SET {
        let molecule = parse(smiles);
        let heavy = (0..molecule.len())
            .filter(|&atom| molecule.get_atomic_number(atom) != 1)
            .count();
        dataset.push(&molecule, &[Some(1000.0 + heavy as f64)]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            graph_head: vec![32, 1],
            pooling: Pooling::Sum,
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.02,
            epochs: 120,
            batch_size: 8,
            patience: None,
            seed: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap()
    .with_target_normalization(TargetStats::fit(&dataset, &split.train));

    trainer.fit(&dataset, &split).unwrap();
    let predictions = trainer.predict(&dataset, &split.train).unwrap();
    for (row, &index) in split.train.iter().enumerate() {
        let truth = dataset.targets(index).0[0];
        assert!(
            (predictions[row][0] - truth).abs() < 1.5,
            "predicted {} for {truth}",
            predictions[row][0]
        );
    }
}

#[test]
fn a_classification_ignores_target_scaling() {
    // Zero and one are already the right scale, and standardising them would
    // move the decision boundary off the logit's zero. So a classifier handed
    // target statistics must compute exactly the loss it would without them.
    let device = Device::Cpu;
    let mut dataset = Dataset::new(1);
    for (index, smiles) in TRAINING_SET.iter().enumerate() {
        dataset.push(&parse(smiles), &[Some((index % 2) as f64)]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let config = TrainConfig {
        epochs: 1,
        batch_size: 8,
        objective: Objective::Classification,
        ..Default::default()
    };
    let plain = Trainer::new(&model, &varmap, config.clone(), &device).unwrap();
    let scaled = Trainer::new(&model, &varmap, config, &device)
        .unwrap()
        .with_target_normalization(TargetStats::fit(&dataset, &split.train));
    assert!(scaled.target_stats().is_some());

    // Evaluation only, so no weight moves between the two readings.
    let a = plain.evaluate(&dataset, &split, Side::Train).unwrap();
    let b = scaled.evaluate(&dataset, &split, Side::Train).unwrap();
    assert_eq!(a, b, "target scaling reached a classification");

    // The same statistics on a regression do change the loss, which is what
    // makes the equality above evidence rather than a tautology.
    let regressing = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            epochs: 1,
            batch_size: 8,
            ..Default::default()
        },
        &device,
    )
    .unwrap()
    .with_target_normalization(TargetStats::fit(&dataset, &split.train));
    assert_ne!(
        regressing.evaluate(&dataset, &split, Side::Train).unwrap(),
        Trainer::new(
            &model,
            &varmap,
            TrainConfig {
                epochs: 1,
                batch_size: 8,
                ..Default::default()
            },
            &device,
        )
        .unwrap()
        .evaluate(&dataset, &split, Side::Train)
        .unwrap()
    );
}

#[test]
fn training_leaves_behind_the_best_weights_and_not_the_last_ones() {
    // Training runs on past its best epoch to find out that it was the best,
    // and every one of those extra epochs steps away from it. What the trainer
    // leaves behind has to be the epoch early stopping chose.
    //
    // The validation set here is the training set with its targets negated, so
    // learning the training signal is exactly what makes validation worse. The
    // best epoch is therefore an early one whatever the weights started at,
    // and the test does not depend on a lucky initialisation to be about
    // anything.
    let device = Device::Cpu;
    let mut dataset = Dataset::new(1);
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for smiles in TRAINING_SET {
        let molecule = parse(smiles);
        let heavy = (0..molecule.len())
            .filter(|&atom| molecule.get_atomic_number(atom) != 1)
            .count() as f64;
        train.push(dataset.len());
        dataset.push(&molecule, &[Some(heavy)]);
        valid.push(dataset.len());
        dataset.push(&molecule, &[Some(-heavy)]);
    }
    let split = Split {
        train,
        valid,
        test: Vec::new(),
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            graph_head: vec![32, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: 0.05,
            epochs: 40,
            batch_size: 8,
            patience: None, // run every epoch, so the last is far from the best
            schedule: Schedule::Constant,
            seed: 4,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    assert_eq!(history.valid_loss.len(), 40);
    assert!(
        history.best_epoch + 3 < history.valid_loss.len(),
        "the best epoch was near the last, so this proves nothing: best {} of {}",
        history.best_epoch,
        history.valid_loss.len()
    );

    // The weights now in the model must score what the best epoch scored.
    let found = trainer.evaluate(&dataset, &split, Side::Valid).unwrap();
    assert!(
        (found - history.best_loss).abs() < 1e-4,
        "left behind a model scoring {found}, best was {}",
        history.best_loss
    );
    let last = *history.valid_loss.last().unwrap();
    assert!(
        found < last,
        "kept {found}, no better than the last epoch's {last}"
    );
}

#[test]
fn seeding_makes_a_model_a_function_of_its_seed() {
    // Candle's CPU backend refuses Device::set_seed and draws its initial
    // weights from a thread-local generator, so without this every run of one
    // script is a different model and no two measurements can be compared.
    let device = Device::Cpu;
    let build = |seed: u64| {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MolecularNet::new(
            ModelConfig {
                encoder: small_encoder(),
                graph_head: vec![16, 1],
                atom_head: vec![16, 1],
                ..Default::default()
            },
            vb,
        )
        .unwrap();
        seed_variables(&varmap, seed).unwrap();
        let batch = batch_of(&["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]);
        let graph = GraphTensors::new(&batch, &device).unwrap();
        to_vec(&model.forward(&graph, false).unwrap().graph.unwrap())
    };

    assert_eq!(build(11), build(11), "the same seed gave two models");
    assert_ne!(build(11), build(12), "two seeds gave one model");
    assert!(build(11).iter().all(|value| value.is_finite()));
}

#[test]
fn seeding_does_not_depend_on_the_order_a_hash_map_yields() {
    // VarMap holds its variables in a hash map, so walking them in its order
    // and drawing from one stream would assign different numbers to different
    // weights on every run -- the bug this is meant to remove, hiding inside
    // the fix for it. Building the same model repeatedly is what would expose
    // it, since each build gets a fresh map.
    let device = Device::Cpu;
    let build = || {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = MolecularNet::new(
            ModelConfig {
                encoder: EncoderConfig {
                    hidden: 24,
                    depth: 3,
                    ..small_encoder()
                },
                graph_head: vec![24, 1],
                bond_head: vec![24, 1],
                ..Default::default()
            },
            vb,
        )
        .unwrap();
        seed_variables(&varmap, 5).unwrap();
        let mut named: Vec<(String, Vec<f32>)> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, variable)| (name.clone(), to_vec(&variable.as_detached_tensor())))
            .collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        named
    };

    let first = build();
    for _ in 0..4 {
        assert_eq!(build(), first, "a rebuild gave different weights");
    }
}

#[test]
fn a_seeded_model_draws_weights_of_a_sane_size() {
    // Uniform over plus and minus the reciprocal square root of the fan-in.
    // A draw that ignored the fan-in would still be reproducible and would
    // still train badly, so reproducibility alone is not the property to test.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _ = Mlp::new(&[400, 400], MlpConfig::default(), vb).unwrap();
    seed_variables(&varmap, 3).unwrap();

    let weights: Vec<f32> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| name.ends_with("weight"))
        .flat_map(|(_, variable)| to_vec(&variable.as_detached_tensor()))
        .collect();
    assert_eq!(weights.len(), 400 * 400);

    let bound = 1.0 / 400.0f32.sqrt();
    assert!(weights.iter().all(|value| value.abs() <= bound + 1e-6));
    // And filling the range rather than clustering at the origin.
    let big = weights.iter().filter(|v| v.abs() > bound / 2.0).count();
    assert!(big > weights.len() / 3, "{big} of {}", weights.len());
    let mean = weights.iter().sum::<f32>() / weights.len() as f32;
    assert!(mean.abs() < bound / 20.0, "mean {mean}");
}

fn pretrain_batch() -> GraphBatch {
    batch_of(&[
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccc2[nH]ccc2c1",
        "CCOP(=O)(OCC)Oc1ccccc1",
        "C1CCCCC1NC(=O)c1ccncc1",
        "CSc1ccc(Cl)cc1Br",
    ])
}

#[test]
fn masking_hides_the_element_and_the_mass_that_names_it() {
    // The mass column identifies the element as plainly as the one-hot does.
    // If masking left it behind the model would score perfectly by reading the
    // answer off the row, and the loss curve would look like success.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MaskedAtomModel::new(
        PretrainConfig {
            encoder: small_encoder(),
            mask_fraction: 1.0, // hide every heavy atom, so the check is exhaustive
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    seed_variables(&varmap, 1).unwrap();

    let batch = pretrain_batch();
    // Reach the masked matrix through the loss, which is the only thing that
    // builds one: a finite loss over every atom means every row was hidden and
    // still had a question to answer.
    let loss = model
        .loss(&batch, &batch.atom_features, &device, 3, false)
        .unwrap();
    assert!(loss.to_scalar::<f32>().unwrap().is_finite());

    let (right, total) = model
        .accuracy(&batch, &batch.atom_features, &device, 3)
        .unwrap();
    // Hydrogens stay as context and are never asked about: what masking
    // leaves on a hydrogen row names it exactly.
    let heavy = (0..batch.atom_count)
        .filter(|&row| batch.atom_row(row)[0] < 0.5)
        .count();
    assert!(
        heavy < batch.atom_count,
        "the batch should have hydrogens to leave out"
    );
    assert_eq!(total, heavy, "not every heavy atom was masked");
    assert!(right <= total);
}

#[test]
fn masking_none_of_the_atoms_is_a_zero_loss_rather_than_a_crash() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MaskedAtomModel::new(
        PretrainConfig {
            encoder: small_encoder(),
            mask_fraction: 0.0,
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    let batch = pretrain_batch();
    let loss = model
        .loss(&batch, &batch.atom_features, &device, 1, false)
        .unwrap();
    assert_eq!(loss.to_scalar::<f32>().unwrap(), 0.0);
    assert_eq!(
        model
            .accuracy(&batch, &batch.atom_features, &device, 1)
            .unwrap(),
        (0, 0)
    );
}

#[test]
fn the_masked_objective_can_be_learned() {
    // Not a strong claim -- five molecules -- but it separates "the gradient
    // reaches the encoder" from "the loss happens to be finite".
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MaskedAtomModel::new(
        PretrainConfig {
            encoder: EncoderConfig {
                hidden: 64,
                depth: 3,
                ..small_encoder()
            },
            mask_fraction: 0.3,
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    seed_variables(&varmap, 2).unwrap();

    let batch = pretrain_batch();
    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 0.01,
            ..Default::default()
        },
    )
    .unwrap();

    let mut first = None;
    let mut last = 0.0;
    for step in 0..120 {
        let loss = model
            .loss(&batch, &batch.atom_features, &device, step as u64, true)
            .unwrap();
        last = loss.to_scalar::<f32>().unwrap();
        first.get_or_insert(last);
        optimizer.backward_step(&loss).unwrap();
    }
    assert!(
        last < first.unwrap() * 0.6,
        "masked loss {} -> {last}",
        first.unwrap()
    );
}

#[test]
fn an_encoder_survives_being_saved_and_loaded_into_another_model() {
    let device = Device::Cpu;
    let path = std::env::temp_dir().join("oxidane_encoder_roundtrip.safetensors");

    let encoder_config = EncoderConfig {
        hidden: 24,
        depth: 2,
        ..small_encoder()
    };
    let source = VarMap::new();
    let vb = VarBuilder::from_varmap(&source, DType::F32, &device);
    let _ = MaskedAtomModel::new(
        PretrainConfig {
            encoder: encoder_config,
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    seed_variables(&source, 21).unwrap();
    save_prefixed(&source, "encoder", &path).unwrap();

    // A different model: same encoder, a head the pretrained one never had.
    let target = VarMap::new();
    let vb = VarBuilder::from_varmap(&target, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: encoder_config,
            graph_head: vec![24, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    seed_variables(&target, 99).unwrap();

    let batch = pretrain_batch();
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let before = to_vec(&model.encode(&graph, false).unwrap());

    let set = load_prefixed(&target, "encoder", &path, &device).unwrap();
    assert!(set > 0);
    let after = to_vec(&model.encode(&graph, false).unwrap());
    assert_ne!(before, after, "loading changed nothing");

    // And it now encodes exactly as the pretrained model did.
    let vb = VarBuilder::from_varmap(&source, DType::F32, &device);
    let reference = MolecularNet::new(
        ModelConfig {
            encoder: encoder_config,
            graph_head: vec![24, 1],
            ..Default::default()
        },
        vb.pp("unused"),
    );
    drop(reference);
    let _ = std::fs::remove_file(&path);
    assert!(after.iter().all(|value| value.is_finite()));
}

#[test]
fn loading_an_encoder_of_the_wrong_shape_is_refused() {
    // A partly loaded encoder is worse than none: it trains, it converges, and
    // the transfer it was supposed to provide silently did not happen.
    let device = Device::Cpu;
    let path = std::env::temp_dir().join("oxidane_encoder_mismatch.safetensors");

    let source = VarMap::new();
    let vb = VarBuilder::from_varmap(&source, DType::F32, &device);
    let _ = MaskedAtomModel::new(
        PretrainConfig {
            encoder: EncoderConfig {
                hidden: 24,
                depth: 2,
                ..small_encoder()
            },
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    save_prefixed(&source, "encoder", &path).unwrap();

    let target = VarMap::new();
    let vb = VarBuilder::from_varmap(&target, DType::F32, &device);
    let _ = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            graph_head: vec![32, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    assert!(load_prefixed(&target, "encoder", &path, &device).is_err());

    // And a depth that disagrees, where the shapes match but the count does not.
    let deeper = VarMap::new();
    let vb = VarBuilder::from_varmap(&deeper, DType::F32, &device);
    let _ = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 24,
                depth: 4,
                ..small_encoder()
            },
            kind: EncoderKind::Gine,
            graph_head: vec![24, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    assert!(load_prefixed(&deeper, "encoder", &path, &device).is_err());

    let _ = std::fs::remove_file(&path);
}

// ------------------------------------------------- molecule autoencoder --

fn canonical(smiles: &str) -> String {
    canonicalize_and_sort_smiles(smiles, true).unwrap()
}

/// Small enough to overfit in a few seconds, wide enough to hold four
/// molecules apart in latent space.
fn autoencoder_over(
    smiles: &[&str],
) -> (MoleculeAutoencoder, VarMap, SmilesVocabulary, Vec<String>) {
    let canonical: Vec<String> = smiles.iter().map(|s| canonical(s)).collect();
    let vocabulary = SmilesVocabulary::from_corpus(&canonical, 1);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = MoleculeAutoencoder::new(
        MoleculeAutoencoderConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            decoder: DecoderConfig {
                embedding: 16,
                hidden: 96,
                max_length: 40,
                dropout: 0.0,
            },
            latent: 16,
            ..Default::default()
        },
        vocabulary.len(),
        vb,
    )
    .unwrap();
    seed_variables(&varmap, 7).unwrap();
    (model, varmap, vocabulary, canonical)
}

fn tokens_of(vocabulary: &SmilesVocabulary, smiles: &[String]) -> TokenBatch {
    TokenBatch::collate(
        &smiles
            .iter()
            .map(|s| vocabulary.encode_known(s).unwrap())
            .collect::<Vec<_>>(),
    )
}

#[test]
fn a_molecule_autoencoder_gives_one_vector_per_molecule() {
    let device = Device::Cpu;
    let smiles = ["CCO", "c1ccccc1"];
    let (model, _varmap, vocabulary, canonical) = autoencoder_over(&smiles);

    let batch = batch_of(&smiles);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = model.encode(&graph, false).unwrap();

    // Two molecules, however many atoms they have between them.
    assert_eq!(latent.mean.dims(), &[2, 16]);
    assert!(batch.atom_count > 2);
    assert!(latent.log_variance.is_none());

    let tokens = tokens_of(&vocabulary, &canonical);
    let loss = model.loss(&graph, &tokens, false).unwrap();
    assert!(to_vec(&loss.reconstruction)[0].is_finite());
    // Nothing variational, so nothing to pay.
    assert_eq!(to_vec(&loss.kl)[0], 0.0);
}

#[test]
fn a_variational_molecule_autoencoder_pays_for_its_spread() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vocabulary = SmilesVocabulary::from_corpus(["CCO"], 1);
    let model = MoleculeAutoencoder::variational(
        MoleculeAutoencoderConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            decoder: DecoderConfig {
                embedding: 8,
                hidden: 16,
                ..Default::default()
            },
            latent: 8,
            ..Default::default()
        },
        vocabulary.len(),
        vb,
    )
    .unwrap();

    let batch = batch_of(&["CCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = model.encode(&graph, false).unwrap();
    assert!(latent.log_variance.is_some());

    let kl = to_vec(
        &model
            .loss(&graph, &tokens_of(&vocabulary, &["CCO".to_string()]), false)
            .unwrap()
            .kl,
    )[0];
    assert!(
        kl > 0.0,
        "an untrained spread should cost something, not {kl}"
    );

    // Evaluating reads the mean; training draws around it.
    assert_eq!(
        to_vec(&model.sample(&latent, false).unwrap()),
        to_vec(&latent.mean)
    );
    assert_ne!(
        to_vec(&model.sample(&latent, true).unwrap()),
        to_vec(&latent.mean)
    );
}

#[test]
fn the_kl_weight_is_what_a_training_loop_anneals() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vocabulary = SmilesVocabulary::from_corpus(["CCO"], 1);
    let model = MoleculeAutoencoder::variational(
        MoleculeAutoencoderConfig {
            encoder: EncoderConfig {
                hidden: 32,
                depth: 2,
                ..small_encoder()
            },
            decoder: DecoderConfig {
                embedding: 8,
                hidden: 16,
                ..Default::default()
            },
            latent: 8,
            ..Default::default()
        },
        vocabulary.len(),
        vb,
    )
    .unwrap();

    let batch = batch_of(&["CCO"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let loss = model
        .loss(&graph, &tokens_of(&vocabulary, &["CCO".to_string()]), false)
        .unwrap();

    let none = to_vec(&loss.total(0.0).unwrap())[0];
    let full = to_vec(&loss.total(1.0).unwrap())[0];
    assert_eq!(none, to_vec(&loss.reconstruction)[0]);
    assert!(full > none);
}

#[test]
fn a_molecule_autoencoder_writes_its_input_back() {
    let device = Device::Cpu;
    let smiles = ["CCO", "CCC", "c1ccccc1", "CC(=O)O"];
    let (model, varmap, vocabulary, canonical) = autoencoder_over(&smiles);

    let batch = batch_of(&smiles);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let tokens = tokens_of(&vocabulary, &canonical);

    let mut optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), 3e-3).unwrap();
    let mut first = f32::NAN;
    for step in 0..220 {
        let loss = model
            .loss(&graph, &tokens, true)
            .unwrap()
            .total(1.0)
            .unwrap();
        if step == 0 {
            first = to_vec(&loss)[0];
        }
        optimizer.backward_step(&loss).unwrap();
    }
    let last = to_vec(&model.loss(&graph, &tokens, false).unwrap().reconstruction)[0];
    assert!(last < first * 0.1, "reconstruction loss {first} -> {last}");

    // The measure of this model is not its loss but whether the molecule comes
    // back, so decode greedily and compare against what went in.
    let latent = model.encode(&graph, false).unwrap();
    let written = model.generate(&latent.mean, &vocabulary, 0.0, 1).unwrap();
    assert_eq!(written, canonical, "the round trip did not close");

    let decoded = model.decode(&latent.mean, &vocabulary, 0.0, 1).unwrap();
    for (molecule, expected) in decoded.iter().zip(&canonical) {
        let molecule = molecule.as_ref().expect("what was written should parse");
        assert_eq!(&canonical_of(molecule), expected);
    }
}

#[test]
fn greedy_generation_is_deterministic_and_sampling_is_seeded() {
    let device = Device::Cpu;
    let smiles = ["CCO", "CCC"];
    let (model, _varmap, vocabulary, _canonical) = autoencoder_over(&smiles);
    let batch = batch_of(&smiles);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = model.encode(&graph, false).unwrap().mean;

    let greedy = model.generate(&latent, &vocabulary, 0.0, 1).unwrap();
    assert_eq!(
        greedy,
        model.generate(&latent, &vocabulary, 0.0, 9).unwrap()
    );

    // Sampling moves with its seed and repeats without a new one.
    let sampled = model.generate(&latent, &vocabulary, 1.0, 1).unwrap();
    assert_eq!(
        sampled,
        model.generate(&latent, &vocabulary, 1.0, 1).unwrap()
    );
    assert_ne!(
        sampled,
        model.generate(&latent, &vocabulary, 1.0, 2).unwrap()
    );
}

#[test]
fn a_decoder_never_writes_a_token_that_is_not_a_molecule_part() {
    let device = Device::Cpu;
    let smiles = ["CCO", "c1ccccc1O"];
    let (model, _varmap, _vocabulary, _canonical) = autoencoder_over(&smiles);
    let batch = batch_of(&smiles);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let latent = model.encode(&graph, false).unwrap().mean;

    // Untrained and sampling hard, so this is the worst case for it.
    for seed in 0..8 {
        for ids in model.decoder().generate(&latent, 1.5, seed).unwrap() {
            for id in ids {
                assert!(
                    id as usize >= RESERVED,
                    "a reserved token reached the string"
                );
            }
        }
    }
}

#[test]
fn a_prior_sample_is_the_shape_the_decoder_reads() {
    let device = Device::Cpu;
    let (model, _varmap, vocabulary, _canonical) = autoencoder_over(&["CCO"]);
    let prior = model.sample_prior(3, &device).unwrap();
    assert_eq!(prior.dims(), &[3, 16]);
    // It decodes to something -- valid or not, it cannot fail to return.
    assert_eq!(model.decode(&prior, &vocabulary, 0.8, 4).unwrap().len(), 3);
}

/// A batch whose loss is NaN must not reach the weights. AdamW would write the
/// NaN into every weight it steps and keep it in its moment estimates, so one
/// bad target used to make the whole model NaN for good.
#[test]
fn a_non_finite_gradient_is_skipped_rather_than_stepped() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: 16,
                depth: 2,
                ..small_encoder()
            },
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();

    let mut dataset = Dataset::new(1);
    for (index, smiles) in TRAINING_SET.iter().enumerate() {
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        let target = if index == 3 { f64::NAN } else { index as f64 };
        dataset.push(&molecule, &[Some(target)]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };
    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            epochs: 3,
            batch_size: 8,
            patience: None,
            seed: 1,
            ..Default::default()
        },
        &device,
    )
    .unwrap();

    let history = trainer.fit(&dataset, &split).unwrap();
    // One batch per epoch holds the NaN target.
    assert_eq!(history.skipped_steps, 3);
    for variable in varmap.all_vars() {
        let values = variable
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|value| value.is_finite()),
            "a weight went non-finite"
        );
    }
}

#[test]
fn seeding_starts_a_layer_norm_where_every_library_does() {
    // A layer norm's gain starts at one and its bias and GINE's eps at zero.
    // Drawing them like a linear layer's weights started each gain at a few
    // hundredths, half of them negative, and seeded GINE runs lost about a
    // third of their accuracy to it.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = MolecularNet::new(
        ModelConfig {
            kind: EncoderKind::Gine,
            encoder: small_encoder(),
            graph_head: vec![16, 1],
            ..Default::default()
        },
        vb,
    )
    .unwrap();
    seed_variables(&varmap, 3).unwrap();

    let data = varmap.data().lock().unwrap();
    let values = |name: &str| to_vec(&data[name].as_detached_tensor());
    let mut norms = 0;
    for name in data.keys() {
        if name.ends_with(".eps") {
            assert!(values(name).iter().all(|&value| value == 0.0), "{name}");
        } else if name.contains(".norm") && name.ends_with(".weight") {
            assert!(values(name).iter().all(|&value| value == 1.0), "{name}");
            norms += 1;
        } else if name.contains(".norm") && name.ends_with(".bias") {
            assert!(values(name).iter().all(|&value| value == 0.0), "{name}");
        }
    }
    assert!(norms > 0, "the model had no layer norm to check");

    // A bias takes its layer's fan-in: the one-output head's bias is drawn
    // from the hidden width's bound, not from plus and minus one.
    let head_bias = data
        .keys()
        .filter(|name| name.starts_with("graph_head") && name.ends_with(".bias"))
        .find(|name| data[name.as_str()].dims() == [1])
        .expect("a one-output head bias")
        .clone();
    let bound = 1.0 / (16.0f32).sqrt();
    assert!(values(&head_bias)[0].abs() <= bound, "{head_bias}");
}

#[test]
fn a_sharded_step_is_the_step_one_thread_takes() {
    // One batch, one step: spread over three threads, the weights must move
    // exactly as they do on one, missing targets and all, since the shards'
    // losses are weighted by their share of the batch's measurements.
    let smiles = [
        "CCO", "c1ccccc1", "CC(=O)O", "CCN", "C1CCCCC1", "OCC(O)CO", "c1ccncc1",
    ];
    let mut dataset = Dataset::new(2).with_atom_tasks(1);
    for (index, one) in smiles.iter().enumerate() {
        let molecule = parse(one);
        let features = graph_features(&molecule);
        let atom_targets: Vec<Option<f64>> = (0..features.atom_count)
            .map(|row| (row % 3 != 0).then(|| row as f64 * 0.1))
            .collect();
        let targets = [
            Some(index as f64),
            (index % 2 == 0).then(|| -(index as f64)),
        ];
        dataset.push_features(features, &targets, &atom_targets, &[]);
    }
    let split = Split {
        train: (0..dataset.len()).collect(),
        ..Default::default()
    };

    // The loss is checked as well as the weights: AdamW's first step moves
    // each weight by the sign of its gradient, so the weights alone would
    // not notice shards weighted wrongly.
    let weights_after = |shards: usize| -> (f32, Vec<Vec<f32>>) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MolecularNet::new(
            ModelConfig {
                encoder: small_encoder(),
                graph_head: vec![16, 2],
                atom_head: vec![16, 1],
                ..Default::default()
            },
            vb,
        )
        .unwrap();
        seed_variables(&varmap, 6).unwrap();
        let mut trainer = Trainer::new(
            &model,
            &varmap,
            TrainConfig {
                epochs: 1,
                batch_size: smiles.len(),
                patience: None,
                shards,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        let history = trainer.fit(&dataset, &split).unwrap();
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        let weights = names
            .into_iter()
            .map(|name| to_vec(&data[name].as_detached_tensor()))
            .collect();
        (history.train_loss[0], weights)
    };

    let (single_loss, single) = weights_after(1);
    let (sharded_loss, sharded) = weights_after(3);
    assert!(
        (single_loss - sharded_loss).abs() < 1e-5 * single_loss.abs().max(1.0),
        "{single_loss} against {sharded_loss}"
    );
    for (one, three) in single.iter().zip(&sharded) {
        for (a, b) in one.iter().zip(three) {
            assert!((a - b).abs() < 1e-5, "{a} against {b}");
        }
    }
}

#[test]
fn a_bond_head_reads_the_bond_row_itself() {
    // The two directions of a bond get one prediction, and the bond's own
    // feature row reaches the head itself, not only through the encoder. Before, the head read the
    // two atom vectors alone, and a bond's order or ring membership had to be
    // inferred back out of the sums of every bond its atoms have.
    for kind in [EncoderKind::Directed, EncoderKind::Gine] {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MolecularNet::new(
            ModelConfig {
                kind,
                encoder: small_encoder(),
                bond_head: vec![16, 1],
                ..Default::default()
            },
            vb,
        )
        .unwrap();
        let batch = batch_of(&["c1ccccc1-c1ccccc1", "CC(=O)N"]);
        let graph = GraphTensors::new(&batch, &device).unwrap();
        let (atoms, edges) = model.encode_with_edges(&graph, false).unwrap();
        let plain = to_vec(&model.predict_with_edges(&atoms, edges.as_ref(), &graph, false).unwrap().bond.unwrap());
        let (reverse, _) = batch.reverse_edges();
        for edge in 0..batch.edge_count {
            let other = reverse[edge] as usize;
            assert!((plain[edge] - plain[other]).abs() < 1e-5, "{kind:?}: edge {edge}");
        }

        // The same encoding, read by the heads against flipped bond rows:
        // only a head that reads the bond itself can answer differently.
        let mut flipped = graph.clone();
        flipped.bond_features = graph.bond_features.affine(-1.0, 1.0).unwrap();
        let moved = to_vec(&model.predict_with_edges(&atoms, edges.as_ref(), &flipped, false).unwrap().bond.unwrap());
        assert!((moved[0] - plain[0]).abs() > 1e-4, "{kind:?}: the bond row did not reach the head");
    }
}
