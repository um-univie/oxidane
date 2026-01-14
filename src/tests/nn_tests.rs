//! The neural half. Behind the `ml` feature, like the code it exercises.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use crate::ml::batch::GraphBatch;
use crate::ml::dataset::Dataset;
use crate::ml::nn::autoencoder::{AutoencoderConfig, GraphAutoencoder};
use crate::ml::nn::layers::{Activation, Mlp, MlpConfig};
use crate::ml::nn::loss::{mask_from_flags, masked_bce_with_logits, masked_mse};
use crate::ml::nn::model::{ModelConfig, MolecularNet, bond_predictions};
use crate::ml::nn::mpnn::{DMpnn, EncoderConfig, EncoderKind};
use crate::ml::nn::tensors::{GraphTensors, Pooling};
use crate::ml::nn::train::{Objective, TrainConfig, Trainer, clip_global_norm};
use crate::ml::split::{Fractions, Split};
use crate::molecule::base::MoleculeCore;
use crate::molecule::featurize::graph_features;
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

    assert_eq!(to_vec(&graph.pool_sum(&atoms).unwrap()), batch.segment_sum(&values, 1));
    let mean = to_vec(&graph.pool_mean(&atoms).unwrap());
    for (ours, theirs) in mean.iter().zip(batch.segment_mean(&values, 1)) {
        assert!((ours - theirs).abs() < 1e-4, "{ours} != {theirs}");
    }

    // And the edge scatter against `scatter_edges_to_atoms`.
    let edge_values: Vec<f32> = (0..batch.edge_count).map(|edge| edge as f32 + 1.0).collect();
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
    let device = Device::Cpu;
    let batch = batch_of(&["C"]);
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
    // Ethane has one bond, so after the subtraction there is nothing for
    // either edge to carry and every message stays at its initial value. If
    // the reverse-edge subtraction were missing, each round would instead feed
    // the other edge's message back in and the two would grow.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let shallow = DMpnn::new(EncoderConfig { depth: 1, ..small_encoder() }, vb.pp("e")).unwrap();
    let deep = DMpnn::new(EncoderConfig { depth: 4, ..small_encoder() }, vb.pp("e")).unwrap();

    let batch = batch_of(&["CC"]);
    let graph = GraphTensors::new(&batch, &device).unwrap();
    let one = to_vec(&shallow.forward(&graph, false).unwrap());
    let four = to_vec(&deep.forward(&graph, false).unwrap());
    for (a, b) in one.iter().zip(&four) {
        assert!((a - b).abs() < 1e-5, "depth changed a two-atom molecule: {a} != {b}");
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
    assert_eq!(model.encode_molecules(&graph, false).unwrap().dims(), &[1, 16]);
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
    // Nine atoms: the mean is the sum divided by nine.
    for (sum, mean) in pooled[0].iter().zip(&pooled[1]) {
        assert!((sum / 9.0 - mean).abs() < 1e-4, "{sum} / 9 != {mean}");
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
    "C", "CC", "CCC", "CCCC", "CCCCC", "CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccccc1C",
    "c1ccccc1CC", "CC(=O)O", "CC(=O)OC", "CCN", "CCCN", "C1CCCCC1", "C1CCCC1", "C1CCC1",
    "CC(C)C", "CC(C)CC", "OCCO", "OCCCO", "CCOCC", "CN(C)C", "c1ccncc1", "c1ccncc1C",
    "CC(=O)Oc1ccccc1", "CCCCCC", "CCCCCCC", "CCCCCCCC",
];

#[test]
fn a_model_learns_something_it_can_represent_exactly() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig { hidden: 32, depth: 2, ..small_encoder() },
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
    assert!(
        last < first * 0.05,
        "loss barely moved: {first} -> {last}"
    );
    // Counting atoms to within half an atom.
    assert!(last < 0.25, "final mean squared error {last}");
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
        for (was, now) in original.iter().zip(to_vec(gradients.get(variable).unwrap())) {
            assert!((was * scale as f32 - now).abs() < 1e-3, "{was} * {scale} != {now}");
        }
    }
}

fn global_norm(gradients: &candle_core::backprop::GradStore, variables: &[candle_core::Var]) -> f64 {
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
    assert!(history.train_loss.len() <= 5, "{}", history.train_loss.len());
}

#[test]
fn a_classifier_trains_on_logits() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig { hidden: 32, ..small_encoder() },
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
    let kl = autoencoder.kl_divergence(&latent).unwrap().to_scalar::<f32>().unwrap();
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
            encoder: EncoderConfig { hidden: 32, depth: 3, ..small_encoder() },
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

    let batch = batch_of(&["C"]);
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
