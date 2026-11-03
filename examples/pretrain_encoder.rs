//! Pretrains an encoder on unlabelled SMILES by hiding atoms and asking what
//! they were, then writes the encoder out for a supervised run to start from.
//!
//! ```text
//! cargo run --release --features ml --example pretrain_encoder -- \
//!     --out encoder.safetensors --epochs 5 < structures.smi
//! ```
//!
//! Reads one SMILES per line and ignores anything after the first whitespace or
//! tab, so a labelled file can be fed in as-is. Labels are ignored:
//! pretraining uses no measurements.

use std::io::BufRead;

use candle_core::{DType, Device};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use oxidane::ml::batch::GraphBatch;
use oxidane::ml::nn::pretrain::{MaskedAtomModel, PretrainConfig};
use oxidane::ml::nn::{EncoderConfig, EncoderKind, save_prefixed, seed_variables};
use oxidane::ml::normalize::FeatureStats;
use oxidane::molecule::featurize::{GraphFeatures, graph_features};
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = "encoder.safetensors".to_string();
    let (mut hidden, mut depth, mut epochs, mut batch_size) = (300usize, 3usize, 5usize, 50usize);
    let (mut learning_rate, mut mask, mut seed, mut limit) = (1e-3f64, 0.15f64, 1u64, usize::MAX);
    let mut gine = false;

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let value = || arguments.get(index + 1).cloned().unwrap_or_default();
        let mut consumed = 2;
        match arguments[index].as_str() {
            "--out" => out = value(),
            "--hidden" => hidden = value().parse().unwrap_or(hidden),
            "--depth" => depth = value().parse().unwrap_or(depth),
            "--epochs" => epochs = value().parse().unwrap_or(epochs),
            "--batch" => batch_size = value().parse().unwrap_or(batch_size),
            "--lr" => learning_rate = value().parse().unwrap_or(learning_rate),
            "--mask" => mask = value().parse().unwrap_or(mask),
            "--seed" => seed = value().parse().unwrap_or(seed),
            "--limit" => limit = value().parse().unwrap_or(limit),
            "--gine" => {
                gine = true;
                consumed = 1;
            }
            other => {
                eprintln!("ignoring unknown argument {other}");
                consumed = 1;
            }
        }
        index += consumed;
    }

    let device = Device::Cpu;

    // ---- read, keeping only the structures ---------------------------------
    let mut molecules: Vec<GraphFeatures> = Vec::new();
    let mut unreadable = 0usize;
    for line in std::io::stdin().lock().lines() {
        if molecules.len() >= limit {
            break;
        }
        let line = line?;
        let smiles = line.split_whitespace().next().unwrap_or("");
        if smiles.is_empty() {
            continue;
        }
        let Ok(fragments) = MolecularGraph::from_smiles(smiles) else {
            unreadable += 1;
            continue;
        };
        if fragments.is_empty() {
            continue;
        }
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        let features = graph_features(&molecule);
        // A molecule with one heavy atom has nothing to infer an element from.
        if features.atom_count >= 3 {
            molecules.push(features);
        }
    }
    eprintln!("{} molecules, {unreadable} unreadable", molecules.len());
    if molecules.is_empty() {
        return Ok(());
    }

    // The scaling a supervised run will use, measured here so the encoder is
    // pretrained on the numbers it will later be given.
    let all = GraphBatch::collate(&molecules);
    let stats = FeatureStats::fit(&all);
    drop(all);

    // ---- build -------------------------------------------------------------
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = PretrainConfig {
        encoder: EncoderConfig {
            hidden,
            depth,
            ..Default::default()
        },
        kind: if gine {
            EncoderKind::Gine
        } else {
            EncoderKind::Directed
        },
        mask_fraction: mask,
    };
    let model = MaskedAtomModel::new(config, vb)?;
    seed_variables(&varmap, seed)?;

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        },
    )?;

    // ---- train -------------------------------------------------------------
    let mut order: Vec<usize> = (0..molecules.len()).collect();
    for epoch in 0..epochs {
        oxidane::ml::shuffled(&mut order, seed.wrapping_add(epoch as u64 + 1));
        let (mut total, mut steps) = (0.0f32, 0usize);
        let (mut right, mut asked) = (0usize, 0usize);

        for chunk in order.chunks(batch_size) {
            let batch = GraphBatch::collate_iter(chunk.iter().map(|&index| &molecules[index]));
            let mut features = batch.atom_features.clone();
            stats.apply(&mut features);

            let step_seed = (epoch as u64) << 32 | steps as u64;
            let loss = model.loss(&batch, &features, &device, step_seed, true)?;
            let value = loss.to_scalar::<f32>()?;
            if value > 0.0 {
                optimizer.backward_step(&loss)?;
            }
            total += value;
            steps += 1;

            // Measured on the same masking the step just trained on, so this is
            // a training accuracy and not a held-out one.
            let (correct, count) = model.accuracy(&batch, &features, &device, step_seed)?;
            right += correct;
            asked += count;
        }

        eprintln!(
            "epoch {epoch:3}  loss {:.4}  masked-atom accuracy {:.3}  ({asked} atoms asked)",
            total / steps.max(1) as f32,
            right as f64 / asked.max(1) as f64
        );
    }

    save_prefixed(&varmap, "encoder", &out)?;
    eprintln!("wrote {out}");
    println!("{out}");
    Ok(())
}
