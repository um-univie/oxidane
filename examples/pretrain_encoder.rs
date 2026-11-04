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
//!
//! * `--heavy` pretrains on graphs of the heavy atoms alone, which is what
//!   keeps a hidden atom's hydrogen count -- and so its valence and its
//!   element -- from reaching it through hydrogen rows. Fine-tune with
//!   `train_mpnn --heavy` on the same.
//! * `--exclude a.tsv,b.tsv` drops every structure whose canonical SMILES is the
//!   first column of a line of those files: the benchmark sets a pretrained
//!   encoder will be scored on, so its test molecules are not in what it
//!   pretrained on.
//! * `--shards N` spreads each step over N threads, as
//!   `TrainConfig::shards` does for supervised training.

use std::io::BufRead;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use oxidane::graph_algorithms::mol_canonization::canonize_mol;
use oxidane::io::ToSMILES;

use oxidane::ml::batch::GraphBatch;
use oxidane::ml::nn::pretrain::{MaskedAtomModel, PretrainConfig};
use oxidane::ml::nn::train::{clip_global_norm, global_norm};
use oxidane::ml::nn::{EncoderConfig, EncoderKind, save_prefixed, seed_variables};
use oxidane::ml::normalize::FeatureStats;
use oxidane::molecule::featurize::{GraphFeatures, graph_features, graph_features_heavy};
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = "encoder.safetensors".to_string();
    let (mut hidden, mut depth, mut epochs, mut batch_size) = (300usize, 3usize, 5usize, 50usize);
    let (mut learning_rate, mut mask, mut seed, mut limit) = (1e-3f64, 0.15f64, 1u64, usize::MAX);
    let mut gine = false;
    let (mut heavy, mut shards) = (false, 1usize);
    let mut exclude: Vec<String> = Vec::new();

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
            "--heavy" => {
                heavy = true;
                consumed = 1;
            }
            "--shards" => shards = value().parse().unwrap_or(shards).max(1),
            "--exclude" => exclude = value().split(',').map(str::to_string).collect(),
            other => {
                eprintln!("ignoring unknown argument {other}");
                consumed = 1;
            }
        }
        index += consumed;
    }

    let device = Device::Cpu;

    // ---- the structures to keep out ----------------------------------------
    let canonical = |molecule: &MolecularGraph| canonize_mol(molecule).to_smiles();
    let mut excluded = std::collections::HashSet::new();
    for path in &exclude {
        for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
            let line = line?;
            let smiles = line.split_whitespace().next().unwrap_or("");
            if let Ok(fragments) = MolecularGraph::from_smiles(smiles)
                && !fragments.is_empty()
            {
                excluded.insert(canonical(&MolecularGraph::combine(
                    &fragments.iter().collect::<Vec<_>>(),
                )));
            }
        }
    }

    // ---- read, keeping only the structures ---------------------------------
    let mut molecules: Vec<GraphFeatures> = Vec::new();
    let (mut unreadable, mut held_out) = (0usize, 0usize);
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
        if !excluded.is_empty() && excluded.contains(&canonical(&molecule)) {
            held_out += 1;
            continue;
        }
        let features = if heavy {
            graph_features_heavy(&molecule)
        } else {
            graph_features(&molecule)
        };
        // A molecule with one heavy atom has nothing to infer an element from.
        if features.atom_count >= 3 {
            molecules.push(features);
        }
    }
    eprintln!(
        "{} molecules, {unreadable} unreadable, {held_out} held out as benchmark structures",
        molecules.len()
    );
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

    let variables = varmap.all_vars();
    let mut optimizer = candle_nn::AdamW::new(
        variables.clone(),
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
            let step_seed = (epoch as u64) << 32 | steps as u64;
            // Each shard masks its own atoms and its loss counts equally:
            // shards of one batch are near enough the same size that
            // weighting them by their masked atoms would change little.
            let size = chunk.len().div_ceil(shards.min(chunk.len()));
            let parts: Vec<&[usize]> = chunk.chunks(size).collect();
            let share = 1.0 / parts.len() as f64;
            let run = |(part, indices): (usize, &&[usize])| -> candle_core::Result<(GradStore, f32, usize, usize)> {
                let batch = GraphBatch::collate_iter(indices.iter().map(|&index| &molecules[index]));
                let mut features = batch.atom_features.clone();
                stats.apply(&mut features);
                let seed = step_seed.wrapping_mul(31).wrapping_add(part as u64);
                let loss = model.loss(&batch, &features, &device, seed, true)?.affine(share, 0.0)?;
                let value = loss.to_scalar::<f32>()?;
                let gradients = loss.backward()?;
                // Measured on the same masking the step just trained on, so
                // this is a training accuracy and not a held-out one.
                let (correct, count) = model.accuracy(&batch, &features, &device, seed)?;
                Ok((gradients, value, correct, count))
            };
            let run = &run;
            let results: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = parts
                    .iter()
                    .enumerate()
                    .map(|part| scope.spawn(move || run(part)))
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("a shard panicked"))
                    .collect()
            });

            let mut summed: Option<GradStore> = None;
            for result in results {
                let (gradients, value, correct, count) = result?;
                total += value;
                right += correct;
                asked += count;
                match &mut summed {
                    None => summed = Some(gradients),
                    Some(sum) => {
                        for variable in &variables {
                            if let Some(gradient) = gradients.get(variable) {
                                let added = match sum.get(variable) {
                                    Some(previous) => previous.add(gradient)?,
                                    None => gradient.clone(),
                                };
                                sum.insert(variable, added);
                            }
                        }
                    }
                }
            }
            if let Some(mut gradients) = summed
                && global_norm(&gradients, &variables)?.is_finite()
            {
                clip_global_norm(&mut gradients, &variables, 5.0)?;
                optimizer.step(&gradients)?;
            }
            steps += 1;
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
