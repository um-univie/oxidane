//! Pretrains an encoder on many labelled assays at once, then writes it out.
//!
//! ```text
//! cargo run --release --features ml --example pretrain_supervised -- \
//!     --data pcba.csv --out encoder.safetensors --epochs 10
//! ```
//!
//! This is the transfer that helps in practice. Self-supervised
//! masking on unlabelled structures did not help here and mostly hurt, which
//! is what the systematic reproductions of it report as well; what does carry
//! across is having been trained to predict *something* measured, on enough
//! assays that no single one can be memorised.
//!
//! The input is a CSV with a `smiles` column and every other column a task.
//! Blanks are missing measurements, which is the normal case -- PCBA is 128
//! assays and almost no molecule was run in all of them -- and they are masked
//! out of the loss rather than read as zeros.
//!
//! Molecules listed in `--exclude` are dropped, so the set a model will later
//! be tested on can be kept out of what its encoder saw.

use std::collections::HashSet;
use std::io::BufRead;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_nn::VarMap;

use oxidane::graph_algorithms::mol_canonization::canonize_mol;
use oxidane::io::ToSMILES;
use oxidane::ml::dataset::Dataset;
use oxidane::ml::nn::{
    EncoderConfig, EncoderKind, ModelConfig, MolecularNet, Objective, Pooling, TrainConfig,
    Trainer, save_prefixed, seed_variables,
};
use oxidane::ml::normalize::{FeatureStats, GlobalStats};
use oxidane::ml::split::{Fractions, random_split};
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = String::new();
    let mut out = "encoder.safetensors".to_string();
    let mut exclude: Vec<String> = Vec::new();
    let (mut hidden, mut depth, mut epochs, mut batch_size) = (300usize, 3usize, 10usize, 64usize);
    let (mut learning_rate, mut seed, mut limit) = (1e-3f64, 1u64, usize::MAX);
    let mut gine = false;

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let value = || arguments.get(index + 1).cloned().unwrap_or_default();
        let mut consumed = 2;
        match arguments[index].as_str() {
            "--data" => data = value(),
            "--out" => out = value(),
            "--exclude" => exclude.push(value()),
            "--hidden" => hidden = value().parse().unwrap_or(hidden),
            "--depth" => depth = value().parse().unwrap_or(depth),
            "--epochs" => epochs = value().parse().unwrap_or(epochs),
            "--batch" => batch_size = value().parse().unwrap_or(batch_size),
            "--lr" => learning_rate = value().parse().unwrap_or(learning_rate),
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
    if data.is_empty() {
        eprintln!("--data <csv> is required");
        return Ok(());
    }

    // ---- what must not be seen ---------------------------------------------
    let mut banned: HashSet<String> = HashSet::new();
    for path in &exclude {
        for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
            let line = line?;
            let smiles = line.split('\t').next().unwrap_or("").trim();
            if let Some(key) = canonical(smiles) {
                banned.insert(key);
            }
        }
    }
    eprintln!("{} molecules held out of pretraining", banned.len());

    // ---- read ---------------------------------------------------------------
    let file = std::io::BufReader::new(std::fs::File::open(&data)?);
    let mut lines = file.lines();
    let header = lines.next().transpose()?.unwrap_or_default();
    let columns: Vec<String> = split_csv(&header);
    let smiles_at = columns
        .iter()
        .position(|name| name == "smiles")
        .ok_or("the CSV has no smiles column")?;
    let tasks: Vec<usize> = (0..columns.len())
        .filter(|&column| {
            column != smiles_at
                && !matches!(columns[column].as_str(), "mol_id" | "Unnamed: 0" | "index")
        })
        .collect();
    eprintln!("{} tasks", tasks.len());

    let mut dataset = Dataset::new(tasks.len());
    let (mut dropped, mut unreadable, mut measurements) = (0usize, 0usize, 0usize);
    for line in lines {
        if dataset.len() >= limit {
            break;
        }
        let line = line?;
        let fields = split_csv(&line);
        if fields.len() != columns.len() {
            continue;
        }
        let Some(key) = canonical(&fields[smiles_at]) else {
            unreadable += 1;
            continue;
        };
        if banned.contains(&key) {
            dropped += 1;
            continue;
        }
        let targets: Vec<Option<f64>> = tasks
            .iter()
            .map(|&column| fields[column].trim().parse::<f64>().ok())
            .collect();
        let measured = targets.iter().filter(|value| value.is_some()).count();
        if measured == 0 {
            continue;
        }
        measurements += measured;
        let Ok(fragments) = MolecularGraph::from_smiles(&fields[smiles_at]) else {
            unreadable += 1;
            continue;
        };
        if fragments.is_empty() {
            continue;
        }
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        dataset.push_with_globals(&molecule, &targets);
    }
    eprintln!(
        "{} molecules, {measurements} measurements, {dropped} held out, {unreadable} unreadable",
        dataset.len()
    );
    if dataset.is_empty() {
        return Ok(());
    }

    // ---- train --------------------------------------------------------------
    // A validation slice only so that early stopping and the best-weight
    // restore have something to look at; there is no test set here, because
    // the thing being produced is an encoder rather than a score.
    let split = random_split(dataset.len(), Fractions::new(0.95, 0.05, 0.0), seed);
    let training = dataset.batch(&split.train);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device_of());
    let model = MolecularNet::new(
        ModelConfig {
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
            pooling: Pooling::Mean,
            graph_head: vec![hidden, dataset.task_count()],
            global_width: dataset.global_width(),
            ..Default::default()
        },
        vb,
    )?;
    seed_variables(&varmap, seed)?;

    let device = device_of();
    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate,
            epochs,
            batch_size,
            patience: Some(5),
            objective: Objective::Classification,
            seed,
            ..Default::default()
        },
        &device,
    )?
    .with_normalization(FeatureStats::fit(&training.graph))
    .with_global_normalization(GlobalStats::fit(&training.graph));

    let history = trainer.fit(&dataset, &split)?;
    for (epoch, loss) in history.train_loss.iter().enumerate() {
        let validation = history
            .valid_loss
            .get(epoch)
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "-".into());
        eprintln!("epoch {epoch:3}  train {loss:.4}  valid {validation}");
    }

    save_prefixed(&varmap, "encoder", &out)?;
    eprintln!("wrote {out}");
    println!("{out}");
    Ok(())
}

fn device_of() -> Device {
    Device::Cpu
}

/// The canonical form, for deciding whether two files name the same molecule.
fn canonical(smiles: &str) -> Option<String> {
    let fragments = MolecularGraph::from_smiles(smiles).ok()?;
    if fragments.is_empty() {
        return None;
    }
    let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
    Some(canonize_mol(&molecule).to_smiles())
}

/// A CSV line, honouring quotes -- a SMILES can hold a comma inside brackets.
fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for character in line.chars() {
        match character {
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut current)),
            _ => current.push(character),
        }
    }
    fields.push(current);
    fields
}
