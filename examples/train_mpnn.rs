//! Trains a message-passing network on `smiles<TAB>value` lines from stdin.
//!
//! ```text
//! cargo run --release --features ml --example train_mpnn -- [options] < data.tsv
//!
//!   --hidden N       width of an atom vector          (default 300)
//!   --depth N        rounds of message passing        (default 3)
//!   --dropout F      dropout, 0 to 1                  (default 0.0)
//!   --lr F           learning rate                    (default 1e-3)
//!   --epochs N       maximum epochs                   (default 50)
//!   --batch N        molecules per step               (default 50)
//!   --patience N     epochs without improvement       (default 15)
//!   --seeds N        independent scaffold splits      (default 3)
//!   --pool sum|mean  readout                          (default mean)
//!   --gine           use GINE instead of D-MPNN
//!   --classify       binary targets, report ROC-AUC
//!   --random-split   split at random rather than by scaffold
//!   --globals        add whole-molecule descriptors to the readout
//! ```
//!
//! The default split is by Bemis--Murcko scaffold, so the reported test error
//! is the error on skeletons the model never saw. A random split of the same
//! data always looks better and always means less; `--random-split` is there to
//! measure how much better, which is a number worth knowing.

use std::io::{BufRead, Write};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use oxidane::ml::dataset::Dataset;
use oxidane::ml::nn::train::Objective;
use oxidane::ml::nn::{
    EncoderConfig, EncoderKind, ModelConfig, MolecularNet, Pooling, TrainConfig, Trainer,
};
use oxidane::ml::normalize::{FeatureStats, GlobalStats};
use oxidane::ml::split::{Fractions, Split, random_split, scaffold_split_balanced};
use oxidane::molecule::scaffold::scaffold_key;
use oxidane::prelude::*;

struct Options {
    hidden: usize,
    depth: usize,
    dropout: f32,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    patience: usize,
    seeds: u64,
    pooling: Pooling,
    kind: EncoderKind,
    classify: bool,
    random_split: bool,
    globals: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            hidden: 300,
            depth: 3,
            dropout: 0.0,
            learning_rate: 1e-3,
            epochs: 50,
            batch_size: 50,
            patience: 15,
            seeds: 3,
            pooling: Pooling::Mean,
            kind: EncoderKind::Directed,
            classify: false,
            random_split: false,
            globals: false,
        }
    }
}

fn parse_options() -> Options {
    let mut options = Options::default();
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        // The flag and its value are read separately rather than through a
        // closure: a closure that advanced `index` would hold it borrowed
        // across the match that reads it.
        let value = || arguments.get(index + 1).cloned().unwrap_or_default();
        let mut consumed = 2;
        match arguments[index].as_str() {
            "--hidden" => options.hidden = value().parse().unwrap_or(options.hidden),
            "--depth" => options.depth = value().parse().unwrap_or(options.depth),
            "--dropout" => options.dropout = value().parse().unwrap_or(options.dropout),
            "--lr" => options.learning_rate = value().parse().unwrap_or(options.learning_rate),
            "--epochs" => options.epochs = value().parse().unwrap_or(options.epochs),
            "--batch" => options.batch_size = value().parse().unwrap_or(options.batch_size),
            "--patience" => options.patience = value().parse().unwrap_or(options.patience),
            "--seeds" => options.seeds = value().parse().unwrap_or(options.seeds),
            "--pool" => {
                options.pooling = if value() == "sum" {
                    Pooling::Sum
                } else {
                    Pooling::Mean
                }
            }
            "--gine" => {
                options.kind = EncoderKind::Gine;
                consumed = 1;
            }
            "--classify" => {
                options.classify = true;
                consumed = 1;
            }
            "--random-split" => {
                options.random_split = true;
                consumed = 1;
            }
            "--globals" => {
                options.globals = true;
                consumed = 1;
            }
            other => {
                eprintln!("ignoring unknown argument {other}");
                consumed = 1;
            }
        }
        index += consumed;
    }
    options
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options();
    let device = Device::Cpu;

    // ---- read -------------------------------------------------------------
    let globals = options.globals;
    let mut dataset = Dataset::new(1);
    let mut keys: Vec<String> = Vec::new();
    let mut skipped = 0usize;

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let mut fields = line.trim().split('\t');
        let (Some(smiles), Some(value)) = (fields.next(), fields.next()) else {
            continue;
        };
        // The whole SMILES, not its first fragment. Half of BBBP is written
        // as a salt, and `from_smiles_single` would hand the model the
        // counterion and throw the drug away.
        let (Ok(fragments), Ok(value)) = (
            MolecularGraph::from_smiles(smiles),
            value.trim().parse::<f64>(),
        ) else {
            skipped += 1;
            continue;
        };
        if fragments.is_empty() {
            skipped += 1;
            continue;
        }
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        keys.push(scaffold_key(&molecule));
        if globals {
            dataset.push_with_globals(&molecule, &[Some(value)]);
        } else {
            dataset.push(&molecule, &[Some(value)]);
        }
    }

    if dataset.is_empty() {
        eprintln!("nothing to train on: expected `smiles<TAB>value` lines on stdin");
        return Ok(());
    }
    eprintln!(
        "{} molecules, {skipped} unreadable, {} distinct scaffolds",
        dataset.len(),
        keys.iter().collect::<std::collections::HashSet<_>>().len()
    );

    // ---- run, once per split ----------------------------------------------
    let mut scores = Vec::new();
    for seed in 0..options.seeds {
        let split = if options.random_split {
            random_split(dataset.len(), Fractions::default(), seed)
        } else {
            scaffold_split_balanced(&keys, Fractions::default(), seed)
        };
        let score = run(&dataset, &split, &options, seed, &device)?;
        eprintln!(
            "  split {seed}: test {} = {:.4}   (train {} / valid {} / test {})",
            if options.classify { "ROC-AUC" } else { "RMSE" },
            score,
            split.train.len(),
            split.valid.len(),
            split.test.len()
        );
        scores.push(score);
    }

    // ---- report -----------------------------------------------------------
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let deviation = (scores
        .iter()
        .map(|score| (score - mean).powi(2))
        .sum::<f64>()
        / scores.len() as f64)
        .sqrt();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    writeln!(
        out,
        "{}\t{:.4} +/- {:.4}\tover {} {} splits",
        if options.classify { "ROC-AUC" } else { "RMSE" },
        mean,
        deviation,
        scores.len(),
        if options.random_split {
            "random"
        } else {
            "scaffold"
        }
    )?;
    Ok(())
}

fn run(
    dataset: &Dataset,
    split: &Split,
    options: &Options,
    seed: u64,
    device: &Device,
) -> Result<f64, Box<dyn std::error::Error>> {
    // Fit the feature scaling on the training set alone. Fitting on all of it
    // leaks the test set's distribution into training.
    let training = dataset.batch(&split.train);
    let stats = FeatureStats::fit(&training.graph);
    let global_stats = GlobalStats::fit(&training.graph);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MolecularNet::new(
        ModelConfig {
            encoder: EncoderConfig {
                hidden: options.hidden,
                depth: options.depth,
                dropout: options.dropout,
                ..Default::default()
            },
            kind: options.kind,
            pooling: options.pooling,
            graph_head: vec![options.hidden, 1],
            global_width: dataset.global_width(),
            ..Default::default()
        },
        vb,
    )?;

    let mut trainer = Trainer::new(
        &model,
        &varmap,
        TrainConfig {
            learning_rate: options.learning_rate,
            epochs: options.epochs,
            batch_size: options.batch_size,
            patience: Some(options.patience),
            objective: if options.classify {
                Objective::Classification
            } else {
                Objective::Regression
            },
            seed: seed + 1,
            ..Default::default()
        },
        device,
    )?
    .with_normalization(stats)
    .with_global_normalization(global_stats);

    trainer.fit(dataset, split)?;

    let predictions = trainer.predict(dataset, &split.test)?;
    let truth: Vec<f64> = split
        .test
        .iter()
        .map(|&index| dataset.targets(index).0[0] as f64)
        .collect();
    let predicted: Vec<f64> = predictions.iter().map(|row| row[0] as f64).collect();

    Ok(if options.classify {
        roc_auc(&predicted, &truth)
    } else {
        root_mean_squared_error(&predicted, &truth)
    })
}

fn root_mean_squared_error(predicted: &[f64], truth: &[f64]) -> f64 {
    if predicted.is_empty() {
        return f64::NAN;
    }
    let total: f64 = predicted
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    (total / predicted.len() as f64).sqrt()
}

/// The area under the ROC curve, by the rank-sum identity.
///
/// Equal to the chance that a randomly chosen positive scores above a randomly
/// chosen negative, which is why ties count a half. Returns `NaN` when one of
/// the classes is missing, since the question has no answer then.
fn roc_auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Average the ranks of tied scores, so that a model predicting one constant
    // scores 0.5 rather than whatever the input order happened to give.
    let mut ranks = vec![0.0f64; scores.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && scores[order[end]] == scores[order[start]] {
            end += 1;
        }
        let average = (start + end - 1) as f64 / 2.0 + 1.0;
        for &index in &order[start..end] {
            ranks[index] = average;
        }
        start = end;
    }

    let positives: f64 = labels.iter().filter(|&&label| label > 0.5).count() as f64;
    let negatives = labels.len() as f64 - positives;
    if positives == 0.0 || negatives == 0.0 {
        return f64::NAN;
    }
    let rank_sum: f64 = labels
        .iter()
        .zip(&ranks)
        .filter(|(label, _)| **label > 0.5)
        .map(|(_, rank)| *rank)
        .sum();
    (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)
}
