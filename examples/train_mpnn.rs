//! Trains a message-passing network on labelled molecules from stdin.
//!
//! ```text
//! cargo run --release --features ml --example train_mpnn -- [options] < data.tsv
//!
//!   --level graph|atom|bond  what to predict, and the input format (below)
//!                                                    (default graph)
//!   --hidden N       width of an atom vector          (default 300)
//!   --depth N        rounds of message passing        (default 3)
//!   --dropout F      dropout, 0 to 1                  (default 0.0)
//!   --lr F           learning rate                    (default 1e-3)
//!   --epochs N       maximum epochs                   (default 50)
//!   --batch N        molecules per step                (default 50)
//!   --patience N     epochs without improvement        (default 15)
//!   --seeds N        independent scaffold splits       (default 3)
//!   --pool sum|mean  readout                            (default mean)
//!   --gine           use GINE instead of D-MPNN
//!   --classify       binary targets, report ROC-AUC
//!   --random-split   split at random rather than by scaffold
//!   --globals        add whole-molecule descriptors to the readout (graph level only)
//!   --ensemble N     average N independently initialised models   (default 1)
//!   --train-fraction F  keep only F of the training set, for a learning curve
//!   --pretrained FILE   start from an encoder written by pretrain_encoder
//!   --jobs N         splits*ensemble members to train at once (default: that many, capped at the core count)
//!   --freeze N       train the heads alone for N epochs before releasing the encoder
//!   --schedule       warm the learning rate up and decay it, rather than holding it
//!   --raw-targets    train on the targets as given, without standardising them
//!   --residual       subtract a per-atom/bond-class mean before training, add it back for predictions (atom/bond level only)
//!   --geometry       embed a 3D conformer per molecule and add bond length / local bond angle features
//!   --shards N       spread each training step over N threads (default 1; worth raising for a single run,
//!                    not when --jobs already fills the machine)
//!   --heavy          give rows to heavy atoms only, hydrogens counted on their neighbour (about twice as fast;
//!                    atom/bond labels on hydrogens are dropped)
//! ```
//!
//! `--residual` is the training trick behind ALFABET's near-chemical-accuracy
//! bond dissociation energy model (St John et al. 2020): rather than the raw
//! target, the model predicts the residual on top of a mean value for atoms
//! or bonds of the same coarse class (element, hybridization/order,
//! aromaticity -- see [`oxidane::molecule::featurize::atom_class`] and
//! [`bond_class`](oxidane::molecule::featurize::bond_class)), which removes
//! most of a bond-type-driven swing (O-H vs. C-C vs. aromatic C-H) before the
//! network has to reason about anything finer.
//!
//! The default split is by Bemis--Murcko scaffold, so the reported test error
//! is the error on skeletons the model never saw. A random split of the same
//! data always looks better and always means less; `--random-split` is there to
//! measure how much better, which is a number worth knowing.
//!
//! # Input format
//!
//! One line per molecule, tab-separated. What follows the SMILES depends on
//! `--level`:
//!
//! ```text
//! graph   smiles<TAB>value
//! atom    smiles<TAB>0:-0.049,1:0.038,2:-0.398
//! bond    smiles<TAB>0-1:88.9,1-2:91.9
//! ```
//!
//! Atom and bond indices are 0-based, in the order the atoms appear in the
//! SMILES -- what every [`MoleculeCore`](oxidane::molecule::base::MoleculeCore)
//! method takes. Not every atom or bond needs a label; only the ones listed
//! are trained on. Atom and bond level lines are restricted to
//! single-fragment SMILES -- a salt's counterion has no meaningful index to
//! label, and combining fragments would shift the ones given -- so a
//! multi-fragment line is skipped and counted like an unparseable one.

use std::io::{BufRead, Write};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use oxidane::ml::dataset::Dataset;
use oxidane::ml::nn::train::Objective;
use oxidane::ml::nn::{
    EncoderConfig, EncoderKind, ModelConfig, MolecularNet, Pooling, Schedule, TrainConfig, Trainer,
    load_prefixed, seed_variables, variables_outside,
};
use oxidane::ml::normalize::{
    BondFeatureStats, ClassMeans, FeatureStats, GlobalStats, TargetStats,
};
use oxidane::ml::split::{Fractions, Split, random_split, scaffold_split_balanced};
use oxidane::molecule::embed::Embed3D;
use oxidane::molecule::featurize::{
    GraphFeatures, graph_features, graph_features_heavy, graph_features_heavy_with_geometry,
    graph_features_with_geometry, molecule_features,
};
use oxidane::molecule::scaffold::scaffold_key;
use oxidane::prelude::*;
use rayon::prelude::*;

/// What the model is asked to predict, and so which head is built and which
/// input format is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Level {
    #[default]
    Graph,
    Atom,
    Bond,
}

impl Level {
    fn parse(value: &str) -> Self {
        match value {
            "atom" => Level::Atom,
            "bond" => Level::Bond,
            _ => Level::Graph,
        }
    }
}

struct Options {
    level: Level,
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
    ensemble: usize,
    schedule: bool,
    scale_targets: bool,
    train_fraction: f64,
    pretrained: Option<String>,
    jobs: usize,
    freeze_epochs: usize,
    residual: bool,
    geometry: bool,
    heavy: bool,
    shards: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            level: Level::default(),
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
            ensemble: 1,
            schedule: false,
            scale_targets: true,
            train_fraction: 1.0,
            pretrained: None,
            jobs: 0,
            freeze_epochs: 0,
            residual: false,
            geometry: false,
            heavy: false,
            shards: 1,
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
            "--level" => options.level = Level::parse(&value()),
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
            "--ensemble" => options.ensemble = value().parse().unwrap_or(options.ensemble).max(1),
            "--pretrained" => options.pretrained = Some(value()),
            "--jobs" => options.jobs = value().parse().unwrap_or(options.jobs),
            "--freeze" => options.freeze_epochs = value().parse().unwrap_or(options.freeze_epochs),
            "--train-fraction" => {
                options.train_fraction = value()
                    .parse()
                    .unwrap_or(options.train_fraction)
                    .clamp(0.01, 1.0)
            }
            "--schedule" => {
                options.schedule = true;
                consumed = 1;
            }
            "--raw-targets" => {
                options.scale_targets = false;
                consumed = 1;
            }
            "--residual" => {
                options.residual = true;
                consumed = 1;
            }
            "--geometry" => {
                options.geometry = true;
                consumed = 1;
            }
            "--heavy" => {
                options.heavy = true;
                consumed = 1;
            }
            "--shards" => {
                options.shards = value().parse().unwrap_or(options.shards);
            }
            other => {
                eprintln!("ignoring unknown argument {other}");
                consumed = 1;
            }
        }
        index += consumed;
    }
    if options.globals && options.level != Level::Graph {
        eprintln!(
            "--globals only reaches the graph head; ignoring it at --level {:?}",
            options.level
        );
        options.globals = false;
    }
    if options.residual && options.level == Level::Graph {
        eprintln!("--residual has no bond/atom class at --level graph; ignoring it");
        options.residual = false;
    }
    options
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options();
    let device = Device::Cpu;

    // ---- read -------------------------------------------------------------
    // Parsing is cheap and stays sequential (stdin is a line at a time
    // anyway); featurizing is not, once `--geometry` is asking for a 3D
    // embedding per molecule (measured at ~200ms each -- hours, sequentially,
    // over a real BDE-scale set). So parsing only collects a molecule and its
    // targets, and featurizing -- the part `--geometry` makes expensive --
    // runs as a `rayon` map over all of them at once afterward, same as the
    // splits and ensemble members below.
    let mut parsed: Vec<Parsed> = Vec::new();
    let mut skipped = 0usize;

    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let mut fields = line.trim().split('\t');
        let (Some(smiles), Some(rest)) = (fields.next(), fields.next()) else {
            continue;
        };

        match options.level {
            Level::Graph => {
                // The whole SMILES, not its first fragment. Half of BBBP is
                // written as a salt, and `from_smiles_single` would hand the
                // model the counterion and throw the drug away.
                let (Ok(fragments), Ok(value)) = (
                    MolecularGraph::from_smiles(smiles),
                    rest.trim().parse::<f64>(),
                ) else {
                    skipped += 1;
                    continue;
                };
                if fragments.is_empty() {
                    skipped += 1;
                    continue;
                }
                let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
                parsed.push(Parsed {
                    molecule,
                    extra: Extra::Value(value),
                });
            }
            Level::Atom | Level::Bond => {
                let Ok(mut fragments) = MolecularGraph::from_smiles(smiles) else {
                    skipped += 1;
                    continue;
                };
                if fragments.len() != 1 {
                    skipped += 1;
                    continue;
                }
                let molecule = fragments.swap_remove(0);

                if options.level == Level::Atom {
                    let Some(labels) = parse_atom_labels(rest) else {
                        skipped += 1;
                        continue;
                    };
                    parsed.push(Parsed {
                        molecule,
                        extra: Extra::AtomLabels(labels),
                    });
                } else {
                    let Some(labels) = parse_bond_labels(rest) else {
                        skipped += 1;
                        continue;
                    };
                    parsed.push(Parsed {
                        molecule,
                        extra: Extra::BondLabels(labels),
                    });
                }
            }
        }
    }

    let featurized: Vec<(GraphFeatures, bool)> = if options.geometry {
        parsed
            .par_iter()
            .map(|entry| featurize(&entry.molecule, true, options.heavy))
            .collect()
    } else {
        parsed
            .iter()
            .map(|entry| featurize(&entry.molecule, false, options.heavy))
            .collect()
    };

    let mut dataset = match options.level {
        Level::Graph => Dataset::new(1),
        Level::Atom => Dataset::new(0).with_atom_tasks(1),
        Level::Bond => Dataset::new(0).with_bond_tasks(1),
    };
    let mut keys: Vec<String> = Vec::new();
    let mut geometry_failures = 0usize;

    for (entry, (features, embed_failed)) in parsed.into_iter().zip(featurized) {
        if embed_failed {
            geometry_failures += 1;
        }
        keys.push(scaffold_key(&entry.molecule));
        match entry.extra {
            Extra::Value(value) => {
                if options.globals {
                    let globals = molecule_features(&entry.molecule);
                    dataset.push_features_with_globals(features, &globals, &[Some(value)]);
                } else {
                    dataset.push_features(features, &[Some(value)], &[], &[]);
                }
            }
            Extra::AtomLabels(labels) => {
                let atom_targets = features.atom_targets_from(&labels);
                dataset.push_features(features, &[], &atom_targets, &[]);
            }
            Extra::BondLabels(labels) => {
                let bond_targets = features.bond_targets_from(&labels);
                dataset.push_features(features, &[], &[], &bond_targets);
            }
        }
    }

    if dataset.is_empty() {
        eprintln!(
            "nothing to train on: check the input format for --level {:?}",
            options.level
        );
        return Ok(());
    }
    eprintln!(
        "{} molecules, {skipped} unreadable, {} distinct scaffolds{}",
        dataset.len(),
        keys.iter().collect::<std::collections::HashSet<_>>().len(),
        if options.geometry {
            format!(", {geometry_failures} fell back to 2D features (embedding failed)")
        } else {
            String::new()
        }
    );

    // ---- run, once per split ----------------------------------------------
    // The splits share nothing -- each builds its own weights and reads the
    // dataset immutably -- so they are run at once. One split at a time left
    // this at about two of ten cores busy, because most of the work is small
    // gather and scatter operations that no matmul library threads for us.
    let splits: Vec<Split> = (0..options.seeds)
        .map(|seed| {
            if options.random_split {
                random_split(dataset.len(), Fractions::default(), seed)
            } else {
                scaffold_split_balanced(&keys, Fractions::default(), seed)
            }
        })
        .map(|mut split| {
            // A learning curve needs the same held-out sets and less of the
            // training one, so the thinning happens here and not in the split.
            if options.train_fraction < 1.0 {
                let keep = ((split.train.len() as f64) * options.train_fraction).round() as usize;
                let mut order = split.train.clone();
                oxidane::ml::shuffled(&mut order, 7717);
                order.truncate(keep.max(1));
                order.sort_unstable();
                split.train = order;
            }
            split
        })
        .collect();

    // Every split trains its own ensemble in parallel too (see `run`), so the
    // independent training runs going on at once are splits times ensemble
    // members, not just splits -- the pool needs enough workers for both
    // layers to actually overlap rather than queue behind one another.
    // Capped at the machine's own core count: asking for more workers than
    // there are cores buys nothing and just adds scheduling overhead.
    let jobs = if options.jobs == 0 {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        (splits.len() * options.ensemble.max(1)).min(available)
    } else {
        options.jobs
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()?;

    let outcomes: Vec<(f64, f64, f64, f64)> = pool.install(|| {
        splits
            .par_iter()
            .enumerate()
            .map(|(seed, split)| {
                run(&dataset, split, &options, seed as u64, &device).unwrap_or((
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                ))
            })
            .collect()
    });

    let mut scores = Vec::new();
    let mut maes = Vec::new();
    for (seed, ((score, train_score, mae, train_mae), split)) in
        outcomes.iter().zip(&splits).enumerate()
    {
        eprintln!(
            "  split {seed}: train {:.4}  test {} = {:.4}{}   (train {} / valid {} / test {})",
            train_score,
            if options.classify { "ROC-AUC" } else { "RMSE" },
            score,
            if options.classify {
                String::new()
            } else {
                format!("  (MAE train {train_mae:.4} test {mae:.4})")
            },
            split.train.len(),
            split.valid.len(),
            split.test.len()
        );
        scores.push(*score);
        maes.push(*mae);
    }

    // ---- report -----------------------------------------------------------
    let summarize = |values: &[f64]| {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let deviation = (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt();
        (mean, deviation)
    };
    let (mean, deviation) = summarize(&scores);
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let split_kind = if options.random_split {
        "random"
    } else {
        "scaffold"
    };
    writeln!(
        out,
        "{}\t{:.4} +/- {:.4}\tover {} {split_kind} splits",
        if options.classify { "ROC-AUC" } else { "RMSE" },
        mean,
        deviation,
        scores.len(),
    )?;
    if !options.classify {
        let (mae_mean, mae_deviation) = summarize(&maes);
        writeln!(
            out,
            "MAE\t{mae_mean:.4} +/- {mae_deviation:.4}\tover {} {split_kind} splits",
            maes.len()
        )?;
    }
    Ok(())
}

/// A parsed input line, not yet featurized -- see the comment in `main`
/// on why featurizing (specifically, the embedding `--geometry` asks for)
/// happens afterward, over all of these at once, rather than inline here.
struct Parsed {
    molecule: MolecularGraph,
    extra: Extra,
}

/// What a [`Parsed`] line still needs to become a [`Dataset`] row -- the part
/// that's different per [`Level`].
enum Extra {
    Value(f64),
    AtomLabels(std::collections::HashMap<usize, f64>),
    BondLabels(std::collections::HashMap<(usize, usize), f64>),
}

/// [`graph_features`], embedding a 3D conformer first when `geometry` is set
/// -- see [`graph_features_with_geometry`]. The second element of the return
/// says the embedding failed and the plain 2D features were used instead; the
/// caller counts it rather than sharing a mutable counter, since this runs
/// across a `rayon` map over every molecule at once when `geometry` is set.
fn featurize(molecule: &MolecularGraph, geometry: bool, heavy: bool) -> (GraphFeatures, bool) {
    let plain = || {
        if heavy {
            graph_features_heavy(molecule)
        } else {
            graph_features(molecule)
        }
    };
    if !geometry {
        return (plain(), false);
    }
    match molecule.embed_3d() {
        Ok(embedded) if heavy => (
            graph_features_heavy_with_geometry(molecule, &embedded.positions),
            false,
        ),
        Ok(embedded) => (
            graph_features_with_geometry(molecule, &embedded.positions),
            false,
        ),
        Err(_) => (plain(), true),
    }
}

/// `0:-0.049,1:0.038,2:-0.398` into atom-index-keyed labels. `None` if any
/// entry does not parse, so the caller can skip the whole line rather than
/// train on a partly-misread one.
fn parse_atom_labels(field: &str) -> Option<std::collections::HashMap<usize, f64>> {
    field
        .trim()
        .split(',')
        .map(|entry| {
            let (index, value) = entry.split_once(':')?;
            Some((
                index.trim().parse::<usize>().ok()?,
                value.trim().parse::<f64>().ok()?,
            ))
        })
        .collect()
}

/// `0-1:88.9,1-2:91.9` into unordered-atom-pair-keyed labels.
fn parse_bond_labels(field: &str) -> Option<std::collections::HashMap<(usize, usize), f64>> {
    field
        .trim()
        .split(',')
        .map(|entry| {
            let (pair, value) = entry.split_once(':')?;
            let (a, b) = pair.split_once('-')?;
            let (a, b) = (
                a.trim().parse::<usize>().ok()?,
                b.trim().parse::<usize>().ok()?,
            );
            Some(((a.min(b), a.max(b)), value.trim().parse::<f64>().ok()?))
        })
        .collect()
}

/// One prediction per row of `indices` at `level` -- one row per molecule for
/// [`Level::Graph`], one per atom row or directed edge otherwise. See
/// [`Trainer::predict`], [`Trainer::predict_atoms`], [`Trainer::predict_bonds`].
fn predict_at_level(
    trainer: &Trainer<'_>,
    dataset: &Dataset,
    indices: &[usize],
    level: Level,
) -> candle_core::Result<Vec<Vec<f32>>> {
    match level {
        Level::Graph => trainer.predict(dataset, indices),
        Level::Atom => trainer.predict_atoms(dataset, indices),
        Level::Bond => trainer.predict_bonds(dataset, indices),
    }
}

/// The truth and measured-mask lined up with [`predict_at_level`]'s output
/// for the same `indices`.
fn truth_at_level(dataset: &Dataset, indices: &[usize], level: Level) -> (Vec<f64>, Vec<bool>) {
    let mut truth = Vec::new();
    let mut mask = Vec::new();
    for &index in indices {
        let (values, measured) = match level {
            Level::Graph => dataset.targets(index),
            Level::Atom => dataset.atom_targets(index),
            Level::Bond => dataset.bond_targets(index),
        };
        truth.extend(values.iter().map(|&value| value as f64));
        mask.extend(measured.iter().copied());
    }
    (truth, mask)
}

fn sigmoid(logit: f64) -> f64 {
    1.0 / (1.0 + (-logit).exp())
}

/// `(test_score, train_score, test_mae, train_mae)`. `test_score`/
/// `train_score` are ROC-AUC when `--classify`, RMSE otherwise; the MAE
/// fields are `NAN` for `--classify`, where they don't apply.
fn run(
    dataset: &Dataset,
    split: &Split,
    options: &Options,
    seed: u64,
    device: &Device,
) -> Result<(f64, f64, f64, f64), Box<dyn std::error::Error>> {
    // Fit every scaling on the training set alone. Fitting on all of it leaks
    // the test set's distribution into training.
    let training = dataset.batch(&split.train);
    let stats = FeatureStats::fit(&training.graph);
    let global_stats = GlobalStats::fit(&training.graph);
    let bond_feature_stats = options
        .geometry
        .then(|| BondFeatureStats::fit(&training.graph));
    let class_means = if options.residual {
        match options.level {
            Level::Atom => Some(ClassMeans::fit_atom(dataset, &split.train)),
            Level::Bond => Some(ClassMeans::fit_bond(dataset, &split.train)),
            Level::Graph => None,
        }
    } else {
        None
    };
    // With `--residual`, the target statistics below are fit over what the
    // model is actually being asked to predict -- the residual after the
    // class mean, not the raw value -- so its scaling matches the much
    // tighter distribution the class mean leaves behind rather than the
    // full bond-type-driven spread of the raw target.
    let target_stats = match (options.level, &class_means) {
        (Level::Atom, Some(means)) => TargetStats::fit_atom_residual(dataset, &split.train, means),
        (Level::Bond, Some(means)) => TargetStats::fit_bond_residual(dataset, &split.train, means),
        (Level::Graph, _) => TargetStats::fit(dataset, &split.train),
        (Level::Atom, None) => TargetStats::fit_atom(dataset, &split.train),
        (Level::Bond, None) => TargetStats::fit_bond(dataset, &split.train),
    };

    let (graph_head, atom_head, bond_head) = match options.level {
        Level::Graph => (vec![options.hidden, 1], vec![], vec![]),
        Level::Atom => (vec![], vec![options.hidden, 1], vec![]),
        Level::Bond => (vec![], vec![], vec![options.hidden, 1]),
    };

    // An ensemble of independently initialised models, averaged. They see the
    // same split and the same data; what differs is where they start and the
    // order they are shown it, and averaging over that is worth more than any
    // one of them can be. Every member trains its own weights end to end
    // with nothing shared, exactly like the splits above -- so it runs the
    // same way, in parallel over `options.ensemble` rather than in a loop.
    // A single member's own training is not itself parallel: message passing
    // on a batch of small organic molecules keeps every matrix multiply too
    // small for candle's threaded matmul to bother splitting up, so the only
    // parallelism worth having is across independent runs, not inside one.
    let train_member = |member: usize| -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
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
                graph_head: graph_head.clone(),
                atom_head: atom_head.clone(),
                bond_head: bond_head.clone(),
                global_width: dataset.global_width(),
                ..Default::default()
            },
            vb,
        )?;
        // Candle's CPU backend cannot be seeded, so the weights are drawn
        // again from a stream this crate controls. Without it two runs of one
        // command give two answers and no comparison here means anything.
        seed_variables(&varmap, seed * 1000 + member as u64 + 1)?;
        // After seeding, so the pretrained weights are not then overwritten.
        if let Some(path) = &options.pretrained {
            load_prefixed(&varmap, "encoder", path, device)?;
        }

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
                schedule: if options.schedule {
                    Schedule::WarmupDecay {
                        warmup: 2,
                        final_fraction: 0.1,
                    }
                } else {
                    Schedule::Constant
                },
                seed: seed * 100 + member as u64 + 1,
                shards: options.shards,
                ..Default::default()
            },
            device,
        )?
        .with_normalization(stats.clone())
        .with_global_normalization(global_stats.clone());
        if let Some(bond_stats) = &bond_feature_stats {
            trainer = trainer.with_bond_feature_normalization(bond_stats.clone());
        }
        if options.scale_targets {
            trainer = match options.level {
                Level::Graph => trainer.with_target_normalization(target_stats.clone()),
                Level::Atom => trainer.with_atom_target_normalization(target_stats.clone()),
                Level::Bond => trainer.with_bond_target_normalization(target_stats.clone()),
            };
        }
        if let Some(means) = &class_means {
            trainer = match options.level {
                Level::Atom => trainer.with_atom_class_means(means.clone()),
                Level::Bond => trainer.with_bond_class_means(means.clone()),
                Level::Graph => trainer,
            };
        }

        // A freshly initialised head sends large, meaningless gradients back
        // into a pretrained encoder for its first epochs. Letting the head
        // settle against a fixed encoder first is what separates transferring
        // a representation from destroying one.
        if options.freeze_epochs > 0 && options.pretrained.is_some() {
            let mut warm = Trainer::over(
                &model,
                variables_outside(&varmap, "encoder")?,
                TrainConfig {
                    learning_rate: options.learning_rate,
                    epochs: options.freeze_epochs,
                    batch_size: options.batch_size,
                    patience: None,
                    objective: if options.classify {
                        Objective::Classification
                    } else {
                        Objective::Regression
                    },
                    seed: seed * 100 + member as u64 + 1,
                    shards: options.shards,
                    ..Default::default()
                },
                device,
            )?
            .with_normalization(stats.clone())
            .with_global_normalization(global_stats.clone());
            if let Some(bond_stats) = &bond_feature_stats {
                warm = warm.with_bond_feature_normalization(bond_stats.clone());
            }
            if options.scale_targets {
                warm = match options.level {
                    Level::Graph => warm.with_target_normalization(target_stats.clone()),
                    Level::Atom => warm.with_atom_target_normalization(target_stats.clone()),
                    Level::Bond => warm.with_bond_target_normalization(target_stats.clone()),
                };
            }
            if let Some(means) = &class_means {
                warm = match options.level {
                    Level::Atom => warm.with_atom_class_means(means.clone()),
                    Level::Bond => warm.with_bond_class_means(means.clone()),
                    Level::Graph => warm,
                };
            }
            warm.fit(dataset, split)?;
        }

        // Nothing else here prints until a member is entirely done, which for
        // a big dataset can be tens of minutes -- indistinguishable from a
        // hang. `fit_with` calls back after every epoch; a line every fifth
        // one (plus the first) is enough to show real progress -- current
        // loss and how far training has actually gotten -- without eighteen
        // members flooding stderr with one line apiece per epoch.
        let history = trainer.fit_with(dataset, split, |history| {
            let epoch = history.train_loss.len();
            if epoch != 1 && epoch % 5 != 0 {
                return;
            }
            let train_loss = history.train_loss.last().copied().unwrap_or(f32::NAN);
            match history.valid_loss.last() {
                Some(&valid_loss) => eprintln!(
                    "    split {seed} member {member}: epoch {epoch}  train {train_loss:.4}  valid {valid_loss:.4}  (best {:.4} @ epoch {})",
                    history.best_loss,
                    history.best_epoch + 1
                ),
                None => eprintln!("    split {seed} member {member}: epoch {epoch}  train {train_loss:.4}"),
            }
        })?;
        eprintln!(
            "  split {seed} member {member}: {} epochs{}, best {}",
            history.train_loss.len(),
            if history.stopped_early {
                " (stopped early)"
            } else {
                ""
            },
            history.best_loss
        );

        let train_predictions = predict_at_level(&trainer, dataset, &split.train, options.level)?;
        let test_predictions = predict_at_level(&trainer, dataset, &split.test, options.level)?;
        // A classifier's head reports a logit; averaging those is not the
        // same as averaging the probabilities they stand for, and it is the
        // probabilities the ensemble is meant to pool.
        let to_probability = |prediction: &Vec<f32>| {
            if options.classify {
                sigmoid(prediction[0] as f64)
            } else {
                prediction[0] as f64
            }
        };
        Ok((
            train_predictions.iter().map(to_probability).collect(),
            test_predictions.iter().map(to_probability).collect(),
        ))
    };

    let outcomes: Vec<(Vec<f64>, Vec<f64>)> = (0..options.ensemble)
        .into_par_iter()
        .map(|member| {
            train_member(member).unwrap_or_else(|error| {
                eprintln!("ensemble member {member} failed: {error}");
                (Vec::new(), Vec::new())
            })
        })
        .collect();

    let mut summed: Vec<f64> = Vec::new();
    let mut summed_train: Vec<f64> = Vec::new();
    for (train_probabilities, test_probabilities) in &outcomes {
        if summed_train.is_empty() {
            summed_train = vec![0.0; train_probabilities.len()];
        }
        if summed.is_empty() {
            summed = vec![0.0; test_probabilities.len()];
        }
        for (row, &value) in train_probabilities.iter().enumerate() {
            summed_train[row] += value;
        }
        for (row, &value) in test_probabilities.iter().enumerate() {
            summed[row] += value;
        }
    }

    let predicted: Vec<f64> = summed
        .iter()
        .map(|total| total / options.ensemble as f64)
        .collect();
    let (truth, mask) = truth_at_level(dataset, &split.test, options.level);
    let (predicted, truth) = filter_measured(&predicted, &truth, &mask);

    let train_predicted: Vec<f64> = summed_train
        .iter()
        .map(|total| total / options.ensemble as f64)
        .collect();
    let (train_truth, train_mask) = truth_at_level(dataset, &split.train, options.level);
    let (train_predicted, train_truth) =
        filter_measured(&train_predicted, &train_truth, &train_mask);

    // MAE alongside RMSE: it's what results reported elsewhere for this kind
    // of target (BDE-db's own alfabet model included) tend to use, and RMSE
    // is always at least as large as MAE for the same errors, so the two
    // are not interchangeable when comparing against a published number.
    Ok(if options.classify {
        (
            roc_auc(&predicted, &truth),
            roc_auc(&train_predicted, &train_truth),
            f64::NAN,
            f64::NAN,
        )
    } else {
        (
            root_mean_squared_error(&predicted, &truth),
            root_mean_squared_error(&train_predicted, &train_truth),
            mean_absolute_error(&predicted, &truth),
            mean_absolute_error(&train_predicted, &train_truth),
        )
    })
}

/// Keeps only the rows a label actually measured -- every graph-level row
/// does today, but an atom or bond level dataset labels a fraction of its
/// rows and the rest must not count towards the score.
fn filter_measured(predicted: &[f64], truth: &[f64], mask: &[bool]) -> (Vec<f64>, Vec<f64>) {
    predicted
        .iter()
        .zip(truth)
        .zip(mask)
        .filter(|&(_, &measured)| measured)
        .map(|((&p, &t), _)| (p, t))
        .unzip()
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

fn mean_absolute_error(predicted: &[f64], truth: &[f64]) -> f64 {
    if predicted.is_empty() {
        return f64::NAN;
    }
    let total: f64 = predicted
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).abs())
        .sum();
    total / predicted.len() as f64
}

/// The area under the ROC curve, by the rank-sum identity.
///
/// Equal to the chance that a randomly chosen positive scores above a randomly
/// chosen negative, which is why ties count a half. Returns `NaN` when one of
/// the classes is missing, since the question has no answer then.
fn roc_auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
