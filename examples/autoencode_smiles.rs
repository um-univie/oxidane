//! Trains a molecule autoencoder: graph in, one vector, SMILES back out.
//!
//! ```text
//! cargo run --release --features ml --example autoencode_smiles -- \
//!     --data ../molecules-data/CID-SMILES --limit 200000 --epochs 20 --variational
//! ```
//!
//! Reads one molecule per line. PubChem's `CID-SMILES` is `<id><TAB><SMILES>`
//! and a `.smi` file is usually the SMILES first, so the column is worked out
//! per line: two or more fields with a number in front means the SMILES is the
//! second, otherwise it is the first. `--column` overrides that.
//!
//! Every molecule is canonicalised before it is used, because the decoder is
//! being asked to reproduce a string and only one spelling of a molecule is
//! learnable from a vector. The same canonical form is what the round-trip
//! count at the end of each epoch compares against.
//!
//! What the numbers mean:
//!
//! * `loss` -- cross entropy per token, on the training molecules.
//! * `kl` -- how far the encoded distributions are from a standard normal.
//!   Watch this on a variational run: if it falls to near zero while the loss
//!   keeps improving, the decoder has stopped reading the latent and the model
//!   has become a language model. Lengthen `--kl-warmup` or lower `--kl`.
//! * `valid` -- the share of held-out molecules whose decoded string parses.
//! * `exact` -- the share that come back as the molecule that went in. This is
//!   the number the model exists to make large; the loss is only its proxy.
//!
//! Features are used unscaled, unlike `pretrain_encoder`. The scaling is fitted
//! from a corpus and is not written out with the weights, and a model whose
//! encoder silently depends on statistics nobody saved is worse than a model
//! that converges a little slower.

use std::io::{BufRead, BufReader, Read};

use candle_core::{DType, Device};
use candle_nn::{Optimizer, VarBuilder, VarMap};

use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use oxidane::io::ToSMILES;
use oxidane::ml::batch::GraphBatch;
use oxidane::ml::nn::decoder::DecoderConfig;
use oxidane::ml::nn::molecule_autoencoder::{MoleculeAutoencoder, MoleculeAutoencoderConfig};
use oxidane::ml::nn::{EncoderConfig, EncoderKind, GraphTensors, save_prefixed, seed_variables};
use oxidane::ml::tokenize::{SmilesVocabulary, TokenBatch, tokenize};
use oxidane::molecule::featurize::graph_features;
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut out, mut vocabulary_out) = (
        "autoencoder.safetensors".to_string(),
        "autoencoder.vocabulary".to_string(),
    );
    let mut data = String::new();
    let (mut hidden, mut depth, mut latent) = (300usize, 3usize, 128usize);
    let (mut decoder_hidden, mut embedding) = (512usize, 64usize);
    let (mut epochs, mut batch_size, mut learning_rate) = (10usize, 64usize, 1e-3f64);
    let (mut kl, mut kl_warmup) = (1.0f64, 5usize);
    let (mut limit, mut max_tokens, mut minimum_count) = (100_000usize, 120usize, 5usize);
    let (mut holdout, mut seed, mut column) = (1000usize, 1u64, 0usize);
    let (mut variational, mut gine, mut samples) = (false, false, 0usize);

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let value = || arguments.get(index + 1).cloned().unwrap_or_default();
        let mut consumed = 2;
        match arguments[index].as_str() {
            "--data" => data = value(),
            "--out" => out = value(),
            "--vocabulary" => vocabulary_out = value(),
            "--hidden" => hidden = value().parse().unwrap_or(hidden),
            "--depth" => depth = value().parse().unwrap_or(depth),
            "--latent" => latent = value().parse().unwrap_or(latent),
            "--decoder-hidden" => decoder_hidden = value().parse().unwrap_or(decoder_hidden),
            "--embedding" => embedding = value().parse().unwrap_or(embedding),
            "--epochs" => epochs = value().parse().unwrap_or(epochs),
            "--batch" => batch_size = value().parse().unwrap_or(batch_size),
            "--lr" => learning_rate = value().parse().unwrap_or(learning_rate),
            "--kl" => kl = value().parse().unwrap_or(kl),
            "--kl-warmup" => kl_warmup = value().parse().unwrap_or(kl_warmup),
            "--limit" => limit = value().parse().unwrap_or(limit),
            "--max-tokens" => max_tokens = value().parse().unwrap_or(max_tokens),
            "--min-count" => minimum_count = value().parse().unwrap_or(minimum_count),
            "--holdout" => holdout = value().parse().unwrap_or(holdout),
            "--seed" => seed = value().parse().unwrap_or(seed),
            "--column" => column = value().parse().unwrap_or(column),
            "--samples" => samples = value().parse().unwrap_or(samples),
            "--variational" => {
                variational = true;
                consumed = 1;
            }
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

    // ---- read, canonicalise, and keep only the string -----------------------
    //
    // The canonical SMILES is enough to rebuild the graph, and a corpus this
    // size does not fit in memory as feature matrices. Featurising happens per
    // batch instead, which costs a parse and saves several gigabytes.
    let reader: Box<dyn Read> = if data.is_empty() {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&data)?)
    };
    let mut molecules: Vec<String> = Vec::new();
    let (mut unreadable, mut too_long) = (0usize, 0usize);
    for line in BufReader::new(reader).lines() {
        if molecules.len() >= limit {
            break;
        }
        let line = line?;
        let Some(smiles) = smiles_of(&line, column) else {
            continue;
        };
        let Ok(canonical) = canonicalize_and_sort_smiles(smiles, true) else {
            unreadable += 1;
            continue;
        };
        if tokenize(&canonical).len() > max_tokens {
            too_long += 1;
            continue;
        }
        molecules.push(canonical);
    }
    eprintln!(
        "{} molecules, {unreadable} unreadable, {too_long} over {max_tokens} tokens",
        molecules.len()
    );
    if molecules.len() <= holdout {
        eprintln!("not enough molecules to train on and hold out {holdout}");
        return Ok(());
    }

    // ---- vocabulary ---------------------------------------------------------
    let vocabulary = SmilesVocabulary::from_corpus(&molecules, minimum_count);
    let before = molecules.len();
    molecules.retain(|smiles| vocabulary.encode_known(smiles).is_some());
    eprintln!(
        "{} tokens; {} molecules dropped for using a rarer one",
        vocabulary.len(),
        before - molecules.len()
    );

    // Shuffled before the split, so that the held-out molecules are not
    // whichever ones happened to sit at the end of the file -- in PubChem's
    // order that would be a chemically distinct slice, not a sample.
    let mut order: Vec<usize> = (0..molecules.len()).collect();
    oxidane::ml::shuffled(&mut order, seed);
    let mut shuffled: Vec<String> = Vec::with_capacity(molecules.len());
    for index in order {
        shuffled.push(std::mem::take(&mut molecules[index]));
    }
    let (train, evaluate) = shuffled.split_at(shuffled.len() - holdout);
    let (train, evaluate) = (train.to_vec(), evaluate.to_vec());
    eprintln!("{} to train on, {} held out", train.len(), evaluate.len());

    // ---- build --------------------------------------------------------------
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config = MoleculeAutoencoderConfig {
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
        latent,
        decoder: DecoderConfig {
            embedding,
            hidden: decoder_hidden,
            // Room for the longest molecule that was kept, plus the end marker.
            max_length: max_tokens + 1,
            dropout: 0.0,
        },
        variational,
        ..Default::default()
    };
    let model = MoleculeAutoencoder::new(config, vocabulary.len(), vb)?;
    seed_variables(&varmap, seed)?;

    let mut optimizer = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        },
    )?;

    // ---- train --------------------------------------------------------------
    let lengths: Vec<usize> = train.iter().map(|smiles| tokenize(smiles).len()).collect();
    let mut order: Vec<usize> = (0..train.len()).collect();
    for epoch in 0..epochs {
        oxidane::ml::shuffled(&mut order, seed.wrapping_add(epoch as u64 + 1));
        bucket_by_length(&mut order, &lengths, batch_size);
        // Zero at first, so that the decoder learns to read the latent before
        // it is charged for carrying one.
        let kl_weight = if kl_warmup == 0 {
            kl
        } else {
            kl * (epoch as f64 / kl_warmup as f64).min(1.0)
        };

        let (mut total, mut divergence, mut steps) = (0.0f32, 0.0f32, 0usize);
        for chunk in order.chunks(batch_size) {
            let smiles: Vec<&str> = chunk.iter().map(|&row| train[row].as_str()).collect();
            let Some((graph, tokens)) = prepare(&smiles, &vocabulary, &device)? else {
                continue;
            };

            let loss = model.loss(&graph, &tokens, true)?;
            let step = loss.total(kl_weight)?;
            optimizer.backward_step(&step)?;

            total += loss.reconstruction.to_scalar::<f32>()?;
            divergence += loss.kl.to_scalar::<f32>()?;
            steps += 1;
        }

        let (valid, exact) = round_trip(&model, &vocabulary, &evaluate, &device, batch_size)?;
        eprintln!(
            "epoch {epoch:3}  loss {:.4}  kl {:.4} (weight {kl_weight:.2})  \
             valid {:.3}  exact {:.3}",
            total / steps.max(1) as f32,
            divergence / steps.max(1) as f32,
            valid as f64 / evaluate.len().max(1) as f64,
            exact as f64 / evaluate.len().max(1) as f64,
        );
    }

    // ---- what came out ------------------------------------------------------
    if samples > 0 {
        eprintln!("--- {samples} draws from the prior ---");
        let prior = model.sample_prior(samples, &device)?;
        for (text, molecule) in model
            .generate(&prior, &vocabulary, 1.0, seed)?
            .into_iter()
            .zip(model.decode(&prior, &vocabulary, 1.0, seed)?)
        {
            match molecule {
                Some(_) => println!("{text}"),
                None => eprintln!("(not a molecule) {text}"),
            }
        }
    }

    save_prefixed(&varmap, "", &out)?;
    std::fs::write(&vocabulary_out, vocabulary.to_lines())?;
    eprintln!("wrote {out} and {vocabulary_out}");
    Ok(())
}

/// Sorts each window of the epoch order by molecule length.
///
/// A batch is padded to its longest member and the decoder runs a step for
/// every column, so a batch holding one 90-token molecule and sixty-three
/// 20-token ones does seventy steps of arithmetic on padding that is masked
/// out of the loss. Sorting the whole corpus by length would fix that and
/// destroy the shuffle -- every batch would then hold near-identical molecules
/// for the whole run. Sorting inside a window of a few dozen batches keeps the
/// shuffle and takes most of the saving.
fn bucket_by_length(order: &mut [usize], lengths: &[usize], batch_size: usize) {
    let window = batch_size * 32;
    for chunk in order.chunks_mut(window.max(1)) {
        chunk.sort_unstable_by_key(|&row| lengths[row]);
    }
}

/// The SMILES field of one line, or `None` if there is nothing usable on it.
///
/// `column` is one-based and overrides the guess. The guess: a line whose
/// first field is a number is an identifier followed by the structure, which
/// is what PubChem's `CID-SMILES` is; anything else leads with the structure.
fn smiles_of(line: &str, column: usize) -> Option<&str> {
    let mut fields = line.split_whitespace();
    if column > 0 {
        return fields.nth(column - 1).filter(|field| !field.is_empty());
    }
    let first = fields.next()?;
    match fields.next() {
        Some(second) if first.bytes().all(|byte| byte.is_ascii_digit()) => Some(second),
        _ => Some(first),
    }
}

/// One batch, as the two halves the model reads: the graphs and the strings.
///
/// `None` when nothing in the chunk parsed, which a caller should skip rather
/// than take a gradient of.
fn prepare(
    smiles: &[&str],
    vocabulary: &SmilesVocabulary,
    device: &Device,
) -> Result<Option<(GraphTensors, TokenBatch)>, Box<dyn std::error::Error>> {
    let mut features = Vec::with_capacity(smiles.len());
    let mut sequences = Vec::with_capacity(smiles.len());
    for &text in smiles {
        let Ok(molecule) = MolecularGraph::from_smiles_single(text) else {
            continue;
        };
        let Some(ids) = vocabulary.encode_known(text) else {
            continue;
        };
        features.push(graph_features(&molecule));
        sequences.push(ids);
    }
    if features.is_empty() {
        return Ok(None);
    }
    let batch = GraphBatch::collate(&features);
    Ok(Some((
        GraphTensors::new(&batch, device)?,
        TokenBatch::collate(&sequences),
    )))
}

/// How many held-out molecules decode to something, and how many decode to
/// themselves.
///
/// Greedy, so this is the model's best single answer rather than a draw from
/// it, and it is the same answer every time -- two runs that differ here
/// differ in the model.
fn round_trip(
    model: &MoleculeAutoencoder,
    vocabulary: &SmilesVocabulary,
    molecules: &[String],
    device: &Device,
    batch_size: usize,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let (mut valid, mut exact) = (0usize, 0usize);
    for chunk in molecules.chunks(batch_size) {
        let smiles: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let Some((graph, _)) = prepare(&smiles, vocabulary, device)? else {
            continue;
        };
        let latent = model.encode(&graph, false)?;
        for (written, expected) in model
            .decode(&latent.mean, vocabulary, 0.0, 0)?
            .into_iter()
            .zip(&smiles)
        {
            let Some(molecule) = written else { continue };
            valid += 1;
            // Compared as canonical strings on both sides: the decoder's
            // output is a spelling, and two spellings of one molecule are the
            // same answer.
            if canonicalize_and_sort_smiles(&molecule.to_smiles(), true)
                .is_ok_and(|written| written == *expected)
            {
                exact += 1;
            }
        }
    }
    Ok((valid, exact))
}
