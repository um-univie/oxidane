//! Writes an encoder's weights and the vectors it produces, as an oracle for a
//! reimplementation of the encoder elsewhere -- a PyTorch mirror, say, to try
//! the architecture on a GPU candle cannot use.
//!
//! ```text
//! cargo run --release --features ml --example encoder_reference -- \
//!     --data molecules.smi --weights reference.safetensors --out reference.latents
//! ```
//!
//! A mirror is only worth anything if it is the same architecture, and
//! comparing loss curves cannot tell: two networks that differ in the
//! reverse-message subtraction, or in how atoms are pooled, both train and
//! both converge to something. Comparing the actual numbers can, so this
//! writes the weights and the latent vectors they give, for the mirror to load
//! and check that it agrees to floating-point noise.
//!
//! The same input file should be handed to `dump_graphs`, so the other side
//! reads the features this crate computed for the same molecules in the same
//! order.

use std::io::{BufRead, BufReader, Read, Write};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use oxidane::ml::batch::GraphBatch;
use oxidane::ml::nn::decoder::DecoderConfig;
use oxidane::ml::nn::molecule_autoencoder::{MoleculeAutoencoder, MoleculeAutoencoderConfig};
use oxidane::ml::nn::{EncoderConfig, GraphTensors, save_prefixed, seed_variables};
use oxidane::ml::tokenize::TokenBatch;
use oxidane::molecule::featurize::graph_features;
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = String::new();
    let mut weights = "reference.safetensors".to_string();
    let mut out = "reference.latents".to_string();
    let mut logits_out = "reference.logits".to_string();
    let (mut hidden, mut depth, mut latent, mut seed) = (32usize, 3usize, 16usize, 7u64);
    // The same default `dump_graphs` uses, and it has to stay the same: the
    // Python side reads that dump, so a molecule one of the two keeps and the
    // other drops puts the two orderings out of step.
    let mut max_tokens = 120usize;

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let value = || arguments.get(index + 1).cloned().unwrap_or_default();
        let mut consumed = 2;
        match arguments[index].as_str() {
            "--data" => data = value(),
            "--weights" => weights = value(),
            "--out" => out = value(),
            "--logits" => logits_out = value(),
            "--hidden" => hidden = value().parse().unwrap_or(hidden),
            "--depth" => depth = value().parse().unwrap_or(depth),
            "--latent" => latent = value().parse().unwrap_or(latent),
            "--seed" => seed = value().parse().unwrap_or(seed),
            "--max-tokens" => max_tokens = value().parse().unwrap_or(max_tokens),
            other => {
                eprintln!("ignoring unknown argument {other}");
                consumed = 1;
            }
        }
        index += consumed;
    }

    let device = Device::Cpu;
    let reader: Box<dyn Read> = if data.is_empty() {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&data)?)
    };

    // Canonicalised and parsed exactly as `dump_graphs` does it, so the two
    // sides are looking at the same graphs in the same order.
    let mut molecules = Vec::new();
    for line in BufReader::new(reader).lines() {
        let line = line?;
        let Some(smiles) = line.split_whitespace().next() else {
            continue;
        };
        let Ok(canonical) = canonicalize_and_sort_smiles(smiles, true) else {
            continue;
        };
        if oxidane::ml::tokenize::tokenize(&canonical).len() > max_tokens {
            continue;
        }
        let Ok(molecule) = MolecularGraph::from_smiles_single(&canonical) else {
            continue;
        };
        molecules.push(graph_features(&molecule));
    }
    if molecules.is_empty() {
        eprintln!("nothing to encode");
        return Ok(());
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MoleculeAutoencoder::new(
        MoleculeAutoencoderConfig {
            encoder: EncoderConfig {
                hidden,
                depth,
                ..Default::default()
            },
            latent,
            decoder: DecoderConfig::default(),
            ..Default::default()
        },
        // A width the fixed probe sequence below stays inside.
        VOCABULARY,
        vb,
    )?;
    seed_variables(&varmap, seed)?;
    save_prefixed(&varmap, "", &weights)?;

    // One molecule per batch, so that the comparison cannot be passed by a
    // model that happens to get batching right and message passing wrong.
    let mut file = std::fs::File::create(&out)?;
    let mut logits_file = std::fs::File::create(&logits_out)?;
    // A fixed sequence rather than the molecule's own tokens: the point here
    // is the recurrence, and a made-up sequence needs no vocabulary to be
    // agreed on before the two sides can be compared.
    let probe = TokenBatch::collate(&[PROBE.to_vec()]);
    for features in &molecules {
        let batch = GraphBatch::collate(std::slice::from_ref(features));
        let graph = GraphTensors::new(&batch, &device)?;
        let latent_row = model.encode(&graph, false)?.mean;
        let row = latent_row.flatten_all()?.to_vec1::<f32>()?;
        let text: Vec<String> = row.iter().map(|value| format!("{value:.7e}")).collect();
        writeln!(file, "{}", text.join(" "))?;

        let logits = model
            .decoder()
            .logits(&latent_row, &probe, false)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let text: Vec<String> = logits.iter().map(|value| format!("{value:.7e}")).collect();
        writeln!(logits_file, "{}", text.join(" "))?;
    }

    eprintln!(
        "wrote {} latents of width {latent} to {out}, {} logits per molecule to {logits_out}, \
         weights to {weights}",
        molecules.len(),
        probe.width * VOCABULARY
    );
    Ok(())
}

/// How many tokens the reference decoder is built for.
const VOCABULARY: usize = 8;

/// The sequence the decoder is asked to score. Every id is a real token rather
/// than one of the reserved four, and the collation puts the start marker in
/// front of it.
const PROBE: [u32; 6] = [4, 5, 6, 7, 4, 5];
