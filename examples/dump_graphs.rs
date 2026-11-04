//! Writes featurised molecules to a file a Python trainer can read.
//!
//! ```text
//! cargo run --release --features ml --example dump_graphs -- \
//!     --data ../molecules-data/CID-SMILES --limit 200000 --out corpus
//! ```
//!
//! Produces `corpus.bin`, `corpus.idx` and `corpus.meta`. The point is that the
//! *features* come from this crate rather than from whatever the Python side
//! would otherwise invent: a prototype trained on different columns is not a
//! prototype of the model that will ship.
//!
//! # The format
//!
//! `corpus.meta` is `key value` lines: the schema version and the two feature
//! widths, so a reader can check it is looking at the layout it expects.
//!
//! `corpus.idx` is `count + 1` little-endian `u64` byte offsets into the data
//! file, so a record can be read without walking the ones before it.
//!
//! `corpus.bin` is one record per molecule, little-endian throughout:
//!
//! ```text
//! u32 atom_count
//! u32 edge_count          -- directed, so twice the bonds
//! u32 smiles_length       -- bytes of canonical SMILES
//! f32 * atom_count * atom_width
//! f32 * edge_count * bond_width
//! u32 * edge_count * 2    -- source, target
//! u8  * smiles_length
//! ```
//!
//! Hydrogens have rows like any other atom, as everywhere else in this crate,
//! so `atom_count` is every atom and not just the heavy ones.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use oxidane::ml::tokenize::tokenize;
use oxidane::molecule::featurize::{
    ATOM_FEATURE_WIDTH, BOND_FEATURE_WIDTH, FEATURE_SCHEMA_VERSION, graph_features,
};
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = "corpus".to_string();
    let mut data = String::new();
    let (mut limit, mut max_tokens, mut column) = (usize::MAX, 120usize, 0usize);

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < arguments.len() {
        let value = || arguments.get(index + 1).cloned().unwrap_or_default();
        let mut consumed = 2;
        match arguments[index].as_str() {
            "--data" => data = value(),
            "--out" => out = value(),
            "--limit" => limit = value().parse().unwrap_or(limit),
            "--max-tokens" => max_tokens = value().parse().unwrap_or(max_tokens),
            "--column" => column = value().parse().unwrap_or(column),
            other => {
                eprintln!("ignoring unknown argument {other}");
                consumed = 1;
            }
        }
        index += consumed;
    }

    let reader: Box<dyn Read> = if data.is_empty() {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&data)?)
    };

    let mut records = BufWriter::new(std::fs::File::create(format!("{out}.bin"))?);
    let mut offsets: Vec<u64> = vec![0];
    let (mut written, mut unreadable, mut too_long) = (0usize, 0usize, 0usize);
    let mut cursor = 0u64;

    for line in BufReader::new(reader).lines() {
        if written >= limit {
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
        // Parsed from the canonical form, so the graph and the string the
        // decoder is asked for are the same molecule written the same way.
        let Ok(molecule) = MolecularGraph::from_smiles_single(&canonical) else {
            unreadable += 1;
            continue;
        };
        let features = graph_features(&molecule);

        let mut bytes: Vec<u8> = Vec::with_capacity(
            12 + features.atom_features.len() * 4
                + features.bond_features.len() * 4
                + features.edge_index.len() * 4
                + canonical.len(),
        );
        bytes.extend((features.atom_count as u32).to_le_bytes());
        bytes.extend((features.edge_count as u32).to_le_bytes());
        bytes.extend((canonical.len() as u32).to_le_bytes());
        for value in &features.atom_features {
            bytes.extend(value.to_le_bytes());
        }
        for value in &features.bond_features {
            bytes.extend(value.to_le_bytes());
        }
        for value in &features.edge_index {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(canonical.as_bytes());

        records.write_all(&bytes)?;
        cursor += bytes.len() as u64;
        offsets.push(cursor);
        written += 1;
        if written % 50_000 == 0 {
            eprintln!("{written} molecules");
        }
    }
    records.flush()?;

    let mut index_file = BufWriter::new(std::fs::File::create(format!("{out}.idx"))?);
    for offset in &offsets {
        index_file.write_all(&offset.to_le_bytes())?;
    }
    index_file.flush()?;

    std::fs::write(
        format!("{out}.meta"),
        format!(
            "schema {FEATURE_SCHEMA_VERSION}\natom_width {ATOM_FEATURE_WIDTH}\n\
             bond_width {BOND_FEATURE_WIDTH}\ncount {written}\n"
        ),
    )?;

    eprintln!(
        "wrote {written} molecules to {out}.bin ({unreadable} unreadable, \
         {too_long} over {max_tokens} tokens)"
    );
    Ok(())
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
