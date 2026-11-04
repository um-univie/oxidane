//! Writes this crate's scaffold splits out as CSV, so another toolkit can be
//! measured on exactly the same ones.
//!
//! ```text
//! cargo run --release --example dump_splits -- <name> <dir> [seeds] < data.tsv
//! ```
//!
//! Produces `<dir>/<name>_<seed>_{train,val,test}.csv`, each with a `smiles`
//! and a `target` column. Comparing two toolkits that each computed their own
//! scaffold split compares the splits as much as the models -- Bemis--Murcko
//! agrees between implementations but the placement of groups into sides need
//! not, and on a thousand-molecule set that is worth more than the difference
//! being looked for.

use std::io::{BufRead, Write};

use oxidane::ml::split::{Fractions, Split, scaffold_split_balanced};
use oxidane::molecule::scaffold::scaffold_key;
use oxidane::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().unwrap_or_else(|| "dataset".into());
    let directory = arguments.next().unwrap_or_else(|| ".".into());
    let seeds: u64 = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    // The molecules are kept exactly as the harness keeps them: a row it could
    // not read is dropped, so the indices line up with the ones it splits.
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let mut fields = line.trim().split('\t');
        let (Some(smiles), Some(value)) = (fields.next(), fields.next()) else {
            continue;
        };
        let (Ok(fragments), Ok(_)) = (
            MolecularGraph::from_smiles(smiles),
            value.trim().parse::<f64>(),
        ) else {
            continue;
        };
        if fragments.is_empty() {
            continue;
        }
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        keys.push(scaffold_key(&molecule));
        rows.push((smiles.to_string(), value.trim().to_string()));
    }
    eprintln!("{} molecules", rows.len());

    for seed in 0..seeds {
        let split = scaffold_split_balanced(&keys, Fractions::default(), seed);
        write(
            &rows,
            &split.train,
            &format!("{directory}/{name}_{seed}_train.csv"),
        )?;
        write(
            &rows,
            &split.valid,
            &format!("{directory}/{name}_{seed}_val.csv"),
        )?;
        write(
            &rows,
            &split.test,
            &format!("{directory}/{name}_{seed}_test.csv"),
        )?;
        eprintln!(
            "seed {seed}: {} / {} / {}",
            split.train.len(),
            split.valid.len(),
            split.test.len()
        );
        let _: &Split = &split;
    }
    Ok(())
}

fn write(
    rows: &[(String, String)],
    indices: &[usize],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "smiles,target")?;
    for &index in indices {
        let (smiles, value) = &rows[index];
        // A SMILES can hold a comma inside brackets, so quote the field rather
        // than hope.
        writeln!(out, "\"{smiles}\",{value}")?;
    }
    Ok(())
}
