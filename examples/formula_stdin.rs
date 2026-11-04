//! Reads one SMILES per line from stdin, prints its molecular formula (or
//! `ERROR`) per line. Used to compare the crate's implicit-hydrogen model
//! against RDKit's over a corpus.
use oxidane::prelude::*;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            let _ = writeln!(out);
            continue;
        }
        let answer = std::panic::catch_unwind(|| match MolecularGraph::from_smiles(smiles) {
            Ok(molecules) => {
                let mut counts: std::collections::BTreeMap<u8, usize> =
                    std::collections::BTreeMap::new();
                for molecule in &molecules {
                    for &atomic_number in molecule.atomic_numbers() {
                        *counts.entry(atomic_number).or_default() += 1;
                    }
                }
                counts
                    .into_iter()
                    .map(|(element, count)| format!("{element}:{count}"))
                    .collect::<Vec<_>>()
                    .join(",")
            }
            Err(_) => "ERROR".to_string(),
        });
        let _ = writeln!(out, "{}", answer.unwrap_or_else(|_| "PANIC".to_string()));
    }
}
