//! Parses (which perceives) and writes the molecule back, without canonizing.
use oxidane::prelude::*;
use std::io::{BufRead, Write};
use std::time::Instant;

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }
        let started = Instant::now();
        match MolecularGraph::from_smiles_single(smiles) {
            Ok(molecule) => {
                let written = molecule.to_smiles();
                let _ = writeln!(out, "{:?}\t{}", started.elapsed(), written);
            }
            Err(error) => {
                let _ = writeln!(out, "{:?}\tERROR {error:?}", started.elapsed());
            }
        }
    }
}
