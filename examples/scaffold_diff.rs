//! Bemis--Murcko scaffolds for SMILES on stdin, one per line.
//!
//! Prints `input<TAB>scaffold<TAB>framework`, for comparison against
//! `tests/scaffold_oracle.py`.

use oxidane::molecule::scaffold::{framework_key, scaffold_key};
use oxidane::prelude::*;
use std::io::{BufRead, Write};

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let Ok(smiles) = line else { break };
        let smiles = smiles.trim();
        if smiles.is_empty() {
            continue;
        }
        match MolecularGraph::from_smiles(smiles) {
            Ok(fragments) if !fragments.is_empty() => {
                let refs: Vec<&MolecularGraph> = fragments.iter().collect();
                let molecule = MolecularGraph::combine(&refs);
                let _ = writeln!(
                    out,
                    "{}\t{}\t{}",
                    smiles,
                    scaffold_key(&molecule),
                    framework_key(&molecule)
                );
            }
            _ => {
                let _ = writeln!(out, "{}\tPARSE_ERROR\tPARSE_ERROR", smiles);
            }
        }
    }
}
