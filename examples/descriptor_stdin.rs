//! Reads one SMILES per line, prints a tab-separated row of descriptors.
//! Used to diff against RDKit.
use oxidane::molecule::descriptors::MolecularDescriptors;
use oxidane::prelude::*;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            let _ = writeln!(out);
            continue;
        }
        let row = std::panic::catch_unwind(|| {
            let fragments = MolecularGraph::from_smiles(smiles)?;
            let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
            Ok::<String, oxidane::io::SMILESParseError>(format!(
                "{:.2}\t{}\t{}\t{}\t{}\t{:.4}",
                molecule.topological_polar_surface_area(),
                molecule.heavy_atom_count(),
                molecule.ring_count(),
                molecule.rotatable_bond_count(),
                molecule.total_formal_charge(),
                molecule.fraction_saturated_carbons(),
            ))
        });
        match row {
            Ok(Ok(text)) => {
                let _ = writeln!(out, "{text}");
            }
            _ => {
                let _ = writeln!(out, "ERROR");
            }
        }
    }
}
