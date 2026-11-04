//! Times `embed_3d` on each SMILES read from stdin, one per line, printing
//! `smiles<TAB>atoms<TAB>milliseconds` (or `ERROR` in place of the time).
//!
//! The timing is taken inside the process and around the embedding alone, so
//! that a comparison against another toolkit is a comparison of embedders
//! rather than of process startup: `tests/embed_speed.py` calls this once
//! for a whole corpus and times RDKit in its own process the same way.
//!
//! `embed_timed [trials] [nosearch]` sets `max_attempts` and can switch the
//! torsion search off, which is how its share of the time is measured.
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::embed::{Embed3D, EmbedConfig};
use oxidane::prelude::*;
use std::io::{BufRead, Write};
use std::time::Instant;

fn main() {
    let trials: Option<usize> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    let search = std::env::args().nth(2).as_deref() != Some("nosearch");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }
        let config = EmbedConfig {
            max_attempts: trials.unwrap_or(EmbedConfig::default().max_attempts),
            torsion_search: search,
            combined_refinement: !search,
            ..Default::default()
        };
        let start = Instant::now();
        let embedded = MolecularGraph::from_smiles_single(smiles)
            .ok()
            .and_then(|molecule| molecule.embed_3d_with_config(&config).ok());
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        match embedded {
            Some(molecule) => {
                let _ = writeln!(out, "{smiles}\t{}\t{elapsed:.3}", molecule.len());
            }
            None => {
                let _ = writeln!(out, "{smiles}\t0\tERROR");
            }
        }
        let _ = out.flush();
    }
}
