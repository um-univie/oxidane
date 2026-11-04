//! One SMILES per line on stdin -> comma-joined bits/ids (or `ERR`) per line.
//! arg: `maccs` (default) or `ecfp` (radius-2 circular identifiers).
//! For validating oxidane's fingerprints against RDKit.
use oxidane::molecule::fingerprint::{circular_fingerprint, maccs_keys};
use oxidane::prelude::*;
use std::io::{self, BufRead, Write};

fn main() {
    let kind = std::env::args().nth(1).unwrap_or_else(|| "maccs".into());
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
        let res = std::panic::catch_unwind(|| {
            let frags = MolecularGraph::from_smiles(smiles).ok()?;
            let mol = MolecularGraph::combine(&frags.iter().collect::<Vec<_>>());
            if kind == "ecfp" {
                let fp = circular_fingerprint(&mol, 2);
                let mut ids: Vec<u32> = fp.fingerprint.ids().collect();
                ids.sort_unstable();
                ids.dedup();
                return Some(
                    ids.iter()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                );
            }
            let keys = maccs_keys(&mol);
            Some(
                keys.set_bits()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        });
        match res {
            Ok(Some(bits)) => {
                let _ = writeln!(out, "{bits}");
            }
            _ => {
                let _ = writeln!(out, "ERR");
            }
        }
    }
}
