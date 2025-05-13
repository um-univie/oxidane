//! Reads one SMILES per line from stdin, prints the canonical form (or
//! `ERROR: <msg>`) per line. Used as an oracle-comparison driver against RDKit.
use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let smiles = line.trim();
        if smiles.is_empty() {
            let _ = writeln!(out);
            continue;
        }
        match std::panic::catch_unwind(|| canonicalize_and_sort_smiles(smiles, false)) {
            Ok(Ok(canon)) => {
                let _ = writeln!(out, "{}", canon);
            }
            Ok(Err(e)) => {
                let _ = writeln!(out, "ERROR: {:?}", e);
            }
            Err(_) => {
                let _ = writeln!(out, "PANIC");
            }
        }
    }
}
