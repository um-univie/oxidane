//! Differential SMARTS check against RDKit.
//!
//! Reads `smarts<TAB>target<TAB>count` (see `notes/smarts_oracle.py`, which
//! makes hydrogens explicit so both sides count the same graph) and reports
//! every disagreement.
use oxidane::io::find_cxsmarts_matches;
use oxidane::prelude::*;
use std::io::BufRead;

fn main() {
    let (mut total, mut agree, mut parse_error) = (0, 0, 0);
    let mut rows: Vec<String> = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let (pattern, target, want): (&str, &str, usize) =
            (parts[0], parts[1], parts[2].parse().unwrap());
        let Ok(mol) = MolecularGraph::from_smiles_single(target) else { continue };
        total += 1;
        match find_cxsmarts_matches(&mol, pattern) {
            Ok(hits) if hits.len() == want => agree += 1,
            Ok(hits) => {
                if rows.len() < 30 {
                    rows.push(format!("  {pattern:12} in {target:14} ours {:5}  rdkit {want:5}", hits.len()));
                }
            }
            Err(e) => {
                parse_error += 1;
                if rows.len() < 30 {
                    rows.push(format!("  {pattern:12} in {target:14} ERR {e:?}"));
                }
            }
        }
    }
    println!("{agree}/{total} agree with RDKit   ({parse_error} pattern errors)");
    for row in rows {
        println!("{row}");
    }
}
