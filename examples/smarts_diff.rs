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
    let mut by_pattern: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let (pattern, target, want): (&str, &str, usize) =
            (parts[0], parts[1], parts[2].parse().unwrap());
        // Every fragment, not the first: a salt's counter-ion is part of the
        // graph RDKit counts against.
        let Ok(fragments) = MolecularGraph::from_smiles(target) else { continue };
        let mol = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        total += 1;
        match find_cxsmarts_matches(&mol, pattern) {
            Ok(hits) if hits.len() == want => agree += 1,
            Ok(hits) => {
                *by_pattern.entry(pattern.to_string()).or_default() += 1;
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
    let mut tally: Vec<_> = by_pattern.into_iter().collect();
    tally.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    for (pattern, count) in tally {
        println!("  {count:6} disagreements on {pattern}");
    }
    for row in rows {
        println!("{row}");
    }
}
