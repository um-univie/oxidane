//! Differential structure-substructure check against RDKit.
//! Reads `pattern<TAB>target<TAB>count` from `notes/submol_oracle.py`.
use molecules::prelude::*;
use std::io::BufRead;

fn main() {
    let (mut total, mut agree) = (0, 0);
    let mut rows: Vec<String> = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let want: usize = parts[2].parse().unwrap();
        let (Ok(pattern), Ok(target)) = (
            MolecularGraph::from_smiles_single_raw(parts[0]),
            MolecularGraph::from_smiles_single(parts[1]),
        ) else {
            continue;
        };
        total += 1;
        let got = target.match_submolecule(&pattern).map(|m| m.len()).unwrap_or(0);
        if got == want {
            agree += 1;
        } else if rows.len() < 25 {
            rows.push(format!("  {:14} in {:16} ours {got:5}  rdkit {want:5}", parts[0], parts[1]));
        }
    }
    println!("{agree}/{total} agree with RDKit");
    for row in rows {
        println!("{row}");
    }
}
