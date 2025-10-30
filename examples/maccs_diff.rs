//! Differential check of the MACCS keys against RDKit.
//!
//! Reads `smiles<TAB>set bits` from `tests/maccs_oracle.py` and reports which
//! keys we set that RDKit does not, and the other way round.
use oxidane::molecule::fingerprint::{MACCS_KEY_COUNT, maccs_keys};
use oxidane::prelude::*;
use std::io::BufRead;

fn main() {
    let mut total = 0usize;
    let mut agreeing = 0usize;
    let mut extra = vec![0usize; MACCS_KEY_COUNT];
    let mut missing = vec![0usize; MACCS_KEY_COUNT];
    let mut examples: Vec<String> = vec![String::new(); MACCS_KEY_COUNT];
    let mut from_aromaticity = 0usize;

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        if fields.is_empty() {
            continue;
        }
        let Ok(fragments) = MolecularGraph::from_smiles(fields[0]) else {
            continue;
        };
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        total += 1;

        let ours = maccs_keys(&molecule);
        let mut theirs = [false; MACCS_KEY_COUNT];
        for bit in fields.get(1).unwrap_or(&"").split(',') {
            if let Ok(bit) = bit.parse::<usize>()
                && bit < MACCS_KEY_COUNT
            {
                theirs[bit] = true;
            }
        }

        let mut same = true;
        for key in 0..MACCS_KEY_COUNT {
            match (ours.get(key), theirs[key]) {
                (true, false) => {
                    extra[key] += 1;
                    same = false;
                    if examples[key].is_empty() {
                        examples[key] = fields[0].to_string();
                    }
                }
                (false, true) => {
                    missing[key] += 1;
                    same = false;
                    if examples[key].is_empty() {
                        examples[key] = fields[0].to_string();
                    }
                }
                _ => (),
            }
        }
        if same {
            agreeing += 1;
        } else {
            let ours_aromatic = (0..molecule.len())
                .filter(|&atom| molecule.is_atom_aromatic(atom))
                .count();
            let theirs_aromatic: usize = fields.get(2).and_then(|it| it.parse().ok()).unwrap_or(0);
            if ours_aromatic != theirs_aromatic {
                from_aromaticity += 1;
            }
        }
    }

    println!("{agreeing}/{total} molecules with identical keys");
    println!("{from_aromaticity} of the differences are molecules the two perceive differently");
    let mut rows: Vec<(usize, usize, usize)> = (0..MACCS_KEY_COUNT)
        .map(|key| (extra[key] + missing[key], key, 0))
        .filter(|&(count, _, _)| count > 0)
        .collect();
    rows.sort_by_key(|&(count, key, _)| (std::cmp::Reverse(count), key));
    for (_, key, _) in rows.into_iter().take(25) {
        println!(
            "  key {key:3}  ours-only {:6}  theirs-only {:6}   {}",
            extra[key], missing[key], examples[key]
        );
    }
}
