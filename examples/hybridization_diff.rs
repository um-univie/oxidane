//! Differential check of the hybridization against RDKit.
//! Reads `smiles<TAB>sorted labels` from `tests/hybridization_oracle.py`.
use oxidane::molecule::featurize::{Hybridization, hybridization};
use oxidane::prelude::*;
use std::collections::BTreeMap;
use std::io::BufRead;

fn main() {
    let (mut total, mut agreeing, mut atoms, mut wrong) = (0usize, 0usize, 0usize, 0usize);
    let mut confusion: BTreeMap<(String, String), (usize, String)> = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        if fields.len() < 2 {
            continue;
        }
        let Ok(fragments) = MolecularGraph::from_smiles(fields[0]) else {
            continue;
        };
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        total += 1;

        let mut ours: Vec<String> = (0..molecule.len())
            .filter(|&atom| molecule.get_atomic_number(atom) != 1)
            .map(|atom| {
                match hybridization(&molecule, atom) {
                    Hybridization::S => "S",
                    Hybridization::Sp => "SP",
                    Hybridization::Sp2 => "SP2",
                    Hybridization::Sp3 => "SP3",
                    Hybridization::Sp3d => "SP3D",
                    Hybridization::Sp3d2 => "SP3D2",
                    Hybridization::Unspecified => "?",
                }
                .to_string()
            })
            .collect();
        ours.sort();
        let theirs: Vec<String> = fields[1]
            .split(',')
            .filter(|label| !label.is_empty())
            .map(str::to_string)
            .collect();
        atoms += theirs.len();
        if ours == theirs {
            agreeing += 1;
            continue;
        }
        let mut counts: BTreeMap<&str, i32> = BTreeMap::new();
        for label in &ours {
            *counts.entry(label.as_str()).or_default() += 1;
        }
        for label in &theirs {
            *counts.entry(label.as_str()).or_default() -= 1;
        }
        wrong += counts.values().filter(|&&count| count > 0).sum::<i32>() as usize;
        let extra: Vec<&str> = counts
            .iter()
            .filter(|&(_, &count)| count > 0)
            .map(|(&label, _)| label)
            .collect();
        let missing: Vec<&str> = counts
            .iter()
            .filter(|&(_, &count)| count < 0)
            .map(|(&label, _)| label)
            .collect();
        let entry = confusion
            .entry((extra.join("+"), missing.join("+")))
            .or_insert((0, fields[0].to_string()));
        entry.0 += 1;
    }
    println!("{agreeing}/{total} molecules, {wrong} of {atoms} atoms differ");
    let mut rows: Vec<_> = confusion.into_iter().collect();
    rows.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
    for ((extra, missing), (count, example)) in rows.into_iter().take(12) {
        println!("  {count:6}  ours {extra:14} theirs {missing:14}  {example}");
    }
}
