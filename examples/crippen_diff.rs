//! Differential check of the Wildman-Crippen atom typing against RDKit.
//!
//! Reads `smiles<TAB>sorted labels<TAB>logp<TAB>mr` from `tests/crippen_oracle.py`
//! and reports, for every molecule whose type counts differ, which types we
//! give too many or too few of -- which is what points at the rule that is
//! wrong, since the two sides need not agree on atom order.
use oxidane::molecule::crippen::crippen_types;
use oxidane::molecule::descriptors::MolecularDescriptors;
use oxidane::prelude::*;
use std::collections::BTreeMap;
use std::io::BufRead;

fn main() {
    let mut total = 0usize;
    let mut agreeing = 0usize;
    let mut worst_log_p: f64 = 0.0;
    let mut from_aromaticity = 0usize;
    let mut confusion: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut examples: BTreeMap<(String, String), String> = BTreeMap::new();

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        if fields.len() < 4 {
            continue;
        }
        // Every fragment, not the first: a salt's counter-ion contributes to
        // both numbers, and RDKit's do.
        let Ok(fragments) = MolecularGraph::from_smiles(fields[0]) else {
            continue;
        };
        let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        total += 1;

        let mut ours: Vec<String> = crippen_types(&molecule)
            .into_iter()
            .filter(|kind| *kind != oxidane::molecule::crippen::CrippenType::None)
            .map(|kind| kind.label().to_string())
            .collect();
        ours.sort();
        let theirs: Vec<String> = fields[1]
            .split(',')
            .filter(|label| !label.is_empty() && !label.starts_with('?'))
            .map(str::to_string)
            .collect();

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
        let ours_aromatic = (0..molecule.len())
            .filter(|&atom| molecule.is_atom_aromatic(atom))
            .count();
        let theirs_aromatic: usize = fields.get(4).and_then(|it| it.parse().ok()).unwrap_or(0);
        if ours_aromatic != theirs_aromatic {
            from_aromaticity += 1;
        }

        let key = (extra.join("+"), missing.join("+"));
        *confusion.entry(key.clone()).or_default() += 1;
        examples.entry(key).or_insert_with(|| fields[0].to_string());

        let (log_p, _) = molecule.crippen_descriptors();
        let theirs_log_p: f64 = fields[2].parse().unwrap_or(0.0);
        worst_log_p = worst_log_p.max((log_p - theirs_log_p).abs());
    }

    println!("{agreeing}/{total} molecules typed identically");
    println!("worst log P difference {worst_log_p:.4}");
    println!("{from_aromaticity} of the differences are molecules the two perceive differently");
    let mut rows: Vec<_> = confusion.into_iter().collect();
    rows.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for ((extra, missing), count) in rows.into_iter().take(30) {
        let example = examples
            .get(&(extra.clone(), missing.clone()))
            .cloned()
            .unwrap_or_default();
        println!("  {count:6}  ours {extra:22} theirs {missing:22}  {example}");
    }
}
