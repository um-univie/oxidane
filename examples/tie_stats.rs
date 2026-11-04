//! How many rings of the family one bond contributes -- the number the old tie
//! cap used to truncate.
//!
//! ```text
//! cargo run --release --example tie_stats -- <corpus> [limit]
//! ```
use oxidane::molecule::ring_aromaticity::{RingFinder, ring_size_limit};
use oxidane::prelude::*;
use rustc_hash::FxHashMap;
use std::io::{BufRead, BufReader};

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let path = &arguments[1];
    let limit: usize = arguments
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let reader = BufReader::new(std::fs::File::open(path).expect("the corpus opens"));

    let mut finder = RingFinder::new();
    let mut worst = (0usize, String::new());
    let mut biggest_family = (0usize, String::new());
    let mut histogram: FxHashMap<usize, u64> = FxHashMap::default();
    let mut records = 0u64;

    for line in reader.lines().take(limit) {
        let Ok(line) = line else { break };
        let smiles = line.split('\t').next_back().unwrap_or("").trim();
        let Ok(molecule) = MolecularGraph::from_smiles_single(smiles) else {
            continue;
        };
        records += 1;
        let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
        let size_limit = match ring_size_limit() {
            0 => usize::MAX,
            limit => limit,
        };
        finder.perceive(&molecule, &cyclic, size_limit);

        let mut through: FxHashMap<(usize, usize), usize> = FxHashMap::default();
        let mut family = 0;
        for ring in finder.rings() {
            family += 1;
            for bond in ring.bonds {
                *through.entry(bond).or_default() += 1;
            }
        }
        let most = through.values().copied().max().unwrap_or(0);
        *histogram.entry(most).or_default() += 1;
        if most > worst.0 {
            worst = (most, smiles.to_string());
        }
        if family > biggest_family.0 {
            biggest_family = (family, smiles.to_string());
        }
    }

    let mut counts: Vec<(usize, u64)> = histogram.into_iter().collect();
    counts.sort_unstable();
    println!("{records} records");
    for (ties, count) in counts {
        println!("  {ties} rings through the busiest bond: {count} records");
    }
    println!("worst {} on {}", worst.0, &worst.1[..worst.1.len().min(70)]);
    println!(
        "largest family {} on {}",
        biggest_family.0,
        &biggest_family.1[..biggest_family.1.len().min(70)]
    );
}
