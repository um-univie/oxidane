//! How long the canonical ring family takes, against the parse it follows.
//!
//! ```text
//! cargo run --release --example ring_bench -- <corpus> [limit] [repeats]
//! ```
//! The corpus is `<cid>\t<smiles>` per line, as `pubchem_sweep` reads. Molecules
//! are parsed once and timed separately, so what is reported for the family is
//! the family alone.
use oxidane::molecule::ring_aromaticity::RingFinder;
use oxidane::prelude::*;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let path = &arguments[1];
    let limit: usize = arguments
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let repeats: usize = arguments.get(3).and_then(|v| v.parse().ok()).unwrap_or(3);

    let reader = BufReader::new(std::fs::File::open(path).expect("the corpus opens"));
    let corpus: Vec<String> = reader
        .lines()
        .take(limit)
        .filter_map(|line| {
            let line = line.ok()?;
            Some(line.split('\t').nth(1)?.to_string())
        })
        .collect();

    let mut parsing = f64::MAX;
    let mut molecules = Vec::new();
    for _ in 0..repeats {
        let started = Instant::now();
        molecules = corpus
            .iter()
            .filter_map(|smiles| MolecularGraph::from_smiles_single(smiles).ok())
            .collect();
        parsing = parsing.min(started.elapsed().as_secs_f64());
    }

    // The ring bonds perception hands the finder, found once per molecule as
    // `aromatize_with` finds them.
    let rings: Vec<_> = molecules
        .iter()
        .map(|molecule| molecule.find_cyclic_bonds_and_atoms().1)
        .collect();

    let mut finding = f64::MAX;
    let mut found = 0usize;
    for _ in 0..repeats {
        let mut finder = RingFinder::new();
        let started = Instant::now();
        let mut count = 0;
        for (molecule, cyclic) in molecules.iter().zip(&rings) {
            finder.perceive(molecule, cyclic, 24);
            count += finder.rings().count();
        }
        finding = finding.min(started.elapsed().as_secs_f64());
        found = black_box(count);
    }

    let records = molecules.len() as f64;
    println!(
        "{} molecules, {found} rings\n  parse        {:9.0} rec/s ({:5.2} us)\n  ring family  {:9.0} rec/s ({:5.2} us)",
        molecules.len(),
        records / parsing,
        parsing * 1e6 / records,
        records / finding,
        finding * 1e6 / records,
    );
}
