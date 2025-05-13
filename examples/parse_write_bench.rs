//! Parse and write throughput, so isotope handling can be changed without
//! silently costing either.
use oxidane::prelude::*;
use std::time::Instant;

fn main() {
    let mut corpus: Vec<String> = std::fs::read_to_string("tests/smiles.txt")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    corpus.extend(
        std::fs::read_to_string("notes/steroids.smi")
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .filter(|line| !line.is_empty()),
    );
    // A few isotope-labelled structures, the case being changed.
    corpus.extend(
        [
            "[13CH3][C@H](C)O",
            "[2H]O[2H]",
            "[13C]1=CC=CC=C1",
            "[15NH2]CC(=O)O",
            "[235U]",
            "[13CH3][13CH2]O",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    corpus.retain(|s| MolecularGraph::from_smiles_single(s).is_ok());
    println!("{} molecules", corpus.len());

    let rounds = 200;

    let start = Instant::now();
    let mut atoms = 0usize;
    for _ in 0..rounds {
        for smiles in &corpus {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            atoms += mol.atomic_numbers().len();
        }
    }
    let parse = start.elapsed();
    println!(
        "parse  {:>9.2?} total   {:>7.1?} per molecule   ({atoms} atoms)",
        parse,
        parse / (rounds * corpus.len()) as u32
    );

    let molecules: Vec<MolecularGraph> = corpus
        .iter()
        .map(|s| MolecularGraph::from_smiles_single(s).unwrap())
        .collect();

    let start = Instant::now();
    let mut written = 0usize;
    for _ in 0..rounds {
        for mol in &molecules {
            written += mol.to_smiles().len();
        }
    }
    let write = start.elapsed();
    println!(
        "write  {:>9.2?} total   {:>7.1?} per molecule   ({written} bytes)",
        write,
        write / (rounds * molecules.len()) as u32
    );
}
