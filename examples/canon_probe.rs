//! Times canonization of one molecule under several numberings, printing each
//! result as it lands.
//!
//! ```text
//! cargo run --release --example canon_probe -- <smiles> [--seeds N]
//! ```
use oxidane::graph_algorithms::mol_canonization::{canonize_mol, reorder_molecule};
use oxidane::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::io::Write;
use std::time::Instant;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let smiles = arguments
        .next()
        .expect("usage: canon_probe <smiles> [--seeds N]");
    let mut seeds = 8u64;
    while let Some(option) = arguments.next() {
        let value = arguments.next().expect("each option requires a value");
        match option.as_str() {
            "--seeds" => seeds = value.parse().expect("--seeds takes a number"),
            _ => panic!("unknown option: {option}"),
        }
    }
    let molecule = MolecularGraph::from_smiles_single(&smiles).expect("the input parses");
    let heavy = molecule.atomic_numbers().iter().filter(|&&z| z > 1).count();
    println!("{} atoms, {heavy} heavy", molecule.len());

    let started = Instant::now();
    let base = canonize_mol(&molecule).to_smiles();
    println!("base  {:8.3}s  {base}", started.elapsed().as_secs_f64());
    let _ = std::io::stdout().flush();

    for seed in 0..seeds {
        let mut labeling: Vec<usize> = (0..molecule.len()).collect();
        labeling.shuffle(&mut StdRng::seed_from_u64(seed));
        let mut permuted = molecule.clone();
        reorder_molecule(&mut permuted, &labeling);
        let started = Instant::now();
        let form = canonize_mol(&permuted).to_smiles();
        println!(
            "seed {seed:2} {:8.3}s  {}",
            started.elapsed().as_secs_f64(),
            if form == base {
                "same".to_string()
            } else {
                form
            }
        );
        let _ = std::io::stdout().flush();
    }
}
