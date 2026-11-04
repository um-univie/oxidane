//! Canonizes the sigma skeleton -- the molecule with every bond order made
//! single -- under several numberings. Isolates graph labelling from
//! aromatic perception: the skeleton carries no bond orders for perception to
//! read, so anything unstable here is the labelling itself.
//!
//! ```text
//! cargo run --release --example canon_skeleton_probe -- <smiles> [--seeds N]
//! ```
use oxidane::graph_algorithms::mol_canonization::{canonize_mol, reorder_molecule};
use oxidane::molecule::bond::BondOrder;
use oxidane::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::time::Instant;

fn flatten(molecule: &MolecularGraph) -> MolecularGraph {
    let mut skeleton = molecule.clone();
    for bonds in skeleton.atom_bonds_mut() {
        for bond in bonds.iter_mut() {
            bond.bond_order = BondOrder::Single;
        }
    }
    skeleton
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let smiles = arguments
        .next()
        .expect("usage: canon_skeleton_probe <smiles> [--seeds N]");
    let mut seeds = 16u64;
    while let Some(option) = arguments.next() {
        let value = arguments.next().expect("each option requires a value");
        match option.as_str() {
            "--seeds" => seeds = value.parse().expect("--seeds takes a number"),
            _ => panic!("unknown option: {option}"),
        }
    }
    let molecule = MolecularGraph::from_smiles_single(&smiles).expect("the input parses");
    let started = Instant::now();
    let base = canonize_mol(&flatten(&molecule)).to_smiles();
    println!(
        "skeleton base {:.3}s {base}",
        started.elapsed().as_secs_f64()
    );
    let mut differing = 0;
    for seed in 0..seeds {
        let mut labeling: Vec<usize> = (0..molecule.len()).collect();
        labeling.shuffle(&mut StdRng::seed_from_u64(seed));
        let mut permuted = molecule.clone();
        reorder_molecule(&mut permuted, &labeling);
        let form = canonize_mol(&flatten(&permuted)).to_smiles();
        if form != base {
            differing += 1;
            println!("seed {seed} differs: {form}");
        }
    }
    println!("{differing}/{seeds} numberings disagree on the skeleton");
}
