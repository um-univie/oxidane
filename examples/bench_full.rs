//! Wall-clock cost of `embed_3d`, averaged over seeds.
//!
//! Averaged rather than repeated, because per-trial cost varies a lot with
//! the seed -- a single seed once made atorvastatin look four times quicker
//! than it is. The comparable RDKit configuration is `ETKDGv3` *plus* a UFF
//! optimisation, since `embed_3d` relaxes by default; timing against bare
//! ETKDGv3 compares a relaxed structure with an unrelaxed one.
use oxidane::molecule::embed::{Embed3D, EmbedConfig};
use oxidane::prelude::*;
use std::time::Instant;

const SEEDS: usize = 20;

fn main() {
    println!("{:<14}{:>12}{:>16}", "", "1 trial", "default (5)");
    for (name, smiles) in [
        ("benzene", "c1ccccc1"),
        ("naphthalene", "c1ccc2ccccc2c1"),
        ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
        ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
        (
            "atorvastatin",
            "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        ),
        (
            "peptide",
            "CC(C)CC(NC(=O)C(C)NC(=O)C(CC(C)C)NC(=O)C(C)NC(=O)C(CC(C)C)NC(=O)C(C)N)C(=O)O",
        ),
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("parse");
        let _ = molecule.embed_3d().expect("embed"); // warm the one-time rule compilation
        print!("{name:<14}");
        for attempts in [1, 5] {
            let start = Instant::now();
            for seed in 0..SEEDS {
                let config = EmbedConfig {
                    max_attempts: attempts,
                    seed: (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    ..Default::default()
                };
                molecule.embed_3d_with_config(&config).expect("embed");
            }
            let each = start.elapsed().as_secs_f64() / SEEDS as f64 * 1e3;
            print!("{each:>9.1} ms   ");
        }
        println!();
    }
}
