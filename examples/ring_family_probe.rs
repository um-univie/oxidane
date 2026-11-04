//! Prints the canonical ring family of each SMILES on stdin, and checks it is
//! the same family when the molecule is renumbered.
//!
//! ```text
//! cargo run --release --example ring_family_probe -- [permutations] [max_size] [--dump]
//! ```
//!
//! `--dump` prints the family itself, one ring per `|`-separated field as
//! comma-separated atom indices, so that it can be compared against another
//! toolkit's ring perception on the same input.
use oxidane::graph_algorithms::mol_canonization::reorder_molecule;
use oxidane::molecule::ring_aromaticity::ring_family;
use oxidane::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::io::{self, BufRead};
use std::time::Instant;

fn signature(
    molecule: &MolecularGraph,
    order: Option<&[usize]>,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let mut rings: Vec<Vec<usize>> = ring_family(molecule, max_size)
        .into_iter()
        .map(|ring| {
            let mut atoms: Vec<usize> = match order {
                // `order[new] = old`, so an atom's original index is read off
                // directly and the family is compared in the original numbering.
                Some(order) => ring.atoms.iter().map(|&atom| order[atom]).collect(),
                None => ring.atoms.clone(),
            };
            atoms.sort_unstable();
            atoms
        })
        .collect();
    rings.sort();
    rings
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let dump = arguments.iter().any(|argument| argument == "--dump");
    let numbers: Vec<usize> = arguments
        .iter()
        .filter_map(|argument| argument.parse().ok())
        .collect();
    let permutations = numbers.first().copied().unwrap_or(16);
    let max_size = numbers.get(1).copied().unwrap_or(usize::MAX);

    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }
        let Ok(molecule) = MolecularGraph::from_smiles_single(smiles) else {
            println!("unparsed");
            continue;
        };

        let started = Instant::now();
        let reference = signature(&molecule, None, max_size);
        let elapsed = started.elapsed();

        let mut moved = 0;
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..permutations {
            let mut order: Vec<usize> = (0..molecule.len()).collect();
            order.shuffle(&mut rng);
            let mut permuted = molecule.clone();
            reorder_molecule(&mut permuted, &order);
            // `reorder_molecule(order)` puts old atom `order[i]` at new index
            // `i`, so the inverse is what maps a new index back.
            if signature(&permuted, Some(&order), max_size) != reference {
                moved += 1;
            }
        }

        if dump {
            let rings: Vec<String> = reference
                .iter()
                .map(|ring| {
                    ring.iter()
                        .map(|atom| atom.to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                })
                .collect();
            println!("{}", rings.join("|"));
            continue;
        }

        let mut sizes: Vec<usize> = reference.iter().map(|ring| ring.len()).collect();
        sizes.sort_unstable();
        println!(
            "{} rings, sizes {:?}, {} of {permutations} numberings differ, {:?}",
            reference.len(),
            sizes,
            moved,
            elapsed
        );
    }
}
