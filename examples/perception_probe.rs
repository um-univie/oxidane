//! Whether perception gives the same aromatic bonds however the atoms are
//! numbered.
//!
//! ```text
//! cargo run --release --example perception_probe -- [permutations] < smiles
//! ```
use oxidane::graph_algorithms::mol_canonization::reorder_molecule;
use oxidane::molecule::aromaticity::{MoleculeAromaticity, default_aromaticity_model};
use oxidane::molecule::bond::BondOrder;
use oxidane::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::io::{BufRead, Write};

/// The aromatic bonds, named by the atoms they joined before any renumbering.
fn aromatic_bonds(molecule: &MolecularGraph, order: Option<&[usize]>) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for atom in 0..molecule.len() {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            if bond.bond_order() != BondOrder::Aromatic || bond.target() < atom {
                continue;
            }
            let (first, second) = match order {
                Some(order) => (order[atom], order[bond.target()]),
                None => (atom, bond.target()),
            };
            bonds.push((first.min(second), first.max(second)));
        }
    }
    bonds.sort_unstable();
    bonds
}

fn main() {
    let permutations: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }
        let Ok(molecule) = MolecularGraph::from_smiles_single(smiles) else {
            let _ = writeln!(out, "unparsed");
            continue;
        };
        let reference = aromatic_bonds(&molecule, None);
        let mut generator = StdRng::seed_from_u64(20260917);
        let mut moved = 0;
        for _ in 0..permutations {
            let mut order: Vec<usize> = (0..molecule.len()).collect();
            order.shuffle(&mut generator);
            let mut permuted = molecule.clone();
            reorder_molecule(&mut permuted, &order);
            permuted
                .aromatize(default_aromaticity_model(), None, false)
                .expect("perception answers");
            if aromatic_bonds(&permuted, Some(&order)) != reference {
                moved += 1;
            }
        }
        let _ = writeln!(
            out,
            "{} aromatic bonds, {moved} of {permutations} numberings disagree",
            reference.len()
        );
    }
}
