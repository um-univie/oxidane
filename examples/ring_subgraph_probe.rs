//! Shows what changes when the ring family is built over the π-capable bonds
//! instead of over the molecule: same cycles exist, different ones are
//! *smallest*, and the family is defined by smallest.
use oxidane::molecule::pi_system::conjugated_ring_bonds;
use oxidane::molecule::ring_aromaticity::RingFinder;
use oxidane::prelude::*;
use std::io::BufRead;

fn show(label: &str, finder: &RingFinder) {
    let mut rings: Vec<(usize, Vec<usize>)> = finder
        .rings()
        .map(|ring| (ring.atoms.len(), ring.atoms))
        .collect();
    rings.sort();
    println!("  {label}: {} rings", rings.len());
    for (size, atoms) in rings {
        println!("     {size}-ring {atoms:?}");
    }
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }
        let Ok(molecule) = MolecularGraph::from_smiles_single(smiles) else {
            println!("unparsed");
            continue;
        };
        println!("{smiles}");

        let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
        let conjugated = conjugated_ring_bonds(&molecule, &cyclic);
        let mut dropped: Vec<(usize, usize)> = cyclic
            .iter()
            .copied()
            .filter(|bond| !conjugated.contains(bond))
            .collect();
        dropped.sort_unstable();
        println!(
            "  ring bonds {}, of which π-capable {} (dropped {dropped:?})",
            cyclic.len(),
            conjugated.len()
        );

        let mut finder = RingFinder::new();
        finder.perceive(&molecule, &cyclic, 24);
        show("family over the molecule", &finder);
        finder.perceive(&molecule, &conjugated, 24);
        show("family over the π-capable bonds", &finder);
    }
}
