//! Prints the distance-bounds matrix a molecule gets, after smoothing:
//! `i<TAB>j<TAB>lower<TAB>upper`, one line per pair, preceded by the atom
//! count and the SMILES.
//!
//! The bounds are what the embedder knows about a molecule before it has a
//! single coordinate, so they are the thing to measure when a bound changes
//! -- `tests/bounds_quality.py` checks them against real geometry, which an
//! end-to-end energy cannot resolve.
use chemistry_consts::ElementProperties;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::embed::bounds_for_dump;
use oxidane::prelude::*;

fn molecule_symbol(molecule: &MolecularGraph, atom: usize) -> String {
    molecule
        .atomic_numbers()
        .get(atom)
        .and_then(|number| number.atomic_symbol())
        .unwrap_or("?")
        .to_string()
}

fn main() {
    let smiles = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: bounds_dump <SMILES>");
        std::process::exit(2);
    });
    let molecule = match MolecularGraph::from_smiles_single(&smiles) {
        Ok(molecule) => molecule,
        Err(error) => {
            eprintln!("parse error: {error:?}");
            std::process::exit(1);
        }
    };
    let (atoms, bounds) = match bounds_for_dump(&molecule) {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("bounds error: {error}");
            std::process::exit(1);
        }
    };
    println!("{atoms}");
    println!("{smiles}");
    // The element of every atom, in our order, so a reader can check that it
    // is talking about the same atoms: our hydrogens are appended in a
    // different order than RDKit's `AddHs` gives, and comparing bounds
    // index-by-index without checking that is how this probe first reported
    // an N-H bond as a 5 Angstrom violation.
    println!(
        "{}",
        (0..atoms)
            .map(|atom| molecule_symbol(&molecule, atom))
            .collect::<Vec<_>>()
            .join(" ")
    );
    for (i, j, lower, upper) in bounds {
        println!("{i}\t{j}\t{lower:.4}\t{upper:.4}");
    }
}
