//! Prints this crate's MMFF94 atom type and partial charge for every atom of
//! a SMILES, so `tests/mmff_oracle.py` can ask RDKit what it thinks of the
//! same molecules.
//!
//! Takes SMILES as arguments, or one per line on standard input, and prints
//! a block per molecule: the SMILES, the atom count, then a line per atom.
//! Heavy atoms come out in the order the SMILES gave them, which is the
//! order RDKit uses too; hydrogens are appended, and each one prints the
//! heavy atom it hangs off so the two sides can be matched without relying
//! on hydrogens being added in the same order.
use chemistry_consts::ElementProperties;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::forcefield::mmff;
use oxidane::prelude::*;
use std::io::BufRead;

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.is_empty() {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines().map_while(Result::ok) {
            let smiles = line.trim();
            if !smiles.is_empty() {
                report(smiles);
            }
        }
    } else {
        for smiles in &arguments {
            report(smiles);
        }
    }
}

fn report(smiles: &str) {
    println!("SMILES {smiles}");
    let Ok(mut molecule) = MolecularGraph::from_smiles_single(smiles) else {
        println!("UNPARSED");
        return;
    };
    molecule.add_hydrogens();

    let types = match mmff::atom_types(&molecule) {
        Ok(types) => types,
        Err(error) => {
            println!("UNTYPED {error}");
            return;
        }
    };
    let charges = mmff::partial_charges(&molecule).expect("typed, so charged");

    println!("{}", molecule.len());
    for atom in 0..molecule.len() {
        let atomic_number = molecule.atomic_numbers()[atom];
        let symbol = atomic_number.atomic_symbol().unwrap_or("X");
        let parent = if atomic_number == 1 {
            molecule
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .first()
                .map_or(-1, |bond| bond.target() as i64)
        } else {
            -1
        };
        println!(
            "{atom} {symbol} {} {} {parent} {:.4}",
            types[atom].number(),
            types[atom].symbol(),
            charges[atom]
        );
    }
}
