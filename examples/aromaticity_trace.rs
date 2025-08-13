//! Prints, for each SMILES given on the command line, every decision
//! aromaticity perception makes about a ring atom: the cyclic bond set, the
//! per-element lookup index the handler returns, the electron count that index
//! maps to under the OpenSMILES model, and the perceived output.
//!
//! Diagnostic only -- nothing in the library reads it.
use oxidane::molecule::aromaticity::{AromaticityFlag, MoleculeAromaticity};
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::bond::BondOrder;
use oxidane::prelude::*;
use rustc_hash::FxHashSet;

/// The OpenSMILES table, transcribed from `aromaticity.rs` so the example can
/// show what each lookup index means without the const being made public.
const OPENSMILES: [&str; 34] = [
    "NotAromatic",      // 0
    "0e",               // 1 B(0), v3, deg3
    "1e",               // 2 B(0) v3 deg2 / B(-1) v4 deg3
    "1e",               // 3 C(0) standard
    "1e",               // 4 C(0) many aromatic bonds, or exocyclic C=C
    "0e",               // 5 C(0) exocyclic double bond to non-carbon
    "1e",               // 6 C(-1) v3 deg2
    "2e",               // 7 C(-1) v3 deg3
    "1e",               // 8 C(+1) v3 deg2
    "0e",               // 9 C(+1) v3 deg3
    "1e",               // 10 N(0) v3/4 deg2
    "2e",               // 11 N(0) v3/4 deg3
    "1e",               // 12 N(0) v5 deg3 N-oxide
    "0e",               // 13 N(+1) v4 deg3 exo =O
    "2e",               // 14 N(-1) v2
    "1e",               // 15 N(+1) v4 deg3 no exo =O
    "2e",               // 16 O(0) v2
    "1e",               // 17 O(+1) v3 deg2
    "2e",               // 18 S(0) v2
    "1e",               // 19 S(+1) v3 deg2
    "2e",               // 20 S(0) v4 deg3 exo =O
    "2e",               // 21 S(+1) v3 deg3 exo O
    "1e",               // 22 P(0) v3 deg2
    "2e",               // 23 P(0) v3 deg3
    "1e",               // 24 P(0) v5 deg3 exo O
    "0e",               // 25 P(+1) v4 deg3 exo =O
    "2e",               // 26 P(-1) v2
    "1e",               // 27 P(+1) v4 deg3 no exo =O
    "2e",               // 28 Se(0) v2
    "1e",               // 29 Se(+1) v4 deg2
    "2e",               // 30 Te(0) v2
    "1e",               // 31 Te(+1) v3 deg2
    "2e",               // 32 As(0) v3 deg3
    "1e",               // 33 As(+1) v4 deg3
];

fn symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        15 => "P",
        16 => "S",
        17 => "Cl",
        _ => "?",
    }
}

fn main() {
    for smiles in std::env::args().skip(1) {
        println!("================ {smiles}");
        // Parse without perceiving, so the decisions below are read off the
        // structure as written rather than off one perception's output.
        let mut molecule = oxidane::io::SMILESParser::parse_smiles_single_with(
            &smiles,
            oxidane::io::SmilesParseOptions { aromaticity: None },
        )
        .unwrap();

        let mut atoms_in_rings = bit_set::BitSet::new();
        let mut bonds_in_rings: FxHashSet<(usize, usize)> = FxHashSet::default();
        molecule.find_cyclic_bonds_and_atoms_into(&mut atoms_in_rings, &mut bonds_in_rings);
        let bonds_in_rings =
            oxidane::molecule::pi_system::conjugated_ring_bonds(&molecule, &bonds_in_rings);

        println!("-- ring atoms and their handler decisions");
        for atom in 0..molecule.len() {
            if !atoms_in_rings.contains(atom) {
                continue;
            }
            let z = molecule.get_atomic_number(atom);
            let charge = molecule.get_atom_charge(atom);
            let valency = molecule.actual_valency(atom);
            let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();

            let index = match z {
                5 => molecule.handle_boron_aromaticity(atom),
                6 => molecule.handle_carbon_aromaticity(atom, &bonds_in_rings),
                7 => molecule.handle_nitrogen_aromaticity(atom, &bonds_in_rings),
                8 => molecule.handle_oxygen_aromaticity(atom),
                15 => molecule.handle_phosphorus_aromaticity(atom, &bonds_in_rings),
                16 => molecule.handle_sulfur_aromaticity(atom, &bonds_in_rings),
                _ => 0,
            };

            let mut detail = String::new();
            for bond in bonds.iter() {
                let target = bond.target();
                if molecule.get_atomic_number(target) == 1 {
                    continue;
                }
                let key = (atom.min(target), atom.max(target));
                let order = match bond.bond_order() {
                    BondOrder::Single => "-",
                    BondOrder::Double => "=",
                    BondOrder::Triple => "#",
                    BondOrder::Aromatic => ":",
                    _ => "?",
                };
                detail.push_str(&format!(
                    " {order}{}{target}{}",
                    symbol(molecule.get_atomic_number(target)),
                    if bonds_in_rings.contains(&key) {
                        "[cyclic]"
                    } else {
                        "[acyclic]"
                    },
                ));
            }

            println!(
                "  atom {atom:>2} {:>2} q{charge} v{valency} deg{} -> index {index:>2} = {:<12} |{detail}",
                symbol(z),
                bonds.len(),
                OPENSMILES[index as usize],
            );
        }

        molecule.aromatize(oxidane::molecule::aromaticity::AromaticityModel::OpenSMILES, None, false)
            .unwrap();
        let aromatic: Vec<usize> = (0..molecule.len())
            .filter(|&a| molecule.is_atom_aromatic(a))
            .collect();
        println!("-- perceived aromatic atoms: {aromatic:?}");
        println!("-- output: {}", molecule.to_smiles());
        let _ = AromaticityFlag::NotAromatic;
    }
}
