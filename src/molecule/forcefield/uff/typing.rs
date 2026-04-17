//! Which UFF atom type an atom is.
//!
//! UFF names types by element, hybridization and -- for a few elements --
//! oxidation state, in a five-character convention: element symbol padded to
//! two characters, then a hybridization digit, then a charge state where the
//! element has more than one. `C_3` is tetrahedral carbon, `C_R` aromatic,
//! `N_2` trigonal nitrogen, `S_3+6` the sulfur of a sulfone.
//!
//! Typing is where a force field is most often quietly wrong, because a
//! mistyped atom still produces a number. The rules below assign the
//! paper's types by the paper's chemistry: the `R` types belong to the
//! atoms of an aromatic ring, a carbonyl oxygen is `O_2` and an amide
//! nitrogen `N_2`, each with the radius and the equilibrium angle fitted
//! to that chemistry. RDKit instead suffixes `R` onto any sp2 carbon,
//! nitrogen, oxygen or sulfur with a conjugated bond, which hands a
//! carbonyl oxygen furan's longer bond radius; `tests/uff_oracle.py` is
//! where the energy consequences of that difference are measured.

use super::super::UntypedAtom;
use super::params::{UffType, lookup};
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::featurize::{Hybridization, hybridization};

/// The number of bonds to oxygen, counting a double bond twice -- what
/// distinguishes UFF's sulfur and phosphorus oxidation states.
fn oxygen_bond_order_sum<M: MoleculeCore + ?Sized>(molecule: &M, atom: usize) -> u32 {
    molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .filter(|bond| molecule.get_atomic_number(bond.target()) == 8)
        .map(|bond| match bond.bond_order() {
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            _ => 1,
        })
        .sum()
}

/// The UFF type of every atom, in atom order.
///
/// # Errors
///
/// [`UntypedAtom`] for the first atom whose element these rules do not
/// cover. The parameter table itself spans the periodic table; what stops
/// here is the perception -- a metal's UFF type depends on its coordination
/// geometry, which this crate does not derive.
pub(super) fn type_atoms<M: MoleculeCore + ?Sized>(
    molecule: &M,
) -> Result<Vec<&'static UffType>, UntypedAtom> {
    (0..molecule.len())
        .map(|atom| {
            let atomic_number = molecule.get_atomic_number(atom);
            label_for(molecule, atom)
                .and_then(lookup)
                .ok_or_else(|| UntypedAtom::uff(atom, atomic_number))
        })
        .collect()
}

/// The UFF type label of `atom`, or `None` where these rules have none.
pub(crate) fn label_for<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atom: usize,
) -> Option<&'static str> {
    let aromatic = molecule.is_atom_aromatic(atom);
    let shape = hybridization(molecule, atom);

    Some(match molecule.get_atomic_number(atom) {
        1 => "H_",
        5 => match shape {
            Hybridization::Sp2 => "B_2",
            _ => "B_3",
        },
        6 if aromatic => "C_R",
        6 => match shape {
            Hybridization::Sp => "C_1",
            Hybridization::Sp2 => "C_2",
            _ => "C_3",
        },
        7 if aromatic => "N_R",
        7 => match shape {
            Hybridization::Sp => "N_1",
            Hybridization::Sp2 => "N_2",
            _ => "N_3",
        },
        8 if aromatic => "O_R",
        8 => match shape {
            Hybridization::Sp => "O_1",
            Hybridization::Sp2 => "O_2",
            _ => "O_3",
        },
        9 => "F_",
        14 => "Si3",
        // Phosphorus and sulfur carry their oxidation state in the type: a
        // phosphate's phosphorus is not a phosphine's, and a sulfone's
        // sulfur is not a thioether's.
        15 => match oxygen_bond_order_sum(molecule, atom) {
            0..=1 => {
                // A four-coordinate phosphorus with no oxygens is a
                // phosphonium: tetrahedral, the paper's `P_3+q`, not the
                // pyramidal `P_3+3` a phosphine gets.
                if molecule.get_atom_bonds(atom).unwrap_or_default().len() == 4 {
                    "P_3+q"
                } else {
                    "P_3+3"
                }
            }
            _ => "P_3+5",
        },
        16 if aromatic => "S_R",
        // The oxidation state decides before the hybridization does. `S_2`
        // is the sulfur of a thiocarbonyl, not of a sulfoxide -- and a
        // sulfoxide's sulfur reads as sp2 to a perception that sees a
        // conjugated `S=O` and a lone pair, which typed dimethyl sulfoxide
        // `S_2` and put its equilibrium angle at 120 degrees instead of
        // 103.2. That single mistyping was worth 63 kcal/mol against
        // RDKit's UFF on the same coordinates.
        16 => match oxygen_bond_order_sum(molecule, atom) {
            2..=3 => "S_3+4",
            4.. => "S_3+6",
            _ => match shape {
                Hybridization::Sp2 => "S_2",
                _ => "S_3+2",
            },
        },
        17 => "Cl",
        34 => "Se3+2",
        35 => "Br",
        53 => "I_",
        _ => return None,
    })
}
