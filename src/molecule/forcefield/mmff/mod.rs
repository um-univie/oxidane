//! MMFF94 -- so far, the half of it that says what each atom is.
//!
//! MMFF94 is the force field this crate is moving towards. Measured against
//! GFN2-xTB on structures relaxed from the same starting geometry, it sits
//! about ten times closer to the semiempirical answer than [`super::uff`]
//! does; the price is that it covers organic main-group chemistry only, and
//! that it needs about a hundred symbolic atom types where UFF needs a
//! handful of rules over the periodic table.
//!
//! Atom typing, the charge model and the energy itself all live here: this
//! is a complete MMFF94, checked term by term against RDKit's. It is built and
//! checked on its own, before any energy term exists, because a force field
//! reports no error when it is mistyped -- it reports a number. Typing is
//! also where an MMFF implementation is most likely to be wrong: RDKit's own
//! typing is some two and a half thousand lines, and this module follows it
//! rather than the paper's prose table, since RDKit is also the oracle that
//! `tests/mmff_oracle.py` checks every atom against.
//!
//! [`partial_charges`] is the second stage: MMFF's charges are not a
//! property of the geometry but of the types, so they are available as soon
//! as the typing is, and are checked the same way. [`ForceField`] is the
//! third: seven terms, of which UFF has four, and each one comparable on its
//! own because RDKit will switch the other six off.
//!
//! What is not here is a decision. [`super::uff`] is still what
//! [`crate::molecule::embed`] relaxes against; whether MMFF94 should replace
//! it is a measurement, not a preference, and `tests/xtb_oracle.py` is where
//! that measurement lives.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::forcefield::mmff;
//!
//! let mut molecule = MolecularGraph::from_smiles_single("CC(=O)O").unwrap();
//! molecule.add_hydrogens();
//! let types = mmff::atom_types(&molecule).unwrap();
//! // Alkyl carbon, carbonyl carbon, carbonyl oxygen, carboxyl oxygen.
//! assert_eq!(types[0].number(), 1);
//! assert_eq!(types[1].number(), 3);
//! assert_eq!(types[2].number(), 7);
//! assert_eq!(types[3].symbol(), "OR");
//!
//! // The two oxygens of a carboxylate share the charge, whichever one the
//! // SMILES wrote it on.
//! let mut acetate = MolecularGraph::from_smiles_single("CC(=O)[O-]").unwrap();
//! acetate.add_hydrogens();
//! let charges = mmff::partial_charges(&acetate).unwrap();
//! assert!((charges[2] - charges[3]).abs() < 1e-9);
//! ```
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::embed::Embed3D;
//! use oxidane::molecule::forcefield::mmff;
//!
//! let molecule = MolecularGraph::from_smiles_single("CCO").unwrap();
//! let mut embedded = molecule.embed_3d().unwrap();
//! let field = mmff::ForceField::new(&embedded).unwrap();
//! let before = field.energy(&embedded.positions);
//! let after = field.minimize(&mut embedded.positions, 200);
//! // Not `after <= before`: `embed_3d` already relaxed this structure, so
//! // there is nothing left to gain, and the two numbers are the same sum
//! // in different orders -- `energy` adds term by term where the
//! // minimiser's last evaluation adds as it goes. They can differ by an
//! // ulp, and once did.
//! assert!(after <= before + 1e-12);
//! // Every term separately, which is how the field is checked.
//! let terms = field.energies(&embedded.positions);
//! assert!((terms.total() - after).abs() < 1e-9);
//! ```
//!
//! # Hydrogens must be atoms
//!
//! Every MMFF type distinction that counts connections counts hydrogens
//! among them, and hydrogens are atoms in this crate rather than a count on
//! their heavy neighbour. A molecule typed without
//! [`MoleculeCore::add_hydrogens`]
//! is a different molecule as far as these rules are concerned -- an amine
//! nitrogen with its hydrogens left implicit reads as a two-connection
//! nitrogen, and types as one.

mod assign;
mod charges;
mod field;
mod params;
mod perception;
mod terms;
mod typing;

pub use charges::partial_charges;
pub use field::{EquilibriumAngle, EquilibriumLength, ForceField, TermEnergies};
pub use typing::{MmffType, atom_types};

use crate::molecule::base::MoleculeCore;
use super::UntypedAtom;

/// MMFF94's equilibrium bond angles for `molecule`, as `(first, centre,
/// last, degrees)` with `first < last`.
///
/// The companion of [`equilibrium_bond_lengths`], and wanted for the same
/// reason: the 1-3 distance bounds are an angle in disguise, and MMFF's
/// angle types already know what a plain hybridisation estimate does not --
/// that an amide nitrogen is planar, that an ether's angle is 113 rather
/// than 109.5, that a sulfoxide closes to 97, and that an angle inside a
/// three- or four-membered ring is the ring's own.
///
/// # Errors
///
/// [`UntypedAtom`], as [`equilibrium_bond_lengths`].
pub fn equilibrium_angles<M: MoleculeCore + ?Sized>(
    molecule: &M,
) -> Result<Vec<(usize, usize, usize, f64)>, UntypedAtom> {
    let (perception, types) = typing::typed(molecule)?;
    let count = perception.atom_count();
    let mut angles = Vec::new();
    for centre in 0..count {
        let neighbours = perception.neighbours(centre);
        for (position, first) in neighbours.iter().enumerate() {
            for last in &neighbours[position + 1..] {
                let (i, j) = (first.atom.min(last.atom), first.atom.max(last.atom));
                if let Some(parameters) = assign::angle_parameters(&perception, &types, i, centre, j)
                {
                    angles.push((i, centre, j, parameters.angle));
                }
            }
        }
    }
    Ok(angles)
}

/// MMFF94's equilibrium bond lengths for `molecule`, as `(i, j, length)` with
/// `i < j` and lengths in Angstroms.
///
/// Available without building a whole [`ForceField`] because the distance
/// bounds in [`crate::molecule::embed`] want the lengths of the field their
/// structure is about to be relaxed against, and want them before any
/// coordinate exists. Every one is either the parameter table's or MMFF.V's
/// own empirical rule, which is the same two-step [`ForceField::new`] takes.
///
/// # Errors
///
/// [`UntypedAtom`] if any atom is outside MMFF94's typing rules; a caller
/// with a fallback of its own should treat that as "no lengths", not as a
/// failure.
pub fn equilibrium_bond_lengths<M: MoleculeCore + ?Sized>(
    molecule: &M,
) -> Result<Vec<(usize, usize, f64)>, UntypedAtom> {
    let (perception, types) = typing::typed(molecule)?;
    let count = perception.atom_count();
    let mut lengths = Vec::with_capacity(count);
    for first in 0..count {
        for neighbour in perception.neighbours(first) {
            let second = neighbour.atom;
            if second <= first {
                continue;
            }
            if let Some(parameters) = assign::bond_parameters(&perception, &types, first, second) {
                lengths.push((first, second, parameters.length));
            }
        }
    }
    Ok(lengths)
}
