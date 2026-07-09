//! Per-atom and per-bond feature matrices, for models that read a molecule as
//! a graph.
//!
//! A message-passing network wants three things: a row of numbers for every
//! atom, a row for every bond, and a list of which atoms each bond joins.
//! Nothing here is chemistry the rest of the crate does not already know; what
//! it adds is a *schema* -- a fixed order of fixed-width fields, so that a model
//! trained on one batch can be given another.
//!
//! The schema is versioned by [`FEATURE_SCHEMA_VERSION`]. Changing the meaning
//! or the order of a column invalidates every model trained against it, so the
//! version must go up whenever it changes, and columns should be added at the
//! end rather than inserted.
//!
//! Hydrogens do not get rows. They are counted into the atom that carries them,
//! which is what nearly every published featurization does and what makes a
//! molecule parsed from SMILES and one read from a file give the same matrix.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::featurize::{ATOM_FEATURE_WIDTH, graph_features};
//!
//! let caffeine =
//!     MolecularGraph::from_smiles_single("CN1C=NC2=C1C(=O)N(C)C(=O)N2C").unwrap();
//! let features = graph_features(&caffeine);
//! assert_eq!(features.atom_count, 14);
//! assert_eq!(features.atom_features.len(), 14 * ATOM_FEATURE_WIDTH);
//! // Every bond is given both ways round, since a message has a direction.
//! assert_eq!(features.edge_index.len(), features.edge_count * 2);
//! ```

use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::chirality::ChiralClass;
use crate::molecule::descriptors::MolecularDescriptors;

/// The version of the column layout below.
///
/// A model is trained against a schema, not against a crate, so this is what a
/// stored model should record.
// Version 3 changes the acceptor column from a N/O proxy to typed N/O/S counts.
pub const FEATURE_SCHEMA_VERSION: u32 = 3;

/// The elements that get a column of their own. Anything else shares the last.
pub const FEATURED_ELEMENTS: [u8; 12] = [5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53];

/// How many numbers describe one atom.
pub const ATOM_FEATURE_WIDTH: usize = FEATURED_ELEMENTS.len()
    + 1  // an element not in the list
    + 6  // heavy neighbours, none through five
    + 6  // formal charge, -2 through +2 and anything else
    + 5  // attached hydrogens, none through four
    + 6  // hybridization
    + 1  // aromatic
    + 1  // in a ring
    + 4  // chirality
    + 1; // mass

/// Which atom columns hold a quantity rather than an indicator.
///
/// Everything else is one hot, and means what it means at the scale it is
/// already on. These are the columns [`crate::ml::normalize::FeatureStats`]
/// standardises; the positions are part of the schema, so a column added here
/// is a schema change like any other.
pub const ATOM_CONTINUOUS_COLUMNS: [usize; 1] = [ATOM_FEATURE_WIDTH - 1]; // mass

/// How many numbers describe one bond.
pub const BOND_FEATURE_WIDTH: usize = 4  // order
    + 1  // conjugated
    + 1  // in a ring
    + 4; // stereochemistry

/// How many orbitals an atom's bonding uses, as far as a graph can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hybridization {
    /// One orbital: an atom with nothing to hybridise -- a lone ion, a bare
    /// metal.
    S,
    /// Two directions: a triple bond, or two double bonds, or two neighbours
    /// and no lone pair.
    Sp,
    /// Three directions in a plane.
    Sp2,
    /// Four directions.
    Sp3,
    /// Five.
    Sp3d,
    /// Six.
    Sp3d2,
    /// Not modelled: a hydrogen, which has no shape of its own, or an atom with
    /// more orbitals than any of the above.
    Unspecified,
}

/// What an atom's bonding looks like: how many orbitals it needs.
///
/// Its neighbours plus its lone pairs, which is the count that decides the
/// shape -- water is `Sp3` on two bonds because two lone pairs take the other
/// corners, and a sulfone's sulfur is `Sp3` on four bonds however many of them
/// are double. Reading the double bonds instead, which is the obvious thing to
/// do from a graph, calls that sulfur `Sp` and is wrong about every sulfone,
/// sulfoxide and phosphate there is.
///
/// One exception to the count: an atom whose lone pair is delocalised into a
/// neighbouring π system is flat, not tetrahedral. That is what makes an
/// amide's nitrogen and a phenol's oxygen `Sp2`, and it is the difference that
/// matters most in practice.
///
/// This is what RDKit reports, checked atom for atom against it; the elements
/// where it is only approximate are the transition metals, whose lone pairs a
/// group number does not describe.
pub fn hybridization<M: MoleculeCore + ?Sized>(molecule: &M, atom: usize) -> Hybridization {
    let atomic_number = molecule.get_atomic_number(atom);
    // A hydrogen has one orbital and no geometry; every toolkit leaves it out.
    if atomic_number <= 1 {
        return Hybridization::Unspecified;
    }
    let outer = chemistry_consts::VALENCE_ELECTRONS
        .get(atomic_number as usize)
        .copied()
        .unwrap_or(0) as i32;
    let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
    let neighbours = bonds.len() as i32;
    // What is left of the outer shell once the bonds and the charge have taken
    // their share, rounded up into pairs.
    let spare = outer - molecule.actual_valency(atom) as i32 - molecule.get_atom_charge(atom) as i32;
    let lone_pairs = if spare > 0 { (spare + 1) / 2 } else { 0 };

    match neighbours + lone_pairs {
        0 | 1 => Hybridization::S,
        2 => Hybridization::Sp,
        3 => Hybridization::Sp2,
        4 => {
            if lone_pairs > 0 && has_conjugated_bond(molecule, atom) {
                Hybridization::Sp2
            } else {
                Hybridization::Sp3
            }
        }
        5 => Hybridization::Sp3d,
        6 => Hybridization::Sp3d2,
        _ => Hybridization::Unspecified,
    }
}

/// Whether any of the atom's bonds is part of a conjugated system.
fn has_conjugated_bond<M: MoleculeCore + ?Sized>(molecule: &M, atom: usize) -> bool {
    molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .any(|bond| is_conjugated(molecule, atom, bond.target(), bond.bond_order()))
}

/// A molecule as a model reads it.
///
/// Every vector is flat, in row-major order, so that it can be handed to a
/// tensor library without being walked again.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphFeatures {
    /// How many atoms have rows -- the heavy atoms.
    pub atom_count: usize,
    /// `atom_count` rows of [`ATOM_FEATURE_WIDTH`] numbers.
    pub atom_features: Vec<f32>,
    /// How many directed edges there are: twice the number of bonds.
    pub edge_count: usize,
    /// `edge_count` rows of [`BOND_FEATURE_WIDTH`] numbers, in the same order
    /// as `edge_index`.
    pub bond_features: Vec<f32>,
    /// `edge_count` pairs of row numbers, source first.
    pub edge_index: Vec<u32>,
    /// Which atom of the molecule each row stands for.
    pub atoms: Vec<usize>,
}

impl GraphFeatures {
    /// One atom's row.
    pub fn atom_row(&self, row: usize) -> &[f32] {
        &self.atom_features[row * ATOM_FEATURE_WIDTH..(row + 1) * ATOM_FEATURE_WIDTH]
    }

    /// One directed edge's row.
    pub fn bond_row(&self, edge: usize) -> &[f32] {
        &self.bond_features[edge * BOND_FEATURE_WIDTH..(edge + 1) * BOND_FEATURE_WIDTH]
    }

    /// The two atoms a directed edge joins, as row numbers.
    pub fn edge(&self, edge: usize) -> (usize, usize) {
        (
            self.edge_index[edge * 2] as usize,
            self.edge_index[edge * 2 + 1] as usize,
        )
    }
}

/// Builds the feature matrices of a molecule.
pub fn graph_features<M: MoleculeCore + ?Sized>(molecule: &M) -> GraphFeatures {
    let atom_count = molecule.len();
    let cyclic_atoms = molecule.cyclic_atoms();
    let (_, cyclic_bonds) = molecule.find_cyclic_bonds_and_atoms();

    let mut row_of = vec![usize::MAX; atom_count];
    let mut atoms = Vec::new();
    for atom in 0..atom_count {
        if molecule.get_atomic_number(atom) != 1 {
            row_of[atom] = atoms.len();
            atoms.push(atom);
        }
    }

    let mut atom_features = vec![0.0f32; atoms.len() * ATOM_FEATURE_WIDTH];
    for (row, &atom) in atoms.iter().enumerate() {
        let target = &mut atom_features[row * ATOM_FEATURE_WIDTH..(row + 1) * ATOM_FEATURE_WIDTH];
        write_atom_row(molecule, atom, cyclic_atoms.contains(atom), target);
    }

    let mut bond_features: Vec<f32> = Vec::new();
    let mut edge_index: Vec<u32> = Vec::new();
    for &atom in &atoms {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let target = bond.target();
            if target >= atom_count || row_of[target] == usize::MAX {
                continue;
            }
            edge_index.push(row_of[atom] as u32);
            edge_index.push(row_of[target] as u32);
            let start = bond_features.len();
            bond_features.resize(start + BOND_FEATURE_WIDTH, 0.0);
            let in_ring = cyclic_bonds.contains(&(atom.min(target), atom.max(target)));
            write_bond_row(
                molecule,
                atom,
                target,
                bond.bond_order(),
                in_ring,
                &mut bond_features[start..start + BOND_FEATURE_WIDTH],
            );
        }
    }

    GraphFeatures {
        atom_count: atoms.len(),
        atom_features,
        edge_count: edge_index.len() / 2,
        bond_features,
        edge_index,
        atoms,
    }
}

/// How many numbers describe a whole molecule. See [`molecule_features`].
pub const MOLECULE_FEATURE_WIDTH: usize = 12;

/// One row of numbers for the molecule as a whole.
///
/// Message passing sees a neighbourhood at a time. Some of what a property
/// depends on is not in any neighbourhood -- how large the molecule is, how
/// much of its surface is polar, how many bonds it can turn about -- and
/// reaching it by message passing means enough rounds that every atom's vector
/// has converged on every other's. Handing those quantities to the readout
/// directly is cheaper and works better, and it is what the strongest
/// published variants of this architecture do.
///
/// Nothing here is new chemistry: every entry is a descriptor the crate
/// already computes. The point is the fixed order, which is part of
/// [`FEATURE_SCHEMA_VERSION`] like the atom columns are.
///
/// The columns, in order: heavy atoms, molecular weight, Crippen logP, Crippen
/// molar refractivity, topological polar surface area, hydrogen-bond donors,
/// hydrogen-bond acceptors, rotatable bonds, rings, fragments, fraction of
/// saturated carbons, total formal charge.
///
/// Every one is a quantity rather than an indicator, so all of them want
/// scaling -- see [`crate::ml::normalize::GlobalStats`].
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::featurize::{MOLECULE_FEATURE_WIDTH, molecule_features};
///
/// let caffeine = MolecularGraph::from_smiles_single("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();
/// let row = molecule_features(&caffeine);
/// assert_eq!(row.len(), MOLECULE_FEATURE_WIDTH);
/// assert_eq!(row[0], 14.0); // heavy atoms
/// ```
pub fn molecule_features<M: MolecularDescriptors + ?Sized>(molecule: &M) -> Vec<f32> {
    let (log_p, refractivity) = molecule.crippen_descriptors();
    vec![
        molecule.heavy_atom_count() as f32,
        molecule.molecular_weight() as f32,
        log_p as f32,
        refractivity as f32,
        molecule.topological_polar_surface_area() as f32,
        molecule.number_of_hydrogen_bond_donors() as f32,
        molecule.number_of_hydrogen_bond_acceptors() as f32,
        molecule.rotatable_bond_count() as f32,
        molecule.ring_count() as f32,
        molecule.fragment_count() as f32,
        molecule.fraction_saturated_carbons() as f32,
        molecule.total_formal_charge() as f32,
    ]
}

/// Fills one atom's row. See the module documentation for what the columns are.
fn write_atom_row<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atom: usize,
    in_ring: bool,
    row: &mut [f32],
) {
    let atomic_number = molecule.get_atomic_number(atom);
    let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
    let hydrogens = bonds
        .iter()
        .filter(|bond| molecule.get_atomic_number(bond.target()) == 1)
        .count();
    let heavy_degree = bonds.len() - hydrogens;

    let mut cursor = 0usize;
    let element = FEATURED_ELEMENTS
        .iter()
        .position(|&candidate| candidate == atomic_number)
        .unwrap_or(FEATURED_ELEMENTS.len());
    one_hot(row, &mut cursor, FEATURED_ELEMENTS.len() + 1, element);
    one_hot(row, &mut cursor, 6, heavy_degree.min(5));
    let charge = molecule.get_atom_charge(atom);
    let charge_slot = if (-2..=2).contains(&charge) {
        (charge + 2) as usize
    } else {
        5
    };
    one_hot(row, &mut cursor, 6, charge_slot);
    one_hot(row, &mut cursor, 5, hydrogens.min(4));
    let hybridization_slot = match hybridization(molecule, atom) {
        Hybridization::Sp => 0,
        Hybridization::Sp2 => 1,
        Hybridization::Sp3 => 2,
        Hybridization::Sp3d => 3,
        Hybridization::Sp3d2 => 4,
        // One column for the states a molecular graph does not describe. No
        // hydrogen ever reaches it, since hydrogens are given no row at all.
        Hybridization::S | Hybridization::Unspecified => 5,
    };
    one_hot(row, &mut cursor, 6, hybridization_slot);
    row[cursor] = f32::from(molecule.is_atom_aromatic(atom));
    cursor += 1;
    row[cursor] = f32::from(in_ring);
    cursor += 1;
    let chirality_slot = match molecule.get_chiral_class(atom) {
        ChiralClass::None => 0,
        ChiralClass::Clockwise => 1,
        ChiralClass::Counterclockwise => 2,
        _ => 3,
    };
    one_hot(row, &mut cursor, 4, chirality_slot);
    // Scaled so that every column is of the same order, which a network that
    // has not been told otherwise will assume.
    row[cursor] = atomic_number as f32 / 100.0;
}

/// Fills one directed bond's row.
fn write_bond_row<M: MoleculeCore + ?Sized>(
    molecule: &M,
    first: usize,
    second: usize,
    order: BondOrder,
    in_ring: bool,
    row: &mut [f32],
) {
    let mut cursor = 0usize;
    let order_slot = match order {
        BondOrder::Double => 1,
        BondOrder::Triple => 2,
        BondOrder::Aromatic => 3,
        _ => 0,
    };
    one_hot(row, &mut cursor, 4, order_slot);
    row[cursor] = f32::from(is_conjugated(molecule, first, second, order));
    cursor += 1;
    row[cursor] = f32::from(in_ring);
    cursor += 1;
    let stereo_slot = match (
        molecule.get_chiral_class(first),
        molecule.get_chiral_class(second),
    ) {
        (ChiralClass::CisTransClockwise, _) | (_, ChiralClass::CisTransClockwise) => 1,
        (ChiralClass::CisTransCounterclockwise, _) | (_, ChiralClass::CisTransCounterclockwise) => 2,
        _ => 0,
    };
    one_hot(row, &mut cursor, 4, stereo_slot);
}

/// Whether a bond is part of a run of overlapping p orbitals.
///
/// A multiple or aromatic bond always is. A single bond is when both of its
/// ends have a p orbital to offer -- either because they carry a π bond of
/// their own or because they are a heteroatom with a lone pair -- which is what
/// makes an amide `C-N` conjugated and an ethane `C-C` not.
fn is_conjugated<M: MoleculeCore + ?Sized>(
    molecule: &M,
    first: usize,
    second: usize,
    order: BondOrder,
) -> bool {
    if order != BondOrder::Single {
        return true;
    }
    [first, second].iter().all(|&atom| {
        let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
        let atomic_number = molecule.get_atomic_number(atom);
        let charge = molecule.get_atom_charge(atom);
        // Below the second row a lone pair is the wrong size to overlap with a
        // neighbouring π system, and only a charge makes it reach: a
        // sulfonamide's nitrogen stays tetrahedral and a thioether's sulfur
        // beside a vinyl group does too, while a thiolate on a ring is flat.
        // An aromatic or multiple bond is conjugated whatever it joins, which
        // is why thiophene's sulfur is flat and a thioether's is not.
        if atomic_number > 10 {
            return charge < 0 && bonds.len() <= 3;
        }
        let carries_pi = bonds.iter().any(|bond| {
            matches!(
                bond.bond_order(),
                BondOrder::Double | BondOrder::Triple | BondOrder::Aromatic
            )
        });
        let lone_pair = matches!(atomic_number, 7 | 8) || charge < 0;
        carries_pi || (lone_pair && bonds.len() <= 3)
    }) && [first, second].iter().any(|&atom| {
        molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .any(|bond| {
                matches!(
                    bond.bond_order(),
                    BondOrder::Double | BondOrder::Triple | BondOrder::Aromatic
                )
            })
    })
}

/// Sets one of `width` columns and advances past all of them.
fn one_hot(row: &mut [f32], cursor: &mut usize, width: usize, slot: usize) {
    if slot < width {
        row[*cursor + slot] = 1.0;
    }
    *cursor += width;
}
