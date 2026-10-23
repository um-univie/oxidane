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
//! Hydrogens get a row like any other atom, and an X-H bond gets an edge like
//! any other bond -- there is nothing about a hydrogen a molecular graph does
//! not already know, and a model that reads one directly can answer a
//! question ("which of this atom's two hydrogens is the reactive one") that
//! folding hydrogens into their neighbour's row cannot even ask.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::featurize::{ATOM_FEATURE_WIDTH, graph_features};
//!
//! let caffeine =
//!     MolecularGraph::from_smiles_single("CN1C=NC2=C1C(=O)N(C)C(=O)N2C").unwrap();
//! let features = graph_features(&caffeine);
//! // 14 heavy atoms (C8H10N4O2) plus its 10 hydrogens.
//! assert_eq!(features.atom_count, 24);
//! assert_eq!(features.atom_features.len(), 24 * ATOM_FEATURE_WIDTH);
//! // Every bond is given both ways round, since a message has a direction.
//! assert_eq!(features.edge_index.len(), features.edge_count * 2);
//! ```

use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::chirality::ChiralClass;
use crate::molecule::descriptors::MolecularDescriptors;
use crate::vector::Vector3D;

/// The version of the column layout below.
///
/// A model is trained against a schema, not against a crate, so this is what a
/// stored model should record.
// Version 5 changes the acceptor column from a N/O proxy to typed N/O/S counts.
// Version 6 makes the molecular-weight column honor explicit isotope labels.
// Version 7 gives a double bond's stereo column its geometry relative to the
// highest-ranked substituents, and no other bond a stereo column at all; and the
// donor column follows RDKit's typed definition (thiols in, water out).
// Version 8 fills the chirality columns: a tetrahedral centre's hand relative
// to its neighbours' ranks, where before every centre landed in "other".
pub const FEATURE_SCHEMA_VERSION: u32 = 8;

/// The elements that get a column of their own. Anything else shares the last.
pub const FEATURED_ELEMENTS: [u8; 13] = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53];

/// How many numbers describe one atom.
pub const ATOM_FEATURE_WIDTH: usize = FEATURED_ELEMENTS.len()
    + 1  // an element not in the list
    + 6  // heavy neighbours, none through five
    + 6  // formal charge, -2 through +2 and anything else
    + 5  // attached hydrogens, none through four
    + 6  // hybridization
    + 1  // aromatic
    + 1  // in a ring
    + 4  // chirality: none, clockwise, anticlockwise, other -- see tetrahedral_hands
    + 1  // mass
    + 1; // mean local bond angle (radians; 0 without 3D geometry -- see graph_features_with_geometry)

/// How many element classes an atom row distinguishes: the featured elements,
/// and one more for everything else.
pub const ATOM_ELEMENT_CLASSES: usize = FEATURED_ELEMENTS.len() + 1;

/// The columns that say which element an atom is, taken together as a one-hot
/// block at the front of the row.
///
/// Named because a pretraining objective has to be able to hide the answer it
/// is asking for. Masking an atom's element means clearing these *and* the
/// mass -- see [`ATOM_CONTINUOUS_COLUMNS`], which for schema 2 is the mass
/// alone. Leaving the mass behind would leave the element written on the row
/// in another hand, and the model would learn to read it rather than to infer
/// anything from the neighbourhood.
pub const ATOM_ELEMENT_COLUMNS: std::ops::Range<usize> = 0..ATOM_ELEMENT_CLASSES;

/// Which atom columns hold a quantity rather than an indicator.
///
/// Everything else is one hot, and means what it means at the scale it is
/// already on. These are the columns [`crate::ml::normalize::FeatureStats`]
/// standardises; the positions are part of the schema, so a column added here
/// is a schema change like any other.
pub const ATOM_CONTINUOUS_COLUMNS: [usize; 2] = [ATOM_FEATURE_WIDTH - 2, ATOM_FEATURE_WIDTH - 1]; // mass, mean local bond angle

/// How many numbers describe one bond.
pub const BOND_FEATURE_WIDTH: usize = 4  // order
    + 1  // conjugated
    + 1  // in a ring
    + 4  // double-bond geometry: none, cis, trans, given but unrankable -- see double_bond_geometry
    + 1; // 3D distance (angstroms; 0 without geometry -- see graph_features_with_geometry)

/// Which bond columns hold a quantity rather than an indicator -- the 3D
/// distance alone, standardised by [`crate::ml::normalize::BondFeatureStats`]
/// the same way [`ATOM_CONTINUOUS_COLUMNS`] are.
pub const BOND_CONTINUOUS_COLUMNS: [usize; 1] = [BOND_FEATURE_WIDTH - 1];

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
    let spare =
        outer - molecule.actual_valency(atom) as i32 - molecule.get_atom_charge(atom) as i32;
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

    /// The distinct bonds, each named by its two atom rows, in the order they
    /// are first seen among the directed edges above.
    ///
    /// A bond-level dataset wants one label per bond rather than one per
    /// directed edge; this is the canonical order to attach them in --
    /// the same one `crate::ml::nn::model::bond_predictions` reports
    /// predictions in.
    pub fn bonds(&self) -> Vec<(usize, usize)> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for edge in 0..self.edge_count {
            let (first, second) = self.edge(edge);
            let key = (first.min(second), first.max(second));
            if seen.insert(key) {
                out.push(key);
            }
        }
        out
    }

    /// Atom-indexed labels, mapped onto one entry per atom row.
    ///
    /// `labels` is keyed by the atom's index in the original molecule (what
    /// [`crate::molecule::base::MoleculeCore`] methods take), not by row --
    /// that is what a caller reading an external dataset has. An index with
    /// no label gets `None`.
    pub fn atom_targets_from(
        &self,
        labels: &std::collections::HashMap<usize, f64>,
    ) -> Vec<Option<f64>> {
        self.atoms
            .iter()
            .map(|atom| labels.get(atom).copied())
            .collect()
    }

    /// Bond-indexed labels, mapped onto one entry per **directed edge**.
    ///
    /// `labels` is keyed by an unordered pair of original molecule atom
    /// indices. Both directions of a bond get the same value, matching the
    /// duplication [`GraphFeatures::bond_features`] already has, so the
    /// result lines up with a bond head's predictions row for row.
    pub fn bond_targets_from(
        &self,
        labels: &std::collections::HashMap<(usize, usize), f64>,
    ) -> Vec<Option<f64>> {
        (0..self.edge_count)
            .map(|edge| {
                let (first, second) = self.edge(edge);
                let (a, b) = (self.atoms[first], self.atoms[second]);
                labels.get(&(a.min(b), a.max(b))).copied()
            })
            .collect()
    }
}

/// Builds the feature matrices of a molecule.
pub fn graph_features<M: MoleculeCore + ?Sized>(molecule: &M) -> GraphFeatures {
    build_graph_features(molecule, None)
}

/// [`graph_features`], with two extra columns filled in from a 3D embedding:
/// each atom's mean local bond angle, and each bond's 3D length -- the
/// information a 2D graph alone doesn't have, in the spirit of the
/// higher-body-order terms an equivariant model like MACE would use, without
/// building the equivariance itself.
///
/// `positions` is indexed exactly like `molecule`'s own atoms: what
/// [`crate::molecule::embed::Embed3D::embed_3d`] gives back as the `positions`
/// field of the [`crate::molecule::molecule3d::Molecule3D`] it returns. Its
/// own `add_hydrogens()` call is a no-op on a molecule from this crate's own
/// parsers, which already carry explicit hydrogens as real atoms, so the
/// indices already line up with no remapping.
pub fn graph_features_with_geometry<M: MoleculeCore + ?Sized>(
    molecule: &M,
    positions: &[Vector3D],
) -> GraphFeatures {
    build_graph_features(molecule, Some(positions))
}

fn build_graph_features<M: MoleculeCore + ?Sized>(
    molecule: &M,
    positions: Option<&[Vector3D]>,
) -> GraphFeatures {
    let atom_count = molecule.len();
    let cyclic_atoms = molecule.cyclic_atoms();
    let (_, cyclic_bonds) = molecule.find_cyclic_bonds_and_atoms();

    // Every atom gets a row now, so the row a given atom occupies is just its
    // own index -- there is no longer a heavy/light distinction to track.
    let atoms: Vec<usize> = (0..atom_count).collect();

    // Both stereo features read the stated configuration against these, and
    // a molecule stating none needs neither.
    let ranks = (0..atom_count)
        .any(|atom| molecule.get_chiral_class(atom) != ChiralClass::None)
        .then(|| refined_ranks(molecule));
    let hands = tetrahedral_hands(molecule, ranks.as_deref());

    let mut atom_features = vec![0.0f32; atom_count * ATOM_FEATURE_WIDTH];
    for atom in 0..atom_count {
        let angle = positions.map_or(0.0, |p| mean_local_angle(molecule, atom, p));
        let target = &mut atom_features[atom * ATOM_FEATURE_WIDTH..(atom + 1) * ATOM_FEATURE_WIDTH];
        write_atom_row(
            molecule,
            atom,
            cyclic_atoms.contains(atom),
            angle,
            hands[atom],
            target,
        );
    }

    let geometry = double_bond_geometry(molecule, ranks.as_deref());
    let mut bond_features: Vec<f32> = Vec::new();
    let mut edge_index: Vec<u32> = Vec::new();
    for atom in 0..atom_count {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let target = bond.target();
            if target >= atom_count {
                continue;
            }
            edge_index.push(atom as u32);
            edge_index.push(target as u32);
            let start = bond_features.len();
            bond_features.resize(start + BOND_FEATURE_WIDTH, 0.0);
            let in_ring = cyclic_bonds.contains(&(atom.min(target), atom.max(target)));
            let distance = positions.map_or(0.0, |p| p[atom].distance(&p[target]) as f32);
            let stereo = geometry
                .get(&(atom.min(target), atom.max(target)))
                .copied()
                .unwrap_or(0);
            write_bond_row(
                molecule,
                atom,
                target,
                bond.bond_order(),
                in_ring,
                stereo,
                distance,
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

/// The mean of every bond angle centred on `atom` (`positions` indexed like
/// `molecule`'s own atoms) -- `0.0` if it has fewer than two neighbours,
/// since no angle exists there. True of every hydrogen, and of any terminal
/// heavy atom.
fn mean_local_angle<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atom: usize,
    positions: &[Vector3D],
) -> f32 {
    let neighbours: Vec<usize> = molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .map(|bond| bond.target())
        .collect();
    if neighbours.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0f64;
    let mut count = 0u32;
    for i in 0..neighbours.len() {
        for &other in &neighbours[i + 1..] {
            if let Some(angle) =
                positions[neighbours[i]].angle_between_points(&positions[atom], &positions[other])
            {
                total += angle;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64) as f32
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

/// Fills one atom's row. See the module documentation for what the columns
/// are. `mean_angle` is this atom's mean local bond angle in radians, or
/// `0.0` when no 3D geometry was given -- see [`graph_features_with_geometry`].
/// `chirality_slot` is the atom's entry from [`tetrahedral_hands`].
fn write_atom_row<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atom: usize,
    in_ring: bool,
    mean_angle: f32,
    chirality_slot: usize,
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
        // One column for the states a molecular graph does not describe --
        // every hydrogen row lands here too, since `hybridization` gives a
        // hydrogen no shape of its own.
        Hybridization::S | Hybridization::Unspecified => 5,
    };
    one_hot(row, &mut cursor, 6, hybridization_slot);
    row[cursor] = f32::from(molecule.is_atom_aromatic(atom));
    cursor += 1;
    row[cursor] = f32::from(in_ring);
    cursor += 1;
    one_hot(row, &mut cursor, 4, chirality_slot);
    // Scaled so that every column is of the same order, which a network that
    // has not been told otherwise will assume.
    row[cursor] = atomic_number as f32 / 100.0;
    cursor += 1;
    row[cursor] = mean_angle;
}

/// Fills one directed bond's row. `distance` is the bond's 3D length in
/// angstroms, or `0.0` when no geometry was given -- see
/// [`graph_features_with_geometry`].
fn write_bond_row<M: MoleculeCore + ?Sized>(
    molecule: &M,
    first: usize,
    second: usize,
    order: BondOrder,
    in_ring: bool,
    stereo_slot: usize,
    distance: f32,
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
    one_hot(row, &mut cursor, 4, stereo_slot);
    row[cursor] = distance;
}

/// The stereo column of every double bond whose geometry the molecule states,
/// keyed by its two atoms in ascending order: 1 when the highest-ranked
/// substituents on its two ends are cis, 2 when they are trans, 3 when the
/// geometry is given but an end carries two substituents nothing tells apart.
///
/// The stored cis/trans classes are relative to whichever neighbour happens to
/// come first in each end's bond list, which is an accident of how the SMILES
/// was written -- reading the class straight into the column gave the same
/// molecule opposite features from two spellings. Re-reading it against
/// [`refined_ranks`] makes it a property of the molecule. Only the double bond
/// itself is labelled; the single bonds beside it carry no geometry.
fn double_bond_geometry<M: MoleculeCore + ?Sized>(
    molecule: &M,
    ranks: Option<&[(u64, u64)]>,
) -> rustc_hash::FxHashMap<(usize, usize), usize> {
    let mut geometry = rustc_hash::FxHashMap::default();
    let is_cis_trans = |atom: usize| {
        matches!(
            molecule.get_chiral_class(atom),
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        )
    };
    let Some(ranks) = ranks else {
        return geometry;
    };
    if !(0..molecule.len()).any(is_cis_trans) {
        return geometry;
    }
    // The substituents of a double-bond end, in bond-list order: everything
    // but the double bond itself.
    let carriers = |end: usize| -> Vec<usize> {
        molecule
            .get_atom_bonds(end)
            .unwrap_or_default()
            .iter()
            .filter(|bond| bond.bond_order() != BondOrder::Double)
            .map(|bond| bond.target())
            .collect()
    };
    // Which of them ranks highest, or `None` when the top two tie.
    let highest = |list: &[usize]| -> Option<usize> {
        let best = (0..list.len()).max_by_key(|&index| ranks[list[index]])?;
        let tied =
            (0..list.len()).any(|index| index != best && ranks[list[index]] == ranks[list[best]]);
        (!tied).then_some(best)
    };

    for atom in 0..molecule.len() {
        if !is_cis_trans(atom) {
            continue;
        }
        let Some(other) = molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .find(|bond| bond.bond_order() == BondOrder::Double)
            .map(|bond| bond.target())
        else {
            continue;
        };
        if other <= atom || !is_cis_trans(other) {
            continue;
        }
        let (first, second) = (carriers(atom), carriers(other));
        if first.is_empty() || second.is_empty() {
            continue;
        }
        // The convention the parser and canonisation keep: equal classes put
        // the first carriers of the two ends on the same side.
        let first_carriers_trans =
            molecule.get_chiral_class(atom) != molecule.get_chiral_class(other);
        let slot = match (highest(&first), highest(&second)) {
            (Some(a), Some(b)) => {
                // Taking an end's other carrier instead of its first flips
                // which side the reference sits on.
                let flips = usize::from(a != 0) + usize::from(b != 0);
                if first_carriers_trans ^ (flips % 2 == 1) {
                    2
                } else {
                    1
                }
            }
            _ => 3,
        };
        geometry.insert((atom, other), slot);
    }
    geometry
}

/// The chirality column of every atom: 0 when it states no stereo, 1 or 2
/// when it is a tetrahedral centre whose neighbours, taken highest
/// [`refined_ranks`] first, run clockwise or anticlockwise, and 3 for any
/// other stated stereo -- a centre two of whose neighbours nothing tells
/// apart, the end of a stereogenic double bond, or a non-tetrahedral class.
///
/// The stored `TH` tag is relative to the neighbours in index order, which is
/// an accident of how the molecule was written; schema 7 and before matched
/// on classes the parser never stores, and put every centre in column 3, so a
/// model could not tell an enantiomer from its mirror image. Re-reading the
/// tag against the ranks makes it a property of the molecule, the way
/// [`double_bond_geometry`] does for a double bond. A three-coordinate centre's
/// missing neighbour -- its lone pair or implicit hydrogen -- stays last in
/// both orders, as it does in the stored tag.
fn tetrahedral_hands<M: MoleculeCore + ?Sized>(
    molecule: &M,
    ranks: Option<&[(u64, u64)]>,
) -> Vec<usize> {
    let mut hands = vec![0; molecule.len()];
    let Some(ranks) = ranks else {
        return hands;
    };
    for (atom, hand) in hands.iter_mut().enumerate() {
        *hand = match molecule.get_chiral_class(atom) {
            ChiralClass::None => 0,
            ChiralClass::TH(tag @ 1..=2) => {
                let mut neighbours: Vec<usize> = molecule
                    .get_atom_bonds(atom)
                    .unwrap_or_default()
                    .iter()
                    .map(|bond| bond.target())
                    .collect();
                neighbours.sort_unstable();
                let mut by_rank: Vec<usize> = (0..neighbours.len()).collect();
                by_rank.sort_by(|&a, &b| ranks[neighbours[b]].cmp(&ranks[neighbours[a]]));
                let tied = by_rank
                    .windows(2)
                    .any(|pair| ranks[neighbours[pair[0]]] == ranks[neighbours[pair[1]]]);
                if !(3..=4).contains(&neighbours.len()) || tied {
                    3
                } else {
                    // Each inversion of the index order is one swap, and each
                    // swap turns the hand over.
                    let inversions = (0..by_rank.len())
                        .flat_map(|i| (i + 1..by_rank.len()).map(move |j| (i, j)))
                        .filter(|&(i, j)| by_rank[i] > by_rank[j])
                        .count();
                    let clockwise = (tag == 2) ^ (inversions % 2 == 1);
                    if clockwise { 1 } else { 2 }
                }
            }
            _ => 3,
        };
    }
    hands
}

/// A rank for every atom that depends on the molecule and not on its
/// numbering: element, charge, aromaticity and degree first -- so a heavier
/// substituent outranks a lighter one, as it would under CIP's first rule --
/// then refined by the ranks of the neighbours, Weisfeiler-Lehman style, until
/// a round splits no more classes.
///
/// Directional single bonds are read as single: which way a `/` leans is part
/// of how the SMILES was written, not of the molecule.
fn refined_ranks<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<(u64, u64)> {
    use crate::molecule::fingerprint::StableHasher;
    const SEED: u64 = 0x4442_5354_4552_454f;

    let count = molecule.len();
    let order_code = |order: BondOrder| -> u64 {
        match order {
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => 4,
            _ => 1,
        }
    };
    let element: Vec<u64> = (0..count)
        .map(|atom| {
            let degree = molecule.get_atom_bonds(atom).unwrap_or_default().len() as u64;
            ((molecule.get_atomic_number(atom) as u64) << 24)
                | (((molecule.get_atom_charge(atom) as i64 + 128) as u64 & 0xFF) << 16)
                | ((molecule.is_atom_aromatic(atom) as u64) << 8)
                | degree.min(0xFF)
        })
        .collect();
    let distinct = |values: &[u64]| {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };

    let mut refined = element.clone();
    let mut classes = distinct(&refined);
    for _ in 0..count {
        let next: Vec<u64> = (0..count)
            .map(|atom| {
                let mut neighbours: Vec<(u64, u64)> = molecule
                    .get_atom_bonds(atom)
                    .unwrap_or_default()
                    .iter()
                    .map(|bond| (refined[bond.target()], order_code(bond.bond_order())))
                    .collect();
                neighbours.sort_unstable();
                let mut hasher = StableHasher::new(SEED);
                hasher.write_u64(refined[atom]);
                for (rank, order) in neighbours {
                    hasher.write_u64(rank);
                    hasher.write_u64(order);
                }
                hasher.finish_u64()
            })
            .collect();
        let next_classes = distinct(&next);
        refined = next;
        if next_classes <= classes {
            break;
        }
        classes = next_classes;
    }
    element.into_iter().zip(refined).collect()
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

/// A coarse class for a heavy atom -- element, hybridization, and
/// aromaticity -- for grouping atoms whose bond to a hydrogen shares a
/// baseline dissociation energy before a model has to learn anything about
/// its specific surroundings. See [`crate::ml::normalize::ClassMeans`].
///
/// `row` is one atom's row, in `write_atom_row`'s column order: element
/// one-hot, heavy-neighbour count, formal charge, attached hydrogens,
/// hybridization, aromatic, in-ring, chirality, mass. Only the first three
/// blocks matter here -- an atom's degree, charge and exact neighbourhood are
/// exactly the finer structure a class is meant to leave for the residual to
/// learn, not bake in.
pub fn atom_class(row: &[f32]) -> u32 {
    let element = argmax(&row[ATOM_ELEMENT_COLUMNS]);
    let hybridization_start = ATOM_ELEMENT_CLASSES + 6 + 6 + 5;
    let hybridization = argmax(&row[hybridization_start..hybridization_start + 6]);
    let aromatic = row[hybridization_start + 6] > 0.5;
    (element as u32 * 6 + hybridization as u32) * 2 + u32::from(aromatic)
}

/// A coarse class for a bond between two heavy atoms -- their elements,
/// order-independent, and the bond's own order -- for the same reason as
/// [`atom_class`].
///
/// `source_atom_row`/`target_atom_row` are the two atoms' own rows;
/// `bond_row` is the directed edge's row, in `write_bond_row`'s column
/// order: order one-hot, then conjugated, in-ring, stereochemistry.
pub fn bond_class(source_atom_row: &[f32], target_atom_row: &[f32], bond_row: &[f32]) -> u32 {
    let first = argmax(&source_atom_row[ATOM_ELEMENT_COLUMNS]) as u32;
    let second = argmax(&target_atom_row[ATOM_ELEMENT_COLUMNS]) as u32;
    let (low, high) = (first.min(second), first.max(second));
    let order = argmax(&bond_row[0..4]) as u32;
    (low * ATOM_ELEMENT_CLASSES as u32 + high) * 4 + order
}

/// The position of the largest entry in a one-hot (or nearly one-hot) slice.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}
