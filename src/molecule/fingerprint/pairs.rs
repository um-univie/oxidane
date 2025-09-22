//! Atom-pair and topological-torsion fingerprints.
//!
//! Where a circular fingerprint says *what surrounds this atom*, these two say
//! *how these two atoms are arranged relative to each other*. That makes them
//! complementary rather than competing: a circular fingerprint of a molecule
//! whose scaffold has been rearranged looks entirely different, while the
//! distances between its features barely move.
//!
//! Both are built on Carhart's atom type -- element, how many π electrons the
//! atom holds, and how many heavy neighbours it has -- which is coarse enough
//! that an ester oxygen and an ether oxygen are told apart by their
//! surroundings rather than by a table.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::fingerprint::{atom_pair_fingerprint, topological_torsion_fingerprint};
//!
//! let propane = MolecularGraph::from_smiles_single("CCC").unwrap();
//! // Two end-to-middle pairs one bond apart, and one end-to-end pair two
//! // bonds apart.
//! assert_eq!(atom_pair_fingerprint(&propane, 30).total_count(), 3);
//! // A torsion needs four atoms in a row, and propane has three.
//! assert!(topological_torsion_fingerprint(&propane).is_empty());
//! ```

use super::heavy::HeavyGraph;
use super::{Fingerprint, StableHasher};
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::topology::DistanceMatrix;

/// Keeps each kind's identifiers out of the others' space.
const ATOM_PAIR_SEED: u64 = 0x4150_4149_0000_0001;
const TORSION_SEED: u64 = 0x544f_5253_0000_0001;

/// How many bonds apart two atoms may be and still be counted as a pair.
///
/// Beyond about thirty bonds a pair says almost nothing that the pairs inside
/// it have not already said, and the count of them grows as the square of the
/// molecule.
pub const DEFAULT_MAX_DISTANCE: usize = 30;

/// Carhart's atom type: what an atom is, in the three numbers a pair or a
/// torsion needs.
///
/// Element, π electron count and heavy-neighbour count, packed so that the
/// value can be compared and sorted directly. An atom in an aromatic ring
/// holds one π electron however many aromatic bonds meet at it, which is what
/// makes a benzene carbon and an alkene carbon the same type -- as they should
/// be, since nothing a pair records tells them apart.
pub fn atom_pair_types<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<u32> {
    let atom_count = molecule.len();
    (0..atom_count)
        .map(|atom| {
            let atomic_number = molecule.get_atomic_number(atom);
            if atomic_number == 1 {
                return 0;
            }
            let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
            let mut heavy_degree = 0u32;
            let mut pi_electrons = 0u32;
            let mut aromatic = false;
            for bond in bonds {
                if molecule.get_atomic_number(bond.target()) == 1 {
                    continue;
                }
                heavy_degree += 1;
                match bond.bond_order() {
                    BondOrder::Double => pi_electrons += 1,
                    BondOrder::Triple => pi_electrons += 2,
                    BondOrder::Quadruple => pi_electrons += 3,
                    // An atom puts *one* p orbital into the ring it is part of,
                    // however many aromatic bonds meet at it. Counting one per
                    // bond gave a benzene carbon two π electrons and made it a
                    // different type from the alkene carbon it is otherwise
                    // indistinguishable from.
                    BondOrder::Aromatic => aromatic = true,
                    _ => (),
                }
            }
            if aromatic {
                pi_electrons += 1;
            }
            (atomic_number as u32) << 8 | pi_electrons.min(7) << 4 | heavy_degree.min(15)
        })
        .collect()
}

/// Every pair of heavy atoms, as their two types and the number of bonds
/// between them.
///
/// Pairs further apart than `max_distance` are left out, as are pairs in
/// different fragments, which are at no distance from each other at all.
pub fn atom_pair_fingerprint<M: MoleculeCore + ?Sized>(
    molecule: &M,
    max_distance: usize,
) -> Fingerprint {
    let graph = HeavyGraph::of(molecule);
    let types = atom_pair_types(molecule);
    let distances = graph.distances();
    let count = graph.len();
    let mut ids: Vec<u32> = Vec::with_capacity(count * count / 2);

    for first in 0..count {
        for second in (first + 1)..count {
            let distance = distances.get(first, second);
            if distance == DistanceMatrix::UNREACHABLE || distance as usize > max_distance {
                continue;
            }
            ids.push(pair_id(
                types[graph.atoms[first]],
                types[graph.atoms[second]],
                distance,
            ));
        }
    }

    Fingerprint::from_ids(ids)
}

/// The identifier of one pair, from the two atom types and their distance.
///
/// The types are ordered before hashing, since a pair has no direction.
pub fn pair_id(first: u32, second: u32, distance: u8) -> u32 {
    let mut hasher = StableHasher::new(ATOM_PAIR_SEED);
    hasher.write_u64((first.min(second) as u64) << 32 | first.max(second) as u64);
    hasher.write_u32(distance as u32);
    hasher.finish_u32()
}

/// Every path of four heavy atoms joined in a row.
///
/// Four is the classic length, one for each atom of a torsion angle, and it is
/// what makes this a shape descriptor: a chain that can fold and a ring that
/// cannot contain quite different sets of them even when their atom pairs
/// agree.
pub fn topological_torsion_fingerprint<M: MoleculeCore + ?Sized>(molecule: &M) -> Fingerprint {
    let graph = HeavyGraph::of(molecule);
    let types = atom_pair_types(molecule);
    let mut ids: Vec<u32> = Vec::new();

    // Every four-atom path has exactly one middle bond, so walking the bonds
    // and hanging an atom off each end generates each path once and no path
    // twice.
    for second in 0..graph.len() {
        for &(_, third) in graph.edges(second) {
            if third <= second {
                continue;
            }
            for &(_, first) in graph.edges(second) {
                if first == third {
                    continue;
                }
                for &(_, fourth) in graph.edges(third) {
                    // Four atoms, not three: a path that comes back on itself
                    // is a ring bond seen twice rather than a torsion.
                    if fourth == second || fourth == first {
                        continue;
                    }
                    ids.push(torsion_id([
                        types[graph.atoms[first]],
                        types[graph.atoms[second]],
                        types[graph.atoms[third]],
                        types[graph.atoms[fourth]],
                    ]));
                }
            }
        }
    }

    Fingerprint::from_ids(ids)
}

/// The identifier of one four-atom path.
///
/// A path read backwards is the same path, so whichever direction gives the
/// smaller sequence of types is the one hashed.
pub fn torsion_id(types: [u32; 4]) -> u32 {
    let reversed = [types[3], types[2], types[1], types[0]];
    let canonical = if reversed < types { reversed } else { types };
    let mut hasher = StableHasher::new(TORSION_SEED);
    for atom_type in canonical {
        hasher.write_u32(atom_type);
    }
    hasher.finish_u32()
}
