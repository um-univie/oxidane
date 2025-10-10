//! Minhashed shingle fingerprints: MHFP and MAP4.
//!
//! Both answer the same complaint. A folded bit vector has a fixed number of
//! bits and a molecule has as many identifiers as it has atoms, so past a few
//! hundred heavy atoms the vector is saturated and two unrelated peptides look
//! alike -- precisely where a fingerprint was supposed to help. Sketching the
//! *unfolded* identifier set with [`MinHash`] instead keeps a fixed number of
//! numbers without ever letting two identifiers collide into one bit, and the
//! agreement between two sketches estimates the Jaccard index of the sets
//! themselves.
//!
//! [`mhfp_fingerprint`] sketches the circular environments. [`map4_fingerprint`]
//! sketches something strictly richer: every *pair* of atoms, named by the
//! circular environment of each end and the number of bonds between them. That
//! is the atom-pair idea and the circular idea in one shingle, and it is what
//! makes MAP4 work across the range where neither alone does -- small molecules,
//! where the pair distances carry the shape, and large ones, where the
//! environments do.
//!
//! Where these differ from the published fingerprints is in how a substructure
//! is named. Probst and Reymond name it by the canonical SMILES of the
//! subgraph; here it is named by the Morgan identifier of
//! [`super::circular_fingerprint`], which is the same information arrived at
//! without writing a string and reading it back. Two molecules agree on a
//! shingle under one naming exactly when they agree under the other, so the
//! Jaccard estimates match; the individual numbers do not, and a sketch from
//! this crate must not be compared against one from `mhfp` or `map4`.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::fingerprint::{map4_fingerprint, mhfp_fingerprint};
//!
//! let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
//! let salicylic = MolecularGraph::from_smiles_single("OC(=O)c1ccccc1O").unwrap();
//! let unrelated = MolecularGraph::from_smiles_single("CCCCCCCCCCCC").unwrap();
//!
//! let first = map4_fingerprint(&aspirin, 2, 30, 512);
//! let second = map4_fingerprint(&salicylic, 2, 30, 512);
//! let third = map4_fingerprint(&unrelated, 2, 30, 512);
//! assert!(first.jaccard(&second) > first.jaccard(&third));
//!
//! // The same molecule written another way sketches identically.
//! let rewritten = MolecularGraph::from_smiles_single("O=C(O)c1ccccc1OC(C)=O").unwrap();
//! assert_eq!(map4_fingerprint(&rewritten, 2, 30, 512), first);
//! assert_eq!(mhfp_fingerprint(&rewritten, 3, 512), mhfp_fingerprint(&aspirin, 3, 512));
//! ```

use super::circular::{layer_identifiers, standard_invariants};
use super::heavy::HeavyGraph;
use super::minhash::MinHash;
use super::{Fingerprint, StableHasher};
use crate::molecule::base::MoleculeCore;
use crate::molecule::topology::DistanceMatrix;

/// Keeps this kind's identifiers out of every other's space.
const MAP4_SEED: u64 = 0x4d41_5034_0000_0001;

/// The radius MAP4 was published with: environments of one and two bonds.
pub const DEFAULT_MAP4_RADIUS: usize = 2;

/// Every pair of heavy atoms, named by the circular environment of each end and
/// the distance between them, before sketching.
///
/// One shingle per pair per layer from one out to `radius`. Layer zero is left
/// out because a pair of bare atom invariants at a distance is
/// [`super::atom_pair_fingerprint`], which exists in its own right; the point
/// of this fingerprint is that each end brings its surroundings with it.
///
/// Pairs further apart than `max_distance` are left out, as are pairs in
/// different fragments. The cap matters more here than for atom pairs: the
/// number of pairs grows as the square of the molecule, and this is the
/// fingerprint meant for the molecules big enough for that to hurt.
pub fn map4_shingles<M: MoleculeCore + ?Sized>(
    molecule: &M,
    radius: usize,
    max_distance: usize,
) -> Fingerprint {
    let graph = HeavyGraph::of(molecule);
    let count = graph.len();
    if count == 0 || radius == 0 {
        return Fingerprint::new();
    }

    let invariants = standard_invariants(molecule);
    let identifiers = layer_identifiers(&graph, radius, &invariants);
    let distances = graph.distances();

    let mut ids: Vec<u32> = Vec::with_capacity(count * count / 2 * radius);
    for first in 0..count {
        let row = distances.row(first);
        for second in (first + 1)..count {
            let distance = row[second];
            if distance == DistanceMatrix::UNREACHABLE || distance as usize > max_distance {
                continue;
            }
            for layer in 1..=radius {
                let base = layer * count;
                ids.push(shingle_id(
                    identifiers[base + first],
                    identifiers[base + second],
                    distance,
                    layer,
                ));
            }
        }
    }

    Fingerprint::from_ids(ids)
}

/// A MAP4 sketch of `permutations` slots.
///
/// `radius` 2 and `max_distance` 30 are the published settings; 512 or 1024
/// permutations are the usual lengths, and the estimate's error falls as one
/// over their square root.
pub fn map4_fingerprint<M: MoleculeCore + ?Sized>(
    molecule: &M,
    radius: usize,
    max_distance: usize,
    permutations: usize,
) -> MinHash {
    MinHash::of(&map4_shingles(molecule, radius, max_distance), permutations)
}

/// The identifier of one pair shingle.
///
/// The two environments are ordered before hashing, since a pair has no
/// direction, and the layer goes in alongside the distance so that a pair of
/// one-bond environments three apart cannot be confused with a pair of two-bond
/// environments.
pub fn shingle_id(first: u32, second: u32, distance: u8, radius: usize) -> u32 {
    let mut hasher = StableHasher::new(MAP4_SEED);
    hasher.write_u64((first.min(second) as u64) << 32 | first.max(second) as u64);
    hasher.write_u32(distance as u32);
    hasher.write_u32(radius as u32);
    hasher.finish_u32()
}

/// An MHFP sketch of `permutations` slots: the circular environments out to
/// `radius`, minhashed.
///
/// `radius` 3 is what the paper calls MHFP6, counting bonds across an
/// environment rather than out from its centre, and is the published default.
///
/// The sketch ignores how often an environment occurs, so this is the set of
/// substructures a molecule contains rather than its composition -- which is
/// the whole point when comparing molecules of very different sizes.
pub fn mhfp_fingerprint<M: MoleculeCore + ?Sized>(
    molecule: &M,
    radius: usize,
    permutations: usize,
) -> MinHash {
    MinHash::of(
        &super::circular_fingerprint(molecule, radius).fingerprint,
        permutations,
    )
}
