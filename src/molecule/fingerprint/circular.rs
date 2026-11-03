//! Circular (Morgan / ECFP-style) fingerprints.
//!
//! Every heavy atom is given an identifier standing for what it is, and then,
//! round by round, an identifier standing for what it *and everything within
//! one more bond* are. After `radius` rounds each atom has named the
//! substructure centred on it out to that distance, and the molecule is the
//! multiset of every name produced along the way -- so `radius` 2, the usual
//! choice, is what other toolkits call ECFP4, counting bonds across the
//! environment rather than out from its centre.
//!
//! Two environments covering exactly the same atoms are the same substructure
//! however they were reached, so the second one to appear is dropped. Without
//! that a symmetric ring contributes one identifier per atom for an environment
//! that has already grown to swallow the whole ring.
//!
//! Hydrogens are atoms in this crate, and they take no part: a hydrogen is a
//! property of the atom it hangs off, counted in that atom's invariant, and
//! letting it be a centre of its own would say the same thing twice.
//!
//! What an identifier means depends on the aromaticity the molecule was
//! perceived with, so a fingerprint is only reproducible to the extent that
//! perception is. That is what makes the writing-independence of
//! [`crate::molecule::pi_system`] a prerequisite for this module rather than a
//! nicety.

use super::heavy::HeavyGraph;
use super::{Fingerprint, StableHasher};
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;

/// Keeps this module's identifiers out of every other fingerprint kind's space.
const CIRCULAR_SEED: u64 = 0x4543_4650_0000_0001;

/// One named substructure: which atoms it covers, and how far out it reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    /// The identifier this environment contributed.
    pub id: u32,
    /// The atom it is centred on.
    pub center: usize,
    /// How many bonds out it reached.
    pub radius: usize,
    /// Every atom it covers, in index order. Hydrogens are not listed.
    pub atoms: Vec<usize>,
}

/// A circular fingerprint, and the substructures that produced it.
///
/// The environments are what makes a bit explicable: a hit on identifier `x`
/// can be pointed at the atoms responsible for it, which is the difference
/// between a model one can argue with and a number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircularFingerprint {
    /// The identifiers and how often each occurred.
    pub fingerprint: Fingerprint,
    /// Every environment kept, in the order it was found.
    pub environments: Vec<Environment>,
}

/// A circular fingerprint of the given radius, using [`standard_invariants`].
///
/// `radius` 2 is ECFP4, `radius` 3 is ECFP6.
pub fn circular_fingerprint<M: MoleculeCore + ?Sized>(
    molecule: &M,
    radius: usize,
) -> CircularFingerprint {
    let invariants = standard_invariants(molecule);
    circular_fingerprint_with_invariants(molecule, radius, &invariants)
}

/// A circular fingerprint over invariants the caller chose.
///
/// The invariant is the whole of what a radius-zero identifier means, so
/// replacing it is how the same algorithm becomes a different fingerprint:
/// [`standard_invariants`] gives ECFP, [`feature_invariants`] gives the
/// pharmacophore-typed FCFP, and a caller with its own atom typing -- a
/// force-field type, a partial charge bin, a learned label -- can hand that in
/// instead. `invariants` is indexed by atom; entries for hydrogens are ignored.
pub fn circular_fingerprint_with_invariants<M: MoleculeCore + ?Sized>(
    molecule: &M,
    radius: usize,
    invariants: &[u32],
) -> CircularFingerprint {
    let graph = HeavyGraph::of(molecule);
    let count = graph.atoms.len();
    if count == 0 {
        return CircularFingerprint {
            fingerprint: Fingerprint::new(),
            environments: Vec::new(),
        };
    }

    // Every layer's identifiers and coverage are kept, because which of two
    // environments covering the same atoms survives is decided at the end, over
    // all of them at once.
    let words = count.div_ceil(64);
    let identifiers = layer_identifiers(&graph, radius, invariants);
    let coverage = layer_coverage(&graph, radius);

    // Two environments covering the same atoms are the same substructure
    // however they were reached, and only one of them may be named. Which one
    // cannot be decided by atom index: the indices are an accident of how the
    // molecule was written, and choosing by them made one compound fingerprint
    // sixteen different ways. The earliest layer wins, and then the smallest
    // identifier -- both properties of the structure rather than of the string.
    //
    // Finding the duplicates by sorting the coverages themselves means
    // comparing whole bit sets; sorting a cheap summary of each instead brings
    // equal coverages together, and only within a run of equal summaries does
    // anything have to be compared in full. A summary that collides says
    // nothing, since the exact comparison still decides.
    let mut summaries: Vec<(u64, u32)> = (0..(count * (radius + 1)) as u32)
        .map(|entry| {
            let slice = &coverage[entry as usize * words..(entry as usize + 1) * words];
            let mut summary = slice.iter().fold(0u64, |state, &word| {
                super::mix64(state ^ word)
            });
            // The population count is the one part of a coverage that cannot
            // collide by accident, so it leads.
            summary = (slice.iter().map(|word| word.count_ones()).sum::<u32>() as u64) << 48
                | (summary >> 16);
            (summary, entry)
        })
        .collect();
    summaries.sort_unstable();

    let mut environments: Vec<Environment> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut taken = vec![false; count * (radius + 1)];
    let mut position = 0;
    while position < summaries.len() {
        let mut end = position + 1;
        while end < summaries.len() && summaries[end].0 == summaries[position].0 {
            end += 1;
        }
        for outer in position..end {
            let entry = summaries[outer].1 as usize;
            if taken[entry] {
                continue;
            }
            let mut best = entry;
            for inner in summaries[outer..end].iter() {
                let other = inner.1 as usize;
                if taken[other]
                    || coverage[other * words..(other + 1) * words]
                        != coverage[entry * words..(entry + 1) * words]
                {
                    continue;
                }
                taken[other] = true;
                // Earliest layer first, then smallest identifier.
                if (other / count, identifiers[other]) < (best / count, identifiers[best]) {
                    best = other;
                }
            }
            ids.push(identifiers[best]);
            environments.push(Environment {
                id: identifiers[best],
                center: graph.atoms[best % count],
                radius: best / count,
                atoms: graph.decode(&coverage[best * words..(best + 1) * words]),
            });
        }
        position = end;
    }
    // The environments came out in summary order, which is meaningless to a
    // reader; centre and radius are what they will look them up by.
    environments.sort_unstable_by_key(|environment| (environment.radius, environment.center));

    CircularFingerprint {
        fingerprint: Fingerprint::from_ids(ids.iter().copied()),
        environments,
    }
}

/// Every atom's identifier at every layer from zero to `radius`, laid out layer
/// by layer: entry `layer * vertices + vertex`.
///
/// This is the whole of the Morgan relabelling, and it is what makes the
/// identifier a property of the substructure rather than of the string -- the
/// neighbours are folded in sorted, so the order the molecule's bond list holds
/// them in cannot be seen in the result. Which environments the *molecule* is
/// then said to contain is a separate question, answered by
/// [`circular_fingerprint_with_invariants`] over the coverages; a caller that
/// wants the atom-wise labels themselves -- a minhashed shingle fingerprint,
/// an atom-resolved model input -- wants these.
pub(super) fn layer_identifiers(
    graph: &HeavyGraph,
    radius: usize,
    invariants: &[u32],
) -> Vec<u32> {
    let count = graph.len();
    let mut identifiers: Vec<u32> = Vec::with_capacity(count * (radius + 1));
    for &atom in &graph.atoms {
        identifiers.push(invariants.get(atom).copied().unwrap_or(0));
    }

    let mut neighbourhood: Vec<u64> = Vec::new();
    for layer in 1..=radius {
        let previous = (layer - 1) * count;
        for vertex in 0..count {
            neighbourhood.clear();
            for edge in graph.edges(vertex) {
                // Bond order and neighbour identifier in one word: sorted
                // together, hashed together, and no chance of a bond order
                // being mistaken for an identifier on the way in.
                neighbourhood.push(((edge.0 as u64) << 32) | identifiers[previous + edge.1] as u64);
            }
            // The neighbours come in whatever order the atom's bond list holds
            // them, which is the order the molecule was written in. Sorting is
            // what makes the identifier a property of the substructure.
            neighbourhood.sort_unstable();

            let mut hasher = StableHasher::new(CIRCULAR_SEED);
            hasher.write_u32(layer as u32);
            hasher.write_u32(identifiers[previous + vertex]);
            for &neighbour in neighbourhood.iter() {
                hasher.write_u64(neighbour);
            }
            identifiers.push(hasher.finish_u32());
        }
    }
    identifiers
}

/// Which atoms each layer's environment covers, as a bit set per environment,
/// laid out to match [`layer_identifiers`].
fn layer_coverage(graph: &HeavyGraph, radius: usize) -> Vec<u64> {
    let count = graph.len();
    let words = count.div_ceil(64);
    let mut coverage: Vec<u64> = vec![0; count * (radius + 1) * words];
    for vertex in 0..count {
        coverage[vertex * words + vertex / 64] |= 1u64 << (vertex % 64);
    }

    for layer in 1..=radius {
        let previous = (layer - 1) * count;
        // The environment grows by one shell, which is the union of what its
        // neighbours reached last time.
        let (older, newer) = coverage.split_at_mut(layer * count * words);
        for vertex in 0..count {
            let target = &mut newer[vertex * words..(vertex + 1) * words];
            target.copy_from_slice(&older[(previous + vertex) * words..(previous + vertex + 1) * words]);
            for edge in graph.edges(vertex) {
                let neighbour = &older[(previous + edge.1) * words..(previous + edge.1 + 1) * words];
                for (into, from) in target.iter_mut().zip(neighbour.iter()) {
                    *into |= from;
                }
            }
        }
    }
    coverage
}

/// The invariant every other toolkit's ECFP is built on: what an atom is,
/// stated without reference to anything beyond its own bonds.
///
/// Six numbers -- how many heavy neighbours it has, how much of its valence
/// goes to them, which element it is, its charge, how many hydrogens it
/// carries, and whether it is in a ring. Isotope is absent, since
/// two samples of the same compound should fingerprint alike; a caller who
/// wants labelled atoms distinguished can build an invariant that says so.
pub fn standard_invariants<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<u32> {
    let atom_count = molecule.len();
    let cyclic_atoms = molecule.cyclic_atoms();

    (0..atom_count)
        .map(|atom| {
            let atomic_number = molecule.get_atomic_number(atom);
            if atomic_number == 1 {
                return 0;
            }
            let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
            let mut heavy_degree = 0u32;
            let mut heavy_valence = 0u32;
            let mut hydrogens = 0u32;
            for bond in bonds {
                if molecule.get_atomic_number(bond.target()) == 1 {
                    hydrogens += 1;
                } else {
                    heavy_degree += 1;
                    heavy_valence += bond_code(bond.bond_order()) as u32;
                }
            }

            let mut hasher = StableHasher::new(CIRCULAR_SEED);
            hasher.write_u32(heavy_degree);
            hasher.write_u32(heavy_valence);
            hasher.write_u32(atomic_number as u32);
            hasher.write_u32(molecule.get_atom_charge(atom) as i32 as u32);
            hasher.write_u32(hydrogens);
            hasher.write_u8(u8::from(cyclic_atoms.contains(atom)));
            hasher.finish_u32()
        })
        .collect()
}

/// The invariant FCFP is built on: not what an atom is, but what it can do.
///
/// Six flags -- hydrogen-bond donor, hydrogen-bond acceptor, aromatic,
/// halogen, basic, acidic -- so that a chloro and a bromo substituent, or an
/// ester and an amide oxygen, are the same feature. This generalises across
/// series where [`standard_invariants`] does not, at the cost of telling two
/// different molecules apart less often.
///
/// The definitions are structural and simple, and specific to this crate: they
/// are not taken from another toolkit's SMARTS and will not agree with one bit
/// for bit.
pub fn feature_invariants<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<u32> {
    let atom_count = molecule.len();

    (0..atom_count)
        .map(|atom| {
            let atomic_number = molecule.get_atomic_number(atom);
            if atomic_number == 1 {
                return 0;
            }
            let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
            let charge = molecule.get_atom_charge(atom);
            let aromatic = molecule.is_atom_aromatic(atom);
            let hydrogens = bonds
                .iter()
                .filter(|bond| molecule.get_atomic_number(bond.target()) == 1)
                .count();
            let heavy_degree = bonds.len() - hydrogens;
            let has_double = bonds
                .iter()
                .any(|bond| bond.bond_order() == BondOrder::Double);

            let nitrogen_or_oxygen = matches!(atomic_number, 7 | 8);
            let donor = nitrogen_or_oxygen && hydrogens > 0;
            // A lone pair to offer, and not already spending it: a quaternary
            // nitrogen has none left, and a positive one is not offering.
            let acceptor = nitrogen_or_oxygen && charge <= 0 && heavy_degree + hydrogens <= 3;
            let halogen = matches!(atomic_number, 9 | 17 | 35 | 53 | 85);
            // Basic in the sense that matters for binding: a nitrogen holding a
            // lone pair that is not delocalised into a ring or a carbonyl.
            let basic = atomic_number == 7 && charge >= 0 && !aromatic && !has_double && {
                !bonds.iter().any(|bond| {
                    let target = bond.target();
                    molecule.get_atomic_number(target) == 6
                        && molecule
                            .get_atom_bonds(target)
                            .unwrap_or_default()
                            .iter()
                            .any(|neighbour| {
                                neighbour.bond_order() == BondOrder::Double
                                    && matches!(molecule.get_atomic_number(neighbour.target()), 7 | 8 | 16)
                            })
                })
            };
            // An acidic hydroxyl: an -OH on a centre that also carries a double
            // bond to oxygen, which covers carboxylic, sulfonic and phosphonic
            // acids alike.
            let acidic = (atomic_number == 8 && hydrogens > 0 && {
                bonds.iter().any(|bond| {
                    let target = bond.target();
                    matches!(molecule.get_atomic_number(target), 6 | 15 | 16)
                        && molecule
                            .get_atom_bonds(target)
                            .unwrap_or_default()
                            .iter()
                            .any(|neighbour| {
                                neighbour.bond_order() == BondOrder::Double
                                    && molecule.get_atomic_number(neighbour.target()) == 8
                            })
                })
            }) || (atomic_number == 8 && charge < 0);

            let flags = u32::from(donor)
                | u32::from(acceptor) << 1
                | u32::from(aromatic) << 2
                | u32::from(halogen) << 3
                | u32::from(basic) << 4
                | u32::from(acidic) << 5;

            let mut hasher = StableHasher::new(CIRCULAR_SEED);
            hasher.write_u32(flags);
            hasher.finish_u32()
        })
        .collect()
}

/// How much of an atom's valence a bond takes, as a small number that is stable
/// across writings.
///
/// An aromatic bond gets its own code rather than being rounded to one or two:
/// it is neither, and rounding it would make a benzene carbon and a butadiene
/// carbon compare equal.
pub(super) fn bond_code(order: BondOrder) -> u8 {
    match order {
        BondOrder::Single | BondOrder::Up | BondOrder::Down | BondOrder::Coordinate => 1,
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        BondOrder::Quadruple => 4,
        BondOrder::Aromatic => 5,
    }
}
