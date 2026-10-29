//! Distances through the bond graph.
//!
//! A great deal of featurization is keyed on how many bonds apart two atoms
//! are: atom-pair and topological-torsion fingerprints, two-dimensional
//! pharmacophore keys, and the whole family of distance-sum descriptors
//! (Wiener, Balaban, the Kier-Hall shape indices). All of them want the same
//! all-pairs matrix, so it is computed once here rather than by each of them.

use crate::molecule::base::MoleculeCore;

/// The number of bonds between every pair of atoms.
///
/// Hydrogens are atoms in this crate and are included, which costs a row each
/// and changes nothing: a hydrogen is a leaf, so no shortest path between two
/// heavy atoms ever goes through one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistanceMatrix {
    size: usize,
    values: Vec<u8>,
}

impl DistanceMatrix {
    /// A matrix from a row-major `size` by `size` block of distances.
    pub fn from_values(size: usize, values: Vec<u8>) -> Self {
        assert_eq!(
            values.len(),
            size * size,
            "a distance matrix must be square"
        );
        Self { size, values }
    }

    /// The distance between two atoms, or [`DistanceMatrix::UNREACHABLE`] when
    /// they are in different fragments.
    ///
    /// Distances are capped at 254 bonds, which no molecule this crate is meant
    /// for comes close to and every consumer bins away long before.
    pub const UNREACHABLE: u8 = u8::MAX;

    /// How many atoms the matrix covers.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The distance between `first` and `second`.
    pub fn get(&self, first: usize, second: usize) -> u8 {
        self.values[first * self.size + second]
    }

    /// The distances from `atom` to every atom, in atom order.
    pub fn row(&self, atom: usize) -> &[u8] {
        &self.values[atom * self.size..(atom + 1) * self.size]
    }

    /// The sum of every distance between two distinct atoms, counted once per
    /// pair -- the Wiener index when taken over heavy atoms alone.
    pub fn distance_sum(&self, include: &[bool]) -> u64 {
        let mut total = 0u64;
        for first in 0..self.size {
            if !include.get(first).copied().unwrap_or(true) {
                continue;
            }
            for second in (first + 1)..self.size {
                if !include.get(second).copied().unwrap_or(true) {
                    continue;
                }
                let distance = self.get(first, second);
                if distance != Self::UNREACHABLE {
                    total += distance as u64;
                }
            }
        }
        total
    }
}

/// Every atom's distance to every other, by a breadth-first search from each.
///
/// `O(atoms x bonds)`, which for the sizes this crate handles is far below the
/// cost of parsing the molecule in the first place.
pub fn topological_distances<M: MoleculeCore + ?Sized>(molecule: &M) -> DistanceMatrix {
    let size = molecule.len();
    let mut values = vec![DistanceMatrix::UNREACHABLE; size * size];
    let mut queue: Vec<usize> = Vec::with_capacity(size);

    for source in 0..size {
        let row = source * size;
        values[row + source] = 0;
        queue.clear();
        queue.push(source);
        let mut head = 0;
        while head < queue.len() {
            let current = queue[head];
            head += 1;
            let distance = values[row + current];
            // The cap is the last value that is not `UNREACHABLE`, so a longer
            // path is reported as being at the cap rather than as absent.
            if distance >= DistanceMatrix::UNREACHABLE - 1 {
                continue;
            }
            let Some(bonds) = molecule.get_atom_bonds(current) else {
                continue;
            };
            for bond in bonds {
                let target = bond.target();
                if target < size && values[row + target] == DistanceMatrix::UNREACHABLE {
                    values[row + target] = distance + 1;
                    queue.push(target);
                }
            }
        }
    }

    DistanceMatrix { size, values }
}

/// The smallest ring through every ring bond, each as a cycle of atoms in the
/// order they are bonded, without duplicates.
///
/// This is the ring set a force field wants. It is not the smallest set of
/// smallest rings: SSSR has exactly `bonds - atoms + fragments` members and
/// so must drop one of the three six-membered rings of
/// bicyclo[2.2.2]octane, which then makes two of those rings "in a ring of
/// size 6" and the third not, purely by which one the algorithm happened to
/// discard. Taking the smallest ring through each bond keeps all three, which
/// is what the symmetrized ring set of every toolkit does and what a typing
/// rule reading "in a ring of size 6" means.
///
/// `O(bonds x (atoms + bonds))`, by a breadth-first search per bond.
pub fn smallest_rings<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<Vec<usize>> {
    let size = molecule.len();
    let mut rings: Vec<Vec<usize>> = Vec::new();
    let mut seen: std::collections::HashSet<Vec<usize>> = std::collections::HashSet::new();

    for first in 0..size {
        for bond in molecule.get_atom_bonds(first).unwrap_or_default() {
            let second = bond.target();
            if second <= first {
                continue;
            }
            let Some(ring) = shortest_cycle_through(molecule, first, second) else {
                continue;
            };
            let mut key = ring.clone();
            key.sort_unstable();
            if seen.insert(key) {
                rings.push(ring);
            }
        }
    }

    rings
}

/// The shortest cycle containing the bond `first`-`second`, which is that
/// bond plus the shortest path between its two atoms that avoids it.
fn shortest_cycle_through<M: MoleculeCore + ?Sized>(
    molecule: &M,
    first: usize,
    second: usize,
) -> Option<Vec<usize>> {
    let size = molecule.len();
    let mut previous = vec![usize::MAX; size];
    let mut queue = Vec::with_capacity(size);
    previous[second] = second;
    queue.push(second);
    let mut head = 0;

    while head < queue.len() {
        let current = queue[head];
        head += 1;
        for bond in molecule.get_atom_bonds(current).unwrap_or_default() {
            let target = bond.target();
            // The bond whose ring is being looked for is the one path back
            // that does not count.
            if (current == second && target == first) || (current == first && target == second) {
                continue;
            }
            if target >= size || previous[target] != usize::MAX {
                continue;
            }
            previous[target] = current;
            if target == first {
                let mut cycle = vec![first];
                let mut atom = first;
                while atom != second {
                    atom = previous[atom];
                    cycle.push(atom);
                }
                return Some(cycle);
            }
            queue.push(target);
        }
    }

    None
}

/// Optional connectivity cache. Bond orders are deliberately excluded. Each
/// clone has its own slot and shares only the immutable cached topology.
#[derive(Debug, Default)]
pub struct RingTopologyCache {
    value: std::sync::RwLock<Option<std::sync::Arc<CachedRings>>>,
}

#[derive(Debug)]
struct CachedRings {
    atom_count: usize,
    neighbours: Vec<Vec<usize>>,
    atoms: bit_set::BitSet,
    bonds: rustc_hash::FxHashSet<(usize, usize)>,
}

impl Clone for RingTopologyCache {
    fn clone(&self) -> Self {
        Self {
            value: std::sync::RwLock::new(self.value.read().unwrap().clone()),
        }
    }
}

impl RingTopologyCache {
    pub(crate) fn fill<M: MoleculeCore + ?Sized>(
        &self,
        molecule: &M,
        atoms: &mut bit_set::BitSet,
        bonds: &mut rustc_hash::FxHashSet<(usize, usize)>,
    ) {
        let matches = |cached: &CachedRings| {
            cached.atom_count == molecule.len()
                && cached.neighbours.len() == molecule.atom_bonds().len()
                && cached
                    .neighbours
                    .iter()
                    .zip(molecule.atom_bonds())
                    .all(|(old, new)| old.iter().copied().eq(new.iter().map(|bond| bond.target())))
        };
        {
            let guard = self.value.read().unwrap();
            if let Some(cached) = guard.as_ref().filter(|cached| matches(cached)) {
                atoms.clone_from(&cached.atoms);
                bonds.clone_from(&cached.bonds);
                return;
            }
        }
        // Exact comparison above also protects callers using raw adjacency
        // mutation. A hash alone would make correctness depend on collisions.
        atoms.clear();
        bonds.clear();
        molecule.walk_cyclic_bonds(|first, second| {
            atoms.insert(first);
            atoms.insert(second);
            bonds.insert(crate::molecule::utils::order_tuple(first, second));
        });
        *self.value.write().unwrap() = Some(std::sync::Arc::new(CachedRings {
            atom_count: molecule.len(),
            neighbours: molecule
                .atom_bonds()
                .iter()
                .map(|row| row.iter().map(|bond| bond.target()).collect())
                .collect(),
            atoms: atoms.clone(),
            bonds: bonds.clone(),
        }));
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::graph_algorithms::mol_canonization::reorder_molecule;
    use crate::prelude::*;

    fn agrees_with_uncached(molecule: &MolecularGraph) {
        let cached = molecule.find_cyclic_bonds_and_atoms();
        let mut uncached = molecule.clone();
        uncached.set_ring_topology_cache_enabled(false);
        assert_eq!(cached, uncached.find_cyclic_bonds_and_atoms());
    }

    #[test]
    fn ring_cache_tracks_raw_connectivity_edits_and_clones() {
        let mut molecule = MolecularGraph::from_smiles_single("C1CCCCC1CC").unwrap();
        molecule.set_ring_topology_cache_enabled(true);
        assert_eq!(molecule.find_cyclic_bonds_and_atoms().1.len(), 6);
        let original = molecule.clone();
        // Opening a ring through the raw public adjacency API must invalidate
        // the answer without relying on callers to remember a cache method.
        molecule.atom_bonds_mut()[0].retain(|bond| bond.target() != 1);
        molecule.atom_bonds_mut()[1].retain(|bond| bond.target() != 0);
        assert!(molecule.find_cyclic_bonds_and_atoms().1.is_empty());
        assert_eq!(original.find_cyclic_bonds_and_atoms().1.len(), 6);
        agrees_with_uncached(&molecule);
    }

    #[test]
    fn ring_cache_survives_bond_order_edits_and_reordering() {
        let mut molecule = MolecularGraph::from_smiles_single("c1ccccc1CC").unwrap();
        molecule.set_ring_topology_cache_enabled(true);
        let rings = molecule.find_cyclic_bonds_and_atoms();
        molecule.localize_aromatic_bonds().unwrap();
        assert_eq!(rings, molecule.find_cyclic_bonds_and_atoms());
        let order: Vec<usize> = (0..molecule.len()).rev().collect();
        reorder_molecule(&mut molecule, &order);
        agrees_with_uncached(&molecule);
    }
}
