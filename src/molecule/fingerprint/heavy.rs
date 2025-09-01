//! The heavy-atom graph every fingerprint in this module is built on.

use super::circular::bond_code;
use crate::molecule::base::MoleculeCore;
use crate::molecule::topology::DistanceMatrix;

/// The molecule's heavy atoms and the bonds between them, in their own index
/// space.
///
/// Hydrogens are atoms in this crate and take no part in any of the graph
/// fingerprints, so walking the molecule's own indexing means skipping about
/// half of it on every pass -- a thousand atoms per twenty molecules against
/// five hundred and forty. Reindexing once pays for itself immediately, and
/// gathering the bond orders at the same time takes the bond-order match out of
/// the inner loop.
pub(super) struct HeavyGraph {
    /// Heavy atom index in the molecule, by vertex.
    pub(super) atoms: Vec<usize>,
    /// Where each vertex's neighbours start in `targets`.
    offsets: Vec<u32>,
    /// `(bond code, neighbouring vertex)`, grouped by vertex.
    targets: Vec<(u8, usize)>,
}

impl HeavyGraph {
    pub(super) fn of<M: MoleculeCore + ?Sized>(molecule: &M) -> Self {
        let atom_count = molecule.len();
        let mut vertex_of = vec![usize::MAX; atom_count];
        let mut atoms = Vec::new();
        for atom in 0..atom_count {
            if molecule.get_atomic_number(atom) != 1 {
                vertex_of[atom] = atoms.len();
                atoms.push(atom);
            }
        }

        let mut offsets = Vec::with_capacity(atoms.len() + 1);
        let mut targets = Vec::new();
        for &atom in &atoms {
            offsets.push(targets.len() as u32);
            for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
                let target = bond.target();
                if target < atom_count && vertex_of[target] != usize::MAX {
                    targets.push((bond_code(bond.bond_order()), vertex_of[target]));
                }
            }
        }
        offsets.push(targets.len() as u32);

        Self {
            atoms,
            offsets,
            targets,
        }
    }

    pub(super) fn edges(&self, vertex: usize) -> &[(u8, usize)] {
        &self.targets[self.offsets[vertex] as usize..self.offsets[vertex + 1] as usize]
    }

    /// The molecule's own indices for the atoms a coverage word set names.
    pub(super) fn decode(&self, words: &[u64]) -> Vec<usize> {
        let mut atoms = Vec::new();
        for (index, &word) in words.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                atoms.push(self.atoms[index * 64 + bit]);
            }
        }
        atoms
    }
}

impl HeavyGraph {
    /// How many heavy atoms there are.
    pub(super) fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Every heavy atom's distance to every other, in vertex space.
    ///
    /// Not [`crate::molecule::topology::topological_distances`], which is over
    /// the molecule's own indexing: nearly half of that matrix is hydrogens,
    /// which no fingerprint asks about and which cost the square of their
    /// number to include.
    pub(super) fn distances(&self) -> DistanceMatrix {
        let size = self.len();
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
                if distance >= DistanceMatrix::UNREACHABLE - 1 {
                    continue;
                }
                for &(_, target) in self.edges(current) {
                    if values[row + target] == DistanceMatrix::UNREACHABLE {
                        values[row + target] = distance + 1;
                        queue.push(target);
                    }
                }
            }
        }

        DistanceMatrix::from_values(size, values)
    }
}
