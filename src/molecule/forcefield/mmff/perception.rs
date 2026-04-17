//! What MMFF94 sees before it types anything: Kekule bond orders, a ring
//! set, and its own idea of which of those rings are aromatic.
//!
//! MMFF does not take a toolkit's aromaticity model. It counts pi electrons
//! ring by ring over an explicitly Kekulized structure, adding two for each
//! ring double bond, one for a double bond into an already-aromatic
//! neighbour, and two more for a lone pair when an odd-sized ring holds a
//! nitrogen, oxygen or divalent sulfur and has no exocyclic double bond.
//! Rings satisfying 4n+2 are aromatic; because a ring's answer can depend on
//! a fused neighbour's, the sweep repeats until it stops learning anything.
//!
//! The distinction matters. MMFF's rules deliberately exclude structures
//! this crate's own perception calls aromatic -- most visibly the exocyclic
//! double bond case, where MMFF wants tropone and its relatives typed as
//! ordinary conjugated systems -- and typing against the wrong aromaticity
//! moves atoms between whole families of types.

use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::featurize::{Hybridization, hybridization};
use crate::molecule::pi_system::kekulize;
use crate::molecule::topology::smallest_rings;
use rustc_hash::FxHashSet;

/// A neighbouring atom and the bond order MMFF reads on the way to it --
/// Kekule everywhere except in a ring MMFF has decided is aromatic, whose
/// bonds it flattens to [`BondOrder::Aromatic`] before typing runs, exactly
/// as RDKit does.
pub(super) struct Neighbour {
    pub atom: usize,
    pub order: BondOrder,
}

pub(super) struct Perception {
    atomic_numbers: Vec<u8>,
    hybridizations: Vec<Hybridization>,
    charges: Vec<i8>,
    neighbours: Vec<Vec<Neighbour>>,
    /// Sum of the Kekule bond orders at each atom, which is the valence the
    /// typing rules mean by "total bond order".
    valence: Vec<u32>,
    rings: Vec<Vec<usize>>,
    ring_is_aromatic: Vec<bool>,
    atom_is_aromatic: Vec<bool>,
    in_ring: Vec<bool>,
}

fn order_value(order: BondOrder) -> u32 {
    match order {
        BondOrder::Double => 2,
        BondOrder::Triple => 3,
        _ => 1,
    }
}

impl Perception {
    pub(super) fn new<M: MoleculeCore + ?Sized>(molecule: &M) -> Self {
        let count = molecule.len();

        let mut aromatic_bonds = FxHashSet::default();
        for atom in 0..count {
            for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
                if atom < bond.target() && bond.bond_order() == BondOrder::Aromatic {
                    aromatic_bonds.insert((atom, bond.target()));
                }
            }
        }
        let kekulization = kekulize(molecule, &aromatic_bonds);

        let mut neighbours: Vec<Vec<Neighbour>> = Vec::with_capacity(count);
        let mut valence = Vec::with_capacity(count);
        for atom in 0..count {
            let mut row = Vec::new();
            let mut total = 0;
            for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
                let target = bond.target();
                let order = match bond.bond_order() {
                    // A run of aromatic bonds with no Kekule structure is
                    // left as it was found -- there is no honest single or
                    // double answer for it, and typing will fail loudly on
                    // the aromatic order rather than quietly on a guess.
                    BondOrder::Aromatic if !kekulization.is_atom_deficient(atom) => {
                        let paired = kekulization
                            .matched
                            .get(atom)
                            .and_then(|partner| *partner)
                            .is_some_and(|partner| partner == target);
                        if paired { BondOrder::Double } else { BondOrder::Single }
                    }
                    other => other,
                };
                total += order_value(order);
                row.push(Neighbour { atom: target, order });
            }
            valence.push(total);
            neighbours.push(row);
        }

        let rings = smallest_rings(molecule);
        let mut in_ring = vec![false; count];
        for ring in &rings {
            for &atom in ring {
                in_ring[atom] = true;
            }
        }

        let mut perception = Self {
            atomic_numbers: (0..count).map(|atom| molecule.get_atomic_number(atom)).collect(),
            charges: molecule.charges().to_vec(),
            hybridizations: (0..count).map(|atom| hybridization(molecule, atom)).collect(),
            neighbours,
            valence,
            rings,
            ring_is_aromatic: Vec::new(),
            atom_is_aromatic: vec![false; count],
            in_ring,
        };
        perception.perceive_aromaticity(molecule);
        perception
    }

    /// MMFF's own aromaticity sweep, and the bond-order flattening that
    /// follows it.
    fn perceive_aromaticity<M: MoleculeCore + ?Sized>(&mut self, molecule: &M) {
        let count = molecule.len();
        self.ring_is_aromatic = vec![false; self.rings.len()];
        // Whether a ring's verdict has been reached at all, as opposed to
        // whether it was aromatic: a ring whose fused neighbour is still
        // undecided has to wait for it, and this is what says so.
        let mut settled = vec![false; count];
        let ring_places: usize = self.rings.iter().map(|ring| ring.len()).sum();
        let mut decided = 0usize;
        let mut previously_decided = usize::MAX;

        while decided != previously_decided && decided < ring_places {
            previously_decided = decided;
            for index in 0..self.rings.len() {
                let Some(electrons) = self.ring_pi_electrons(index, &settled) else {
                    continue;
                };
                let mut can_be_aromatic = true;
                for &atom in &self.rings[index] {
                    settled[atom] = true;
                    let atomic_number = molecule.get_atomic_number(atom);
                    if matches!(atomic_number, 6 | 7)
                        && hybridization(molecule, atom) != Hybridization::Sp2
                    {
                        can_be_aromatic = false;
                    }
                }
                if can_be_aromatic && electrons > 2 && (electrons - 2) % 4 == 0 {
                    self.ring_is_aromatic[index] = true;
                    for &atom in &self.rings[index] {
                        self.atom_is_aromatic[atom] = true;
                    }
                }
            }
            decided = self
                .rings
                .iter()
                .flatten()
                .filter(|&&atom| settled[atom])
                .count();
        }

        // Only now, with every ring decided, do the aromatic rings lose
        // their alternation: the typing rules ask whether a bond is double,
        // and inside an aromatic ring the answer MMFF wants is neither.
        for (index, ring) in self.rings.iter().enumerate() {
            if !self.ring_is_aromatic[index] {
                continue;
            }
            for position in 0..ring.len() {
                let first = ring[position];
                let second = ring[(position + 1) % ring.len()];
                for neighbour in &mut self.neighbours[first] {
                    if neighbour.atom == second {
                        neighbour.order = BondOrder::Aromatic;
                    }
                }
                for neighbour in &mut self.neighbours[second] {
                    if neighbour.atom == first {
                        neighbour.order = BondOrder::Aromatic;
                    }
                }
            }
        }
    }

    /// The pi electron count of one ring, or `None` when it depends on a
    /// ring whose own verdict has not been reached yet.
    fn ring_pi_electrons(&self, index: usize, settled: &[bool]) -> Option<u32> {
        let ring = &self.rings[index];
        let mut electrons = 0u32;
        let mut heteroatom_in_ring = false;
        let mut exocyclic_double_bond = false;

        for (position, &atom) in ring.iter().enumerate() {
            let atomic_number = self.atomic_numbers[atom];
            if matches!(atomic_number, 7 | 8)
                || (atomic_number == 16 && self.neighbours[atom].len() == 2)
            {
                heteroatom_in_ring = true;
            }

            let next = ring[(position + 1) % ring.len()];
            if self.order(atom, next) == BondOrder::Double {
                electrons += 2;
                continue;
            }

            // A ring bond that is not double contributes nothing itself, but
            // the atom may still bring in an electron from outside the ring
            // -- and only carbon, or a nitrogen at four bonds, can.
            if atomic_number != 6 && !(atomic_number == 7 && self.valence[atom] == 4) {
                continue;
            }
            for neighbour in &self.neighbours[atom] {
                if ring.contains(&neighbour.atom) || neighbour.order == BondOrder::Single {
                    continue;
                }
                if self.in_ring[neighbour.atom] && !settled[neighbour.atom] {
                    return None;
                }
                if neighbour.order == BondOrder::Double {
                    if self.atom_is_aromatic[neighbour.atom] {
                        electrons += 1;
                    } else {
                        exocyclic_double_bond = true;
                    }
                }
            }
        }

        // The lone pair of a pyrrole-like ring: only where nothing exocyclic
        // is competing for the same electrons, and only in an odd ring,
        // which is MMFF's way of saying five-membered.
        if heteroatom_in_ring && !exocyclic_double_bond && ring.len() % 2 == 1 {
            electrons += 2;
        }
        Some(electrons)
    }

    /// The shape this crate perceives for the atom, which is what decides
    /// whether a bond is one MMFF hangs torsions off.
    pub(super) fn shape(&self, atom: usize) -> Hybridization {
        self.hybridizations[atom]
    }

    pub(super) fn element(&self, atom: usize) -> u8 {
        self.atomic_numbers[atom]
    }

    pub(super) fn formal_charge(&self, atom: usize) -> i8 {
        self.charges.get(atom).copied().unwrap_or(0)
    }

    pub(super) fn atom_count(&self) -> usize {
        self.atomic_numbers.len()
    }

    /// How many rings an atom belongs to, which a couple of typing rules use
    /// to tell a fused nitrogen from a lone one.
    pub(super) fn ring_count(&self, atom: usize) -> usize {
        self.rings.iter().filter(|ring| ring.contains(&atom)).count()
    }

    pub(super) fn rings(&self) -> &[Vec<usize>] {
        &self.rings
    }

    pub(super) fn neighbours(&self, atom: usize) -> &[Neighbour] {
        &self.neighbours[atom]
    }

    pub(super) fn degree(&self, atom: usize) -> usize {
        self.neighbours[atom].len()
    }

    pub(super) fn valence(&self, atom: usize) -> u32 {
        self.valence[atom]
    }

    pub(super) fn order(&self, first: usize, second: usize) -> BondOrder {
        self.neighbours[first]
            .iter()
            .find(|neighbour| neighbour.atom == second)
            .map_or(BondOrder::Single, |neighbour| neighbour.order)
    }

    pub(super) fn is_aromatic(&self, atom: usize) -> bool {
        self.atom_is_aromatic[atom]
    }

    pub(super) fn in_ring_of_size(&self, atom: usize, size: usize) -> bool {
        self.rings
            .iter()
            .any(|ring| ring.len() == size && ring.contains(&atom))
    }

    pub(super) fn in_aromatic_ring_of_size(&self, atom: usize, size: usize) -> bool {
        self.rings
            .iter()
            .enumerate()
            .any(|(index, ring)| {
                self.ring_is_aromatic[index] && ring.len() == size && ring.contains(&atom)
            })
    }

    pub(super) fn in_same_ring_of_size(&self, size: usize, atoms: &[usize]) -> bool {
        self.rings.iter().any(|ring| {
            ring.len() == size && atoms.iter().all(|atom| ring.contains(atom))
        })
    }
}
