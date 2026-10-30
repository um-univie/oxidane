use crate::{
    atom::Atom,
    molecular_formula::MolecularFormula,
    molecule::{
        bond::{Bond, BondOrder, BondTarget},
        chirality::{ChiralClass, ChiralType},
        graph::Node,
        traits::*,
        utils::order_tuple,
    },
};
use bit_set::BitSet;
use bitvec::prelude::*;
use chemistry_consts::ElementProperties;
use rustc_hash::FxHashSet;
use tinyvec::{ArrayVec, array_vec};

pub trait Molecule:
    MoleculeCore + MoleculeExperimental + MoleculeAromaticity + MoleculeGraph
{
}

pub trait MoleculeCore {
    /// An optional cache for connectivity-only ring membership.
    fn ring_topology_cache(&self) -> Option<&crate::molecule::topology::RingTopologyCache> {
        None
    }

    fn atom_bonds(&self) -> &Vec<ArrayVec<[BondTarget; 10]>>;
    fn atom_bonds_mut(&mut self) -> &mut Vec<ArrayVec<[BondTarget; 10]>>;
    fn atomic_numbers(&self) -> &[u8];
    fn atomic_numbers_mut(&mut self) -> &mut Vec<u8>;
    fn chiral_classes(&self) -> Option<&[ChiralClass]>;
    fn chiral_classes_mut(&mut self) -> Option<&mut Vec<ChiralClass>>;
    fn initialize_chiral_classes(&mut self);
    fn charges(&self) -> &[i8];
    fn charges_mut(&mut self) -> &mut Vec<i8>;
    fn isotopes(&self) -> Option<&Vec<u16>>;
    fn isotopes_mut(&mut self) -> &mut Option<Vec<u16>>;
    fn atom_classes(&self) -> &Option<Vec<usize>>;
    fn atom_classes_mut(&mut self) -> &mut Option<Vec<usize>>;
    fn from_atoms(atoms: &[Atom]) -> Self;
    fn radical_states(&self) -> &[u8];
    fn radical_states_mut(&mut self) -> &mut Vec<u8>;
    /// Extends the per-atom vectors this representation carries beyond the ones
    /// the trait knows about, so that [`MoleculeCore::add_atom`] keeps them in
    /// step with the rest.
    ///
    /// `Molecule3D` indexes `positions` directly, and adding an atom without
    /// this left it shorter than the atom count — reading an SDF that omits its
    /// hydrogens and adding them gave twelve atoms and six positions, and the
    /// next thing to ask for a distance panicked.
    fn push_atom_extras(&mut self, _atom: &Atom) {}

    /// The counterpart of [`MoleculeCore::push_atom_extras`] for the last atom.
    fn pop_atom_extras(&mut self) {}

    /// The counterpart of [`MoleculeCore::push_atom_extras`] for one atom.
    fn remove_atom_extras(&mut self, _atom_index: usize) {}

    fn pop_atom(&mut self) {
        self.atomic_numbers_mut().pop();
        self.charges_mut().pop();
        self.radical_states_mut().pop();
        self.atom_bonds_mut().pop();
        if let Some(classes) = self.atom_classes_mut() {
            classes.pop();
        }
        if let Some(chirals) = self.chiral_classes_mut() {
            chirals.pop();
        }
        if let Some(isotopes) = self.isotopes_mut() {
            isotopes.pop();
        }
        self.pop_atom_extras();
    }

    fn remove_atom(&mut self, atom_index: usize) {
        self.atomic_numbers_mut().remove(atom_index);
        self.charges_mut().remove(atom_index);
        self.radical_states_mut().remove(atom_index);
        self.atom_bonds_mut().remove(atom_index);
        if let Some(classes) = self.atom_classes_mut() {
            classes.remove(atom_index);
        }
        if let Some(chirals) = self.chiral_classes_mut() {
            chirals.remove(atom_index);
        }
        if let Some(isotopes) = self.isotopes_mut() {
            isotopes.remove(atom_index);
        }
        self.remove_atom_extras(atom_index);

        for current_atom_bonds in self.atom_bonds_mut() {
            current_atom_bonds.retain(|bond| bond.target() != atom_index);

            // Update indices of bonds that pointed to atoms after the deleted one
            for bond in current_atom_bonds.iter_mut() {
                if bond.target() > atom_index {
                    bond.set_target(bond.target() - 1);
                }
            }
        }
    }

    fn can_increase_bond(atom_index: usize, neighbor_index: usize, degrees: &[i8]) -> bool {
        degrees[atom_index] < 0 && degrees[neighbor_index] < 0
    }

    fn get_atomic_symbol(&self, atom_index: usize) -> Option<&str> {
        self.atomic_numbers().get(atom_index)?.atomic_symbol()
    }

    fn len(&self) -> usize {
        self.atomic_numbers().len()
    }

    fn is_empty(&self) -> bool {
        self.atomic_numbers().is_empty()
    }

    fn is_radical(&self) -> bool {
        self.radical_states()
            .iter()
            .any(|&radical_state| radical_state > 0)
    }

    /// Difference between expected and actual valency for an atom, or `None` if the element has no expected valency.
    fn valency_delta(&self, atom_index: usize) -> Option<i8> {
        let expected_valency = self.expected_valency(atom_index)?;
        let actual_valency = self.actual_valency(atom_index);
        Some(expected_valency - actual_valency)
    }

    /// Checks if an atom has a plausible valency based on its element type, charge, and bonding.
    ///
    /// Checks an upper bound on occupied valence, including each unpaired electron.
    /// Neutral phosphorus permits up to five and sulfur up to six; other elements
    /// use their largest nonnegative tabulated valence. Formal charges use the
    /// group-based adjustment used by hydrogen perception. Underfilled atoms and
    /// elements without a valence model are allowed, so this is a plausibility
    /// check, not a full chemical validator.
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("C").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// assert!(molecule.has_plausible_valency(0)); // Carbon with 4 bonds is plausible
    ///
    /// let molecule = MolecularGraph::from_smiles("C=O").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// assert!(molecule.has_plausible_valency(1)); // Oxygen with 2 bonds is plausible
    ///
    /// let molecule = MolecularGraph::from_smiles("OS(=O)(=O)O").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// assert!(molecule.has_plausible_valency(1)); // Sulfur with valence six
    /// ```
    fn has_plausible_valency(&self, atom_index: usize) -> bool {
        let Some(&element) = self.atomic_numbers().get(atom_index) else {
            return false;
        };
        let Some(valences) = element.valencies() else {
            // No covalent model for this element (e.g. a transition metal).
            return true;
        };
        let maximum = match element {
            // The constants table also lists P(VII); ordinary covalent
            // phosphorus is limited to five here, sulfur to six.
            15 => 5,
            16 => 6,
            _ => valences.filter(|&v| v >= 0).max().unwrap_or(0) as i16,
        };
        // Use the same group-based charge adjustment as hydrogen perception,
        // but widen before arithmetic so even extreme i8 charges cannot overflow.
        let charge = self.get_atom_charge(atom_index) as i16;
        let shift = match element {
            5 | 13 | 31 | 49 | 81 => -charge,
            6 | 14 | 32 | 50 | 82 => -charge.abs(),
            7 | 15 | 33 | 51 | 83 | 8 | 16 | 34 | 52 | 84 | 9 | 17 | 35 | 53 | 85 => charge,
            _ => -charge.abs(),
        };
        let radicals = self.radical_states().get(atom_index).copied().unwrap_or(0) as i16;
        let occupied = self.actual_valency(atom_index) as i16 + radicals;
        occupied <= maximum + shift
    }

    fn get_atomic_number(&self, atom_index: usize) -> u8 {
        self.atomic_numbers()[atom_index]
    }

    fn set_atom_class(&mut self, atom_index: usize, class: usize) {
        if let Some(classes) = self.atom_classes_mut() {
            classes[atom_index] = class;
        } else {
            let size = self.atomic_numbers().len();
            let mut classes = vec![0; size];
            classes[atom_index] = class;
            *self.atom_classes_mut() = Some(classes);
        }
    }

    fn cmp_atom_charges(&self, atom_index1: usize, atom_index2: usize) -> std::cmp::Ordering {
        self.get_atom_charge(atom_index1)
            .cmp(&self.get_atom_charge(atom_index2))
    }
    /// Compares the electronegativities of two atoms based on the Pauling Scale, for detailed Values see the chemistry_consts crate
    ///
    /// # Arguments
    /// * 'atom1_index' - The index of the first atom.
    /// * 'atom2_index' - The index of the second atom.
    ///
    /// # Example
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_xyz("tests/ethane.xyz").unwrap().remove(0);
    ///
    /// assert_eq!(molecule.cmp_electronegativities(0, 1), std::cmp::Ordering::Equal);
    /// ```
    fn cmp_electronegativities(
        &self,
        atom1_index: usize,
        atom2_index: usize,
    ) -> std::cmp::Ordering {
        self.electronegativity(atom1_index)
            .cmp(&self.electronegativity(atom2_index))
    }

    /// Returns the Pauling electronegativity of the atom at the specified index scaled by 100
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom for which to retrieve the electronegativity.
    ///
    /// # Returns
    /// An `Option` containing the electronegativity value as a `u16` if the atom index is valid, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_xyz("tests/ethane.xyz").unwrap().remove(0);
    /// assert_eq!(molecule.electronegativity(0), Some(254)); // Carbon's electronegativity
    /// assert_eq!(molecule.electronegativity(2), Some(220)); // Hydrogen's electronegativity
    /// ```
    fn electronegativity(&self, atom_index: usize) -> Option<u16> {
        self.atomic_numbers().get(atom_index)?.electronegativity()
    }

    /// Calculates the formal oxidation state of an atom by comparing electronegativities across its bonds.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_xyz("tests/ethane.xyz").unwrap().remove(0);
    /// assert_eq!(molecule.get_oxidation_state(0), -3);
    /// ```
    fn get_oxidation_state(&self, atom_index: usize) -> i8 {
        let mut state = self.get_atom_charge(atom_index);
        for bond in self.get_atom_bonds(atom_index).unwrap_or_default() {
            let target = bond.target();
            match self.cmp_electronegativities(atom_index, target) {
                std::cmp::Ordering::Less => state += 1,
                std::cmp::Ordering::Greater => state -= 1,
                _ => continue,
            }
        }
        state
    }

    /// Number of bonded neighbors matching element `element`, or 0 if `atom_index` is out of bounds.
    fn number_of_bonded_element(&self, atom_index: usize, element: u8) -> usize {
        let Some(bonds) = self.get_atom_bonds(atom_index) else {
            return 0;
        };
        bonds
            .iter()
            .filter(|bond| self.atomic_numbers()[bond.target()] == element)
            .count()
    }

    fn expected_valency(&self, atom_index: usize) -> Option<i8> {
        self.atomic_numbers()[atom_index].valencies()?.next()
    }

    /// Whether `atom_index` is one of an aromatic ring's atoms.
    ///
    /// `is_atom_aromatic` asks whether it still has an aromatic bond, and that
    /// misses the atom whose ring bonds *all* carry directional marks -- it has
    /// none left to be recognised by, so its neighbours' bonds were counted as
    /// single, they gained a hydrogen each, and the ring stopped kekulizing.
    /// Two directional bonds on one atom mean it sits in a ring of them; one
    /// directional bond and no aromatic bond is an ordinary substituent and
    /// stays single.
    fn sits_in_an_aromatic_ring(&self, atom_index: usize) -> bool {
        if self.is_atom_aromatic(atom_index) {
            return true;
        }
        // Two marks and an aromatic neighbour along one of them. Both halves
        // are needed: two marks alone is also what a carbon between two
        // stereogenic double bonds looks like, and calling that aromatic gives
        // it a valence it does not have -- 56 perfectly ordinary dienes stopped
        // parsing when the count alone was the test.
        let mut marks = 0usize;
        let mut an_aromatic_neighbour = false;
        for bond in self.atom_bonds()[atom_index].iter() {
            if bond.bond_order() == BondOrder::Up || bond.bond_order() == BondOrder::Down {
                marks += 1;
                an_aromatic_neighbour |= self.is_atom_aromatic(bond.target());
            }
        }
        // And the atom has to be one an aromatic ring could hold: at most three
        // connections, because a fourth leaves no room for the π system.
        // Without that, a phosphorus carrying two marked phenyls --
        // `N=P(/c1ccccc1)(\c2ccccc2)C` -- was read as sitting in an aromatic
        // ring, its two bonds priced at three half-units each, and it came back
        // a hydrogen heavier than it went in. Those bonds join it to two
        // *separate* rings; they are not ring bonds of one.
        marks >= 2 && self.atom_bonds()[atom_index].len() <= 3 && an_aromatic_neighbour
    }

    /// Whether any of `atom_index`'s bonds is one [`MoleculeCore::actual_valency`]
    /// prices above a single bond -- three half-units rather than two.
    ///
    /// That is the question "is this atom's valence settled?", and it has to be
    /// asked exactly the way the pricing asks it. An aromatic bond is the
    /// obvious case. The other is a directional mark between two atoms of an
    /// aromatic ring, which the pricing also charges three half-units for: a
    /// quinoid thiophene written `=c1/s/c(cc1)=` leaves its sulfur no aromatic
    /// bond of its own, both ring bonds having been marked to place the
    /// exocyclic geometry. Priced as a ring atom it measures three; asked for
    /// aromatic bonds it answers none, and the gap between the two readings
    /// became a hydrogen the molecule has no room for.
    fn has_an_unsettled_bond(&self, atom_index: usize) -> bool {
        self.get_atom_bonds(atom_index)
            .unwrap_or_default()
            .iter()
            .any(|bond| match bond.bond_order() {
                BondOrder::Aromatic => true,
                BondOrder::Up | BondOrder::Down => {
                    self.sits_in_an_aromatic_ring(atom_index)
                        && self.sits_in_an_aromatic_ring(bond.target())
                }
                _ => false,
            })
    }

    /// Actual valency of an atom. Fractional aromatic valencies round down.
    fn actual_valency(&self, atom_index: usize) -> i8 {
        // A directional mark is not a bond order; it says which side of a
        // double bond a substituent sits on, and the bond underneath it is
        // whatever it would otherwise have been. Between two aromatic atoms
        // that is an aromatic bond, worth three half-units rather than the two
        // a single bond gets. Counting it as single leaves the atom looking
        // half a bond short, which rounds up to a whole missing hydrogen:
        // `c1/ccccc1` gained a second hydrogen on the marked carbon, and its
        // ring then had no Kekulé structure at all.
        let directional = |target: usize| {
            if self.sits_in_an_aromatic_ring(atom_index) && self.sits_in_an_aromatic_ring(target) {
                3
            } else {
                2
            }
        };
        (self.atom_bonds()[atom_index]
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Single => 2,
                BondOrder::Up | BondOrder::Down => directional(bond.target()),
                BondOrder::Double => 4,
                BondOrder::Triple => 6,
                BondOrder::Quadruple => 8,
                BondOrder::Aromatic => 3,
                BondOrder::Coordinate => 2,
            })
            .sum::<i8>())
            / 2
    }

    /// Degree of an atom, calculated as actual valency plus radicals minus effective valency.
    fn get_atom_degree(&self, atom_index: usize) -> Option<i8> {
        let actual_valency = self.actual_valency(atom_index);
        let charge = self.get_atom_charge(atom_index);
        // Every unpaired electron takes up a bonding slot, so a diradical
        // accounts for two. An absent entry means none; some sources leave the
        // vector empty rather than filling it with zeros.
        let radicals = self.radical_states().get(atom_index).copied().unwrap_or(0) as i8;
        let atomic_number = self.atomic_numbers()[atom_index];
        let effective = effective_valency(atomic_number, charge)?;
        Some(actual_valency + radicals - effective)
    }

    fn get_atom_isotope(&self, atom_index: usize) -> Option<u16> {
        if atom_index >= self.len() {
            return None;
        }
        let Some(isotope) = self.get_isotope(atom_index) else {
            let atomic_number = self.atomic_numbers()[atom_index];
            let isotope = atomic_number.most_abundant_isotope();
            return isotope;
        };
        Some(isotope)
    }

    /// Degree of all atoms in the molecule.
    fn degrees(&self) -> Vec<i8> {
        self.atomic_numbers()
            .iter()
            .enumerate()
            .map(|(index, _)| self.get_atom_degree(index).unwrap_or(0))
            .collect()
    }

    /// Sums exact masses of explicitly labelled isotopes and default monoisotopic
    /// element masses for unlabelled atoms. Returns NaN if a written isotope has
    /// no tabulated mass. Wildcard labels contribute zero.
    fn monoisotopic_mass(&self) -> f64 {
        self.atomic_numbers()
            .iter()
            .enumerate()
            .map(|(index, &element)| match self.get_isotope(index) {
                Some(mass_number) => {
                    crate::atom::isotope_mass(element, mass_number).unwrap_or(f64::NAN)
                }
                None => element.monoisotopic_mass().unwrap_or(0.0),
            })
            .sum::<f64>()
    }

    /// Number of atoms in the molecule.
    fn number_of_atoms(&self) -> usize {
        self.atomic_numbers().len()
    }

    /// Net formal charge of the molecule, summing charges across all atoms.
    fn formal_charge(&self) -> i32 {
        self.charges().iter().sum::<i8>() as i32
    }

    /// Formal charge of the atom at `atom_index`.
    fn get_atom_charge(&self, atom_index: usize) -> i8 {
        self.charges()[atom_index]
    }

    /// Mutable reference to the formal charge of the atom at `atom_index`.
    fn get_atom_charge_mut(&mut self, atom_index: usize) -> &mut i8 {
        &mut self.charges_mut()[atom_index]
    }

    /// The isotope **explicitly stated** for an atom, if any.
    ///
    /// `None` means the SMILES (or file) left it unspecified, not that the
    /// atom has no mass number — use [`MoleculeCore::get_atom_isotope`] for
    /// the effective mass number, which falls back to the element's most
    /// abundant isotope. The distinction is what lets `to_smiles` write
    /// `[13C]` where it was asked for and plain `C` where it was not.
    fn get_isotope(&self, atom_index: usize) -> Option<u16> {
        self.isotopes()
            .and_then(|isotopes| isotopes.get(atom_index).copied())
            .filter(|&isotope| isotope != 0)
    }

    /// Chiral class of the atom at `atom_index`, or [`ChiralClass::None`] if unspecified.
    fn get_chiral_class(&self, atom_index: usize) -> ChiralClass {
        self.chiral_classes()
            .and_then(|chirals| chirals.get(atom_index).copied())
            .unwrap_or(ChiralClass::None)
    }

    /// Returns `true` if the atom at `atom_index` has a non-zero radical state.
    fn is_atom_radical(&self, atom_index: usize) -> bool {
        self.radical_states()
            .get(atom_index)
            .is_some_and(|&state| state > 0)
    }

    /// Sets the radical state of the atom at `atom_index`.
    fn set_atom_radical_state(&mut self, atom_index: usize, radical_state: u8) {
        if atom_index < self.radical_states().len() {
            self.radical_states_mut()[atom_index] = radical_state;
        } else {
            debug_assert!(
                false,
                "atom index out of bounds, could not set radical state"
            );
        }
    }

    fn set_atom_charge(&mut self, atom_index: usize, charge: i8) {
        if atom_index < self.charges().len() {
            self.charges_mut()[atom_index] = charge;
        } else {
            debug_assert!(false, "atom index out of bounds, could not set charge");
        }
    }
    fn get_atom_class(&self, atom_index: usize) -> usize {
        let reference = self.atom_classes().as_ref();
        reference.map_or(0, |classes| classes[atom_index])
    }

    fn get_edges(&self) -> Vec<(usize, usize)> {
        self.atom_bonds()
            .iter()
            .enumerate()
            .flat_map(|(atom_index, bonds)| {
                bonds.iter().filter_map(move |bond| {
                    if atom_index < bond.target() {
                        Some((atom_index, bond.target()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }
    fn get_edges_with_type(&self) -> Vec<(usize, usize, BondOrder)> {
        self.atom_bonds()
            .iter()
            .enumerate()
            .flat_map(|(atom_index, bonds)| {
                bonds.iter().map(move |bond| {
                    if atom_index < bond.target() {
                        Some((atom_index, bond.target(), bond.bond_order()))
                    } else {
                        None
                    }
                })
            })
            .flatten()
            .collect()
    }

    fn get_chiral_type(&self, atom_index: usize) -> ChiralType {
        let chiral_class = self.get_chiral_class(atom_index);
        let chiral_type = match chiral_class {
            ChiralClass::None => ChiralType::None,
            ChiralClass::Clockwise => ChiralType::TH,
            ChiralClass::Counterclockwise => ChiralType::TH,
            ChiralClass::SP(_) => ChiralType::SP,
            ChiralClass::TB(_) => ChiralType::TB,
            ChiralClass::OH(_) => ChiralType::OH,
            ChiralClass::AL(_) => ChiralType::AL,
            ChiralClass::TH(_) => ChiralType::TH,
            ChiralClass::CisTransClockwise => ChiralType::CisTrans,
            ChiralClass::CisTransCounterclockwise => ChiralType::CisTrans,
        };
        chiral_type
    }

    fn get_atom_bonds(&self, atom_index: usize) -> Option<&[BondTarget]> {
        self.atom_bonds()
            .get(atom_index)
            .map(|bonds| bonds.as_slice())
    }

    fn get_atom_bonds_mut(&mut self, atom_index: usize) -> Option<&mut [BondTarget]> {
        self.atom_bonds_mut()
            .get_mut(atom_index)
            .map(|bonds| bonds.as_mut_slice())
    }

    /// Whether SMILES represents this atom in its neighbour's bracket H count.
    /// Only a terminal, unlabelled, neutral, nonradical hydrogen on a non-H atom
    /// may fold. The traversal, H counts and stereo ordering share this rule.
    fn hydrogen_folds_into_neighbour(&self, atom: usize) -> bool {
        self.atomic_numbers()[atom] == 1
            && self.get_isotope(atom).is_none()
            && self.charges().get(atom).copied().unwrap_or(0) == 0
            && self.radical_states().get(atom).copied().unwrap_or(0) == 0
            && self.atom_bonds()[atom].len() == 1
            && self.atom_bonds()[atom][0].bond_order() == BondOrder::Single
            && self.atomic_numbers()[self.atom_bonds()[atom][0].target()] != 1
    }

    /// Builds a SMILES traversal tree, recording non-tree edges as closures.
    fn build_smiles_tree(
        &self,
        start_index: usize,
        bond_type: BondOrder,
        parent_index: Option<usize>,
        visited: &mut Vec<bool>,
        ring_closures: &mut Vec<(usize, usize, BondOrder)>,
        traversal_order: &mut Vec<usize>,
    ) -> Node {
        let mut root = Node::new(parent_index, start_index, bond_type);
        visited[start_index] = true;
        traversal_order.push(start_index); // Record traversal order

        let mut bonds = self.atom_bonds()[start_index];
        bonds.sort_by_key(|bond| bond.target());

        for neighbor in bonds {
            if Some(neighbor.target()) == parent_index {
                continue;
            }

            // The tree, bracket counts and stereo ordering must use the same
            // folding rule: every other hydrogen is emitted exactly once.
            if self.hydrogen_folds_into_neighbour(neighbor.target()) {
                continue;
            }

            if visited[neighbor.target()] {
                if neighbor.target() < start_index {
                    continue;
                }
                let bond_type = neighbor.bond_order();
                ring_closures.push((start_index, neighbor.target(), bond_type));
            } else {
                let child = self.build_smiles_tree(
                    neighbor.target(),
                    neighbor.bond_order(),
                    Some(start_index),
                    visited,
                    ring_closures,
                    traversal_order,
                );
                root.add_child(child);
            }
        }
        root
    }

    fn number_of_pi_electrons(&self, atom_index: usize) -> usize {
        let mut number_of_pi_electrons = 0;
        // Default to 0 if no bonds are found
        let Some(bonds) = self.get_atom_bonds(atom_index) else {
            return 0;
        };
        for bond in bonds {
            if bond.bond_order() == BondOrder::Double || bond.bond_order() == BondOrder::Aromatic {
                number_of_pi_electrons += 1;
            } else if bond.bond_order() == BondOrder::Triple {
                number_of_pi_electrons += 2;
            }
        }
        match self.get_atom_charge(atom_index) {
            2.. => number_of_pi_electrons += 1,
            ..=-2 => number_of_pi_electrons -= 1,
            _ => {}
        }
        match self.atomic_numbers()[atom_index] {
            7 => number_of_pi_electrons += 2,
            8 => number_of_pi_electrons += 2,
            _ => {}
        }
        number_of_pi_electrons
    }

    /// Checks if the molecule has any stereochemistry
    ///
    /// # Returns
    /// A boolean indicating whether the molecule has stereochemistry
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("C[C@H](N)O").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// assert_eq!(molecule.has_stereochemistry(), true);
    /// ```
    fn has_stereochemistry(&self) -> bool {
        self.chiral_classes().map_or(false, |classes| {
            classes.iter().any(|class| class != &ChiralClass::None)
        })
    }

    fn molecular_formula(&self) -> MolecularFormula
    where
        Self: Molecule + Sized,
    {
        MolecularFormula::from_molecule(self)
    }

    fn is_chiral(&self) -> bool {
        self.chiral_classes().map_or(false, |classes| {
            classes.iter().any(|class| class != &ChiralClass::None)
        })
    }

    /// Gets the path of atoms in the order they would be visited during SMILES string generation
    ///
    /// # Returns
    /// A vector of atom indices in the order they are visited when generating a SMILES string
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let mol = MolecularGraph::from_smiles("CC(=O)C").unwrap()[0].clone();
    /// let path = mol.get_smiles_path();
    /// // Acetone's four heavy atoms, each once; hydrogens are not traversed.
    /// assert_eq!(path.len(), 4);
    /// let mut sorted = path.clone();
    /// sorted.sort_unstable();
    /// assert_eq!(sorted, vec![0, 1, 2, 3]);
    /// ```
    fn get_smiles_path(&self) -> Vec<usize> {
        let mut path = Vec::with_capacity(self.len());
        let mut visited = vec![false; self.len()];
        for i in 0..self.len() {
            if self.atomic_numbers()[i] == 1 {
                visited[i] = true;
            }
        }

        // Start from each unvisited atom to handle disconnected components
        for start in 0..self.len() {
            if !visited[start] {
                self.visit_atom_for_path(start, &mut visited, &mut path);
            }
        }

        path
    }

    fn visit_atom_for_path(&self, current: usize, visited: &mut [bool], path: &mut Vec<usize>) {
        visited[current] = true;
        path.push(current);

        if let Some(bonds) = self.get_atom_bonds(current) {
            for bond in bonds {
                let target = bond.target();

                if !visited[target] {
                    // If target not visited, traverse it
                    self.visit_atom_for_path(target, visited, path);
                }
            }
        }
    }

    fn is_atom_chiral(&self, atom_index: usize) -> bool {
        self.get_chiral_class(atom_index) != ChiralClass::None
    }

    fn is_atom_aromatic(&self, atom_index: usize) -> bool {
        self.get_atom_bonds(atom_index)
            .unwrap()
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Aromatic)
    }

    fn add_atom(&mut self, atom: Atom) {
        let index = self.atomic_numbers().len();
        // An isotope on the new atom has to be recorded even when the molecule
        // was not carrying isotopes at all, which means backfilling the atoms
        // already there with "natural abundance" first.
        if atom.isotope.is_some() && self.isotopes().is_none() {
            *self.isotopes_mut() = Some(vec![0; index]);
        }
        if atom.atom_class.is_some() && self.atom_classes().is_none() {
            *self.atom_classes_mut() = Some(vec![0; index]);
        }
        self.atomic_numbers_mut().push(atom.atomic_number);
        for bond in &atom.bonds {
            self.atom_bonds_mut()[bond.target()].push(BondTarget::single(index));
        }
        self.charges_mut().push(atom.charge);
        self.radical_states_mut().push(atom.radical_state);
        if self.chiral_classes().is_some() {
            self.chiral_classes_mut().unwrap().push(atom.chiral_class);
        }
        if let Some(isotopes) = self.isotopes_mut() {
            isotopes.push(atom.isotope.unwrap_or(0));
        }
        if let Some(classes) = self.atom_classes_mut() {
            classes.push(atom.atom_class.unwrap_or(0));
        }
        self.push_atom_extras(&atom);
        self.atom_bonds_mut().push(atom.bonds);
    }

    fn add_bond(&mut self, source: usize, target: usize, bond_order: BondOrder) {
        self.atom_bonds_mut()[source].push(BondTarget::new(target, bond_order));
        self.atom_bonds_mut()[target].push(BondTarget::new(source, bond_order));
    }

    /// Adds hydrogens to the molecule based on the atom's degree which is defined as as the sum of the valency delta (expected - actual) and the absolute value of the charge and the radical state
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let mut molecule = MolecularGraph::from_smiles_single("C").unwrap();
    /// molecule.add_hydrogens();
    /// assert_eq!(molecule.number_of_bonded_element(0, 1), 4);
    /// ```
    /// Rewrites a neutral hypervalent nitrogen's double bond to a terminal
    /// chalcogen as the charge-separated pair it stands for.
    ///
    /// `c1ccn(=O)cc1` and `CN(C)=O` both put four bonds on a neutral nitrogen,
    /// which nitrogen does not have. What is meant is the N-oxide,
    /// `[n+]([O-])`, and until it is written that way the ring cannot be
    /// kekulized at all -- pyridine N-oxide's five carbons have nothing to
    /// pair the odd one with, because the nitrogen is offering no p orbital.
    ///
    /// The transform is the one RDKit's normaliser applies, and it is not a
    /// guess: a neutral tetravalent nitrogen has no other reading.
    ///
    /// ```
    /// use oxidane::prelude::*;
    ///
    /// // Written as a hypervalent nitrogen, read as the N-oxide.
    /// let oxide = MolecularGraph::from_smiles_single("c1ccn(=O)cc1").unwrap();
    /// assert_eq!(oxide.formal_charge(), 0);
    /// assert_eq!(oxide.molecular_formula().to_string(), "C5H5N1O1");
    /// ```
    fn charge_separate_hypervalent_nitrogen(&mut self) {
        let mut changes: Vec<(usize, usize)> = Vec::new();
        for atom in 0..self.len() {
            if self.get_atomic_number(atom) != 7 || self.get_atom_charge(atom) != 0 {
                continue;
            }
            let Some(bonds) = self.get_atom_bonds(atom) else {
                continue;
            };
            if self.actual_valency(atom) <= 3 {
                continue;
            }
            // Only a *terminal* chalcogen: a double bond to anything that
            // carries further bonds is part of a larger system, and moving its
            // electrons would be a claim about that system rather than about
            // this nitrogen.
            let partner = bonds.iter().find(|bond| {
                bond.bond_order() == BondOrder::Double
                    && matches!(self.get_atomic_number(bond.target()), 8 | 16)
                    && self.get_atom_charge(bond.target()) == 0
                    && self
                        .get_atom_bonds(bond.target())
                        .is_some_and(|other| other.len() == 1)
            });
            if let Some(bond) = partner {
                changes.push((atom, bond.target()));
            }
        }

        for (nitrogen, chalcogen) in changes {
            for (first, second) in [(nitrogen, chalcogen), (chalcogen, nitrogen)] {
                if let Some(bonds) = self.get_atom_bonds_mut(first) {
                    for bond in bonds.iter_mut() {
                        if bond.target() == second {
                            bond.bond_order = BondOrder::Single;
                        }
                    }
                }
            }
            self.set_atom_charge(nitrogen, 1);
            self.set_atom_charge(chalcogen, -1);
        }
    }

    fn add_hydrogens(&mut self) {
        self.add_hydrogens_with_valences(&[]);
    }

    /// Adds hydrogens as [`MoleculeCore::add_hydrogens`] does, except for the
    /// atoms `valences` gives an explicit total valence for.
    ///
    /// A MOL file can state an atom's valence outright, which is how it says
    /// "this one takes no hydrogen" for something the valence model would
    /// otherwise fill: RDKit writes 15, the format's spelling of zero valence,
    /// on a bare potassium, and 3 on a germanium bonded to three ethyls.
    /// Without this both came back with a hydrogen the file had ruled out.
    /// An entry of `None`, or an index past the end, means "use the model".
    fn add_hydrogens_with_valences(&mut self, valences: &[Option<u8>]) {
        // How many hydrogens each atom is short of the valence its bonds have
        // already reached into -- the smallest of its element's normal valences
        // that they have not outgrown, which is what OpenSMILES asks for and
        // what every other toolkit fills by. `OS=O` is sulfurous acid, its
        // sulfur three bonds into a valence of four and so one hydrogen short;
        // against sulfur's valence of two it looked already over and got none.
        // `MoleculeCore::get_atom_degree` keeps the narrower reading, because
        // aromaticity, hybridization and Crippen all measure against it and
        // mean it.
        //
        // The exemption is written against the atom's *bonds*. `actual_valency`
        // prices an aromatic bond above a single one, so thiophene's sulfur
        // measures three and this would find it a bond short of four and hand
        // it a hydrogen it has no room for. The atom's aromatic flag is not the
        // test: a ring perception has **rejected** keeps the orders that made it
        // look aromatic, so the flag reads false while the valency still reads
        // three. Nor is ring membership the test -- that loses a cyclic
        // H-phosphonate, whose phosphorus is four bonds into a valence of five
        // and genuinely one hydrogen short.
        //
        // What this costs is worth stating. Where two writings of one compound
        // leave an atom with different bond orders -- which happens wherever
        // perception refuses a ring one way and not the other -- the rule turns
        // that difference into a hydrogen, and `normalize`'s answer stops
        // matching the molecule it was written from: seven merocyanine dyes and
        // a germanium macrocycle do that, over thirty million records. Taking
        // the rule out costs more. Dialkyl H-phosphonates are then written
        // `[P](OC)(OC)=O`, whose bracket makes `assign_radicals` give the
        // phosphorus an unpaired electron on the way back in, so the answer no
        // longer shares a key with what it came from -- twelve of those in the
        // first four hundred thousand records alone, and `OS=O` loses its
        // answer entirely. Both costs are the same underlying defect: an atom
        // whose valence two readings disagree about. Fixing that upstream is
        // what would settle it.
        let degrees = self.degrees();
        let shortfalls: Vec<i8> = (0..self.len())
            .map(|index| {
                // An aromatic bond has no whole number of π units, and a
                // *directional* one has no settled order at all -- the mark says
                // which side a substituent sits on, and what lies under it is
                // decided later, by perception. `actual_valency` already prices
                // such a bond at three half-units when the atom sits in a ring
                // the input wrote aromatic, so measuring the atom here and then
                // topping it up counts the same bond two ways.
                //
                // Hydrogens are filled before `localize_aromatic_bonds` and
                // `aromatize` have run, so the aromaticity in question is the
                // one the *input asserted*, which perception may go on to
                // refuse. A merocyanine dye writes `/s/c(...)`; its sulfur was
                // read as valence three, given a hydrogen, and then the ring was
                // rejected -- leaving an ordinary sulfide carrying a hydrogen it
                // cannot have, and a formula one heavier than the molecule.
                //
                // What counts as unsettled is whatever `actual_valency` priced
                // above a single bond, which `has_an_unsettled_bond` asks in
                // those same terms -- so the ring atom whose bonds all carry
                // marks is caught too, and not merely the one with an aromatic
                // bond left to be recognised by. Asking the narrower question
                // cost four quinoid macrocycles, whose canonical form came back
                // a hydrogen heavier than it went out.
                //
                // A lone `/` or `\` is still a single bond that has been told
                // which way to lean, worth exactly one either way: counting
                // every mark as unsettled gave `Cl/Cl(=O)=N/N=O` and
                // `ClCl(=O)=N/N=O` different hydrogen counts, hence different
                // canonization invariants, hence different canonical forms for
                // one molecule. `sits_in_an_aromatic_ring` is what keeps the
                // two apart.
                if self.has_an_unsettled_bond(index) {
                    return -degrees[index];
                }
                let radicals = self.radical_states().get(index).copied().unwrap_or(0) as i8;
                let spent = self.actual_valency(index) + radicals;
                saturating_valency(
                    self.atomic_numbers()[index],
                    self.get_atom_charge(index),
                    spent,
                )
                .map_or(0, |valence| valence - spent)
            })
            .collect();
        shortfalls
            .iter()
            .enumerate()
            .for_each(|(index, &shortfall)| {
                let atomic_number = self.atomic_numbers()[index];

                if atomic_number == 1 {
                    return;
                }

                let number_of_hydrogens = match valences.get(index).copied().flatten() {
                    // The stated valence counts the bonds already there, so what is
                    // left over is the hydrogen count. A file may state a valence
                    // the bonds already exceed, which leaves nothing to add rather
                    // than something to remove.
                    Some(valence) => (valence as i8 - self.actual_valency(index)).max(0),
                    None => shortfall,
                };

                for _ in 0..number_of_hydrogens {
                    let _hydrogen_index = self.atomic_numbers().len();
                    let atom = Atom::new(1)
                        .with_charge(0)
                        .with_radical(0)
                        .with_bonds(array_vec!([BondTarget;10] => BondTarget::single(index)))
                        .with_chiral_class(ChiralClass::None);

                    self.add_atom(atom);
                }
            });
    }

    /// Returns true if the molecule contains explicitly defined hydrogen atoms
    fn has_explicit_hydrogens(&self) -> bool {
        self.atomic_numbers().iter().any(|&num| num == 1)
    }

    fn get_aromatic_atoms(&self) -> Vec<usize> {
        self.atom_bonds()
            .iter()
            .enumerate()
            .filter_map(|(index, bonds)| {
                if bonds
                    .iter()
                    .any(|bond| bond.bond_order == BondOrder::Aromatic)
                {
                    Some(index)
                } else {
                    None
                }
            })
            .collect()
    }

    fn change_bond_order(&mut self, atom1: usize, atom2: usize, new_order: BondOrder) {
        let bond1 = self.atom_bonds_mut()[atom1]
            .iter()
            .position(|bond| bond.target == atom2)
            .unwrap();
        let bond2 = self.atom_bonds_mut()[atom2]
            .iter()
            .position(|bond| bond.target == atom1)
            .unwrap();
        self.atom_bonds_mut()[atom1][bond1].bond_order = new_order;
        self.atom_bonds_mut()[atom2][bond2].bond_order = new_order;
    }

    fn can_kekulize_bond(
        &self,
        atom1_idx: usize,
        atom2_idx: usize,
        aromatic_atoms: &BitVec<usize>,
    ) -> bool {
        aromatic_atoms[atom1_idx] && aromatic_atoms[atom2_idx]
    }

    fn get_bond_order(&self, atom1: usize, atom2: usize) -> BondOrder {
        self.atom_bonds()[atom1]
            .iter()
            .find(|bond| bond.target() == atom2)
            .map(|bond| bond.bond_order())
            .unwrap_or(BondOrder::Single)
    }

    fn check_for_metal_center(&self, system: &BitSet) -> bool {
        system.iter().any(|atom_idx| {
            let atomic_number = self.atomic_numbers()[atom_idx];
            // Transition metals: 21-30, 39-48, 57-80, 89-112
            matches!(atomic_number,
                21..=30 | 39..=48 | 57..=80 | 89..=112
            )
        })
    }

    fn aromatize_metallocenic_system(&mut self, _system: &BitSet) {
        // // First, identify the metal center
        // let metal_center = system.iter()
        //     .flat_map(|cycle| cycle.iter())
        //     .find(|&&atom_idx| {
        //         let atomic_number = self.atomic_numbers()[atom_idx];
        //         matches!(atomic_number, 21..=30 | 39..=48 | 57..=80 | 89..=112)
        //     });

        // if let Some(&metal_idx) = metal_center {
        //     // Convert all bonds in cyclopentadienyl rings to aromatic
        //     for cycle in system {
        //         if cycle.len() == 5 { // Cp rings are typically 5-membered
        //             for i in 0..cycle.len() {
        //                 let atom1 = cycle[i];
        //                 let atom2 = cycle[(i + 1) % cycle.len()];
        //                 self.change_bond_order(atom1, atom2, BondOrder::Aromatic);

        //                 // Add coordinate bond from metal to ring atoms
        //                 if atom1 != metal_idx {
        //                     self.change_bond_order(metal_idx, atom1, BondOrder::Coordinate);
        //                 }
        //             }
        //         }
        //     }
        // }
    }

    fn number_of_bond_order(&self, atom_idx: usize, bond_order: BondOrder) -> usize {
        self.atom_bonds()[atom_idx]
            .iter()
            .filter(|bond| bond.bond_order() == bond_order)
            .count()
    }

    /// Gets the union of atoms in a set of cycles
    ///
    /// # Arguments
    /// * `cycles` - Vector of cycles to get the union of
    ///
    /// # Returns
    /// A BitSet representing all atoms present in any of the cycles
    fn get_system_atoms(&self, cycles: &[Vec<usize>]) -> BitSet {
        let mut system_atoms = BitSet::new();
        for cycle in cycles {
            for &atom_idx in cycle {
                system_atoms.insert(atom_idx);
            }
        }
        system_atoms
    }

    fn is_electronegative(&self, atom_idx: usize) -> bool {
        let atomic_number = self.atomic_numbers()[atom_idx];
        matches!(atomic_number, 7 | 8 | 9 | 17 | 35 | 53) // N, O, F, Cl, Br, I
    }

    fn is_amine_nitrogen(&self, atom_idx: usize) -> bool {
        let double_bonds = self.has_double_bonds(atom_idx);
        let hydrogen_count = self.number_of_bonded_element(atom_idx, 1);
        // Amine nitrogen has at least one hydrogen
        !double_bonds || hydrogen_count > 0
    }

    fn has_double_bonds(&self, atom_idx: usize) -> bool {
        self.atom_bonds()[atom_idx]
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Double)
    }

    fn add_aromatic_bonds(&mut self) {
        for atom_idx in 0..self.len() {
            let is_atom_aromatic = self.is_atom_aromatic(atom_idx);
            if !is_atom_aromatic {
                continue;
            }

            let bonds = self.get_atom_bonds(atom_idx).unwrap_or_default().to_vec();

            for neighbor_idx in bonds {
                let is_neighbor_aromatic = self.is_atom_aromatic(neighbor_idx.target());
                if !is_neighbor_aromatic || neighbor_idx.bond_order() == BondOrder::Aromatic {
                    continue;
                }
                self.change_bond_order(atom_idx, neighbor_idx.target(), BondOrder::Aromatic);
            }
        }
    }

    fn get_all_bonds(&self) -> Vec<Bond> {
        let mut bonds = Vec::new();
        for atom_index in 0..self.len() {
            let Some(atom_bonds) = self.get_atom_bonds(atom_index) else {
                continue;
            };

            for bond in atom_bonds {
                if atom_index < bond.target() {
                    bonds.push(Bond::new(atom_index, bond.target(), bond.bond_order()));
                }
            }
        }
        bonds
    }

    fn get_all_heavy_atom_bonds(&self) -> Vec<Bond> {
        let mut bonds = Vec::new();
        for atom_index in 0..self.len() {
            if self.get_atomic_number(atom_index) == 1 {
                continue;
            }
            let Some(atom_bonds) = self.get_atom_bonds(atom_index) else {
                continue;
            };

            for bond in atom_bonds {
                if atom_index < bond.target() && self.get_atomic_number(bond.target()) != 1 {
                    bonds.push(Bond::new(atom_index, bond.target(), bond.bond_order()));
                }
            }
        }
        bonds
    }

    /// Computes the fundamental cycles of the molecule using a spanning tree approach.
    ///
    /// # Returns
    /// A vector of cycles, where each cycle is represented as a BitSet.
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("C1CC1").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let cycles = molecule.compute_fundamental_cycles();
    /// assert_eq!(cycles.len(), 1); // One fundamental cycle for cyclopropane
    /// ```
    fn compute_fundamental_cycles(&self) -> FxHashSet<BitSet> {
        #[derive(Debug, Clone)]
        struct TreeNode {
            parent: Option<usize>,
        }

        let num_atoms = self.len();
        let mut fundamental_cycles = FxHashSet::default();

        // Early return for molecules too small to have cycles
        if num_atoms < 3 {
            return fundamental_cycles;
        }

        let mut tree = Vec::with_capacity(num_atoms);
        for _ in 0..num_atoms {
            tree.push(TreeNode { parent: None });
        }

        // Pre-allocate data structures
        let mut remaining_bonds = self.get_edges();
        let mut node_stack = Vec::with_capacity(num_atoms);
        node_stack.push(0);

        // Pre-allocate paths for reuse
        let mut cycle = Vec::with_capacity(num_atoms);
        let mut other_path = Vec::with_capacity(num_atoms);

        while let Some(current_node) = node_stack.pop() {
            let mut i = 0;
            while i < remaining_bonds.len() {
                let (atom1, atom2) = remaining_bonds[i];

                // More efficient node matching
                let other_node = if atom1 == current_node {
                    atom2
                } else if atom2 == current_node {
                    atom1
                } else {
                    i += 1;
                    continue;
                };

                if tree[other_node].parent.is_some() {
                    // Found a fundamental cycle
                    cycle.clear();
                    other_path.clear();

                    // Trace paths to root more efficiently
                    let mut node = current_node;
                    while let Some(parent) = tree[node].parent {
                        cycle.push(node);
                        node = parent;
                    }
                    cycle.push(node);

                    let mut node = other_node;
                    while let Some(parent) = tree[node].parent {
                        other_path.push(node);
                        node = parent;
                    }
                    other_path.push(node);

                    let mut common_ancestor = None;
                    for (i, &n1) in cycle.iter().enumerate() {
                        if let Some(j) = other_path.iter().position(|&n2| n1 == n2) {
                            common_ancestor = Some((i, j));
                            break;
                        }
                    }

                    if let Some((i, j)) = common_ancestor {
                        let mut fundamental_cycle = BitSet::with_capacity(num_atoms);
                        for &atom in &cycle[..=i] {
                            fundamental_cycle.insert(atom);
                        }
                        for &atom in other_path[..j].iter().rev() {
                            fundamental_cycle.insert(atom);
                        }

                        // Only add if it's a new unique cycle
                        if !fundamental_cycles.contains(&fundamental_cycle) {
                            fundamental_cycles.insert(fundamental_cycle);
                        }
                    }
                } else {
                    tree[other_node].parent = Some(current_node);
                    node_stack.push(other_node);
                }

                remaining_bonds.swap_remove(i);
            }
        }

        fundamental_cycles
    }

    /// Converts a fundamental cycle BitSet to a Vec of atom indices
    ///
    /// # Arguments
    /// * `cycle` - BitSet representing a cycle
    ///
    /// # Returns
    /// A `Vec<usize>` containing the atom indices in order
    fn fundamental_cycle_to_vec(&self, cycle: &BitSet) -> Vec<usize> {
        let mut atoms = cycle.iter().collect::<Vec<_>>();

        // Sort atoms based on connectivity to ensure proper order
        let mut ordered = Vec::with_capacity(atoms.len());
        ordered.push(atoms.remove(0));

        while !atoms.is_empty() {
            let current = *ordered.last().unwrap();
            let next_idx = atoms.iter().position(|&atom| {
                self.atom_bonds()[current]
                    .iter()
                    .any(|bond| bond.target() == atom)
            });

            if let Some(idx) = next_idx {
                ordered.push(atoms.remove(idx));
            } else {
                // If no connected atom found, add the first remaining atom
                // This handles disconnected components
                ordered.push(atoms.remove(0));
            }
        }

        ordered
    }

    /// Enumerates possible cycles by combining fundamental cycles that share at least two atoms.
    ///
    /// This is a simplified approach that combines cycles when they share at least two atoms
    /// and are not identical. This avoids the computational complexity of checking all possible
    /// combinations while still finding most relevant cycles.
    ///
    /// # Returns
    /// A vector of BitSets, where each BitSet represents a unique cycle.
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("c1ccc2ccccc2c1").unwrap(); // naphthalene
    /// let molecule = molecule.first().unwrap();
    /// let cycles = molecule.enumerate_possible_cycles();
    /// println!("{:?}", molecule.compute_fundamental_cycles());
    /// println!("{:?}", cycles);
    /// assert!(cycles.len() >= 3); // Should find at least 3 cycles (2 6-membered rings + 10-membered perimeter)
    /// ```
    fn enumerate_possible_cycles(&self) -> Vec<BitSet> {
        let fundamental_cycles = self.compute_fundamental_cycles();
        if fundamental_cycles.len() < 2 {
            return fundamental_cycles.into_iter().collect();
        }

        let mut all_cycles = fundamental_cycles;
        let mut cycles_vec: Vec<BitSet> = all_cycles.iter().cloned().collect();
        let mut i = 0;
        while i < cycles_vec.len() {
            let mut j = 0;
            while j < cycles_vec.len() {
                let shared_atoms = cycles_vec[i].intersection(&cycles_vec[j]);

                // If cycles share at least 2 atoms and aren't identical
                if shared_atoms.into_iter().sum::<usize>() >= 2 {
                    let new_cycle = cycles_vec[i].union(&cycles_vec[j]).collect::<BitSet>();

                    // Only add if it forms a valid cycle and isn't already present
                    if !all_cycles.contains(&new_cycle) {
                        all_cycles.insert(new_cycle.clone());
                        cycles_vec.push(new_cycle);
                    }
                }
                j += 1;
            }
            i += 1;
        }

        cycles_vec.into_iter().collect()
    }

    /// Validates if a set of atoms forms a valid cycle by checking connectivity
    /// and ensuring each atom has exactly two connections within the cycle.
    fn is_valid_cycle(&self, cycle: &BitSet) -> bool {
        if cycle.len() < 3 {
            return false;
        }

        // Check if all atoms in the cycle have exactly two connections within the cycle
        let mut visited = BitSet::new();
        let mut stack = Vec::new();

        // Start with any atom in the cycle
        if let Some(start) = cycle.iter().next() {
            stack.push(start);
            visited.insert(start);

            while let Some(current) = stack.pop() {
                let mut connections_in_cycle = 0;

                // Check all bonds of the current atom
                for bond in &self.atom_bonds()[current] {
                    let target = bond.target();
                    if cycle.contains(target) {
                        connections_in_cycle += 1;
                        if !visited.contains(target) {
                            visited.insert(target);
                            stack.push(target);
                        }
                    }
                }

                // Each atom in a valid cycle must have exactly two connections within the cycle
                if connections_in_cycle != 2 {
                    return false;
                }
            }

            // All atoms must be visited and form a single connected component
            visited.len() == cycle.len()
        } else {
            false
        }
    }

    /// The atoms and bonds that lie on a ring.
    ///
    /// A bond is on a ring exactly when it is not a *bridge* — when removing
    /// it would not disconnect its two ends — and an atom is on a ring when it
    /// carries such a bond. That equivalence is what makes this exact for
    /// fused and bridged systems, where enumerating cycles is not: the
    /// previous implementation collected a depth-first cycle basis and only
    /// recognised a cycle when the back edge pointed at an atom still on the
    /// current path, so naphthalene lost its two fusion carbons and
    /// bicyclo[2.2.2]octane was reported as having no ring atoms at all.
    ///
    /// This is Tarjan's bridge-finding, iterative so that a large fused
    /// system cannot overflow the stack, and keyed on *edges* rather than
    /// parent atoms so that a doubled bond between the same pair of atoms is
    /// still recognised as a ring.
    fn find_cyclic_bonds_and_atoms(&self) -> (BitSet, FxHashSet<(usize, usize)>) {
        let mut cyclic_atoms = BitSet::new();
        let mut cyclic_bonds = FxHashSet::default();
        self.find_cyclic_bonds_and_atoms_into(&mut cyclic_atoms, &mut cyclic_bonds);
        (cyclic_atoms, cyclic_bonds)
    }

    /// Which atoms lie on a cycle, without working out which bonds do.
    ///
    /// The bond set costs a hash insert per ring bond and a caller that only
    /// wants to know whether an atom is in a ring -- the atom invariant of a
    /// circular fingerprint, for one -- pays it for nothing.
    fn cyclic_atoms(&self) -> BitSet {
        let mut cyclic_atoms = BitSet::with_capacity(self.len());
        self.walk_cyclic_bonds(|first, second| {
            cyclic_atoms.insert(first);
            cyclic_atoms.insert(second);
        });
        cyclic_atoms
    }

    /// [`MoleculeCore::find_cyclic_bonds_and_atoms`] writing into buffers the
    /// caller owns, so a caller that does this for molecule after molecule can
    /// keep the allocations instead of making a fresh pair each time.
    fn find_cyclic_bonds_and_atoms_into(
        &self,
        cyclic_atoms: &mut BitSet,
        cyclic_bonds: &mut FxHashSet<(usize, usize)>,
    ) {
        if let Some(cache) = self.ring_topology_cache() {
            cache.fill(self, cyclic_atoms, cyclic_bonds);
            return;
        }
        cyclic_atoms.clear();
        cyclic_bonds.clear();
        self.walk_cyclic_bonds(|first, second| {
            cyclic_bonds.insert(order_tuple(first, second));
            cyclic_atoms.insert(first);
            cyclic_atoms.insert(second);
        });
    }

    /// Calls `visit` once for every bond that lies on a cycle.
    ///
    /// Everything asked about rings comes from here, and what each caller wants
    /// out of it differs: perception needs the bonds in a set it can look up,
    /// a fingerprint's atom invariant needs only whether an atom was named at
    /// all. Handing the bonds over one at a time lets each build what it needs
    /// and nothing else.
    fn walk_cyclic_bonds<F: FnMut(usize, usize)>(&self, mut visit: F) {
        let atom_count = self.len();
        if atom_count == 0 {
            return;
        }

        // Peel off everything that cannot possibly lie on a cycle before
        // looking for one. A vertex of degree one is on no cycle, and removing
        // it can leave its neighbour with degree one in turn, so this strips
        // whole side chains and every hydrogen; what remains is the 2-core.
        //
        // It matters because the bridge search below is O(V+E) with a stack
        // frame and three arrays per vertex, and in a drug-sized molecule
        // roughly half the atoms are hydrogens and two thirds are not in any
        // ring. Peeling is one pass over the degrees plus one queue.
        let mut degree = vec![0u32; atom_count];
        for atom in 0..atom_count {
            let Some(bonds) = self.get_atom_bonds(atom) else {
                continue;
            };
            degree[atom] = bonds
                .iter()
                .filter(|bond| bond.target() < atom_count)
                .count() as u32;
        }
        let mut peeled: Vec<usize> = (0..atom_count).filter(|&a| degree[a] <= 1).collect();
        while let Some(atom) = peeled.pop() {
            degree[atom] = 0;
            let Some(bonds) = self.get_atom_bonds(atom) else {
                continue;
            };
            for bond in bonds {
                let target = bond.target();
                if target < atom_count && degree[target] > 0 {
                    degree[target] -= 1;
                    if degree[target] == 1 {
                        peeled.push(target);
                    }
                }
            }
        }

        // Each bond once, with an identity, plus adjacency that names it. Held
        // as one array with an index of where each atom's neighbours start,
        // rather than as a vector per atom: after peeling, every atom still
        // standing has neighbours, so a vector per atom is a heap allocation
        // per ring atom and this is called once for every molecule parsed.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut neighbour_start = vec![0u32; atom_count + 1];
        for atom in 0..atom_count {
            if degree[atom] == 0 {
                continue;
            }
            let Some(bonds) = self.get_atom_bonds(atom) else {
                continue;
            };
            for bond in bonds {
                let target = bond.target();
                if atom < target && target < atom_count && degree[target] > 0 {
                    edges.push((atom, target));
                    neighbour_start[atom] += 1;
                    neighbour_start[target] += 1;
                }
            }
        }
        let mut running = 0u32;
        for start in neighbour_start.iter_mut() {
            let count = *start;
            *start = running;
            running += count;
        }
        let mut cursor = neighbour_start.clone();
        let mut adjacency: Vec<(usize, usize)> = vec![(0, 0); running as usize];
        for (edge_id, &(first, second)) in edges.iter().enumerate() {
            adjacency[cursor[first] as usize] = (second, edge_id);
            cursor[first] += 1;
            adjacency[cursor[second] as usize] = (first, edge_id);
            cursor[second] += 1;
        }

        let mut discovery = vec![usize::MAX; atom_count];
        let mut low = vec![usize::MAX; atom_count];
        let mut is_bridge = vec![false; edges.len()];
        let mut timer = 0usize;
        // (atom, the edge it was entered by, index of the next neighbour)
        let mut stack: Vec<(usize, usize, usize)> = Vec::new();

        for start in 0..atom_count {
            if discovery[start] != usize::MAX {
                continue;
            }
            discovery[start] = timer;
            low[start] = timer;
            timer += 1;
            stack.push((start, usize::MAX, 0));

            while let Some(&(atom, entered_by, cursor)) = stack.last() {
                let neighbours =
                    &adjacency[neighbour_start[atom] as usize..neighbour_start[atom + 1] as usize];
                if cursor < neighbours.len() {
                    if let Some(top) = stack.last_mut() {
                        top.2 += 1;
                    }
                    let (neighbour, edge_id) = neighbours[cursor];
                    // Step over the edge we arrived on, but only that one: a
                    // second, parallel bond back to the same atom is a ring.
                    if edge_id == entered_by {
                        continue;
                    }
                    if discovery[neighbour] == usize::MAX {
                        discovery[neighbour] = timer;
                        low[neighbour] = timer;
                        timer += 1;
                        stack.push((neighbour, edge_id, 0));
                    } else {
                        low[atom] = low[atom].min(discovery[neighbour]);
                    }
                } else {
                    stack.pop();
                    if let Some(&(parent, _, _)) = stack.last() {
                        low[parent] = low[parent].min(low[atom]);
                        // Nothing below `atom` reaches back past `parent`, so
                        // the edge between them is the only connection.
                        if low[atom] > discovery[parent] {
                            is_bridge[entered_by] = true;
                        }
                    }
                }
            }
        }

        for (edge_id, &(first, second)) in edges.iter().enumerate() {
            if !is_bridge[edge_id] {
                visit(first, second);
            }
        }
    }

    fn determine_ring_systems(
        &self,
        atoms_in_rings: &BitSet,
        bonds_in_ring: &FxHashSet<(usize, usize)>,
    ) -> (Vec<Option<usize>>, usize) {
        let mut counter = 0;
        let mut ring_assignments = vec![None; self.len()];

        for atom_index in 0..self.len() {
            if atoms_in_rings.contains(atom_index) && ring_assignments[atom_index].is_none() {
                ring_assignments[atom_index] = Some(counter);
                self.traverse_ringsystem(
                    atom_index,
                    &mut ring_assignments,
                    &mut counter,
                    bonds_in_ring,
                );
                counter += 1;
            }
        }
        (ring_assignments, counter)
    }

    fn traverse_ringsystem(
        &self,
        atom_index: usize,
        ring_assignments: &mut [Option<usize>],
        counter: &mut usize,
        bonds_in_ring: &FxHashSet<(usize, usize)>,
    ) {
        let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
        for bond in bonds {
            if bonds_in_ring.contains(&(atom_index, bond.target()))
                && ring_assignments[atom_index].is_none()
            {
                ring_assignments[bond.target()] = Some(*counter);
                self.traverse_ringsystem(atom_index, ring_assignments, counter, bonds_in_ring);
            }
        }
    }
}

// Converts a cycle (represented as a Vec<usize>) to a BitSet for efficient comparison
///
/// # Arguments
/// * `cycle` - Vector of atom indices representing a cycle
///
/// # Returns
/// A BitSet where each set bit represents the presence of an atom in the cycle
///
/// # Example
/// ```
/// use oxidane::molecule::base::cycle_to_bitset;
///
/// let cycle = vec![0, 2, 4];
/// let bitset = cycle_to_bitset(&cycle);
/// assert_eq!(bitset.len(), 3);
/// assert!(bitset.contains(0));
/// assert!(bitset.contains(2));
/// assert!(bitset.contains(4));
/// ```
pub fn cycle_to_bitset(cycle: &[usize]) -> BitSet {
    let mut bitset = BitSet::new();
    for &atom_idx in cycle {
        bitset.insert(atom_idx);
    }
    bitset
}

/// Converts a list of cycles to a vector of BitSets for efficient comparison
///
/// # Returns
/// A vector of BitSets, each representing a cycle in the molecule
/// How a formal charge changes the number of bonds an element forms.
///
/// A charge is not simply "one bond fewer": which way it goes depends on where
/// the element sits in the periodic table, because the charge either spends an
/// electron from a lone pair or fills an empty orbital.
///
/// - Nitrogen and beyond have lone pairs to spend, so a positive charge buys a
///   bond and a negative one costs it: ammonium is tetravalent, the amide
///   anion divalent, hydroxide monovalent.
/// - Boron's group has an empty orbital instead, so the signs reverse:
///   borohydride is tetravalent.
/// - Carbon's group has neither, so either sign costs a bond — the
///   carbocation and the carbanion are both trivalent.
/// - Anything else, metals in particular, forms no bonds to satisfy here; a
///   naive `+ charge` would turn `[Na+]` into `[NaH2+]`.
///
/// Using `charge.abs()` for everything gets the anions right and every cation
/// wrong, which is what this replaced.
#[inline]
pub(crate) fn charge_valency_shift(atomic_number: u8, charge: i8) -> i8 {
    // The overwhelmingly common case, and the answer is zero for every branch
    // below, so it is worth not reaching them.
    if charge == 0 {
        return 0;
    }
    match atomic_number {
        // B, Al, Ga, In, Tl
        5 | 13 | 31 | 49 | 81 => -charge,
        // C, Si, Ge, Sn, Pb
        6 | 14 | 32 | 50 | 82 => -charge.abs(),
        // N, P, As, Sb, Bi
        7 | 15 | 33 | 51 | 83 => charge,
        // O, S, Se, Te, Po
        8 | 16 | 34 | 52 | 84 => charge,
        // F, Cl, Br, I, At
        9 | 17 | 35 | 53 | 85 => charge,
        _ => -charge.abs(),
    }
}

/// The valence a bare atom settles on once it already carries `bonded` bonds'
/// worth of them: the smallest of the element's normal valences that is at
/// least `bonded`.
///
/// This is the rule every other toolkit fills a bare atom's hydrogens by, and
/// it differs from [`effective_valency`] exactly where an element has more than
/// one normal valence and the bonds have already passed the first: a sulfur
/// with three bonds is read as taking one hydrogen, not none, because sulfur's
/// valences are 2, 4 and 6 and three bonds have outgrown the first. Writing
/// such an atom bare therefore hands back a different molecule, whatever this
/// crate's own reader would have made of it.
///
/// `None` when the bonds exceed every valence the element has, and for the
/// elements with no valence entry at all.
pub(crate) fn saturating_valency(atomic_number: u8, charge: i8, bonded: i8) -> Option<i8> {
    use chemistry_consts::ElementProperties;
    let shift = charge_valency_shift(atomic_number, charge);
    atomic_number
        .valencies()?
        // A negative entry is an ionic charge rather than a covalent valence.
        .filter(|valency| *valency > 0)
        .map(|valency| valency + shift)
        .find(|valency| *valency >= bonded)
}

/// The number of bonds an atom of this element and charge is expected to form.
#[inline]
pub(crate) fn effective_valency(atomic_number: u8, charge: i8) -> Option<i8> {
    use chemistry_consts::ElementProperties;
    Some(atomic_number.valencies()?.next()? + charge_valency_shift(atomic_number, charge))
}

pub fn cycles_to_bitsets(cycles: &[Vec<usize>]) -> Vec<BitSet> {
    cycles.iter().map(|cycle| cycle_to_bitset(cycle)).collect()
}
