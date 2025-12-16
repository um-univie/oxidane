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
        // Remove the atom's data
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

        // Update all remaining bonds
        for current_atom_bonds in self.atom_bonds_mut() {
            // Remove any bonds to the deleted atom
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

    /// This function returns the valency_delta of an atom
    ///
    ///
    /// # Examples
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_smiles("C1CCCCC1CC2CCCCC2");
    /// assert_eq!(molecule.valency_delta(0), Some(0));
    fn valency_delta(&self, atom_index: usize) -> Option<i8> {
        let expected_valency = self.expected_valency(atom_index)?;
        let actual_valency = self.actual_valency(atom_index);
        Some(expected_valency - actual_valency)
    }

    /// Checks if an atom has a plausible valency based on its element type, charge, and bonding.
    ///
    /// This function determines whether the actual valency of an atom is chemically reasonable
    /// by comparing it with the expected valency for that element, while accounting for:
    /// - Formal charges (which can modify acceptable valencies)
    /// - Radical states (which can affect bonding capacity)
    /// - Special cases for hypervalent elements (P, S) and transition metals
    ///
    /// # Arguments
    /// * `atom_index` - The index of the atom to check
    ///
    /// # Returns
    /// `true` if the atom's valency is plausible, `false` otherwise
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
    /// assert!(molecule.has_plausible_valency(0)); // Sulfur can be hypervalent
    /// ```
    fn has_plausible_valency(&self, atom_index: usize) -> bool {
        // Get the expected valency for this atom
        let expected_valency = match self.expected_valency(atom_index) {
            Some(val) => val,
            None => return true, // If we don't know the expected valency, assume it's plausible
        };

        // Get the actual valency
        let actual_valency = self.actual_valency(atom_index);
        // Get the charge and radical state
        let charge = self.get_atom_charge(atom_index);
        let is_radical = self.is_atom_radical(atom_index);
        // Calculate the valency delta (expected - actual)
        let valency_delta = expected_valency - actual_valency;

        // For most elements, the valency delta should be non-negative
        // Exceptions are made for charged atoms and radicals
        if valency_delta < 0 {
            // Atoms with positive charge can have higher valency
            if charge > 0 && valency_delta + charge >= 0 {
                return true;
            }

            // Radicals can have one less bond than expected
            if is_radical && valency_delta + 1 >= 0 {
                return true;
            }

            // Special cases for specific elements
            let atomic_number = self.get_atomic_number(atom_index);
            match atomic_number {
                // Hypervalent elements like P, S can exceed their usual valency
                15 => return valency_delta < -4, // Phosphorus, maximum valency is 7
                16 => return valency_delta < -5, // Sulfur standard valency is 2, maximum valency is 6
                // Transition metals often have variable valencies
                21..=30 | 39..=48 | 57..=80 | 89..=112 => return true,
                // Default case - valency should match expected
                _ => return false,
            }
        }

        // If valency_delta is non-negative, the valency is plausible
        true
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

    /// This function returns the oxidation state of an atom
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Example
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_xyz("tests/ethane.xyz").unwrap().remove(0);
    /// assert_eq!(molecule.get_oxidation_state(0), -3);
    ///
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

    /// This function returns the number of bonded elements of a specific type in case of invalid atom index it returns 0 to keep the api simple
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

    /// This function returns the actual valency of an atom. In case of fractional valency it returns the rounded down value (e.g. fused aromatic systems are treated as aromatic)
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Returns
    /// The actual valency of the atom as an `i8`
    ///
    ///
    fn actual_valency(&self, atom_index: usize) -> i8 {
        (self.atom_bonds()[atom_index]
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Single => 2,
                BondOrder::Up => 2,
                BondOrder::Down => 2,
                BondOrder::Double => 4,
                BondOrder::Triple => 6,
                BondOrder::Quadruple => 8,
                BondOrder::Aromatic => 3,
                BondOrder::Coordinate => 2,
            })
            .sum::<i8>())
            / 2
    }

    /// This function returns the degree of an atom, which is the sum of the valency delta, the charge and the radical state
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Returns
    /// The degree of the atom as an `i8`
    ///
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

    /// This function returns the degree of all atoms in the molecule
    ///
    /// # Arguments
    /// * 'self' - The molecule
    ///
    /// # Returns
    /// A vector of degrees for all atoms in the molecule as an `i8`
    ///
    fn degrees(&self) -> Vec<i8> {
        self.atomic_numbers()
            .iter()
            .enumerate()
            .map(|(index, _)| self.get_atom_degree(index).unwrap_or(0))
            .collect()
    }

    /// This function returns the monoisotopic mass of the molecule, based on the most common isotope of each element
    ///
    /// # Arguments
    /// * 'self' - The molecule
    ///
    /// # Returns
    /// The monoisotopic mass of the molecule as an `f64`
    fn monoisotopic_mass(&self) -> f64 {
        self.atomic_numbers()
            .iter()
            .map(|atom| atom.monoisotopic_mass().unwrap_or(0.0))
            .sum::<f64>()
    }

    /// This function returns the number of atoms in the molecule, which is the length of the atomic numbers vector
    ///
    /// # Arguments
    /// * 'self' - The molecule
    ///
    /// # Returns
    /// The number of atoms in the molecule as an `usize`
    fn number_of_atoms(&self) -> usize {
        self.atomic_numbers().len()
    }

    /// This function returns the formal charge of the molecule, which is the sum of the charges of all atoms
    ///
    /// # Arguments
    /// * 'self' - The molecule
    ///
    /// # Returns
    /// The formal charge of the molecule as an `i32`
    fn formal_charge(&self) -> i32 {
        self.charges().iter().sum::<i8>() as i32
    }

    /// This function returns a copy of the formal charge of an atom
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Returns
    /// The charge of the atom as an `i8`
    fn get_atom_charge(&self, atom_index: usize) -> i8 {
        self.charges()[atom_index]
    }

    /// This function returns a mutable reference to the formal charge of an atom
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Returns
    /// The charge of the atom as an `i8`
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

    /// This function returns the chiral class of an atom, if one is specified
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Returns
    /// The chiral class of the atom as an `Option<ChiralClass>`
    fn get_chiral_class(&self, atom_index: usize) -> ChiralClass {
        self.chiral_classes()
            .and_then(|chirals| chirals.get(atom_index).copied())
            .unwrap_or(ChiralClass::None)
    }

    /// This function returns whether an atom is radical, if no radical state is specified we assume the atom is not radical
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    ///
    /// # Returns
    /// The radical state of the atom as a `bool`
    fn is_atom_radical(&self, atom_index: usize) -> bool {
        self.radical_states().get(atom_index).is_some_and(|&state| state > 0)
    }

    /// This function sets the radical state of an atom
    ///
    /// # Arguments
    /// * 'atom_index' - The index of the atom
    /// * 'is_radical' - The new radical state of the atom
    ///
    /// # Returns
    /// The radical state of the atom as a `bool`
    fn set_atom_radical_state(&mut self, atom_index: usize, radical_state: u8) {
        if atom_index < self.radical_states().len() {
            self.radical_states_mut()[atom_index] = radical_state;
        } else {
            debug_assert!(false, "atom index out of bounds, could not set radical state");
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

    /// This function builds a smiles tree from the molecule
    ///
    /// # Arguments
    /// * 'start_index' - The index of the atom to start the tree from
    /// * 'bond_type' - The type of bond to start the tree from
    /// * 'parent_index' - The index of the parent atom
    /// * 'visited' - A vector of visited atoms
    /// * 'ring_closures' - A vector of ring closures
    /// * 'traversal_order' - A vector of the traversal order of the tree
    ///
    /// # Returns
    /// The root of the smiles tree
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

        // A hydrogen folds into its neighbour's `H` count and needs no node of
        // its own — unless it carries an isotope, which can only be written as
        // a separate atom (`O[2H]`). Checking whether the molecule has any
        // isotopes at all first keeps the common case to one `Option` test.
        let has_labelled_hydrogens = self.isotopes().is_some();

        for neighbor in bonds {
            if Some(neighbor.target()) == parent_index {
                continue;
            }

            if self.atomic_numbers()[neighbor.target()] == 1
                && !(has_labelled_hydrogens && self.get_isotope(neighbor.target()).is_some())
            {
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

        // Get all bonds for current atom
        if let Some(bonds) = self.get_atom_bonds(current) {
            // Process each bond
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
        let degrees = self.degrees();
        degrees.iter().enumerate().for_each(|(index, &degree)| {
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
                None => -degree,
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

    // Helper methods
    fn get_bond_order(&self, atom1: usize, atom2: usize) -> BondOrder {
        self.atom_bonds()[atom1]
            .iter()
            .find(|bond| bond.target() == atom2)
            .map(|bond| bond.bond_order())
            .unwrap_or(BondOrder::Single)
    }

    fn check_for_metal_center(&self, system: &BitSet) -> bool {
        // Check if any atom is a transition metal
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

        // Initialize tree nodes with capacity
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

        // Process nodes until stack is empty
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

                // Check if other node is already in tree
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

                    // Find common ancestor more efficiently
                    let mut common_ancestor = None;
                    for (i, &n1) in cycle.iter().enumerate() {
                        if let Some(j) = other_path.iter().position(|&n2| n1 == n2) {
                            common_ancestor = Some((i, j));
                            break;
                        }
                    }

                    if let Some((i, j)) = common_ancestor {
                        // Construct fundamental cycle more efficiently
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
                    // Add other node to tree
                    tree[other_node].parent = Some(current_node);
                    node_stack.push(other_node);
                }

                // Remove processed bond
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
    /// A Vec<usize> containing the atom indices in order
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
                // Find intersection of cycles
                let shared_atoms = cycles_vec[i].intersection(&cycles_vec[j]);

                // If cycles share at least 2 atoms and aren't identical
                if shared_atoms.into_iter().sum::<usize>() >= 2 {
                    // Create new cycle by the union of the two cycles
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

    fn remove_tautomeric_hydrogens(&mut self, tautomers: &mut [bool]) -> Vec<usize> {
        let mut hydrogens_to_remove = Vec::new();
        for (atom_index, &is_tautomeric) in tautomers.iter().enumerate() {
            if is_tautomeric {
                let bond_count = self.atom_bonds()[atom_index].len();
                let expected_valency = self.expected_valency(atom_index).unwrap_or(0);
                if bond_count + 1 > expected_valency as usize {
                    let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
                    for bond in bonds.iter() {
                        if self.get_atomic_number(bond.target()) == 1 {
                            hydrogens_to_remove.push(bond.target());
                            break;
                        }
                    }
                }
            }
        }
        for hydrogen_index in hydrogens_to_remove.iter() {
            self.remove_atom(*hydrogen_index);
        }
        hydrogens_to_remove
    }

    fn find_tautomers(&self) -> Vec<bool> {
        let mut is_tautomeric = vec![false; self.len()];
        let mut stack = Vec::new();
        let mut visited = vec![false; self.len()];

        // Find all oxygen atoms as starting points
        let oxygens = self
            .atomic_numbers()
            .iter()
            .enumerate()
            .filter(|&(_, atomic_number)| *atomic_number == 8)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        // Process each oxygen atom as a potential start of a tautomeric path
        for &start_oxygen in &oxygens {
            // Reset visited for each new oxygen starting point
            visited.fill(false);

            // Mark hydrogens as visited since they can't be part of conjugated system
            for (atom_index, &atomic_number) in self.atomic_numbers().iter().enumerate() {
                if atomic_number == 1 {
                    visited[atom_index] = true;
                }
            }

            // Initialize search from this oxygen
            stack.push((start_oxygen, Vec::new(), None, 0));
            visited[start_oxygen] = true;

            while let Some((current, current_path, last_bond_order, path_length)) = stack.pop() {
                let bonds = self.atom_bonds()[current].clone();

                // Check if there are any unvisited double bonds
                let has_unvisited_double_bonds = bonds
                    .iter()
                    .any(|bond| !visited[bond.target()] && bond.bond_order() == BondOrder::Double);

                // If we have a valid path and no unvisited double bonds that could lead to longer paths
                if !has_unvisited_double_bonds && last_bond_order.is_some() {
                    // Mark the entire path as tautomeric
                    is_tautomeric[start_oxygen] = true;
                    for &(atom, _, _) in &current_path {
                        is_tautomeric[atom] = true;
                    }
                    is_tautomeric[current] = true;
                }

                for bond in bonds {
                    let target = bond.target();

                    if visited[target] {
                        continue;
                    }

                    let is_valid_continuation = match last_bond_order {
                        None => true,
                        Some(BondOrder::Single) => bond.bond_order() == BondOrder::Double,
                        Some(BondOrder::Double) => bond.bond_order() == BondOrder::Single,
                        _ => false,
                    };

                    if is_valid_continuation {
                        let mut new_path = current_path.clone();
                        new_path.push((current, target, bond.bond_order()));

                        stack.push((target, new_path, Some(bond.bond_order()), path_length + 1));
                        visited[target] = true;
                    }
                }
            }
        }

        is_tautomeric
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
