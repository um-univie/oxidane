use crate::io::{SMILESParseError, SMILESParser, SmilesParseOptions};
use crate::{
    atom::Atom,
    molecule::{
        base::{Molecule, MoleculeCore},
        bond::{BondOrder, BondTarget},
        chirality::ChiralClass,
        traits::*,
    },
};
use bit_set::BitSet;
use chemistry_consts::ElementProperties;
use nohash_hasher::IntMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;
use tinyvec::ArrayVec;

impl ToSMILES for MolecularGraph {}
impl MoleculeChirality for MolecularGraph {}
impl MoleculeExperimental for MolecularGraph {}
impl MoleculeAromaticity for MolecularGraph {}
impl MoleculeGraph for MolecularGraph {}
impl Molecule for MolecularGraph {}

/// Error types for molecule splitting operations
#[derive(Debug, Clone, PartialEq)]
pub enum SplitError {
    /// The split would break existing bonds between atom groups
    BondWouldBreak {
        atom1: usize,
        atom2: usize,
        bond_order: BondOrder,
    },
    /// One or more atom indices are out of bounds
    InvalidAtomIndex(usize),
    /// The atom groups are empty
    EmptyGroups,
    /// Atom appears in multiple groups
    OverlappingGroups(usize),
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitError::BondWouldBreak {
                atom1,
                atom2,
                bond_order,
            } => {
                write!(
                    f,
                    "Split would break {:?} bond between atoms {} and {}",
                    bond_order, atom1, atom2
                )
            }
            SplitError::InvalidAtomIndex(idx) => {
                write!(f, "Atom index {} is out of bounds", idx)
            }
            SplitError::EmptyGroups => {
                write!(f, "Cannot split into empty groups")
            }
            SplitError::OverlappingGroups(atom) => {
                write!(f, "Atom {} appears in multiple groups", atom)
            }
        }
    }
}

impl std::error::Error for SplitError {}

/// This function
#[derive(Debug, Default, Clone)]
pub struct MolecularGraph {
    pub atomic_numbers: Vec<u8>,
    pub atom_classes: Option<Vec<usize>>,
    pub charges: Vec<i8>,
    pub chiral_classes: Option<Vec<ChiralClass>>,
    pub isotopes: Option<Vec<u16>>,
    pub partial_charges: Option<Vec<f64>>,
    pub radical_states: Vec<u8>,
    atom_bonds: Vec<ArrayVec<[BondTarget; 10]>>,
}

impl MoleculeCore for MolecularGraph {
    /// `partial_charges` is indexed by atom index when a molecule is split, so
    /// it has to track the atom count. A newly added atom has none of its own —
    /// nothing has computed one for it — and zero is the neutral reading.
    fn push_atom_extras(&mut self, _atom: &Atom) {
        if let Some(partial_charges) = self.partial_charges.as_mut() {
            partial_charges.push(0.0);
        }
    }

    fn pop_atom_extras(&mut self) {
        if let Some(partial_charges) = self.partial_charges.as_mut() {
            partial_charges.pop();
        }
    }

    fn remove_atom_extras(&mut self, atom_index: usize) {
        if let Some(partial_charges) = self.partial_charges.as_mut() {
            if atom_index < partial_charges.len() {
                partial_charges.remove(atom_index);
            }
        }
    }

    fn atom_bonds(&self) -> &Vec<ArrayVec<[BondTarget; 10]>> {
        &self.atom_bonds
    }
    fn atom_bonds_mut(&mut self) -> &mut Vec<ArrayVec<[BondTarget; 10]>> {
        &mut self.atom_bonds
    }
    fn atom_classes(&self) -> &Option<Vec<usize>> {
        &self.atom_classes
    }
    fn atom_classes_mut(&mut self) -> &mut Option<Vec<usize>> {
        &mut self.atom_classes
    }
    fn charges(&self) -> &[i8] {
        &self.charges
    }
    fn charges_mut(&mut self) -> &mut Vec<i8> {
        &mut self.charges
    }
    fn atomic_numbers(&self) -> &[u8] {
        &self.atomic_numbers
    }
    fn atomic_numbers_mut(&mut self) -> &mut Vec<u8> {
        &mut self.atomic_numbers
    }
    fn chiral_classes(&self) -> Option<&[ChiralClass]> {
        self.chiral_classes.as_deref()
    }
    fn chiral_classes_mut(&mut self) -> Option<&mut Vec<ChiralClass>> {
        self.chiral_classes.as_mut()
    }
    fn initialize_chiral_classes(&mut self) {
        if !self.chiral_classes.is_some() {
            self.chiral_classes = Some(vec![ChiralClass::None; self.len()]);
        }
    }
    fn isotopes(&self) -> Option<&Vec<u16>> {
        self.isotopes.as_ref()
    }
    fn isotopes_mut(&mut self) -> &mut Option<Vec<u16>> {
        &mut self.isotopes
    }
    fn radical_states(&self) -> &[u8] {
        &self.radical_states
    }
    fn radical_states_mut(&mut self) -> &mut Vec<u8> {
        &mut self.radical_states
    }

    fn is_atom_radical(&self, atom_index: usize) -> bool {
        self.radical_states[atom_index] > 0
    }

    fn set_atom_radical_state(&mut self, atom_index: usize, radical_state: u8) {
        self.radical_states[atom_index] = radical_state;
    }

    fn from_atoms(atoms: &[Atom]) -> Self {
        let atomic_numbers = atoms.iter().map(|atom| atom.atomic_number).collect();
        let charges = atoms.iter().map(|atom| atom.charge).collect();
        let radical_states = atoms.iter().map(|atom| atom.radical_state).collect();
        let atom_bonds = atoms.iter().map(|atom| atom.bonds).collect();
        let mut isotopes = None;
        let mut atom_classes = None;
        let mut chiral_classes = None;
        let mut partial_charges = None;

        // Zero means "no isotope was written for this atom". Filling in the
        // element's default instead made an unlabelled carbon indistinguishable
        // from one written `[12C]`, which is why `to_smiles` could not emit
        // isotopes at all; it also made an atom's stored isotope depend on
        // whether some *other* atom in the molecule happened to carry one.
        if atoms.iter().any(|atom| atom.isotope.is_some()) {
            isotopes = Some(atoms.iter().map(|atom| atom.isotope.unwrap_or(0)).collect());
        }
        if atoms.iter().any(|atom| atom.atom_class.is_some()) {
            atom_classes = Some(
                atoms
                    .iter()
                    .map(|atom| atom.atom_class.unwrap_or(0))
                    .collect(),
            );
        }
        if atoms.iter().any(|atom| atom.partial_charge.is_some()) {
            partial_charges = Some(
                atoms
                    .iter()
                    .map(|atom| atom.partial_charge.unwrap_or(0.0))
                    .collect(),
            );
        }
        if atoms
            .iter()
            .any(|atom| atom.chiral_class != ChiralClass::None)
        {
            chiral_classes = Some(atoms.into_iter().map(|atom| atom.chiral_class).collect());
        }

        MolecularGraph {
            atomic_numbers,
            charges,
            radical_states,
            atom_bonds,
            isotopes,
            chiral_classes,
            atom_classes,
            partial_charges,
        }
    }
}

impl MolecularGraph {
    pub fn from_smiles(smiles: &str) -> Result<Vec<Self>, SMILESParseError> {
        let parser = SMILESParser::parse_smiles(smiles)?;
        Ok(parser)
    }

    pub fn from_smiles_raw(smiles: &str) -> Result<Vec<Self>, SMILESParseError> {
        let parser = SMILESParser::parse_smiles_raw(smiles)?;
        Ok(parser)
    }

    pub fn from_smiles_single(smiles: &str) -> Result<Self, SMILESParseError> {
        let mut parser = SMILESParser::parse_smiles(smiles)?;
        let mol = parser.swap_remove(0);
        Ok(mol)
    }

    pub fn from_smiles_single_raw(smiles: &str) -> Result<Self, SMILESParseError> {
        let mut parser = SMILESParser::parse_smiles_raw(smiles)?;
        let mol = parser.swap_remove(0);
        Ok(mol)
    }

    /// The disjoint union of several molecules: one graph holding all of
    /// them, unconnected.
    ///
    /// SMILES parsing splits on `.`, so a molecule read from a string is
    /// always a single connected species. Anything that acts on two of them at
    /// once — an intermolecular reaction, say — needs them in one graph first.
    ///
    /// ```rust
    /// # use molecules::prelude::*;
    /// let methanol = MolecularGraph::from_smiles_single("CO").unwrap();
    /// let ammonia = MolecularGraph::from_smiles_single("N").unwrap();
    /// let both = MolecularGraph::combine(&[&methanol, &ammonia]);
    /// assert_eq!(both.to_smiles(), "CO.N");
    /// ```
    pub fn combine(molecules: &[&Self]) -> Self {
        let mut atoms: Vec<Atom> = Vec::new();
        for molecule in molecules {
            let offset = atoms.len();
            for index in 0..molecule.atomic_numbers().len() {
                let mut bonds = molecule.atom_bonds()[index];
                for bond in bonds.iter_mut() {
                    bond.set_target(bond.target() + offset);
                }
                let mut atom = Atom::new(molecule.atomic_numbers()[index])
                    .with_charge(molecule.charges()[index])
                    .with_radical(molecule.radical_states()[index])
                    .with_bonds(bonds);
                if let Some(isotope) = molecule.get_isotope(index) {
                    atom = atom.with_isotope(isotope);
                }
                if let Some(classes) = molecule.chiral_classes() {
                    atom = atom.with_chiral_class(classes[index]);
                }
                if let Some(classes) = molecule.atom_classes() {
                    atom = atom.with_atom_class(classes.get(index).copied());
                }
                atoms.push(atom);
            }
        }
        Self::from_atoms(&atoms)
    }

    /// Like [`MolecularGraph::from_smiles`], with control over what is
    /// perceived after parsing — see [`SmilesParseOptions`].
    ///
    /// ```rust
    /// # use molecules::prelude::*;
    /// use molecules::io::SmilesParseOptions;
    ///
    /// // Kekulé benzene. By default the ring is perceived as aromatic, so it
    /// // answers the same questions as `c1ccccc1`.
    /// let perceived = MolecularGraph::from_smiles_single("C1=CC=CC=C1").unwrap();
    /// assert_eq!(perceived.to_smiles(), "c1ccccc1");
    ///
    /// // Asking for no perception keeps the bonds exactly as written.
    /// let literal = MolecularGraph::from_smiles_single_with(
    ///     "C1=CC=CC=C1",
    ///     SmilesParseOptions { aromaticity: None },
    /// )
    /// .unwrap();
    /// assert_eq!(literal.to_smiles(), "C1=CC=CC=C1");
    /// ```
    pub fn from_smiles_with(
        smiles: &str,
        options: SmilesParseOptions,
    ) -> Result<Vec<Self>, SMILESParseError> {
        SMILESParser::parse_smiles_with(smiles, options)
    }

    /// Like [`MolecularGraph::from_smiles_single`], with control over what is
    /// perceived after parsing — see [`SmilesParseOptions`].
    pub fn from_smiles_single_with(
        smiles: &str,
        options: SmilesParseOptions,
    ) -> Result<Self, SMILESParseError> {
        SMILESParser::parse_smiles_single_with(smiles, options)
    }
    fn count_atom_pi_electrons(&self, atom_idx: usize) -> usize {
        // Count pi electrons from multiple bonds
        let pi_electrons = self.atom_bonds()[atom_idx]
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Double => 1, // One pi electron per double bond
                BondOrder::Triple => 2, // Two pi electrons per triple bond
                _ => 0,
            })
            .sum();

        // Add electrons from lone pairs for certain atoms
        let atomic_number = self.atomic_numbers()[atom_idx];
        match atomic_number {
            7 => {
                // Nitrogen
                if self.get_atom_charge(atom_idx) == 0 && pi_electrons == 0 {
                    2 // Neutral N with no pi bonds contributes lone pair
                } else {
                    pi_electrons
                }
            }
            8 | 16 => {
                // Oxygen or Sulfur
                if self.get_atom_charge(atom_idx) == 0 && pi_electrons == 0 {
                    2 // Neutral O/S with no pi bonds contributes lone pair
                } else {
                    pi_electrons
                }
            }
            _ => pi_electrons,
        }
    }

    /// Split the molecule into multiple molecules based on atom indices.
    ///
    /// This method creates separate molecules from groups of atoms, but only if the split
    /// doesn't break any existing bonds. If any bond would be broken between atoms in
    /// different groups, the method returns an error.
    ///
    /// # Arguments
    ///
    /// * `atom_groups` - A vector of vectors where each inner vector contains the atom
    ///   indices that should form a separate molecule.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<MolecularGraph>, SplitError>` - Ok containing the split molecules if the
    ///   split is valid, or an Error if bonds would be broken or other validation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use molecules::prelude::*;
    /// // Ethanol and ammonia in one graph, split back apart. The groups have
    /// // to be whole connected pieces: splitting through a bond is an error.
    /// let methanol = MolecularGraph::from_smiles_single("CO").unwrap();
    /// let ammonia = MolecularGraph::from_smiles_single("N").unwrap();
    /// let combined = MolecularGraph::combine(&[&methanol, &ammonia]);
    ///
    /// let methanol_atoms: Vec<usize> = (0..methanol.atomic_numbers().len()).collect();
    /// let ammonia_atoms: Vec<usize> =
    ///     (methanol.atomic_numbers().len()..combined.atomic_numbers().len()).collect();
    /// let parts = combined.split(vec![methanol_atoms, ammonia_atoms]).unwrap();
    /// assert_eq!(parts.len(), 2);
    /// ```
    pub fn split(&self, atom_groups: Vec<Vec<usize>>) -> Result<Vec<MolecularGraph>, SplitError> {
        // Validate input groups
        self.validate_split_groups(&atom_groups)?;

        // Check that no bonds would be broken
        self.validate_no_broken_bonds(&atom_groups)?;

        // Perform the actual split
        let mut result_molecules = Vec::new();

        for group in atom_groups {
            let molecule = self.create_molecule_from_atoms(&group)?;
            result_molecules.push(molecule);
        }

        Ok(result_molecules)
    }

    /// Validates that the atom groups are well-formed
    fn validate_split_groups(&self, atom_groups: &[Vec<usize>]) -> Result<(), SplitError> {
        if atom_groups.is_empty() || atom_groups.iter().all(|group| group.is_empty()) {
            return Err(SplitError::EmptyGroups);
        }

        let mut seen_atoms = BitSet::with_capacity(self.atomic_numbers().len());

        for group in atom_groups {
            for &atom_idx in group {
                // Check if atom index is valid
                if atom_idx >= self.atomic_numbers().len() {
                    return Err(SplitError::InvalidAtomIndex(atom_idx));
                }

                // Check for overlapping groups
                if !seen_atoms.insert(atom_idx) {
                    return Err(SplitError::OverlappingGroups(atom_idx));
                }
            }
        }

        Ok(())
    }

    /// Validates that no bonds would be broken by the split
    fn validate_no_broken_bonds(&self, atom_groups: &[Vec<usize>]) -> Result<(), SplitError> {
        // Create a mapping from atom index to group index using IntMap for better performance
        let mut atom_to_group = IntMap::default();
        for (group_idx, group) in atom_groups.iter().enumerate() {
            for &atom_idx in group {
                atom_to_group.insert(atom_idx, group_idx);
            }
        }

        // Check all bonds to ensure they don't cross group boundaries
        for (atom_idx, bonds) in self.atom_bonds().iter().enumerate() {
            if let Some(&group_idx) = atom_to_group.get(&atom_idx) {
                for bond in bonds {
                    let target_atom = bond.target();
                    if let Some(&target_group_idx) = atom_to_group.get(&target_atom) {
                        if group_idx != target_group_idx {
                            // This bond crosses group boundaries, which would break it
                            return Err(SplitError::BondWouldBreak {
                                atom1: atom_idx,
                                atom2: target_atom,
                                bond_order: bond.bond_order(),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Creates a new molecule from a subset of atoms
    fn create_molecule_from_atoms(
        &self,
        atom_indices: &[usize],
    ) -> Result<MolecularGraph, SplitError> {
        if atom_indices.is_empty() {
            return Err(SplitError::EmptyGroups);
        }

        // Create mapping from old indices to new indices using IntMap for better performance
        let mut old_to_new = IntMap::default();
        for (new_idx, &old_idx) in atom_indices.iter().enumerate() {
            old_to_new.insert(old_idx, new_idx);
        }

        // Extract atomic data for selected atoms
        let atomic_numbers: Vec<u8> = atom_indices
            .iter()
            .map(|&idx| self.atomic_numbers[idx])
            .collect();

        let charges: Vec<i8> = atom_indices.iter().map(|&idx| self.charges[idx]).collect();

        let radical_states: Vec<u8> = atom_indices
            .iter()
            .map(|&idx| self.radical_states[idx])
            .collect();

        // Extract optional fields - only create them if the original molecule has them
        // This maintains the same behavior as from_atoms: None when no atoms have the property
        let atom_classes = self
            .atom_classes
            .as_ref()
            .map(|classes| atom_indices.iter().map(|&idx| classes[idx]).collect());

        let chiral_classes = self
            .chiral_classes
            .as_ref()
            .map(|classes| atom_indices.iter().map(|&idx| classes[idx]).collect());

        let isotopes = self
            .isotopes
            .as_ref()
            .map(|isotopes| atom_indices.iter().map(|&idx| isotopes[idx]).collect());

        let partial_charges = self
            .partial_charges
            .as_ref()
            .map(|charges| atom_indices.iter().map(|&idx| charges[idx]).collect());

        // Extract and remap bonds
        let mut atom_bonds = vec![ArrayVec::new(); atom_indices.len()];

        for (new_idx, &old_idx) in atom_indices.iter().enumerate() {
            for bond in &self.atom_bonds[old_idx] {
                let target = bond.target();
                // Only include bonds to atoms that are also in this group
                if let Some(&new_target) = old_to_new.get(&target) {
                    atom_bonds[new_idx].push(BondTarget::new(new_target, bond.bond_order()));
                }
            }
        }

        Ok(MolecularGraph {
            atomic_numbers,
            atom_classes,
            charges,
            chiral_classes,
            isotopes,
            partial_charges,
            radical_states,
            atom_bonds,
        })
    }
}
