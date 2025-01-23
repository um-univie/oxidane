use std::collections::HashMap;

use crate::atom::Atom;
use crate::molecule::aromaticity::{AromaticityModel, MoleculeAromaticity};
use crate::atom::{WILDCARD_ATOMIC_NUMBER, WILDCARD_SYMBOL};
use crate::molecule::Molecule;
use crate::molecule::graph::Node;
pub use crate::{
    molecule::bond::{BondOrder, BondTarget},
    molecule::chirality::ChiralClass,
    molecule::molecular_graph::MolecularGraph,
    molecule::traits::*,
};
pub use chemistry_consts::ElementProperties;
pub use nohash_hasher::IntMap;

#[derive(Debug, Clone)]
pub enum SMILESParseError {
    NoMolecules,
    ElementNotFound(String),
    BondNotFound,
    RingIndexError,
    InvalidBranch,
    InvalidAromatic(String),
    InvalidChiralClass,
    SMILESComplexError,
    Charge,
    ChiralClass(String),
    AtomClassMissing,
    EOL,
    /// A character that is not part of the SMILES language appeared where an
    /// atom, bond or structural symbol was expected.
    UnexpectedCharacter(char, String),
}

/// What to perceive after the atoms and bonds have been read.
///
/// Parsing itself records exactly what the string said; everything here is a
/// *derived* property that a SMILES may leave implicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmilesParseOptions {
    /// Which aromaticity model to run, or `None` to leave bonds exactly as
    /// written.
    ///
    /// On by default, because a Kekulé ring and a lower-case ring are the same
    /// molecule and should answer the same questions: without perception,
    /// `C1=CC=CC=C1` has no aromatic bond and so does not match the SMARTS
    /// `c1ccccc1`, while `c1ccccc1` does. Turn it off to inspect a structure
    /// exactly as it was written, or when the input is already Kekulé and the
    /// distinction matters.
    pub aromaticity: Option<AromaticityModel>,
}

impl Default for SmilesParseOptions {
    fn default() -> Self {
        Self { aromaticity: Some(AromaticityModel::OpenSMILES) }
    }
}

#[derive(Default)]
pub struct SMILESParser<'a> {
    smiles: &'a str,
    atoms: Vec<Atom>,
    bonds: Vec<(usize, usize, Option<BondOrder>)>,
    current_atom_index: usize,
    current_atom_charge: Option<i8>,
    current_atom_is_complex: bool,
    element_buffer: String,
    isotope: Option<u16>,
    chiral_class: ChiralClass,
    current_atom_class: Option<usize>,
    last_bond_type: Option<BondOrder>,
    ring_number: Option<usize>,
    ring_bonds: IntMap<usize, (Option<usize>, Option<usize>, Option<BondOrder>)>,
    rings: Vec<(usize, usize, Option<BondOrder>)>,
    hydrogens: IntMap<usize, u8>,
    branch_stack: Vec<usize>,
    branch_exits: usize,
    is_multiple_branch: bool,
    is_double_digit: bool,
    /// Per-atom neighbor declarations in SMILES written order. Non-negative
    /// values are atom indices; a value `v < 0` is the ring-bond placeholder with
    /// id `-v - 1`, resolved to the partner atom via `write_ring_partner`.
    write_order: IntMap<usize, Vec<i64>>,
    /// Atoms that have an incoming "from" bond (i.e. are not a fragment root),
    /// whose first `write_order` entry is therefore the from-atom.
    write_has_from: IntMap<usize, ()>,
    /// Ring number currently open → placeholder id.
    write_ring_open: IntMap<usize, usize>,
    /// Placeholder id → the two atoms of that ring bond.
    write_ring_partner: Vec<[usize; 2]>,
}

impl<'a> SMILESParser<'a> {
    /// Empties the parser for the next fragment or the next string, keeping
    /// every buffer's allocation.
    ///
    /// Replacing the whole parser instead — which is what the fragment loop
    /// used to do at each `.` — drops the atom, bond and ring tables and
    /// reallocates them from nothing for the next fragment.
    fn reset(&mut self) {
        self.atoms.clear();
        self.bonds.clear();
        self.element_buffer.clear();
        self.ring_bonds.clear();
        self.rings.clear();
        self.hydrogens.clear();
        self.branch_stack.clear();
        self.write_order.clear();
        self.write_has_from.clear();
        self.write_ring_open.clear();
        self.write_ring_partner.clear();

        self.current_atom_index = 0;
        self.current_atom_charge = None;
        self.current_atom_is_complex = false;
        self.isotope = None;
        self.chiral_class = ChiralClass::None;
        self.current_atom_class = None;
        self.last_bond_type = None;
        self.ring_number = None;
        self.branch_exits = 0;
        self.is_multiple_branch = false;
        self.is_double_digit = false;
    }

    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }
    /// Parses a SMILES string and returns a Molecule
    /// # Arguments
    /// * `smiles` - A string slice that holds the SMILES string
    ///
    /// # Example
    /// ```
    /// use molecules::prelude::*;
    /// let molecules = MolecularGraph::from_smiles("C(C(C))COCCl").unwrap();
    /// assert_eq!(molecules[0].atomic_numbers().len(), 18);
    /// assert_eq!(molecules[0].get_edges().len(), 17);
    /// ```
    pub fn parse_smiles(smiles: &str) -> Result<Vec<MolecularGraph>, SMILESParseError> {
        Self::parse_smiles_with(smiles, SmilesParseOptions::default())
    }

    /// Parses every fragment of `smiles`, with control over what is perceived
    /// afterwards — see [`SmilesParseOptions`].
    pub fn parse_smiles_with(
        smiles: &str,
        options: SmilesParseOptions,
    ) -> Result<Vec<MolecularGraph>, SMILESParseError> {
        let (mut molecules, write_orders) = SMILESParser::parse_fragments(smiles)?;
        for (molecule, written) in molecules.iter_mut().zip(write_orders.iter()) {
            molecule.add_hydrogens();

            Self::assign_proper_chirality(molecule);
            crate::molecule::cip::normalize_tetrahedral_references(molecule, written);
            crate::molecule::cip::normalize_allene_references(molecule, written);
            molecule.convert_directional_bonds_to_chiral();

            if let Some(model) = options.aromaticity {
                // A ring written in Kekulé form describes the same molecule as
                // one written in lower case, so without this the two parse to
                // graphs that answer aromaticity questions differently:
                // `c1ccccc1` would match `c1ccccc1` and not `C1=CC=CC=C1`.
                // Failure here means the perception could not be completed,
                // which leaves the bonds exactly as written — the same result
                // as not asking for it.
                let _ = molecule.aromatize(model, None, false);
            }
        }
        Ok(molecules)
    }

    pub fn parse_smiles_single(smiles: &str) -> Result<MolecularGraph, SMILESParseError> {
        Self::parse_smiles_single_with(smiles, SmilesParseOptions::default())
    }

    /// Parses the first fragment of `smiles`, with control over what is
    /// perceived afterwards — see [`SmilesParseOptions`].
    pub fn parse_smiles_single_with(
        smiles: &str,
        options: SmilesParseOptions,
    ) -> Result<MolecularGraph, SMILESParseError> {
        let mut molecules = Self::parse_smiles_with(smiles, options)?;
        if molecules.is_empty() {
            return Err(SMILESParseError::NoMolecules);
        }
        Ok(molecules.swap_remove(0))
    }
    pub fn parse_smiles_raw_single(smiles: &str) -> Result<MolecularGraph, SMILESParseError> {
        let mut molecules = SMILESParser::parse_smiles_raw(smiles)?;
        let mol = molecules.swap_remove(0);
        Ok(mol)
    }

    /// Resolves the captured per-atom written neighbor order for the current
    /// fragment. Each value is `(has_from_atom, neighbor_indices_in_written_order)`
    /// with ring placeholders resolved to their partner atoms.
    fn resolve_write_orders(&self) -> IntMap<usize, (bool, Vec<usize>)> {
        let mut out = IntMap::default();
        for (&atom, decls) in self.write_order.iter() {
            let resolved: Vec<usize> = decls
                .iter()
                .map(|&d| {
                    if d >= 0 {
                        d as usize
                    } else {
                        let id = (-d - 1) as usize;
                        let [a, b] = self.write_ring_partner[id];
                        if a == atom { b } else { a }
                    }
                })
                .collect();
            out.insert(atom, (self.write_has_from.contains_key(&atom), resolved));
        }
        out
    }

    pub fn parse_smiles_raw(smiles: &str) -> Result<Vec<MolecularGraph>, SMILESParseError> {
        Self::parse_fragments(smiles).map(|(molecules, _)| molecules)
    }

    /// Parses a SMILES string into molecules, returning for each the per-atom
    /// SMILES written neighbor order (used to normalize tetrahedral chirality).
    #[allow(clippy::type_complexity)]
    fn parse_fragments(
        smiles: &str,
    ) -> Result<(Vec<MolecularGraph>, Vec<IntMap<usize, (bool, Vec<usize>)>>), SMILESParseError>
    {
        let mut parser = SMILESParser::default();
        parser.smiles = smiles;
        // One atom per two bytes is a generous upper bound for SMILES and
        // costs one allocation instead of the handful a growing vector takes.
        parser.atoms.reserve(smiles.len() / 2 + 1);
        parser.bonds.reserve(smiles.len() / 2 + 1);
        let mut molecules = Vec::with_capacity(smiles.bytes().filter(|&c| c == b'.').count());
        let mut write_orders: Vec<IntMap<usize, (bool, Vec<usize>)>> = Vec::new();

        // SMILES should only be ASCII so we could bypass the UTF-8 checks using bytes()
        let bytes = smiles.as_bytes();
        let mut pointer = 0;
        while pointer < bytes.len() {
            let Some(&byte) = bytes.get(pointer) else {
                break;
            };
            pointer += 1;
            match byte {
                b'A'..=b'Z' => {
                    // Handle the previous element
                    parser.handle_atom(Some(byte))?
                }
                // The wildcard atom: an atom whose element is deliberately
                // unspecified, used for attachment points and polymer end
                // caps. It is a real atom in the graph, carried as atomic
                // number 0.
                b'*' => parser.handle_atom(Some(byte))?,
                b'a'..=b'z' => {
                    // According to the SMILES specification, the lowercase letters are used to denote aromatic atoms if they are not complex
                    match byte {
                        b'b' | b'c' | b'n' | b'o' | b's' | b'p' => {
                            parser.handle_atom(Some(byte))?
                        }
                        b'r' => {
                            if parser.element_buffer == "B" {
                                parser.element_buffer.push('R')
                            } else {
                                return Err(SMILESParseError::InvalidAromatic(format!(
                                    "{} in {}",
                                    parser.element_buffer, smiles
                                )));
                            }
                        }

                        b'l' => {
                            if parser.element_buffer == "C" {
                                parser.element_buffer.push('L')
                            } else {
                                return Err(SMILESParseError::InvalidAromatic(format!(
                                    "{} in {}",
                                    parser.element_buffer, smiles
                                )));
                            }
                        }
                        anything_else => {
                            parser.element_buffer.push(anything_else as char);
                            return Err(SMILESParseError::InvalidAromatic(format!(
                                "{} in {}",
                                parser.element_buffer, smiles
                            )));
                        }
                    }
                }

                b'.' => {
                    parser.handle_atom(None)?;
                    parser.add_all_bonds();
                    // If we still have unsatisfied ring bonds, we need to wait
                    // This does not yet solve the issue as we will need to split mols in the end
                    if !parser
                        .ring_bonds
                        .iter()
                        .any(|(_, (a, b, _))| a.is_none() || b.is_none())
                    {
                        let molecule = MolecularGraph::from_atoms(&parser.atoms);
                        molecules.push(molecule);
                        write_orders.push(parser.resolve_write_orders());
                        parser.reset();
                    }
                }

                b'0'..=b'9' => {
                    parser.handle_number(byte)?;
                }
                b'-' => {
                    parser.last_bond_type = Some(BondOrder::Single);
                }
                b'/' => {
                    parser.last_bond_type = Some(BondOrder::Up);
                }
                b'\\' => {
                    parser.last_bond_type = Some(BondOrder::Down);
                }
                b'=' => {
                    parser.last_bond_type = Some(BondOrder::Double);
                }
                b'#' => {
                    parser.last_bond_type = Some(BondOrder::Triple);
                }
                b'$' => {
                    parser.last_bond_type = Some(BondOrder::Quadruple);
                }
                b':' => {
                    parser.last_bond_type = Some(BondOrder::Aromatic);
                }

                b'(' => {
                    if parser.branch_exits > 0 {
                        parser.is_multiple_branch = true;
                    } else {
                        parser.branch_stack.push(parser.current_atom_index);
                    }
                }
                b')' => parser.branch_exits += 1,
                b'[' => {
                    parser.handle_complex_atom(bytes, &mut pointer)?;
                }

                b'%' => {
                    parser.is_double_digit = true;
                }
                // A SMILES string ends at whitespace; whatever follows is
                // other data (a name, an identifier) and is not structure.
                b' ' | b'\t' | b'\n' | b'\r' => break,

                // Anything else is not SMILES. Stopping quietly here — which
                // is what this used to do — turned a typo into silent data
                // loss: `*C(F)(Cl)Br` returned `Ok` with an empty molecule
                // because parsing broke on the first character, and `C&CC`
                // returned a lone carbon.
                unexpected => {
                    return Err(SMILESParseError::UnexpectedCharacter(
                        unexpected as char,
                        smiles.to_string(),
                    ));
                }
            }
        }

        if !parser.element_buffer.is_empty() {
            parser.handle_atom(None)?
        }

        parser.add_all_bonds();

        // Whatever valence a bracketed atom does not account for is unpaired
        // electrons. Requiring a stated hydrogen count here as well meant
        // `[C]` and `[N+]`, which state none, fell through to having hydrogens
        // invented for them instead.
        for atom in parser.atoms.iter_mut() {
            if atom.is_complex() {
                // The same effective valency `get_atom_degree` uses, against
                // the bonds alone, so that what the radicals do not account
                // for is exactly zero and no hydrogens follow.
                let effective = crate::molecule::base::effective_valency(
                    atom.atomic_number(),
                    atom.charge(),
                )
                .unwrap_or(0);
                let difference = effective - atom.bond_valency();
                if difference > 0 {
                    atom.set_radical(difference as u8);
                }
            }
        }

        let molecule = MolecularGraph::from_atoms(&parser.atoms);
        molecules.push(molecule);
        write_orders.push(parser.resolve_write_orders());

        Ok((molecules, write_orders))
    }

    fn assign_proper_chirality(molecule: &mut MolecularGraph) -> Result<(), SMILESParseError> {
        let Some(mut chiral_classes) = molecule.chiral_classes.clone() else {
            return Ok(());
        };
        // Read the incoming classes from a snapshot: the loop below rewrites
        // them in place, and deciding what a three-coordinate centre means
        // needs to know what its double-bond partner *arrived* as, not what it
        // has already been rewritten to.
        let incoming = chiral_classes.clone();
        let is_directional = |atom: usize| {
            matches!(
                incoming.get(atom),
                Some(ChiralClass::Clockwise) | Some(ChiralClass::Counterclockwise)
            )
        };

        for (index, chiral_class) in chiral_classes.iter_mut().enumerate() {
            match chiral_class {
                ChiralClass::Clockwise | ChiralClass::Counterclockwise => {
                    let Some(bonds) = molecule.get_atom_bonds(index) else {
                        continue;
                    };

                    let number_of_double_bonds = bonds
                        .iter()
                        .filter(|bond| bond.bond_order() == BondOrder::Double)
                        .count();
                    let bond_number = bonds.len();

                    *chiral_class = match (number_of_double_bonds, bond_number) {
                        (2, 2) => {
                            // Allene
                            if *chiral_class == ChiralClass::Clockwise {
                                ChiralClass::AL(2)
                            } else {
                                ChiralClass::AL(1)
                            }
                        }
                        (0, 4) => {
                            if *chiral_class == ChiralClass::Clockwise {
                                ChiralClass::TH(2)
                            } else {
                                ChiralClass::TH(1)
                            }
                        }
                        (0, 5) => {
                            // Trigonal bipyramidal
                            // This should never happen but we'll handle it anyway
                            if *chiral_class == ChiralClass::Clockwise {
                                ChiralClass::TB(2)
                            } else {
                                ChiralClass::TB(1)
                            }
                        }
                        (0, 3) => {
                            // Three-coordinate centre with a lone pair: a
                            // phosphine, arsine or amine written with `@`. The
                            // lone pair occupies the fourth vertex, so the tag
                            // means the same thing a tetrahedral one does.
                            if *chiral_class == ChiralClass::Clockwise {
                                ChiralClass::TH(2)
                            } else {
                                ChiralClass::TH(1)
                            }
                        }
                        (1, 3) => {
                            // Two things reach here. An sp2 atom whose class
                            // was derived from the `/` and `\` around a double
                            // bond — in which case the atom at the far end of
                            // that bond was marked the same way — and a
                            // three-coordinate heteroatom written with `@`
                            // that happens to carry a double bond, i.e. a
                            // sulfoxide or phosphine oxide, whose partner is a
                            // plain terminal oxygen. Only the first is a
                            // double-bond descriptor; the second is a
                            // stereocentre with a lone pair.
                            let partner_is_directional = bonds
                                .iter()
                                .find(|bond| bond.bond_order() == BondOrder::Double)
                                .is_some_and(|bond| is_directional(bond.target()));
                            match (partner_is_directional, *chiral_class == ChiralClass::Clockwise) {
                                (true, true) => ChiralClass::CisTransClockwise,
                                (true, false) => ChiralClass::CisTransCounterclockwise,
                                (false, true) => ChiralClass::TH(2),
                                (false, false) => ChiralClass::TH(1),
                            }
                        }
                        (0, 6) => {
                            // Octahedral
                            // This should never happen but we'll handle it anyway
                            if *chiral_class == ChiralClass::Clockwise {
                                ChiralClass::OH(2)
                            } else {
                                ChiralClass::OH(1)
                            }
                        }
                        _ => {
                            return Err(SMILESParseError::InvalidChiralClass);
                        }
                    };
                }
                _ => (), // Leave other chiral classes unchanged
            }
        }

        // Update the molecule's chiral classes
        molecule.chiral_classes = Some(chiral_classes);
        Ok(())
    }
    /// Adds all bonds to the current molecule, ensuring aromatic bonds are correctly assigned.
    ///
    /// This method processes ring bonds and explicit hydrogen bonds, assigning the appropriate
    /// bond orders based on the aromaticity of the connected atoms.
    ///
    /// All other bonds default to their parsed bond orders.
    fn add_all_bonds(&mut self) {
        for entry in self.ring_bonds.iter() {
            match (entry.1.0, entry.1.1) {
                (Some(start_idx), Some(end_idx)) => {
                    self.rings.push((start_idx, end_idx, entry.1.2))
                }
                (None, None) => (),
                _ => (),
            }
        }

        for (start_idx, end_idx, order) in self.rings.iter() {
            if *start_idx >= self.atoms.len() || *end_idx >= self.atoms.len() {
                continue; // Skip invalid bonds
            }

            // This is false if they are not in the same system
            let bond_order = if self.atoms[*start_idx].aromatic
                && self.atoms[*end_idx].aromatic
                && order.is_none()
            {
                Some(BondOrder::Aromatic)
            } else {
                *order
            };

            self.bonds.push((*start_idx, *end_idx, bond_order));
        }

        for (atom_index, hydrogen_count) in self.hydrogens.iter() {
            for _ in 0..*hydrogen_count {
                let hydrogen_index = self.atoms.len();
                self.atoms.push(Atom::new(1)); // Hydrogen
                self.bonds
                    .push((*atom_index, hydrogen_index, Some(BondOrder::Single)));
            }
        }

        // Assign bonds to the molecule
        for bond in self.bonds.iter_mut() {
            let atom1_index = bond.0;
            let atom2_index = bond.1;
            // Ensure bond indices are within bounds
            if bond.0 >= self.atoms.len() || bond.1 >= self.atoms.len() {
                eprintln!(
                    "Error: Bond indices out of bounds: bond.0={}, bond.1={}, atoms.len={}",
                    bond.0,
                    bond.1,
                    self.atoms.len()
                );
                continue; // Skip invalid bonds
            }

            let is_atom1_aromatic = self.atoms[atom1_index].aromatic;
            let is_atom2_aromatic = self.atoms[atom2_index].aromatic;
            let bond_order = if is_atom1_aromatic && is_atom2_aromatic && bond.2.is_none() {
                BondOrder::Aromatic
            } else {
                bond.2.unwrap_or(BondOrder::Single)
            };

            let atom1 = &mut self.atoms[bond.0];
            atom1.add_bond(BondTarget::new(bond.1, bond_order));
            let atom2 = &mut self.atoms[bond.1];
            atom2.add_bond(BondTarget::new(bond.0, bond_order));
        }
    }

    fn handle_branch(&mut self) -> Result<(), SMILESParseError> {
        // Pop the stack until we find the last branch atom
        let mut branch_atom = None;
        while let Some(branch) = self.branch_stack.pop() {
            branch_atom = Some(branch);
            if self.branch_exits > 0 {
                self.branch_exits -= 1;
            } else {
                return Err(SMILESParseError::InvalidBranch);
            }
            if self.branch_exits == 0 {
                break;
            }
        }

        if self.branch_exits > 0 {
            return Err(SMILESParseError::InvalidBranch);
        }

        if let Some(branch_atom) = branch_atom {
            let current_atom = self.current_atom_index;
            self.bonds
                .push((branch_atom, current_atom, self.last_bond_type));

            // Written order: branch_atom is current_atom's from-atom; current_atom
            // is appended to branch_atom at this (post-branch) written position.
            self.write_order
                .entry(current_atom)
                .or_default()
                .push(branch_atom as i64);
            self.write_has_from.insert(current_atom, ());
            self.write_order
                .entry(branch_atom)
                .or_default()
                .push(current_atom as i64);
            if self.is_multiple_branch {
                // If we have multiple branches, we need to push the current atom back on the stack
                self.branch_stack.push(branch_atom);
                self.is_multiple_branch = false;
            }
        } else {
            return Err(SMILESParseError::BondNotFound);
        }
        Ok(())
    }

    /// Handles the creation of a bond between the current atom and the previous atom.
    ///
    /// This method determines the bond order based on the last bond type parsed
    /// and the aromaticity of the connected atoms. If both atoms are aromatic,
    /// the bond is set to `BondOrder::Aromatic`, overriding any previously assigned bond order.
    ///
    fn handle_bond(&mut self) -> Result<(), SMILESParseError> {
        if self.current_atom_index == 0 {
            return Ok(()); // No bond to handle if it's the first atom
        }

        if self.branch_exits > 0 && !self.branch_stack.is_empty() {
            self.handle_branch()?;
        } else {
            let mut previous_atom = self.current_atom_index - 1;
            let current_atom = self.current_atom_index;

            let bond_order = if let (Some(prev_atom), Some(curr_atom)) =
                (self.atoms.get(previous_atom), self.atoms.get(current_atom))
            {
                // Check if both connected atoms are aromatic
                if prev_atom.aromatic && curr_atom.aromatic && self.last_bond_type.is_none() {
                    Some(BondOrder::Aromatic)
                } else {
                    // Otherwise, use the last bond type parsed
                    self.last_bond_type
                }
            } else {
                self.last_bond_type
            };

            // Add the bond and print debug information
            self.bonds.push((previous_atom, current_atom, bond_order));

            // Record written neighbor order: the from-atom is current_atom's first
            // neighbor; current_atom is appended to previous_atom at this position.
            self.write_order
                .entry(current_atom)
                .or_default()
                .push(previous_atom as i64);
            self.write_has_from.insert(current_atom, ());
            self.write_order
                .entry(previous_atom)
                .or_default()
                .push(current_atom as i64);
        }
        self.last_bond_type = None; // Reset to default bond type
        Ok(())
    }

    /// Handles the parsing of an atom in the SMILES string.
    ///
    /// This method processes the current element buffer, determines the atomic number,
    /// and updates the atom's properties such as aromaticity, isotope, chiral class, and charge.
    ///
    /// If the `byte` parameter is provided, it indicates the next character in the SMILES string
    /// and may influence the bond type.
    ///
    /// # Arguments
    ///
    /// * `byte` - An optional byte representing the next character in the SMILES string.
    ///
    fn handle_atom(&mut self, byte: Option<u8>) -> Result<(), SMILESParseError> {
        if !self.element_buffer.is_empty() {
            // The wildcard `*` is an atom with no element assigned; atomic
            // number 0 is the conventional stand-in (RDKit uses the same), and
            // is what makes it rank below every real element under CIP rule 1a.
            let atomic_number = if self.element_buffer == "*" {
                WILDCARD_ATOMIC_NUMBER
            } else {
                self.element_buffer
                    .as_str()
                    .atomic_number()
                    .ok_or(SMILESParseError::ElementNotFound(
                        self.element_buffer.clone(),
                    ))?
            };

            let mut atom = Atom::new(atomic_number).with_atom_class(self.current_atom_class);

            self.current_atom_index += 1;

            // Determine aromaticity based on lowercase symbol or aromatic flag
            if self.element_buffer.chars().next().unwrap().is_lowercase() {
                atom.aromatic = true;
            }

            // Handle bond if additional byte is present
            if byte.is_some() {
                self.handle_bond()?
            }

            if let Some(isotope) = self.isotope {
                if is_valid_isotope(atomic_number, isotope) {
                    atom = atom.with_isotope(isotope);
                    self.isotope = None;
                } else {
                    eprintln!(
                        "Isotope {} is not valid for atomic number {}, skipping it",
                        isotope, atomic_number
                    );
                }
            }

            if self.chiral_class != ChiralClass::None {
                atom = atom.with_chiral_class(self.chiral_class);
                self.chiral_class = ChiralClass::None;
            }

            if let Some(charge) = self.current_atom_charge {
                atom = atom.with_charge(charge);
                self.current_atom_charge = None;
            }

            if self.current_atom_is_complex {
                atom = atom.with_is_complex(true);
                self.current_atom_is_complex = false;
            }

            self.atoms.push(atom);
            self.element_buffer.clear();
            // Reset atom class after using it
            self.current_atom_class = None;
        }

        if let Some(character_byte) = byte {
            self.element_buffer.push(character_byte as char);
        }

        Ok(())
    }

    /// Handles numeric characters in SMILES strings, particularly for ring closures
    ///
    /// # Arguments
    /// * `byte` - ASCII byte representing a digit (1-9)
    ///
    /// # Returns
    /// * `Result<(), ParseError>` - Ok if number handled successfully, Err if invalid ring format
    fn handle_number(&mut self, byte: u8) -> Result<(), SMILESParseError> {
        // Handle double-digit ring numbers (after %)
        if self.is_double_digit {
            let digit = byte_to_number(byte) as usize;
            self.ring_number = Some(match self.ring_number {
                Some(existing) => existing * 10 + digit,
                None => digit,
            });
            if self.ring_number.unwrap() > 9 {
                self.is_double_digit = false;
            } else {
                return Ok(());
            }
        }

        // Handle single-digit ring numbers
        if self.ring_number.is_none() {
            self.ring_number = Some(byte_to_number(byte) as usize);
        }

        // Process the complete ring number
        let ring = self.ring_number.take().unwrap();

        // Record the ring bond in written order at the current atom. Pairing the
        // opening and closing digit occurrences gives the ring partner.
        let ring_atom = self.current_atom_index;
        let placeholder_id = match self.write_ring_open.remove(&ring) {
            Some(id) => {
                self.write_ring_partner[id][1] = ring_atom;
                id
            }
            None => {
                let id = self.write_ring_partner.len();
                self.write_ring_partner.push([ring_atom, usize::MAX]);
                self.write_ring_open.insert(ring, id);
                id
            }
        };
        self.write_order
            .entry(ring_atom)
            .or_default()
            .push(-(placeholder_id as i64) - 1);

        let entry = self.ring_bonds.entry(ring).or_insert((None, None, None));

        match (entry.0, entry.1) {
            (None, _) => {
                entry.0 = Some(self.current_atom_index);
                entry.2 = self.last_bond_type; // Store bond order when first encountered
            }
            (Some(_), None) => {
                entry.1 = Some(self.current_atom_index);
                entry.2 = self.last_bond_type;
            }
            _ => {
                // Ring closure is complete, create new entry
                self.rings
                    .push((entry.0.unwrap(), entry.1.unwrap(), entry.2));
                *entry = (Some(self.current_atom_index), None, self.last_bond_type);
            }
        }

        // Resetting
        self.last_bond_type = None; // Reset bond type after handling

        // Validate that the ring closure isn't connecting an atom to itself
        if entry.0 == entry.1 && entry.1.is_some() {
            return Err(SMILESParseError::RingIndexError);
        }

        Ok(())
    }

    pub fn handle_complex_atom(
        &mut self,
        bytes: &[u8],
        position: &mut usize,
    ) -> Result<(), SMILESParseError> {
        self.handle_atom(None)?;
        // Brackets mean the atom states its own hydrogen count, so whatever
        // valence is left over is unpaired electrons rather than more
        // hydrogens. This used to be set only when the brackets also carried
        // an atom map, so `[CH3:1]` came out as the methyl radical it is while
        // `[CH3]` silently gained a fourth hydrogen and became methane.
        self.current_atom_is_complex = true;

        if bytes[*position].is_ascii_digit() {
            let mut temp_number: u16 = 0;
            while bytes[*position].is_ascii_digit() {
                let number = byte_to_number(bytes[*position]);
                temp_number = temp_number * 10 + number as u16;
                *position += 1;
            }
            self.isotope = Some(temp_number);
        }

        let mut is_se_or_as = false;
        let test = [bytes[*position], bytes[*position + 1]];
        match &test {
            b"se" => {
                is_se_or_as = true;
                self.element_buffer.push_str("se");
                *position += 1;
            }
            b"as" => {
                is_se_or_as = true;
                self.element_buffer.push_str("as");
                *position += 1;
            }
            _ => (),
        }

        if bytes[*position] == b'*' {
            // Wildcard atom inside brackets, e.g. `[*]` or `[13*]`.
            self.element_buffer.push('*');
        } else if bytes[*position].is_ascii_alphabetic() && !is_se_or_as {
            self.element_buffer.push(bytes[*position] as char);
            if bytes[*position + 1].is_ascii_lowercase() {
                *position += 1;
                self.element_buffer.push(bytes[*position] as char);
            }
        }

        *position += 1;
        let start_position = *position;

        if bytes[*position] == b'@' {
            let mut temp_counter = 0;
            *position += 1;
            while bytes[*position].is_ascii_uppercase() {
                if bytes[*position] == b'H' && temp_counter == 0 {
                    break;
                }
                *position += 1;
                temp_counter += 1;
                if temp_counter > 2 {
                    return Err(SMILESParseError::ChiralClass(
                        "Chiral class is too long".to_string(),
                    ));
                }
            }
            if bytes[*position] == b'@' {
                self.chiral_class = ChiralClass::Clockwise;
                *position += 1;
            } else {
                while bytes[*position].is_ascii_digit() {
                    *position += 1;
                }

                self.chiral_class = parse_chiral_class(&bytes[start_position..*position])?;
            }
        }

        if bytes[*position] == b'H' {
            *position += 1;
            // Theoretically we could have more than 9 hydrogen in extreme cases but it is not accepted in SMILES
            if bytes[*position].is_ascii_digit() {
                self.hydrogens
                    .insert(self.current_atom_index, byte_to_number(bytes[*position]));
                *position += 1;
            } else {
                self.hydrogens.insert(self.current_atom_index, 1);
            }
        }

        if bytes[*position] == b'+' || bytes[*position] == b'-' {
            let sign = bytes[*position];
            *position += 1;
            if bytes[*position].is_ascii_digit() {
                match sign {
                    b'+' => self.current_atom_charge = Some(byte_to_number(bytes[*position]) as i8),
                    b'-' => {
                        self.current_atom_charge = Some(-(byte_to_number(bytes[*position]) as i8))
                    }
                    _ => (), // This cant happen
                }
                *position += 1;
            } else {
                match sign {
                    b'+' => self.current_atom_charge = Some(1),
                    b'-' => self.current_atom_charge = Some(-1),
                    _ => (), // This cant happen
                }
            }
        }

        if bytes[*position] == b':' {
            *position += 1;
            if bytes[*position].is_ascii_digit() {
                let mut atom_class: usize = 0;
                while *position < bytes.len() && bytes[*position].is_ascii_digit() {
                    atom_class = atom_class
                        .saturating_mul(10)
                        .saturating_add(byte_to_number(bytes[*position]) as usize);
                    *position += 1;
                }
                self.current_atom_class = Some(atom_class);
            } else {
                return Err(SMILESParseError::AtomClassMissing);
            }
        }

        if bytes[*position] != b']' {
            eprintln!(
                "SMILESComplexError: {}",
                String::from_utf8_lossy(&bytes[*position..])
            );
            return Err(SMILESParseError::SMILESComplexError);
        }

        // Skip over the ]
        *position += 1;
        self.handle_bond()?;
        Ok(())
    }
}

fn byte_to_number(byte: u8) -> u8 {
    byte - b'0'
}

fn parse_chiral_class(slice: &[u8]) -> Result<ChiralClass, SMILESParseError> {
    match slice {
        s if s.starts_with(b"@@") => Ok(ChiralClass::Clockwise),
        s if s.starts_with(b"@TH") => {
            let number = parse_number_on_end_of_chiral_class(s);
            Ok(ChiralClass::TH(number))
        }
        s if s.starts_with(b"@AL") => {
            let number = parse_number_on_end_of_chiral_class(s);
            Ok(ChiralClass::AL(number))
        }
        s if s.starts_with(b"@SP") => {
            let number = parse_number_on_end_of_chiral_class(s);
            Ok(ChiralClass::SP(number))
        }
        s if s.starts_with(b"@TB") => {
            let number = parse_number_on_end_of_chiral_class(s);
            Ok(ChiralClass::TB(number))
        }
        s if s.starts_with(b"@OH") => {
            let number = parse_number_on_end_of_chiral_class(s);
            Ok(ChiralClass::OH(number))
        }
        s if s.starts_with(b"@") => Ok(ChiralClass::Counterclockwise),
        _ => Err(SMILESParseError::ChiralClass(
            String::from_utf8_lossy(slice).to_string(),
        )),
    }
}

fn parse_number_on_end_of_chiral_class(chiral_class: &[u8]) -> u8 {
    let mut number = 0;
    for &byte in chiral_class {
        if byte.is_ascii_digit() {
            number = number * 10 + byte_to_number(byte);
        }
    }
    number
}

fn is_valid_isotope(atomic_number: u8, isotope: u16) -> bool {
    // On a wildcard the isotope field is not a nuclide at all: `[1*]`, `[2*]`
    // and so on are how attachment points get numbered, so any value is valid.
    if atomic_number == WILDCARD_ATOMIC_NUMBER {
        return true;
    }
    let Some(isotopes) = atomic_number.isotopes() else {
        // If the atomic number is not found, we assume that the isotope is not valid
        // This is a bit of a hack but it should work
        return false;
    };

    for iso in isotopes {
        // TODO check if this is correct for all cases
        if iso.mass.round() as u16 == isotope {
            return true;
        }
    }
    false
}

pub trait ToSMILES: Molecule {
    fn to_smiles(&self) -> String {
        if self.atomic_numbers().is_empty() {
            return String::new();
        }
        let mut smiles = String::new();
        let components = self.get_components();

        for component in components {
            let rings = &mut Vec::new();
            let mut visited = vec![false; self.atomic_numbers().len()];
            // Track traversal order of atoms
            let mut traversal_order = Vec::with_capacity(self.atomic_numbers().len());

            let root = self.build_smiles_tree(
                component[0],
                BondOrder::Single,
                None,
                &mut visited,
                rings,
                &mut traversal_order,
            );

            // Sort rings based on when their first atom was encountered in traversal
            let mut ring_closures: IntMap<usize, Vec<(usize, usize, BondOrder)>> =
                IntMap::default();
            let mut ring_number = 1;

            // Process rings in order of first appearance in traversal
            for &atom_idx in &traversal_order {
                for (start, end, bond_order) in rings.iter() {
                    if *start == atom_idx {
                        ring_closures.entry(*start).or_default().push((
                            *end,
                            ring_number,
                            *bond_order,
                        ));
                        ring_closures.entry(*end).or_default().push((
                            *start,
                            ring_number,
                            *bond_order,
                        ));
                        ring_number += 1;
                    }
                }
            }

            if !smiles.is_empty() {
                smiles.push('.');
            }

            let directional_marks = self.directional_bond_marks(&root);
            let allene_orders = self.allene_emission_orders(&root, &ring_closures);
            smiles.push_str(&self.construct_smiles(
                &root,
                &ring_closures,
                &directional_marks,
                &allene_orders,
            ))
        }
        relabel_smiles_numbers(&smiles)
    }

    /// Computes the directional single bonds (`/`, `\`) that express cis/trans
    /// double-bond stereochemistry, keyed by the child atom of the bond that
    /// carries the mark (the child's parent bond is the directional one).
    ///
    /// The internal representation stores a `CisTransClockwise`/`CisTransCounterclockwise`
    /// class on each sp2 carbon; the first single-bond neighbor of each carbon
    /// (`subs[0]`) are *trans* iff the two carbons carry different classes. Each
    /// reference substituent is placed on a side (down/up) and the mark is chosen
    /// so the emitted geometry matches, accounting for whether the substituent is
    /// written before or after its carbon.
    fn directional_bond_marks(&self, root: &Node) -> IntMap<usize, BondOrder> {
        let mut marks: IntMap<usize, BondOrder> = IntMap::default();
        let Some(classes) = self.chiral_classes() else {
            return marks;
        };

        // Tree parent of each atom.
        let mut parent: IntMap<usize, usize> = IntMap::default();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            for child in node.children() {
                parent.insert(child.index(), node.index());
                stack.push(child);
            }
        }

        let is_cis_trans = |c: ChiralClass| {
            matches!(
                c,
                ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
            )
        };

        let is_tree_edge = |carbon: usize, sub: usize| {
            parent.get(&carbon) == Some(&sub) || parent.get(&sub) == Some(&carbon)
        };
        // The bond's directional mark lives on the child endpoint; `before` is true
        // when the substituent is written before its carbon (the carbon is the child).
        let mark_target = |carbon: usize, sub: usize| -> (usize, bool) {
            let before = parent.get(&carbon) == Some(&sub);
            (if before { carbon } else { sub }, before)
        };

        // One constraint per stereogenic double bond, linking the emitted symbols
        // of its two reference bonds.
        struct DirConstraint {
            child_a: usize,
            before_a: bool,
            child_b: usize,
            before_b: bool,
            opposite: bool,
        }
        let mut constraints: Vec<DirConstraint> = Vec::new();

        for a in 0..self.atomic_numbers().len() {
            if !is_cis_trans(classes[a]) {
                continue;
            }
            let bonds_a = match self.get_atom_bonds(a) {
                Some(b) => b,
                None => continue,
            };
            let b = match bonds_a
                .iter()
                .find(|bo| bo.bond_order() == BondOrder::Double)
            {
                Some(bo) => bo.target(),
                None => continue,
            };
            if b <= a || !is_cis_trans(classes[b]) {
                continue;
            }

            let subs_a: Vec<usize> = bonds_a
                .iter()
                .filter(|bo| bo.bond_order() != BondOrder::Double)
                .map(|bo| bo.target())
                .collect();
            let subs_b: Vec<usize> = self
                .get_atom_bonds(b)
                .unwrap_or_default()
                .iter()
                .filter(|bo| bo.bond_order() != BondOrder::Double && bo.target() != a)
                .map(|bo| bo.target())
                .collect();
            if subs_a.is_empty() || subs_b.is_empty() {
                continue;
            }
            let trans = classes[a] != classes[b];

            // Reference substituent on each carbon: prefer a heavy atom on a tree
            // edge (only those can carry a written directional mark); for conjugated
            // systems this selects the shared single bond, linking the constraints.
            let pick = |subs: &[usize], carbon: usize| -> Option<usize> {
                subs.iter()
                    .copied()
                    .filter(|&s| is_tree_edge(carbon, s) && self.atomic_numbers()[s] != 1)
                    .min()
                    .or_else(|| subs.iter().copied().find(|&s| is_tree_edge(carbon, s)))
            };
            let (Some(ra), Some(rb)) = (pick(&subs_a, a), pick(&subs_b, b)) else {
                continue;
            };

            // The chosen references end up on opposite sides after accounting for
            // whether each is the `subs[0]` reference of its carbon.
            let opposite = trans ^ (ra != subs_a[0]) ^ (rb != subs_b[0]);
            let (child_a, before_a) = mark_target(a, ra);
            let (child_b, before_b) = mark_target(b, rb);
            constraints.push(DirConstraint {
                child_a,
                before_a,
                child_b,
                before_b,
                opposite,
            });
        }

        // Propagate a consistent set of emitted symbols. `up[child]` = true means
        // `/`; the geometric side is `down = if before { up } else { !up }`, and each
        // constraint requires `down_a XOR down_b == opposite`. Propagation is run to
        // a fixpoint before seeding a new connected component, so conjugated bonds
        // are never seeded on both sides independently.
        let mut up: IntMap<usize, bool> = IntMap::default();
        loop {
            let mut progressed = true;
            while progressed {
                progressed = false;
                for c in &constraints {
                    match (up.get(&c.child_a).copied(), up.get(&c.child_b).copied()) {
                        (Some(ua), None) => {
                            let da = if c.before_a { ua } else { !ua };
                            let db = c.opposite ^ da;
                            up.insert(c.child_b, if c.before_b { db } else { !db });
                            progressed = true;
                        }
                        (None, Some(ub)) => {
                            let db = if c.before_b { ub } else { !ub };
                            let da = c.opposite ^ db;
                            up.insert(c.child_a, if c.before_a { da } else { !da });
                            progressed = true;
                        }
                        _ => {}
                    }
                }
            }
            // Seed one fully-unassigned constraint to start the next component.
            match constraints
                .iter()
                .find(|c| !up.contains_key(&c.child_a) && !up.contains_key(&c.child_b))
            {
                Some(c) => {
                    up.insert(c.child_a, true);
                }
                None => break,
            }
        }

        for (child, is_up) in up {
            marks.insert(
                child,
                if is_up { BondOrder::Up } else { BondOrder::Down },
            );
        }
        marks
    }

    /// Constructs the SMILES string recursively from the molecule's tree structure.
    ///
    /// # Arguments
    ///
    /// * `node` - The current node in the SMILES tree.
    /// * `ring_closures` - A map of ring closures for the molecule.
    /// * `directional_marks` - `/`,`\` marks for cis/trans double bonds.
    ///
    /// # Returns
    ///
    /// A `String` representing the SMILES notation of the molecule.
    ///
    /// # Example
    ///
    /// ```rust
    /// use molecules::prelude::*;
    ///
    /// let molecule = MolecularGraph::from_smiles("c1ccccc1").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let smiles = molecule.to_smiles();
    /// assert_eq!(smiles, "c1ccccc1");
    /// ```
    fn construct_smiles(
        &self,
        node: &Node,
        ring_closures: &IntMap<usize, Vec<(usize, usize, BondOrder)>>,
        directional_marks: &IntMap<usize, BondOrder>,
        allene_orders: &IntMap<usize, Vec<usize>>,
    ) -> String {
        let mut smiles = String::new();

        // Determine if the current bond is aromatic
        let is_current_bond_aromatic = node.bond_type() == BondOrder::Aromatic;

        // Add bond type symbol only if the bond is not aromatic
        if !is_current_bond_aromatic {
            match node.bond_type() {
                BondOrder::Single => {
                    // A cis/trans directional mark replaces the plain single bond.
                    match directional_marks.get(&node.index()) {
                        Some(BondOrder::Up) => smiles.push('/'),
                        Some(BondOrder::Down) => smiles.push('\\'),
                        _ => {
                            if let Some(parent) = node.parent() {
                                let are_both_atoms_aromatic = self.is_atom_aromatic(node.index())
                                    && self.is_atom_aromatic(parent);
                                if are_both_atoms_aromatic {
                                    smiles.push('-')
                                }
                            }
                        }
                    }
                }
                BondOrder::Up => smiles.push('/'),
                BondOrder::Down => smiles.push('\\'),
                BondOrder::Double => smiles.push('='),
                BondOrder::Triple => smiles.push('#'),
                BondOrder::Quadruple => smiles.push('$'),
                BondOrder::Aromatic => smiles.push(':'),
                BondOrder::Coordinate => {}
            }
        }

        let charge = self.charges()[node.index()];
        let atomic_number = self.atomic_numbers()[node.index()];
        // An isotope that was written must be written back. Read once: the
        // whole vector is absent for the overwhelmingly common case of a
        // molecule with no labelled atom, which keeps this free there.
        let isotope = self.get_isotope(node.index());

        // Hydrogens normally fold into their neighbour's `H` count, but a
        // labelled one cannot — `[2H]` has to appear as its own atom — so it
        // must not be counted here as well.
        let number_of_hydrogens = match self.isotopes() {
            None => self.number_of_bonded_element(node.index(), 1),
            Some(isotopes) => self.atom_bonds()[node.index()]
                .iter()
                .filter(|bond| {
                    self.atomic_numbers()[bond.target()] == 1
                        && isotopes.get(bond.target()).copied().unwrap_or(0) == 0
                })
                .count(),
        };

        // A hydrogen folds into its neighbour's `H` count only when there is a
        // neighbour to fold into and nothing about it that the count cannot
        // carry. Folding unconditionally on the element made a hydrogen with no
        // heavy neighbour vanish: `[Cl-].[H+]` came out as `[Cl-].`, with the
        // proton gone and an empty fragment where it had been, and `[H][H]`
        // came out as the empty string, both atoms folding into each other.
        // The charge and the radical state have nowhere to go in an `H3` either.
        let folds_into_neighbour = atomic_number == 1
            && isotope.is_none()
            && charge == 0
            && self.radical_states().get(node.index()).copied().unwrap_or(0) == 0
            && self.atom_bonds()[node.index()]
                .iter()
                .any(|bond| self.atomic_numbers()[bond.target()] != 1);

        if folds_into_neighbour {
            // Add children
            if node.children().len() == 1 {
                smiles.push_str(&self.construct_smiles(&node.children()[0], ring_closures, directional_marks, allene_orders));
            } else {
                let length = node.children().len();
                for (index, child) in node.children().iter().enumerate() {
                    if length - 1 == index {
                        // The terminal child does not need braces.
                        smiles.push_str(&self.construct_smiles(child, ring_closures, directional_marks, allene_orders));
                        continue;
                    }
                    smiles.push('(');
                    smiles.push_str(&self.construct_smiles(child, ring_closures, directional_marks, allene_orders));
                    smiles.push(')');
                }
            }
            return smiles;
        }

        // Get atomic symbol. The wildcard atom has no element and so no entry
        // in the periodic table; it is written as `*`.
        let mut atomic_symbol = if atomic_number == WILDCARD_ATOMIC_NUMBER {
            WILDCARD_SYMBOL.to_string()
        } else {
            let Some(symbol) = atomic_number.atomic_symbol() else {
                return "".to_string();
            };
            symbol.to_string()
        };

        // Check for aromaticity: if any bond of the atom is aromatic, make the symbol lowercase
        let is_aromatic = self
            .get_atom_bonds(node.index())
            .unwrap()
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Aromatic);
        if is_aromatic {
            atomic_symbol = atomic_symbol.to_lowercase();
        }

        let is_charged = charge != 0;
        let is_labelled = isotope.is_some();
        // SMILES has no symbol for an unpaired electron: a radical is written
        // as a bracketed atom whose stated hydrogen count falls short of its
        // valence, so `[CH3]` is the methyl radical and `C` is methane.
        // Without the brackets the two are indistinguishable on the way out.
        let is_radical = self.is_atom_radical(node.index());
        // `*` is written bare like the organic subset, since brackets carry no
        // extra information for an atom with no element and no hydrogens.
        // The organic subset, the elements SMILES lets you write without
        // brackets: B, C, N, O, P, S, F, Cl, Br, I -- plus `*`, which carries
        // no information brackets could add. Boron was missing, so `B` came
        // out as `[BH3]`.
        let is_organic_subset =
            matches!(atomic_number, 0 | 5 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53);

        let has_standard_hydrogens = match atomic_number {
            // A wildcard has no valence, so any hydrogen on it must be stated.
            0 => number_of_hydrogens == 0,
            // The count a reader would infer for a bare atom is its lowest
            // valence, so writing one bare is safe up to that many hydrogens.
            // Nitrogen and phosphorus are trivalent, which is why ammonia is
            // `N` and not `[NH3]`; both were capped at two.
            5 => number_of_hydrogens <= 3,            // Boron
            6 => number_of_hydrogens <= 4,            // Carbon
            7 => number_of_hydrogens <= 3,            // Nitrogen
            8 => number_of_hydrogens <= 2,            // Oxygen
            9 => number_of_hydrogens <= 1,            // Fluorine
            15 => number_of_hydrogens <= 3,           // Phosphorus
            16 => number_of_hydrogens <= 2,           // Sulfur
            17 | 35 | 53 => number_of_hydrogens <= 1, // Chlorine/Bromine/Iodine
            _ => false,
        };

        let has_negative_degree = self.get_atom_degree(node.index()).unwrap_or(0) < 0;

        let is_aromatic_nitrogen_with_hydrogen =
            atomic_number == 7 && is_aromatic && number_of_hydrogens > 0;

        // Cis/trans stereochemistry is emitted as directional bonds (`/`,`\`),
        // not as an atom tag, so it does not require brackets or a chiral symbol.
        let chirality = self.get_chiral_class(node.index());
        let is_cis_trans_class = matches!(
            chirality,
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        );
        let is_chiral = self.is_atom_chiral(node.index()) && !is_cis_trans_class;
        // Determine if we need square brackets
        let needs_brackets = is_labelled
            || is_radical
            || !(!is_charged && is_organic_subset && has_standard_hydrogens)
            || (has_negative_degree);
        // Build atom string
        if needs_brackets || is_aromatic_nitrogen_with_hydrogen || is_chiral {
            let chiral_str: String = match chirality {
                ChiralClass::Counterclockwise => "@".to_string(),
                ChiralClass::Clockwise => "@@".to_string(),
                ChiralClass::CisTransClockwise => String::new(),
                ChiralClass::CisTransCounterclockwise => String::new(),
                // The tetrahedral tag is stored against the centre's index-sorted
                // neighbor ordering; re-express it against the order the
                // neighbors are actually emitted in here so ring renumbering or
                // any reordering cannot silently invert the stereochemistry.
                ChiralClass::TH(reference) => {
                    let atom = node.index();
                    let emission = self.chiral_emission_order(node, ring_closures);
                    let sorted = crate::molecule::chirality::index_sorted_neighbors(self, atom);
                    let tag = crate::molecule::chirality::tetrahedral_tag_for_order(
                        reference, &sorted, &emission,
                    );
                    if tag == 2 { "@@".to_string() } else { "@".to_string() }
                }
                // Like the tetrahedral tag, an allene's is stored against a
                // reference ordering of its carriers rather than against the
                // order they happen to be emitted in, so it has to be
                // re-expressed here or a reordered molecule writes the mirror
                // image of itself.
                ChiralClass::AL(reference) => {
                    let atom = node.index();
                    match (
                        crate::molecule::chirality::index_sorted_allene_carriers(self, atom),
                        allene_orders.get(&atom),
                    ) {
                        (Some(sorted), Some(emission)) => {
                            let tag = crate::molecule::chirality::tetrahedral_tag_for_order(
                                reference, &sorted, emission,
                            );
                            if tag == 2 { "@@".to_string() } else { "@".to_string() }
                        }
                        _ => if reference == 2 { "@@".to_string() } else { "@".to_string() },
                    }
                }
                ChiralClass::SP(n) => format!("@SP{}", n),
                ChiralClass::TB(n) => format!("@TB{}", n),
                ChiralClass::OH(n) => format!("@OH{}", n),
                _ => String::new(),
            };

            let hydrogen_str = match number_of_hydrogens {
                0 => "",
                1 => "H",
                _ => &format!("H{}", number_of_hydrogens),
            };

            let charge_string = match charge {
                2.. => format!("+{}", charge.abs()),
                ..=-2 => format!("-{}", charge.abs()),
                1 => "+".to_string(),
                -1 => "-".to_string(),
                0 => String::new(),
            };

            let isotope_string = match isotope {
                Some(mass_number) => mass_number.to_string(),
                None => String::new(),
            };
            smiles.push_str(&format!(
                "[{}{}{}{}{}]",
                isotope_string, atomic_symbol, chiral_str, hydrogen_str, charge_string
            ));
        } else {
            smiles.push_str(&atomic_symbol);
        }

        // Add ring closures
        if let Some(closures) = ring_closures.get(&node.index()) {
            for closure in closures {
                let bond_order = closure.2;
                let ring_number = closure.1;
                let neighbor = closure.0;
                let are_both_atoms_aromatic =
                    self.is_atom_aromatic(node.index()) && self.is_atom_aromatic(neighbor);
                if (bond_order == BondOrder::Single && !are_both_atoms_aromatic)
                    || bond_order == BondOrder::Aromatic
                {
                    if ring_number > 9 {
                        smiles.push_str(&format!("%{}", ring_number));
                    } else {
                        smiles.push_str(&format!("{}", ring_number));
                    }
                } else if ring_number > 9 {
                    smiles.push_str(&format!("{}%{}", bond_order, ring_number));
                } else {
                    smiles.push_str(&format!("{}{}", bond_order, ring_number));
                }
            }
        }

        // Add children
        if node.children().len() == 1 {
            smiles.push_str(&self.construct_smiles(&node.children()[0], ring_closures, directional_marks, allene_orders));
        } else {
            let length = node.children().len();
            for (index, child) in node.children().iter().enumerate() {
                if length - 1 == index {
                    // The terminal child does not need braces.
                    smiles.push_str(&self.construct_smiles(child, ring_closures, directional_marks, allene_orders));
                    continue;
                }
                smiles.push('(');
                smiles.push_str(&self.construct_smiles(child, ring_closures, directional_marks, allene_orders));
                smiles.push(')');
            }
        }

        smiles
    }

    /// The order in which a single node's neighbors appear in the emitted SMILES:
    /// `[parent, folded hydrogen(s), ring-closure partners, branch children]`.
    /// This is the ordering a chirality tag at that atom is relative to.
    /// For every allene centre in the tree, its four carriers in the order the
    /// emitted SMILES will list them.
    ///
    /// Computed up front by walking the tree, because the carriers sit on the
    /// *ends* of the allene and the writer, at the centre, has the node of
    /// neither: one end is its parent, already written, and the other is its
    /// child, not yet. Their own emission orders are what decide where the
    /// four substituents land in the string.
    fn allene_emission_orders(
        &self,
        root: &Node,
        ring_closures: &IntMap<usize, Vec<(usize, usize, BondOrder)>>,
    ) -> IntMap<usize, Vec<usize>> {
        let mut orders: IntMap<usize, Vec<usize>> = IntMap::default();
        let Some(classes) = self.chiral_classes() else {
            return orders;
        };
        if !classes.iter().any(|c| matches!(c, ChiralClass::AL(_))) {
            return orders;
        }

        // Substituents of one end, in emission order, with the centre removed.
        let substituents = |end: &Node, centre: usize| -> Vec<usize> {
            let mut order = self.chiral_emission_order(end, ring_closures);
            order.retain(|&atom| atom != centre);
            order
        };

        let mut stack: Vec<&Node> = vec![root];
        while let Some(node) = stack.pop() {
            for child in node.children() {
                stack.push(child);
            }

            let is_allene = matches!(
                classes.get(node.index()),
                Some(ChiralClass::AL(_))
            );
            if is_allene {
                // The centre started this fragment: both ends are children.
                if node.parent().is_none() && node.children().len() == 2 {
                    let centre = node.index();
                    let mut carriers = substituents(&node.children()[0], centre);
                    carriers.extend(substituents(&node.children()[1], centre));
                    if carriers.len() == 4 {
                        orders.insert(centre, carriers);
                    }
                }
                continue;
            }

            // Otherwise this node may be the *first* end: its child is the
            // centre, and the centre's child is the second end.
            for child in node.children() {
                let centre = child.index();
                if !matches!(classes.get(centre), Some(ChiralClass::AL(_))) {
                    continue;
                }
                let Some(second) = child.children().first() else {
                    continue;
                };
                let mut carriers = substituents(node, centre);
                carriers.extend(substituents(second, centre));
                if carriers.len() == 4 {
                    orders.insert(centre, carriers);
                }
            }
        }
        orders
    }

    fn chiral_emission_order(
        &self,
        node: &Node,
        ring_closures: &IntMap<usize, Vec<(usize, usize, BondOrder)>>,
    ) -> Vec<usize> {
        let atom = node.index();
        let mut order = Vec::with_capacity(6);
        if let Some(parent) = node.parent() {
            order.push(parent);
        }
        let mut hydrogens: Vec<usize> = self.atom_bonds()[atom]
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| self.atomic_numbers()[target] == 1)
            .collect();
        hydrogens.sort_unstable();
        order.extend(hydrogens);
        if let Some(closures) = ring_closures.get(&atom) {
            order.extend(closures.iter().map(|closure| closure.0));
        }
        for child in node.children() {
            order.push(child.index());
        }
        order
    }

    /// For every atom, the order in which its neighbors appear in the SMILES
    /// string that [`to_smiles`](Self::to_smiles) would emit:
    /// `[parent, folded hydrogen(s), ring-closure partners, branch children]`.
    ///
    /// A tetrahedral/cis-trans chirality tag (`@`/`@@`) is defined relative to
    /// this neighbor order, so comparing an atom's order before and after a
    /// relabeling tells us (via permutation parity) whether the stored tag must
    /// flip to keep describing the same 3D arrangement. The traversal mirrors
    /// `to_smiles` exactly (root = `component[0]`, hydrogens folded into the
    /// atom, ring closures emitted before branches).
    fn smiles_emission_orders(&self) -> Vec<Vec<usize>> {
        let n = self.atomic_numbers().len();
        let mut orders: Vec<Vec<usize>> = vec![Vec::new(); n];

        for component in self.get_components() {
            if component.is_empty() {
                continue;
            }
            let mut visited = vec![false; n];
            let mut rings: Vec<(usize, usize, BondOrder)> = Vec::new();
            let mut traversal_order = Vec::with_capacity(n);
            let root = self.build_smiles_tree(
                component[0],
                BondOrder::Single,
                None,
                &mut visited,
                &mut rings,
                &mut traversal_order,
            );

            // Per-atom ring-closure partner order, matching to_smiles: closures
            // are numbered by the traversal order of their opening atom, and
            // both endpoints record the partner in that order.
            let mut ring_partners: IntMap<usize, Vec<usize>> = IntMap::default();
            for &atom_idx in &traversal_order {
                for (start, end, _bond) in rings.iter() {
                    if *start == atom_idx {
                        ring_partners.entry(*start).or_default().push(*end);
                        ring_partners.entry(*end).or_default().push(*start);
                    }
                }
            }

            let mut stack = vec![&root];
            while let Some(node) = stack.pop() {
                let atom = node.index();
                let mut order = Vec::with_capacity(6);
                if let Some(parent) = node.parent() {
                    order.push(parent);
                }
                // Folded hydrogens sit right after the from-atom (or first for a
                // root atom), ascending by index for determinism.
                let mut hydrogens: Vec<usize> = self.atom_bonds()[atom]
                    .iter()
                    .map(|bond| bond.target())
                    .filter(|&target| self.atomic_numbers()[target] == 1)
                    .collect();
                hydrogens.sort_unstable();
                order.extend(hydrogens);
                if let Some(partners) = ring_partners.get(&atom) {
                    order.extend(partners.iter().copied());
                }
                for child in node.children() {
                    order.push(child.index());
                    stack.push(child);
                }
                orders[atom] = order;
            }
        }
        orders
    }

    fn to_smiles_with_implicit_hydrogens(&self) -> String {
        if self.atomic_numbers().is_empty() {
            return String::new();
        }
        let mut smiles = String::new();
        let components = self.get_components();

        for component in components {
            let rings = &mut Vec::new();
            let mut visited = vec![false; self.atomic_numbers().len()];
            // Track traversal order of atoms
            let mut traversal_order = Vec::with_capacity(self.atomic_numbers().len());

            let root_index = component
                .iter()
                .find(|&&x| self.atomic_numbers()[x] != 1)
                .unwrap_or(&component[0]);

            let root = self.build_smiles_tree(
                *root_index,
                BondOrder::Single,
                None,
                &mut visited,
                rings,
                &mut traversal_order,
            );

            // Sort rings based on when their first atom was encountered in traversal
            let mut ring_closures: IntMap<usize, Vec<(usize, usize, BondOrder)>> =
                IntMap::default();
            let mut ring_number = 1;

            // Process rings in order of first appearance in traversal
            for &atom_idx in &traversal_order {
                for (start, end, bond_order) in rings.iter() {
                    if *start == atom_idx {
                        ring_closures.entry(*start).or_default().push((
                            *end,
                            ring_number,
                            *bond_order,
                        ));
                        ring_closures.entry(*end).or_default().push((
                            *start,
                            ring_number,
                            *bond_order,
                        ));
                        ring_number += 1;
                    }
                }
            }

            if !smiles.is_empty() {
                smiles.push('.');
            }

            let directional_marks = self.directional_bond_marks(&root);
            let allene_orders = self.allene_emission_orders(&root, &ring_closures);
            smiles.push_str(&self.construct_smiles(
                &root,
                &ring_closures,
                &directional_marks,
                &allene_orders,
            ))
        }
        relabel_smiles_numbers(&smiles)
    }
}

pub trait FromSMILES {
    fn from_smiles(smiles: &str) -> Result<Vec<Self>, SMILESParseError>
    where
        Self: Sized;
    fn from_smiles_single(smiles: &str) -> Result<Self, SMILESParseError>
    where
        Self: Sized;
    fn from_smiles_with_sanitization(smiles: &str) -> Result<Vec<Self>, SMILESParseError>
    where
        Self: Sized;
}

impl FromSMILES for MolecularGraph {
    fn from_smiles_single(smiles: &str) -> Result<Self, SMILESParseError> {
        let molecules = SMILESParser::parse_smiles(smiles)?;
        Ok(molecules[0].clone())
    }
    fn from_smiles(smiles: &str) -> Result<Vec<MolecularGraph>, SMILESParseError> {
        let molecules = SMILESParser::parse_smiles(smiles)?;
        Ok(molecules)
    }
    fn from_smiles_with_sanitization(
        smiles: &str,
    ) -> Result<Vec<MolecularGraph>, SMILESParseError> {
        let mut molecules = SMILESParser::parse_smiles(smiles)?;
        Ok(molecules)
    }
}

fn relabel_smiles_numbers(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut number_map: HashMap<String, String> = std::collections::HashMap::default();
    let mut next_number = 1;
    let mut chars = input.chars().peekable();
    let mut is_bracket = false;

    while let Some(c) = chars.next() {
        if c == '[' {
            is_bracket = true;
        }
        if c == ']' {
            is_bracket = false;
        }
        if is_bracket {
            result.push(c);
            continue;
        }
        if c == '%' {
            // Handle multi-digit numbers
            let mut number = String::new();
            while let Some(&next_char) = chars.peek() {
                if next_char.is_ascii_digit() {
                    number.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            // Map the number if we haven't seen it before
            let mapped = number_map.entry(number.clone()).or_insert_with(|| {
                let current = next_number;
                next_number += 1;
                current.to_string()
            });

            result.push('%');
            result.push_str(mapped);
        } else if c.is_ascii_digit() {
            // Handle single digit numbers
            let number = c.to_string();
            let mapped = number_map.entry(number).or_insert_with(|| {
                let current = next_number;
                next_number += 1;
                current.to_string()
            });

            result.push_str(mapped);
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::ToSMILES;
    use crate::prelude::*;

    #[test]
    fn test_parse_smiles() {
        let molecules = SMILESParser::parse_smiles("C(C(C))COCCl").unwrap();
        assert_eq!(molecules[0].atomic_numbers[0], 6);
        assert_eq!(molecules[0].atomic_numbers.len(), 18);
    }

    #[test]
    fn test_parse_smiles_with_isotope() {
        let molecules = SMILESParser::parse_smiles("CCCC[13C]").unwrap();
        // The labelled carbon states no hydrogens, so it has none.
        assert_eq!(molecules[0].atomic_numbers.len(), 14);
        assert_eq!(molecules[0].get_atom_isotope(4).unwrap(), 13);
    }

    #[test]
    fn test_parse_smiles_with_chiral_class() {
        let molecules = SMILESParser::parse_smiles("C[C@](F)(Cl)Br").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 8);
        assert_eq!(molecules[0].get_edges().len(), 7);
        assert_eq!(molecules[0].get_chiral_class(1), ChiralClass::TH(1));
    }

    #[test]
    fn test_parse_smiles_with_complex_atom_and_isotope() {
        let molecules = SMILESParser::parse_smiles("C[C@](F)(Cl)[13C](Br)").unwrap();
        println!("{:?}", molecules);
        println!("{:?}", molecules[0].get_edges());
        assert_eq!(molecules[0].atomic_numbers.len(), 9);
        assert_eq!(molecules[0].get_atom_isotope(4).unwrap(), 13);
        assert_eq!(molecules[0].get_chiral_class(1), ChiralClass::TH(1));
        assert_eq!(molecules[0].get_edges().len(), 8);
    }

    #[test]
    fn test_parse_smiles_with_complex_atom_and_hydrogen() {
        let molecules = SMILESParser::parse_smiles("C[C@H](Cl)C").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 11);
        assert_eq!(molecules[0].get_edges().len(), 10);
    }

    #[test]
    fn test_bond_parsing() {
        let molecules = SMILESParser::parse_smiles("C-C-C#C").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 10);
        assert_eq!(molecules[0].get_edges().len(), 9);
    }

    #[test]
    fn parse_test_file() {
        let smiles = std::fs::read_to_string("tests/smiles.txt").unwrap();
        for smile in smiles.lines() {
            println!("Parsing: {}", smile);
            match SMILESParser::parse_smiles(smile) {
                Ok(molecules) => {
                    // println!("{:?}", molecules);
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                    panic!();
                }
            }
        }
    }

    #[test]
    fn test_submolecule() {
        let molecules = SMILESParser::parse_smiles("C(C(C))COCCl.C(C(C))").unwrap();
        let submolecule = molecules[0].match_submolecule(&molecules[1]);
        assert_eq!(submolecule, None);
    }

    #[test]
    fn test_subgraph() {
        // A raw-parsed pattern carries no hydrogen atoms, and match_submolecule
        // matches it as a plain subgraph: hydrogen counts are not compared (a
        // bracket [C] does not enforce zero hydrogens here; use CxSMARTS [CH0]
        // for that). The C-C-C chain therefore matches C(C(C))COCCl.
        let molecule = SMILESParser::parse_smiles_single("C(C(C))COCCl").unwrap();
        let pattern = SMILESParser::parse_smiles_raw_single("[C][C][C]").unwrap();
        println!("Pattern: {}", pattern.to_smiles());
        let submolecule = molecule.match_submolecule(&pattern);
        assert!(
            submolecule.is_some_and(|mappings| !mappings.is_empty()),
            "C-C-C chain should match as a subgraph"
        );
    }

    #[test]
    fn test_ring_closure_with_charge() {
        let molecules =
            SMILESParser::parse_smiles("c1ccc2c(c1)=[NH+]C(=O)C=2Nc1ccc(C(OCCC)=O)cc1").unwrap();

        println!("Molecule: {}", molecules[0].to_smiles());
        assert_eq!(molecules[0].atomic_numbers.len(), 40);
    }

    #[test]
    fn test_parse_smiles_with_charged_atoms() {
        // Test positive charges
        let molecules = SMILESParser::parse_smiles("[NH4+]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 5); // N + 4H
        assert_eq!(molecules[0].get_atom_charge(0), 1);

        // Test negative charges
        let molecules = SMILESParser::parse_smiles("[O-]").unwrap();
        // Brackets state the hydrogen count, so this is the bare oxide anion.
        assert_eq!(molecules[0].atomic_numbers.len(), 1);
        assert_eq!(molecules[0].get_atom_charge(0), -1);

        // Test multiple charges
        let molecules = SMILESParser::parse_smiles("[NH3+][O-]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 5); // N + 3H + O
        assert_eq!(molecules[0].get_atom_charge(0), 1);
        assert_eq!(molecules[0].get_atom_charge(1), -1); // Hydrogens are added last
    }

    #[test]
    fn test_parse_smiles_with_explicit_hydrogens() {
        // Test NH2 group
        let molecules = SMILESParser::parse_smiles("[NH2]").unwrap();
        // The aminyl radical: two hydrogens as written, and one unpaired
        // electron rather than a third hydrogen.
        assert_eq!(molecules[0].atomic_numbers.len(), 3); // N + 2H
        assert_eq!(molecules[0].radical_states[0], 1);

        // Test CH3 group
        let molecules = SMILESParser::parse_smiles("[CH3]").unwrap();
        // The methyl radical, not methane.
        assert_eq!(molecules[0].atomic_numbers.len(), 4); // C + 3H
        assert_eq!(molecules[0].radical_states[0], 1);

        // Test complex molecule with explicit hydrogens
        let molecules = SMILESParser::parse_smiles("[CH3][NH2]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 7); // C + 3H + N + 2H, methylamine
        assert_eq!(molecules[0].get_edges().len(), 6); // All single bonds
    }

    #[test]
    fn test_parse_smiles_with_charged_complex_molecules() {
        // Test zwitterion (amino acid-like structure)
        let molecules = SMILESParser::parse_smiles("[NH3+]CC[O-]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 11); // N + 3H + 2C + O + 4H implicit
        assert_eq!(molecules[0].get_atom_charge(0), 1);
        assert_eq!(molecules[0].get_atom_charge(3), -1);

        // Test multiple charged centers
        let molecules = SMILESParser::parse_smiles("[NH4+]CC[NH3+]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 15); // 2N + 7H + 2C + 4H implicit
        assert_eq!(molecules[0].get_atom_charge(0), 1);
        assert_eq!(molecules[0].get_atom_charge(3), 1);
    }

    #[test]
    fn test_parse_smiles_with_atom_classes() {
        // Test single atom class
        let molecules = SMILESParser::parse_smiles("[C:1]").unwrap();
        // No hydrogens written means none: an atomic carbon with four
        // unpaired electrons.
        assert_eq!(molecules[0].atomic_numbers.len(), 1);
        assert_eq!(molecules[0].radical_states[0], 4);

        // Test multi-digit atom class
        let molecules = SMILESParser::parse_smiles("[C:123][N:456]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 2); // both bare, both radicals

        // Test atom class with other properties
        let molecules = SMILESParser::parse_smiles("[13CH2+1:1]").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 3); // C + 2H
        assert_eq!(molecules[0].get_atom_isotope(0).unwrap(), 13);
        assert_eq!(molecules[0].get_atom_charge(0), 1);
    }

    #[test]
    fn test_parse_smiles_with_ring_fragments() {
        // Test simple ring closure for fragments: C1.C1 should become CC
        let molecules = SMILESParser::parse_smiles("C1.C1").unwrap();
        assert_eq!(molecules.len(), 1); // Should be a single molecule
        assert_eq!(molecules[0].atomic_numbers.len(), 8); // 2C + 6H
        assert_eq!(molecules[0].get_edges().len(), 7); // 6 C-H bonds + 1 C-C bond

        // Test with different atom types: O1.N1 should become ON
        let molecules = SMILESParser::parse_smiles("O1.N1").unwrap();
        assert_eq!(molecules.len(), 1); // Should be a single molecule
        assert_eq!(molecules[0].atomic_numbers.len(), 5); // O + N + 3H
        assert_eq!(molecules[0].get_edges().len(), 4); // 3 N-H bonds + 1 O-N bond
    }
}

#[cfg(test)]
mod aromatic_bond_tests {
    use super::*;

    #[test]
    fn test_parse_aromatic_bonds() {
        let molecules = SMILESParser::parse_smiles("c1ccccc1").unwrap();
        assert_eq!(molecules[0].atomic_numbers.len(), 12);
        assert_eq!(molecules[0].get_edges().len(), 12);
        let mut aromatic_bonds = 0;
        for bond in molecules[0].get_edges_with_type().into_iter() {
            if bond.2 == BondOrder::Aromatic {
                aromatic_bonds += 1;
            }
        }
        assert_eq!(aromatic_bonds, 6);
    }

    #[test]
    fn test_parse_mixed_bonds_with_aromatic_atoms() {
        let molecules = SMILESParser::parse_smiles("C1=CC=CC=C1").unwrap();

        assert_eq!(molecules[0].atomic_numbers.len(), 12);
        assert_eq!(molecules[0].get_edges().len(), 12);
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub struct BondTargetOption {
    pub target: usize,
    pub bond_order: Option<BondOrder>,
}

impl BondTargetOption {
    pub fn new(target: usize, bond_type: Option<BondOrder>) -> BondTargetOption {
        BondTargetOption {
            target,
            bond_order: bond_type,
        }
    }
}
