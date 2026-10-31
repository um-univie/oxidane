use std::fmt::Write as _;

use crate::atom::Atom;
use crate::atom::{BOND_CAPACITY, WILDCARD_ATOMIC_NUMBER, WILDCARD_SYMBOL};
use crate::molecule::aromaticity::{AromaticityModel, default_aromaticity_model};
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
    /// An atom was given more bonds than an atom can be given. Ring-closure
    /// labels that are opened but never closed, or reused across a `.`, pile up
    /// on whichever atom happens to be current, and ten is already past every
    /// coordination number this crate models.
    TooManyBonds(usize),
    /// The lower-case bonds the string states cannot all be written as
    /// alternating single and double ones, so the string does not describe a
    /// molecule. See [`crate::molecule::aromaticity::KekulizationError`].
    Kekulization(String),
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
        Self {
            aromaticity: Some(default_aromaticity_model()),
        }
    }
}

/// Records, as unpaired electrons, the valence a bracketed atom's bonds and
/// stated hydrogens together leave unaccounted for.
///
/// Brackets do not make an atom a radical; they make its hydrogen count
/// explicit, and stop the reader inventing more. What is left over after the
/// bonds and those hydrogens is the radical: `C[CH3]` is ethane, because the
/// three stated hydrogens and the bond to carbon fill carbon's valence, while
/// a lone `[CH3]` is the methyl radical and `[Si]` with three neighbours is a
/// silyl radical, both one electron short. By this point the stated hydrogens
/// are atoms in their own right, so `bond_valency` already counts them.
///
/// Requiring a stated hydrogen count here as well meant `[C]` and `[N+]`,
/// which state none, fell through to having hydrogens invented for them
/// instead.
///
/// This has to run for every fragment. It used to run once, on whatever was
/// left in the parser at the end of the string, so in `C(C)[Ge](CC)CC.[K]`
/// only the potassium was marked and the germanium silently acquired a
/// hydrogen -- and writing the same molecule with the fragments the other way
/// round moved the hydrogen to the other fragment.
///
/// The bonds are priced by the molecule's own [`MoleculeCore::actual_valency`],
/// which is what `add_hydrogens` measures against next, and not by
/// [`Atom::bond_valency`]. The two disagreed about a directional bond between
/// two aromatic ring atoms -- an aromatic bond carrying a mark, three half-units
/// to one and two to the other. A ring closed on a mark, as canonical SMILES
/// writes `[n+]1(ccnc(c\1=C/C)=C)C`, then left the pyridinium nitrogen half a
/// bond short here and full there: it read back with an unpaired electron its
/// Kekulé form does not have, and the tautomer key of the canonical SMILES was
/// not the key of the molecule it was written from.
fn assign_radicals(atoms: &[Atom], molecule: &mut MolecularGraph) {
    for (index, atom) in atoms.iter().enumerate() {
        if !atom.is_complex() {
            continue;
        }
        // The valence the bonds have already reached into, not the element's
        // first one. They differ exactly where an element has several: neutral
        // sulfur's are 2, 4 and 6, so `[S](C)(C)C` is one electron short of
        // four rather than one *over* two, and reading it as the latter left it
        // with no radical and a valence of three -- which is not a valence
        // sulfur has, so the molecule was not a feasible point of the tautomer
        // model and its groups had no solution at all. Every other toolkit
        // reads these the same way: RDKit gives the same atom one unpaired
        // electron, and `[Si]` with three bonds, and `[P]` with four.
        // An aromatic bond has no whole number of π units to it, so an atom
        // carrying one is left to the element's first valence: thiophene's
        // sulfur counts three across its two aromatic bonds, and asking which
        // valence three has reached into answers four and invents an unpaired
        // electron the ring does not have.
        let bonded = molecule.actual_valency(index);
        let effective = if molecule.has_an_unsettled_bond(index) {
            crate::molecule::base::effective_valency(atom.atomic_number(), atom.charge())
        } else {
            crate::molecule::base::saturating_valency(atom.atomic_number(), atom.charge(), bonded)
        }
        .unwrap_or(0);
        let difference = effective - bonded;
        if difference > 0 {
            molecule.radical_states_mut()[index] = difference as u8;
        }
    }
}

pub(crate) struct SmilesWriteOrder {
    pub has_from: bool,
    pub neighbors: Vec<usize>,
    pub ring_closures: usize,
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
    /// Indexed by atom, not keyed by it: atom indices are dense and assigned
    /// in order, so the hash map this used to be paid for hashing and
    /// rehashing on every bond in the string to store what a `Vec` addresses
    /// directly.
    write_order: Vec<Vec<i64>>,
    /// Atoms that have an incoming "from" bond (i.e. are not a fragment root),
    /// whose first `write_order` entry is therefore the from-atom.
    write_has_from: Vec<bool>,
    /// Ring number currently open → placeholder id, indexed by ring number.
    /// `usize::MAX` means the ring is not open.
    write_ring_open: Vec<usize>,
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
    /// use oxidane::prelude::*;
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
            molecule.charge_separate_hypervalent_nitrogen();
            molecule.add_hydrogens();

            Self::assign_proper_chirality(molecule)?;
            crate::molecule::cip::normalize_tetrahedral_references(molecule, written);
            crate::molecule::cip::normalize_allene_references(molecule, written);
            molecule.convert_directional_bonds_to_chiral();

            if let Some(model) = options.aromaticity {
                // Reading is where a string that is not a molecule gets turned
                // away. Perception is tolerant by design, so ask for the Kekulé
                // structure here, where the answer can be refused; the call
                // perception makes later finds the bonds already localized and
                // returns at once.
                molecule
                    .localize_aromatic_bonds()
                    .map_err(|failure| SMILESParseError::Kekulization(failure.to_string()))?;
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

    /// Records `declaration` as the next neighbour of `atom` in written order.
    fn declare_neighbor(&mut self, atom: usize, declaration: i64) {
        if self.write_order.len() <= atom {
            self.write_order.resize(atom + 1, Vec::new());
        }
        self.write_order[atom].push(declaration);
    }

    fn set_has_from(&mut self, atom: usize) {
        if self.write_has_from.len() <= atom {
            self.write_has_from.resize(atom + 1, false);
        }
        self.write_has_from[atom] = true;
    }

    /// Resolve ring placeholders while retaining the written closure count for
    /// three-coordinate stereochemistry.
    fn resolve_write_orders(&self) -> IntMap<usize, SmilesWriteOrder> {
        let mut out = IntMap::default();
        for (atom, decls) in self.write_order.iter().enumerate() {
            if decls.is_empty() {
                continue;
            }
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
            out.insert(
                atom,
                SmilesWriteOrder {
                    has_from: self.write_has_from.get(atom).copied().unwrap_or(false),
                    neighbors: resolved,
                    ring_closures: decls.iter().filter(|&&d| d < 0).count(),
                },
            );
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
    ) -> Result<(Vec<MolecularGraph>, Vec<IntMap<usize, SmilesWriteOrder>>), SMILESParseError> {
        let mut parser = SMILESParser::default();
        parser.smiles = smiles;
        // One atom per two bytes is a generous upper bound for SMILES and
        // costs one allocation instead of the handful a growing vector takes.
        parser.atoms.reserve(smiles.len() / 2 + 1);
        parser.bonds.reserve(smiles.len() / 2 + 1);
        let mut molecules = Vec::with_capacity(smiles.bytes().filter(|&c| c == b'.').count());
        let mut write_orders: Vec<IntMap<usize, SmilesWriteOrder>> = Vec::new();

        // SMILES should only be ASCII so we could bypass the UTF-8 checks using bytes()
        let bytes = smiles.as_bytes();
        let mut pointer = 0;
        while pointer < bytes.len() {
            let Some(&byte) = bytes.get(pointer) else {
                break;
            };
            pointer += 1;
            match byte {
                b'A'..=b'Z' => parser.handle_atom(Some(byte))?,
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
                    parser.add_all_bonds()?;
                    // If we still have unsatisfied ring bonds, we need to wait
                    // This does not yet solve the issue as we will need to split mols in the end
                    if !parser
                        .ring_bonds
                        .iter()
                        .any(|(_, (a, b, _))| a.is_none() || b.is_none())
                    {
                        let mut molecule = MolecularGraph::from_atoms(&parser.atoms);
                        assign_radicals(&parser.atoms, &mut molecule);
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

                // `%(nnn)` -- the extended ring-closure form. Two digits
                // after `%` is all the notation proper allows, which caps a
                // molecule at a hundred open rings, and a fullerene wants more.
                // RDKit reads and writes this form, so an answer using it stays
                // readable by other toolkits.
                // `pointer` has already advanced past the `%`, so the next
                // byte is at `pointer` and the digits begin one further on.
                b'%' if bytes.get(pointer) == Some(&b'(') => {
                    // At most five digits, as RDKit allows. The number indexes
                    // a table the parser grows to fit it, so an unbounded one
                    // overflowed, or asked for terabytes and aborted the
                    // process: `C%(999999999999)CC%(999999999999)` did.
                    let mut cursor = pointer + 1;
                    let mut value = 0usize;
                    let mut digits = 0;
                    while let Some(&digit) = bytes.get(cursor) {
                        if !digit.is_ascii_digit() {
                            break;
                        }
                        if digits == MAX_RING_CLOSURE_DIGITS {
                            return Err(SMILESParseError::RingIndexError);
                        }
                        value = value * 10 + byte_to_number(digit) as usize;
                        cursor += 1;
                        digits += 1;
                    }
                    if digits == 0 || bytes.get(cursor) != Some(&b')') {
                        return Err(SMILESParseError::RingIndexError);
                    }
                    parser.is_double_digit = false;
                    parser.ring_number = Some(value);
                    parser.close_ring_number()?;
                    pointer = cursor + 1;
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

        // A ring-closure label that was opened and never closed does not
        // describe a molecule. Dropping it silently, which is what
        // `add_all_bonds` does, was survivable until a chiral centre was
        // involved: the centre counts the pending closure among its neighbours
        // when it records its tag, so the tag ends up stored against an
        // ordering the finished molecule does not have, and re-expressing it
        // finds a neighbour that is not there. `C[C@H]1C` used to panic.
        if parser
            .ring_bonds
            .iter()
            .any(|(_, (start, end, _))| start.is_some() != end.is_some())
        {
            return Err(SMILESParseError::RingIndexError);
        }

        parser.add_all_bonds()?;

        let mut molecule = MolecularGraph::from_atoms(&parser.atoms);
        assign_radicals(&parser.atoms, &mut molecule);
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
                        // Four neighbours is tetrahedral however they are
                        // bonded. A sulfoximine's sulfur carries `=O` and `=N`
                        // and is still a stereocentre -- a configurationally
                        // stable one, which is the point of the class -- and so
                        // is the phosphorus of a phosphorothioate. Requiring
                        // four single bonds rejected both.
                        (_, 4) => {
                            if *chiral_class == ChiralClass::Clockwise {
                                ChiralClass::TH(2)
                            } else {
                                ChiralClass::TH(1)
                            }
                        }
                        (_, 5) => {
                            // Trigonal bipyramidal
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
                            match (
                                partner_is_directional,
                                *chiral_class == ChiralClass::Clockwise,
                            ) {
                                (true, true) => ChiralClass::CisTransClockwise,
                                (true, false) => ChiralClass::CisTransCounterclockwise,
                                (false, true) => ChiralClass::TH(2),
                                (false, false) => ChiralClass::TH(1),
                            }
                        }
                        (_, 6) => {
                            // Octahedral
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

        molecule.chiral_classes = Some(chiral_classes);
        Ok(())
    }
    /// Adds all bonds to the current molecule, ensuring aromatic bonds are correctly assigned.
    ///
    /// Processes ring bonds and explicit hydrogen bonds, assigning bond orders
    /// based on aromaticity flags.
    fn add_all_bonds(&mut self) -> Result<(), SMILESParseError> {
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

        for bond in self.bonds.iter_mut() {
            let atom1_index = bond.0;
            let atom2_index = bond.1;
            // A bond to an atom that was never read cannot be repaired, and
            // silently dropping it hands back a different molecule than the one
            // the string asked for.
            if bond.0 >= self.atoms.len() || bond.1 >= self.atoms.len() {
                return Err(SMILESParseError::RingIndexError);
            }

            let is_atom1_aromatic = self.atoms[atom1_index].aromatic;
            let is_atom2_aromatic = self.atoms[atom2_index].aromatic;
            let bond_order = if is_atom1_aromatic && is_atom2_aromatic && bond.2.is_none() {
                BondOrder::Aromatic
            } else {
                bond.2.unwrap_or(BondOrder::Single)
            };

            // The bond list is a fixed-capacity inline array, so a string that
            // piles more than it holds onto one atom used to abort the process
            // rather than fail the parse.
            if self.atoms[atom1_index].bonds().len() >= BOND_CAPACITY {
                return Err(SMILESParseError::TooManyBonds(atom1_index));
            }
            if self.atoms[atom2_index].bonds().len() >= BOND_CAPACITY {
                return Err(SMILESParseError::TooManyBonds(atom2_index));
            }

            let atom1 = &mut self.atoms[bond.0];
            atom1.add_bond(BondTarget::new(bond.1, bond_order));
            let atom2 = &mut self.atoms[bond.1];
            atom2.add_bond(BondTarget::new(bond.0, bond_order));
        }
        Ok(())
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
            self.declare_neighbor(current_atom, branch_atom as i64);
            self.set_has_from(current_atom);
            self.declare_neighbor(branch_atom, current_atom as i64);
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

    /// Handles the creation of a bond between the current atom and the previous atom,
    /// assigning aromatic bond orders when both atoms are aromatic.
    fn handle_bond(&mut self) -> Result<(), SMILESParseError> {
        if self.current_atom_index == 0 {
            return Ok(()); // No bond to handle if it's the first atom
        }

        if self.branch_exits > 0 && !self.branch_stack.is_empty() {
            self.handle_branch()?;
        } else {
            let previous_atom = self.current_atom_index - 1;
            let current_atom = self.current_atom_index;

            let bond_order = if let (Some(prev_atom), Some(curr_atom)) =
                (self.atoms.get(previous_atom), self.atoms.get(current_atom))
            {
                if prev_atom.aromatic && curr_atom.aromatic && self.last_bond_type.is_none() {
                    Some(BondOrder::Aromatic)
                } else {
                    // Otherwise, use the last bond type parsed
                    self.last_bond_type
                }
            } else {
                self.last_bond_type
            };

            self.bonds.push((previous_atom, current_atom, bond_order));

            // Record written neighbor order: the from-atom is current_atom's first
            // neighbor; current_atom is appended to previous_atom at this position.
            self.declare_neighbor(current_atom, previous_atom as i64);
            self.set_has_from(current_atom);
            self.declare_neighbor(previous_atom, current_atom as i64);
        }
        self.last_bond_type = None; // Reset to default bond type
        Ok(())
    }

    /// Parses an atom token from the element buffer, updating atomic number, charge,
    /// isotope, chirality, and aromaticity flags.
    fn handle_atom(&mut self, byte: Option<u8>) -> Result<(), SMILESParseError> {
        if !self.element_buffer.is_empty() {
            // The wildcard `*` is an atom with no element assigned; atomic
            // number 0 is the conventional stand-in (RDKit uses the same), and
            // is what makes it rank below every real element under CIP rule 1a.
            let atomic_number = if self.element_buffer == "*" {
                WILDCARD_ATOMIC_NUMBER
            } else {
                self.element_buffer.as_str().atomic_number().ok_or(
                    SMILESParseError::ElementNotFound(self.element_buffer.clone()),
                )?
            };

            let mut atom = Atom::new(atomic_number).with_atom_class(self.current_atom_class);

            self.current_atom_index += 1;

            // Determine aromaticity based on lowercase symbol or aromatic flag
            if self.element_buffer.chars().next().unwrap().is_lowercase() {
                atom.aromatic = true;
            }

            if byte.is_some() {
                self.handle_bond()?
            }

            if let Some(isotope) = self.isotope {
                atom = atom.with_isotope(isotope);
                self.isotope = None;
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
        self.close_ring_number()
    }

    /// Opens or closes the ring whose number is already in `ring_number`.
    fn close_ring_number(&mut self) -> Result<(), SMILESParseError> {
        let ring = self.ring_number.take().unwrap();

        // Record the ring bond in written order at the current atom. Pairing the
        // opening and closing digit occurrences gives the ring partner.
        let ring_atom = self.current_atom_index;
        if self.write_ring_open.len() <= ring {
            self.write_ring_open.resize(ring + 1, usize::MAX);
        }
        let placeholder_id = match std::mem::replace(&mut self.write_ring_open[ring], usize::MAX) {
            usize::MAX => {
                let id = self.write_ring_partner.len();
                self.write_ring_partner.push([ring_atom, usize::MAX]);
                self.write_ring_open[ring] = id;
                id
            }
            id => {
                self.write_ring_partner[id][1] = ring_atom;
                id
            }
        };
        self.declare_neighbor(ring_atom, -(placeholder_id as i64) - 1);

        let entry = self.ring_bonds.entry(ring).or_insert((None, None, None));

        // A bond symbol may be written at either end of a ring closure, and the
        // two ends do not mean the same thing. A directional mark describes the
        // bond running from the atom it is attached to *toward* its partner, so
        // one written at the closing digit points backwards against the written
        // order of the atoms -- and everything downstream
        // (`convert_directional_bonds_to_chiral`) reads a bond's direction from
        // the written order of its two atoms. Flipping it here is what lets a
        // closing mark mean what it says: without it `C1CCCC/C=C\\1` parsed as
        // trans when it is cis.
        let closing_order = |order: Option<BondOrder>| match order {
            Some(BondOrder::Up) => Some(BondOrder::Down),
            Some(BondOrder::Down) => Some(BondOrder::Up),
            other => other,
        };
        match (entry.0, entry.1) {
            (None, _) => {
                entry.0 = Some(self.current_atom_index);
                entry.2 = self.last_bond_type; // Store bond order when first encountered
            }
            (Some(_), None) => {
                entry.1 = Some(self.current_atom_index);
                // A closing end with no symbol of its own says nothing, and must
                // not erase what the opening end said.
                if self.last_bond_type.is_some() {
                    entry.2 = closing_order(self.last_bond_type);
                }
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
        // Everything below reads forward without a bounds check, using the
        // closing bracket as the sentinel it is guaranteed to stop at. An
        // unterminated bracket has no such sentinel, so it walked off the end
        // of the string instead: `[H][` used to panic.
        if !bytes[*position..].contains(&b']') {
            return Err(SMILESParseError::EOL);
        }

        self.handle_atom(None)?;
        // Brackets mean the atom states its own hydrogen count, so whatever
        // valence is left over is unpaired electrons rather than more
        // hydrogens. This used to be set only when the brackets also carried
        // an atom map, so `[CH3:1]` came out as the methyl radical it is while
        // `[CH3]` silently gained a fourth hydrogen and became methane.
        self.current_atom_is_complex = true;

        if bytes[*position].is_ascii_digit() {
            // Any mass number is accepted, as RDKit accepts `[999C]`, but one
            // past what the field holds is refused rather than wrapped:
            // `[65536C]` overflowed it.
            let mut temp_number: u16 = 0;
            while bytes[*position].is_ascii_digit() {
                let number = byte_to_number(bytes[*position]);
                temp_number = temp_number
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(number as u16))
                    .ok_or(SMILESParseError::SMILESComplexError)?;
                *position += 1;
            }
            self.isotope = Some(temp_number);
        }

        // A bracket with no element in it -- `[]`, or an isotope alone as in
        // `[2]` -- names no atom. Reading two bytes ahead here would also run
        // past the closing bracket when it is the last byte of the string.
        if bytes[*position] == b']' {
            return Err(SMILESParseError::ElementNotFound(String::new()));
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
            // A charge is written either as a signed number -- `[Ca+2]` -- or
            // as the sign repeated, `[Ca++]`. The second is the older spelling
            // and the one catalogues and vendor files are still full of, so a
            // reader that only takes the first rejects a calcium salt.
            let magnitude = if *position < bytes.len() && bytes[*position].is_ascii_digit() {
                let mut magnitude: i32 = 0;
                while *position < bytes.len() && bytes[*position].is_ascii_digit() {
                    magnitude = magnitude
                        .saturating_mul(10)
                        .saturating_add(byte_to_number(bytes[*position]) as i32);
                    *position += 1;
                }
                magnitude
            } else {
                let mut magnitude: i32 = 1;
                while *position < bytes.len() && bytes[*position] == sign {
                    magnitude += 1;
                    *position += 1;
                }
                magnitude
            };
            let magnitude = magnitude.clamp(0, i8::MAX as i32) as i8;
            self.current_atom_charge = Some(if sign == b'+' { magnitude } else { -magnitude });
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
            return Err(SMILESParseError::SMILESComplexError);
        }

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

/// Where a cis/trans double bond's `/` and `\` marks are to be written.
///
/// Two kinds of place, because a SMILES has two kinds of bond. A tree edge is
/// written once, between an atom and its parent, and its mark is keyed by the
/// child. A ring closure is written twice -- a digit at each end -- and its mark
/// goes on the second of them, keyed by that atom and its partner, so that `/`
/// means on both what it means on a tree edge: the bond rises from the atom
/// written earlier to the one written later.
#[derive(Default)]
pub struct DirectionalMarks {
    /// Mark for the bond joining this atom to its parent.
    tree: IntMap<usize, BondOrder>,
    /// Mark for the ring closure written at this atom, closing to that partner.
    ring: rustc_hash::FxHashMap<(usize, usize), BondOrder>,
}

impl DirectionalMarks {
    fn get(&self, child: &usize) -> Option<&BondOrder> {
        self.tree.get(child)
    }

    fn ring(&self, at: usize, partner: usize) -> Option<BondOrder> {
        self.ring.get(&(at, partner)).copied()
    }
}

/// The most digits an extended ring-closure number `%(nnnnn)` may carry.
const MAX_RING_CLOSURE_DIGITS: usize = 5;

/// The largest ring-closure number two digits after `%` can hold. Past it the
/// writer reuses numbers where it can, and writes the extended `%(nnn)` form
/// where it cannot.
const LARGEST_RING_NUMBER: usize = 99;

/// Writes a ring-closure label: bare below ten, `%nn` to ninety-nine, and
/// `%(nnn)` above that.
///
/// The parenthesised form is the extended ring closure. Two digits is all the
/// notation proper allows, which caps a molecule at a hundred rings open at
/// once, and a fullerene wants more -- reuse cuts the demand a long way but not
/// below the ceiling for every cage PubChem holds. RDKit reads and writes this
/// same form, so an answer needing it stays readable elsewhere.
fn write_ring_label(out: &mut String, label: usize) {
    use std::fmt::Write;
    if label > LARGEST_RING_NUMBER {
        let _ = write!(out, "%({label})");
    } else if label > 9 {
        let _ = write!(out, "%{label}");
    } else {
        let _ = write!(out, "{label}");
    }
}

/// Gives every ring closure the number it will end up wearing, and puts each
/// atom's closures in that number's order.
///
/// The writer numbers rings as it finds them and a string pass afterwards
/// renumbers them by order of first appearance, which are not the same order:
/// a closure written `3` at an atom can come out as `5` once the pass has run,
/// and the digit beside it written `7` can come out as `2`. Sorting the atom's
/// closures before that happens therefore sorted the wrong numbers, and a
/// dilithium carbanion macrocycle from PubChem wrote `\5=2` from its deposited
/// form and `=2\5` from its own canonical one -- the same two closures, the
/// same two digits, read out in opposite orders.
///
/// So the renumbering is done here instead, against the order the atoms will be
/// emitted in. At each atom the closures whose numbers are already settled come
/// first, smallest first, and the ones opening there take the next free numbers
/// -- which are larger than every number already given out, so the digits at
/// every atom end up ascending and stay that way when the string pass maps them
/// onto themselves.
fn settle_ring_labels(
    root: &Node,
    atom_count: usize,
    ring_closures: &mut IntMap<usize, Vec<(usize, usize, BondOrder)>>,
) {
    let mut emission_position = vec![usize::MAX; atom_count];
    let mut order: Vec<usize> = Vec::with_capacity(atom_count);
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if node.index() < emission_position.len() {
            emission_position[node.index()] = order.len();
        }
        order.push(node.index());
        for child in node.children().iter().rev() {
            stack.push(child);
        }
    }

    let mut settled: rustc_hash::FxHashMap<usize, usize> = rustc_hash::FxHashMap::default();
    let mut next = 1usize;
    for &atom in &order {
        let Some(closures) = ring_closures.get_mut(&atom) else {
            continue;
        };
        closures.sort_by_key(|&(partner, label, _)| match settled.get(&label) {
            Some(&number) => (0usize, number, 0usize),
            None => (
                1,
                0,
                emission_position
                    .get(partner)
                    .copied()
                    .unwrap_or(usize::MAX),
            ),
        });
        for &(_, label, _) in closures.iter() {
            settled.entry(label).or_insert_with(|| {
                let number = next;
                next += 1;
                number
            });
        }
    }

    for closures in ring_closures.values_mut() {
        for entry in closures.iter_mut() {
            if let Some(&number) = settled.get(&entry.1) {
                entry.1 = number;
            }
        }
        closures.sort_by_key(|&(_, number, _)| number);
    }
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

            // A folded hydrogen is not emitted as an atom and therefore
            // cannot be the from-atom of a stereocentre. Starting there would
            // count it both as parent and as folded H in the stereo ordering.
            let root_index = component
                .iter()
                .find(|&&atom| !self.hydrogen_folds_into_neighbour(atom))
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
            // Ring-closure numbers count up, one per ring, and are not reused:
            // reuse would shorten most strings and is what RDKit does, but it
            // changes the canonical form of every molecule with two ring
            // systems in it, and those strings are what the tautomer keys are
            // made of. So the numbering only changes where the plain one
            // cannot be written at all.
            if rings.len() <= LARGEST_RING_NUMBER {
                // Where one atom opens more than one ring, the numbers go to
                // its partners in traversal order rather than in whatever order
                // the rings happened to be collected in. Collection follows the
                // bond lists, which are a fact about how the molecule was
                // written; traversal follows the numbering the writer was
                // handed, which is a fact about the molecule.
                let mut position = vec![usize::MAX; self.atomic_numbers().len()];
                for (place, &atom_index) in traversal_order.iter().enumerate() {
                    if atom_index < position.len() {
                        position[atom_index] = place;
                    }
                }
                let mut ring_number = 1;
                for &atom_index in &traversal_order {
                    let mut opening: Vec<&(usize, usize, BondOrder)> = rings
                        .iter()
                        .filter(|(start, _, _)| *start == atom_index)
                        .collect();
                    opening.sort_by_key(|(_, end, _)| {
                        position.get(*end).copied().unwrap_or(usize::MAX)
                    });
                    for (start, end, bond_order) in opening {
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
            } else {
                // A ring-closure number is given back once its ring has closed,
                // and the smallest free one is handed out next. Counting up and
                // never reusing runs past `%99`, which is as high as the notation
                // goes: PubChem's fullerene cages need hundreds of closures and
                // came out as `%200`, a string this crate's own reader -- and every
                // other -- refuses. None of them needs more than a few dozen live
                // at once.
                //
                // The live interval of a ring is measured in *emission* positions,
                // taken by walking the tree the writer walks, not in the order the
                // traversal happened to record its atoms. Those two differ, and
                // releasing on the second one handed a number back before its
                // closing digit had been written: bicyclo[2.2.2]octane came out as
                // `C1CC1CCC1CC1`, four digits reading as two small rings.
                let atom_count = self.atomic_numbers().len();
                let mut emission_position = vec![usize::MAX; atom_count];
                let mut emitted = 0usize;
                let mut stack = vec![&root];
                while let Some(node) = stack.pop() {
                    emission_position[node.index()] = emitted;
                    emitted += 1;
                    for child in node.children().iter().rev() {
                        stack.push(child);
                    }
                }

                // (first end written, second end written, which ring)
                let mut live: Vec<(usize, usize, usize)> = rings
                    .iter()
                    .enumerate()
                    .map(|(index, (first, second, _))| {
                        let opens = emission_position[*first];
                        let closes = emission_position[*second];
                        (opens.min(closes), opens.max(closes), index)
                    })
                    .collect();
                live.sort_unstable();

                let mut ring_number = 1;
                let mut free: std::collections::BinaryHeap<std::cmp::Reverse<usize>> =
                    std::collections::BinaryHeap::new();
                let mut released: Vec<Vec<usize>> = vec![Vec::new(); emitted.max(1)];
                let mut cursor = 0usize;
                for (opens, closes, index) in live {
                    while cursor < opens {
                        for number in released[cursor].drain(..) {
                            free.push(std::cmp::Reverse(number));
                        }
                        cursor += 1;
                    }
                    let number = match free.pop() {
                        Some(std::cmp::Reverse(number)) => number,
                        None => {
                            let number = ring_number;
                            ring_number += 1;
                            number
                        }
                    };
                    let (first, second, bond_order) = rings[index];
                    ring_closures
                        .entry(first)
                        .or_default()
                        .push((second, number, bond_order));
                    ring_closures
                        .entry(second)
                        .or_default()
                        .push((first, number, bond_order));
                    if closes < released.len() {
                        released[closes].push(number);
                    }
                }
            }

            // The digits an atom carries are written in ascending order, and
            // they are the numbers the string will actually show -- see
            // [`settle_ring_labels`]. `chiral_emission_order` reads these same
            // lists, so the neighbour ordering a tag is written against moves
            // with the digits and the stereochemistry says what it always said.
            settle_ring_labels(&root, self.atomic_numbers().len(), &mut ring_closures);

            if !smiles.is_empty() {
                smiles.push('.');
            }

            let directional_marks = self.directional_bond_marks(&root);
            let allene_orders = self.allene_emission_orders(&root, &ring_closures);
            self.construct_smiles(
                &mut smiles,
                &root,
                &ring_closures,
                &directional_marks,
                &allene_orders,
            )
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
    fn directional_bond_marks(&self, root: &Node) -> DirectionalMarks {
        let mut marks = DirectionalMarks::default();
        let Some(classes) = self.chiral_classes() else {
            return marks;
        };

        // Tree parent of each atom, and the order the atoms are written in.
        // The order matters because a ring-closure bond is written at both of
        // its ends, and its mark goes on the second -- which is what makes it
        // mean the same thing a mark on a tree edge means, namely that the bond
        // rises from the atom written earlier to the one written later.
        let mut parent: IntMap<usize, usize> = IntMap::default();
        let mut order: IntMap<usize, usize> = IntMap::default();
        let mut stack = vec![root];
        let mut next = 0usize;
        while let Some(node) = stack.pop() {
            order.insert(node.index(), next);
            next += 1;
            for child in node.children().iter().rev() {
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
        // A bond that is not a tree edge is written as a ring closure, and a
        // ring closure can carry a directional mark just as a tree edge can.
        // Refusing to use one was why a ring alkene lost its geometry: in a
        // ring exactly one bond is the closure, so the double bond beside it
        // has a carbon whose only substituent lies across it, no reference
        // could be found, and the whole bond was passed over unmarked.
        let is_usable =
            |carbon: usize, sub: usize| is_tree_edge(carbon, sub) || order.contains_key(&sub);
        // The bond's directional mark lives on the child endpoint; `before` is true
        // when the substituent is written before its carbon (the carbon is the child).
        let atom_count = self.atomic_numbers().len();
        let mut ring_sites: Vec<(usize, usize)> = Vec::new();
        let mut mark_target = |carbon: usize, sub: usize| -> (usize, bool) {
            if is_tree_edge(carbon, sub) {
                let before = parent.get(&carbon) == Some(&sub);
                return (if before { carbon } else { sub }, before);
            }
            // A ring closure. The mark is written at the end that comes second,
            // so that `/` means the same "rises from earlier to later" it means
            // on a tree edge; `before` says the substituent is the earlier one.
            let before = order.get(&sub) < order.get(&carbon);
            let site = if before { (carbon, sub) } else { (sub, carbon) };
            let index = ring_sites
                .iter()
                .position(|&s| s == site)
                .unwrap_or_else(|| {
                    ring_sites.push(site);
                    ring_sites.len() - 1
                });
            (atom_count + index, before)
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
        // Every bond some constraint will put a mark on, as (atom, atom).
        let mut marked_bonds: Vec<(usize, usize)> = Vec::new();

        // A double bond inside a small ring has its geometry forced by the ring
        // -- cyclohexene has no cis and trans to choose between -- so it is not
        // stereogenic and takes no mark. Writing one anyway is not merely
        // redundant: the marks around a ring compete with those of the
        // *exocyclic* double bond that is genuinely stereogenic, and a reader
        // resolving the conflict drops the real geometry. Seven is the usual
        // cut-off; beyond it a ring is slack enough for the two to differ.
        const SMALLEST_STRAINED_RING: usize = 7;
        let ring_size_through = |a: usize, b: usize| -> Option<usize> {
            // Shortest path from `a` back to `b` without using their own bond:
            // the smallest ring the bond lies on.
            let mut seen = vec![false; self.atomic_numbers().len()];
            let mut frontier = vec![a];
            seen[a] = true;
            // `step` counts bonds walked, so the ring is `step + 1` bonds
            // round; stopping one short of the cut-off is what keeps an
            // eight-membered ring -- which really can be cis or trans -- out of
            // the "forced by the ring" bucket.
            for step in 1..SMALLEST_STRAINED_RING {
                let mut next = Vec::new();
                for &atom in &frontier {
                    for bond in self.get_atom_bonds(atom).unwrap_or_default() {
                        let target = bond.target();
                        if atom == a && target == b {
                            continue;
                        }
                        if target == b {
                            return Some(step + 1);
                        }
                        if !seen[target] {
                            seen[target] = true;
                            next.push(target);
                        }
                    }
                }
                frontier = next;
            }
            None
        };

        for a in 0..self.atomic_numbers().len() {
            if !is_cis_trans(classes[a]) {
                continue;
            }
            let bonds_a = match self.get_atom_bonds(a) {
                Some(b) => b,
                None => continue,
            };
            // The double bond the class describes, which is not always the
            // first one the atom lists. A phosphaalkene's phosphorus carries
            // two, and taking the first paired it with the end that has no
            // class, so the mark was never written and the geometry the reader
            // had read was lost on the way out.
            let b = match bonds_a
                .iter()
                .filter(|bo| bo.bond_order() == BondOrder::Double)
                .map(|bo| bo.target())
                .find(|&target| is_cis_trans(classes[target]))
            {
                Some(target) => target,
                None => continue,
            };
            if b <= a {
                continue;
            }
            if ring_size_through(a, b).is_some() {
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

            // Whether marking the bond `carbon`-`s` would start to specify a
            // double bond that has no geometry to specify.
            //
            // A directional mark belongs to every double bond at either of its
            // atoms, not only to the one it was written for. So a mark placed
            // on a substituent that itself carries an unmarked double bond is
            // half of a geometry nobody asked for, and if that bond's far end
            // is marked too -- which it will be, when it is the next
            // stereogenic bond along a conjugated chain -- the molecule comes
            // back with a geometry it did not have.
            //
            // PubChem's CID 142764698 is the case: a cyanine whose ring
            // `C=N` sits between the exocyclic `C=C` and the next ring `C=C`.
            // Marking the nitrogen for the exocyclic bond, and the carbon two
            // atoms along for the ring bond, specified the `C=N` between them,
            // and the molecule read back with eight stereogenic bonds where it
            // went in with seven. RDKit writes the same molecule with that
            // nitrogen unmarked, using the other substituent.
            let would_specify_more = |carbon: usize, s: usize| -> bool {
                self.get_atom_bonds(s)
                    .unwrap_or_default()
                    .iter()
                    .any(|bond| {
                        bond.bond_order() == BondOrder::Double
                            && bond.target() != carbon
                            && !(is_cis_trans(classes[s]) && is_cis_trans(classes[bond.target()]))
                    })
            };

            // Reference substituent on each carbon: prefer a heavy atom on a tree
            // edge (only those can carry a written directional mark); for conjugated
            // systems this selects the shared single bond, linking the constraints.
            let pick = |subs: &[usize], carbon: usize| -> Option<usize> {
                subs.iter()
                    .copied()
                    // A substituent that is itself the carbon of another
                    // stereogenic double bond is the bond the two share, and
                    // taking it for both is what keeps them consistent. Any
                    // other choice gives this carbon *two* marked substituents
                    // -- one for each double bond -- with nothing to say they
                    // lie on opposite sides, and the two marks then place them
                    // on the same side. A reader meeting that contradiction
                    // discards the geometry, which is why a diene whose two
                    // bonds differ (E,Z) came back with no stereo at all while
                    // E,E and Z,Z were fine.
                    .filter(|&s| is_tree_edge(carbon, s) && is_cis_trans(classes[s]))
                    .min()
                    // Then one that specifies nothing else, in preference to
                    // one that would.
                    .or_else(|| {
                        subs.iter()
                            .copied()
                            .filter(|&s| {
                                is_tree_edge(carbon, s)
                                    && self.atomic_numbers()[s] != 1
                                    && !would_specify_more(carbon, s)
                            })
                            .min()
                    })
                    .or_else(|| {
                        subs.iter()
                            .copied()
                            .filter(|&s| {
                                is_usable(carbon, s)
                                    && self.atomic_numbers()[s] != 1
                                    && !would_specify_more(carbon, s)
                            })
                            .min()
                    })
                    .or_else(|| {
                        subs.iter()
                            .copied()
                            .filter(|&s| is_tree_edge(carbon, s) && self.atomic_numbers()[s] != 1)
                            .min()
                    })
                    .or_else(|| subs.iter().copied().find(|&s| is_tree_edge(carbon, s)))
                    .or_else(|| {
                        subs.iter()
                            .copied()
                            .filter(|&s| is_usable(carbon, s) && self.atomic_numbers()[s] != 1)
                            .min()
                    })
                    .or_else(|| subs.iter().copied().find(|&s| is_usable(carbon, s)))
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
            marked_bonds.push((a, ra));
            marked_bonds.push((b, rb));

            // The two substituents of one double-bond carbon lie on opposite
            // sides of it, and if both end up marked then their symbols have to
            // say so. Choosing references carefully avoids most collisions but
            // not all: where the traversal forces a second mark onto the same
            // carbon anyway, nothing here required the two to disagree, and
            // they came out on the same side. A reader meeting that
            // contradiction discards the geometry outright -- which is why the
            // failures were whole molecules with no stereo at all rather than
            // molecules with one bond wrong.
            for (carbon, subs, reference) in [(a, &subs_a, ra), (b, &subs_b, rb)] {
                let usable: Vec<usize> = subs
                    .iter()
                    .copied()
                    .filter(|&s| is_usable(carbon, s))
                    .collect();
                // Only where the *other* substituent really will carry a mark
                // of its own, which happens exactly when it is an end of
                // another stereogenic double bond -- that bond's own reference
                // is what puts a second symbol on this carbon. Tying a
                // substituent that nothing else marks does not resolve a
                // collision, it manufactures one more mark, and a mark is a
                // geometry: on a fully conjugated ring every single bond sits
                // between two double bonds, so the extra symbol handed a
                // geometry to a double bond the input had left unstated.
                // `C1=C/C=C\C(=C/C=C1)\C2=C/C=C\C=C/C=C2` came back with six
                // stereogenic bonds where it and RDKit both read five, and the
                // sixth changed the molecule's own canonical form on rereading.
                let Some(other) = usable
                    .iter()
                    .copied()
                    .find(|&s| s != reference && is_cis_trans(classes[s]))
                else {
                    continue;
                };
                let (first, first_before) = mark_target(carbon, reference);
                let (second, second_before) = mark_target(carbon, other);
                if first == second {
                    continue;
                }
                constraints.push(DirConstraint {
                    child_a: first,
                    before_a: first_before,
                    child_b: second,
                    before_b: second_before,
                    opposite: true,
                });
                marked_bonds.push((carbon, other));
            }
        }

        // A mark belongs to every double bond at either of its atoms, so a
        // double bond with no geometry of its own can still end up with a
        // marked substituent at each end -- put there for the stereogenic
        // bonds on either side of it along a conjugated chain -- and two
        // marks, one per end, *are* a geometry. The reader would hand it one
        // the input never stated, and the canonical form of the canonical form
        // would not be the canonical form: PubChem's CID 89159404,
        // `C/C=C\C(=C\C)\C(=N/C=C\C)/C=C\C`, came back with five
        // stereogenic bonds where it went in with four.
        //
        // The input says "no geometry" the way RDKit does, with two marks on
        // one end placing both substituents on the same side, which
        // contradicts itself and specifies nothing. So say it the same way:
        // where one end already carries two marks, require them to agree.
        // The requirement is added only where it is consistent with every
        // geometric one -- a parity union-find over the marks tells -- so it
        // can never cost a bond the input did specify.
        let mut parity_parent: IntMap<usize, (usize, bool)> = IntMap::default();
        fn find(parent: &mut IntMap<usize, (usize, bool)>, site: usize) -> (usize, bool) {
            let (up, parity) = *parent.entry(site).or_insert((site, false));
            if up == site {
                return (site, false);
            }
            let (root, above) = find(parent, up);
            let total = parity ^ above;
            parent.insert(site, (root, total));
            (root, total)
        }
        // Parity of a mark's geometric side relative to its emitted symbol.
        let joins = |parent: &mut IntMap<usize, (usize, bool)>, c: &DirConstraint, apply: bool| {
            // `down = up ^ !before`; constraint `down_a ^ down_b == opposite`
            // becomes `up_a ^ up_b == opposite ^ !before_a ^ !before_b`.
            let wanted = c.opposite ^ !c.before_a ^ !c.before_b;
            let (ra, pa) = find(parent, c.child_a);
            let (rb, pb) = find(parent, c.child_b);
            if ra == rb {
                return (pa ^ pb) == wanted;
            }
            if apply {
                parent.insert(rb, (ra, pa ^ pb ^ wanted));
            }
            true
        };
        for c in &constraints {
            joins(&mut parity_parent, c, true);
        }
        let mut contradictions: Vec<DirConstraint> = Vec::new();
        for x in 0..atom_count {
            let Some(bonds_x) = self.get_atom_bonds(x) else {
                continue;
            };
            for y in bonds_x
                .iter()
                .filter(|bond| bond.bond_order() == BondOrder::Double)
                .map(|bond| bond.target())
                .filter(|&y| y > x)
            {
                if is_cis_trans(classes[x]) && is_cis_trans(classes[y]) {
                    continue;
                }
                let marked_at = |end: usize, far: usize| -> Vec<usize> {
                    let mut subs: Vec<usize> = marked_bonds
                        .iter()
                        .filter_map(|&(p, q)| match (p == end, q == end) {
                            (true, _) if q != far => Some(q),
                            (_, true) if p != far => Some(p),
                            _ => None,
                        })
                        .collect();
                    subs.sort_unstable();
                    subs.dedup();
                    subs
                };
                let (at_x, at_y) = (marked_at(x, y), marked_at(y, x));
                if at_x.is_empty() || at_y.is_empty() {
                    continue;
                }
                let mut settled = false;
                for (end, subs) in [(x, &at_x), (y, &at_y)] {
                    if subs.len() < 2 {
                        continue;
                    }
                    let (child_a, before_a) = mark_target(end, subs[0]);
                    let (child_b, before_b) = mark_target(end, subs[1]);
                    let same_side = DirConstraint {
                        child_a,
                        before_a,
                        child_b,
                        before_b,
                        opposite: false,
                    };
                    if joins(&mut parity_parent, &same_side, false) {
                        joins(&mut parity_parent, &same_side, true);
                        contradictions.push(same_side);
                        settled = true;
                        break;
                    }
                }
                if settled {
                    continue;
                }
                // Neither end has a second mark to contradict the first, so
                // give one end a second mark that does. It has to be a mark
                // that specifies nothing else: on a substituent with no double
                // bond of its own, or on the end of a stereogenic double bond
                // that is already marked, tied to lie opposite that mark so
                // the bond keeps the geometry it has.
                'ends: for (end, far, subs) in [(x, y, &at_x), (y, x, &at_y)] {
                    let candidates: Vec<usize> = self
                        .get_atom_bonds(end)
                        .unwrap_or_default()
                        .iter()
                        .filter(|bond| bond.bond_order() != BondOrder::Double)
                        .map(|bond| bond.target())
                        .filter(|&sub| sub != far && !subs.contains(&sub) && is_usable(end, sub))
                        .collect();
                    for sub in candidates {
                        // A double bond to an atom with nothing else on it --
                        // a carbonyl's oxygen -- has no far substituent to be
                        // placed against, so a mark beside it specifies
                        // nothing. That is the free substituent CID 144402149
                        // had: a cyclooctatetraene whose unspecified ring bond
                        // carries an ester, every other bond at its ends taken
                        // by the neighbouring ring bonds' marks.
                        let doubles: Vec<usize> = self
                            .get_atom_bonds(sub)
                            .unwrap_or_default()
                            .iter()
                            .filter(|bond| bond.bond_order() == BondOrder::Double)
                            .map(|bond| bond.target())
                            .filter(|&partner| {
                                self.get_atom_bonds(partner).unwrap_or_default().len() > 1
                            })
                            .collect();
                        let (existing, existing_before) = mark_target(end, subs[0]);
                        let (added, added_before) = mark_target(end, sub);
                        let mut needed = vec![DirConstraint {
                            child_a: existing,
                            before_a: existing_before,
                            child_b: added,
                            before_b: added_before,
                            opposite: false,
                        }];
                        match doubles.as_slice() {
                            [] => {}
                            &[partner]
                                if is_cis_trans(classes[sub]) && is_cis_trans(classes[partner]) =>
                            {
                                let Some(reference) = marked_at(sub, partner)
                                    .into_iter()
                                    .find(|&other| other != end)
                                else {
                                    continue;
                                };
                                let (held, held_before) = mark_target(sub, reference);
                                let (tied, tied_before) = mark_target(sub, end);
                                needed.push(DirConstraint {
                                    child_a: held,
                                    before_a: held_before,
                                    child_b: tied,
                                    before_b: tied_before,
                                    opposite: true,
                                });
                            }
                            _ => continue,
                        }
                        let mut trial = parity_parent.clone();
                        if !needed.iter().all(|c| {
                            let fits = joins(&mut trial, c, false);
                            if fits {
                                joins(&mut trial, c, true);
                            }
                            fits
                        }) {
                            continue;
                        }
                        parity_parent = trial;
                        marked_bonds.push((end, sub));
                        contradictions.extend(needed);
                        break 'ends;
                    }
                }
            }
        }
        constraints.extend(contradictions);

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

        for (site, is_up) in up {
            let order = if is_up {
                BondOrder::Up
            } else {
                BondOrder::Down
            };
            match site.checked_sub(atom_count) {
                Some(index) => {
                    if let Some(&(at, partner)) = ring_sites.get(index) {
                        marks.ring.insert((at, partner), order);
                    }
                }
                None => {
                    marks.tree.insert(site, order);
                }
            }
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
    /// use oxidane::prelude::*;
    ///
    /// let molecule = MolecularGraph::from_smiles("c1ccccc1").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let smiles = molecule.to_smiles();
    /// assert_eq!(smiles, "c1ccccc1");
    /// ```
    fn construct_smiles(
        &self,
        smiles: &mut String,
        node: &Node,
        ring_closures: &IntMap<usize, Vec<(usize, usize, BondOrder)>>,
        directional_marks: &DirectionalMarks,
        allene_orders: &IntMap<usize, Vec<usize>>,
    ) {
        let is_current_bond_aromatic = node.bond_type() == BondOrder::Aromatic;

        // A directional mark is written even on an aromatic bond. A double bond
        // hanging off an aromatic ring is stereogenic, and its geometry needs a
        // marked substituent on *each* of its carbons -- the one on the ring
        // side is a ring bond. Dropping it writes half a description, and a
        // reader with half a description discards the geometry rather than
        // guess, so two exocyclic double bonds on one ring came back with none
        // at all.
        //
        // This could not be written until the parser could read it back:
        // `MoleculeCore::actual_valency` counted a directional bond as a single
        // one, so an aromatic carbon carrying a mark looked half a bond short,
        // gained a hydrogen, and left its ring without a Kekulé structure.
        if let Some(mark) = directional_marks.get(&node.index())
            && is_current_bond_aromatic
        {
            smiles.push(if *mark == BondOrder::Up { '/' } else { '\\' });
        }

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
        let number_of_hydrogens = self.atom_bonds()[node.index()]
            .iter()
            .filter(|bond| self.hydrogen_folds_into_neighbour(bond.target()))
            .count();

        // A hydrogen folds into its neighbour's `H` count only when there is a
        // neighbour to fold into and nothing about it that the count cannot
        // carry. Folding unconditionally on the element made a hydrogen with no
        // heavy neighbour vanish: `[Cl-].[H+]` came out as `[Cl-].`, with the
        // proton gone and an empty fragment where it had been, and `[H][H]`
        // came out as the empty string, both atoms folding into each other.
        // The charge and the radical state have nowhere to go in an `H3` either.
        let folds_into_neighbour = self.hydrogen_folds_into_neighbour(node.index());

        if folds_into_neighbour {
            if node.children().len() == 1 {
                self.construct_smiles(
                    smiles,
                    &node.children()[0],
                    ring_closures,
                    directional_marks,
                    allene_orders,
                );
            } else {
                let length = node.children().len();
                for (index, child) in node.children().iter().enumerate() {
                    if length - 1 == index {
                        // The terminal child does not need braces.
                        self.construct_smiles(
                            smiles,
                            child,
                            ring_closures,
                            directional_marks,
                            allene_orders,
                        );
                        continue;
                    }
                    smiles.push('(');
                    self.construct_smiles(
                        smiles,
                        child,
                        ring_closures,
                        directional_marks,
                        allene_orders,
                    );
                    smiles.push(')');
                }
            }
            return;
        }

        // Get atomic symbol. The wildcard atom has no element and so no entry
        // in the periodic table; it is written as `*`.
        let atomic_symbol: &str = if atomic_number == WILDCARD_ATOMIC_NUMBER {
            WILDCARD_SYMBOL
        } else {
            let Some(symbol) = atomic_number.atomic_symbol() else {
                return;
            };
            symbol
        };

        // Check for aromaticity: if any bond of the atom is aromatic, make the symbol lowercase
        let is_aromatic = self
            .get_atom_bonds(node.index())
            .unwrap()
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Aromatic);
        // Written lowercase for an aromatic atom. Pushed a byte at a time
        // rather than through `to_lowercase`, which allocates a String for
        // every atom in the molecule and can only ever change ASCII here.
        let push_symbol = |out: &mut String| {
            if is_aromatic {
                for byte in atomic_symbol.bytes() {
                    out.push(byte.to_ascii_lowercase() as char);
                }
            } else {
                out.push_str(atomic_symbol);
            }
        };

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
        let is_organic_subset = matches!(
            atomic_number,
            0 | 5 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 35 | 53
        );

        // Whether a reader handed this atom without brackets would give it back
        // the hydrogens it has. The count is not a property of the element
        // alone: it is what the element's valence leaves over once the bonds
        // the atom already carries are taken off, so `P` is phosphine and
        // `P(C)(C)C` is trimethylphosphine, and the old test -- which asked
        // only whether the hydrogen count was within the element's lowest
        // valence -- wrote `[PH2](C)(C)C` as `P(C)(C)C` and lost two.
        //
        // Two readers have to agree, not one. This crate fills a bare atom to
        // the element's *lowest* valence; every other toolkit fills it to the
        // smallest valence the bonds have not already outgrown, which is the
        // same number until an element with several valences passes the first.
        // A sulfur with three bonds is nought hydrogens here and one hydrogen
        // elsewhere, so `[S](C)=C` written bare comes back as a different
        // molecule -- brackets whenever the two disagree. Aromatic atoms are
        // exempt because the two rules already agree on them: an aromatic bond
        // counts one and a half, so the sum has passed the lowest valence for
        // every heteroatom and both readers infer none.
        let has_standard_hydrogens = if atomic_number == 0 {
            // A wildcard has no valence, so any hydrogen on it must be stated.
            number_of_hydrogens == 0
        } else {
            let hydrogen_bond_valence = 2 * number_of_hydrogens as i8;
            let bare_bond_valence =
                (self.actual_valency(node.index()) * 2 - hydrogen_bond_valence) / 2;
            crate::molecule::base::effective_valency(atomic_number, charge).is_some_and(|valency| {
                let inferred = (valency - bare_bond_valence).max(0);
                number_of_hydrogens as i8 == inferred
                    && (is_aromatic
                        || crate::molecule::base::saturating_valency(
                            atomic_number,
                            charge,
                            bare_bond_valence,
                        )
                        // Bonds past every valence the element has leave a
                        // reader nothing to fill in.
                        .map_or(inferred == 0, |reader| {
                            reader - bare_bond_valence == inferred
                        }))
            })
        };

        let has_negative_degree = self.get_atom_degree(node.index()).unwrap_or(0) < 0;

        // Cis/trans stereochemistry is emitted as directional bonds (`/`,`\`),
        // not as an atom tag, so it does not require brackets or a chiral symbol.
        let chirality = self.get_chiral_class(node.index());
        let is_cis_trans_class = matches!(
            chirality,
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        );
        let is_chiral = self.is_atom_chiral(node.index()) && !is_cis_trans_class;
        let needs_brackets = is_labelled
            || is_radical
            || !(!is_charged && is_organic_subset && has_standard_hydrogens)
            || (has_negative_degree);
        if needs_brackets || is_chiral {
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
                    let mut tag = crate::molecule::chirality::tetrahedral_tag_for_order(
                        reference, &sorted, &emission,
                    );
                    if crate::molecule::chirality::tetrahedral_ring_closure_inverts(
                        self,
                        atom,
                        ring_closures.get(&atom).map_or(0, Vec::len),
                    ) {
                        tag = 3 - tag;
                    }
                    if tag == 2 {
                        "@@".to_string()
                    } else {
                        "@".to_string()
                    }
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
                            if tag == 2 {
                                "@@".to_string()
                            } else {
                                "@".to_string()
                            }
                        }
                        _ => {
                            if reference == 2 {
                                "@@".to_string()
                            } else {
                                "@".to_string()
                            }
                        }
                    }
                }
                ChiralClass::SP(n) => format!("@SP{}", n),
                ChiralClass::TB(n) => format!("@TB{}", n),
                ChiralClass::OH(n) => format!("@OH{}", n),
                _ => String::new(),
            };

            // Pushed field by field. Building this through `format!` cost
            // five allocations for every bracketed atom -- one per field and
            // one for the result -- all of which were copied into `smiles`
            // and dropped immediately.
            smiles.push('[');
            if let Some(mass_number) = isotope {
                let _ = write!(smiles, "{mass_number}");
            }
            push_symbol(smiles);
            smiles.push_str(&chiral_str);
            match number_of_hydrogens {
                0 => {}
                1 => smiles.push('H'),
                count => {
                    let _ = write!(smiles, "H{count}");
                }
            }
            match charge {
                0 => {}
                1 => smiles.push('+'),
                -1 => smiles.push('-'),
                2.. => {
                    let _ = write!(smiles, "+{}", charge.abs());
                }
                ..=-2 => {
                    let _ = write!(smiles, "-{}", charge.abs());
                }
            }
            smiles.push(']');
        } else {
            push_symbol(smiles);
        }

        if let Some(closures) = ring_closures.get(&node.index()) {
            for closure in closures {
                let bond_order = closure.2;
                let ring_number = closure.1;
                let neighbor = closure.0;
                let are_both_atoms_aromatic =
                    self.is_atom_aromatic(node.index()) && self.is_atom_aromatic(neighbor);
                // A ring closure carrying a double bond's geometry is written
                // with its mark in place of the plain bond symbol.
                if let Some(mark) = directional_marks.ring(node.index(), neighbor) {
                    // Written at the closing end, and a mark on a ring-closure
                    // digit describes the bond running from the atom it is
                    // attached to *toward* its partner. The partner here was
                    // written earlier, so the symbol points the opposite way to
                    // the one a tree edge would carry for the same geometry,
                    // and has to be flipped. Preserving the marks without this
                    // kept the count and inverted the compound: cis,cis
                    // cycloocta-1,5-diene came back as cis,trans.
                    smiles.push(if mark == BondOrder::Up { '\\' } else { '/' });
                    write_ring_label(smiles, ring_number);
                    continue;
                }
                if (bond_order == BondOrder::Single && !are_both_atoms_aromatic)
                    || bond_order == BondOrder::Aromatic
                {
                    write_ring_label(smiles, ring_number);
                } else {
                    let _ = write!(smiles, "{bond_order}");
                    write_ring_label(smiles, ring_number);
                }
            }
        }

        if node.children().len() == 1 {
            self.construct_smiles(
                smiles,
                &node.children()[0],
                ring_closures,
                directional_marks,
                allene_orders,
            );
        } else {
            let length = node.children().len();
            for (index, child) in node.children().iter().enumerate() {
                if length - 1 == index {
                    // The terminal child does not need braces.
                    self.construct_smiles(
                        smiles,
                        child,
                        ring_closures,
                        directional_marks,
                        allene_orders,
                    );
                    continue;
                }
                smiles.push('(');
                self.construct_smiles(
                    smiles,
                    child,
                    ring_closures,
                    directional_marks,
                    allene_orders,
                );
                smiles.push(')');
            }
        }
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

            let is_allene = matches!(classes.get(node.index()), Some(ChiralClass::AL(_)));
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

    /// Neighbours in emitted SMILES order. Folded H follows the from-atom;
    /// an explicit isotope-H instead occupies its actual position in the tree.
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
            .filter(|&target| self.hydrogen_folds_into_neighbour(target))
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
    /// `to_smiles` exactly (root = first non-folded atom, hydrogens folded into the
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
            // A folded hydrogen is not emitted as an atom and therefore
            // cannot be the from-atom of a stereocentre. Starting there would
            // count it both as parent and as folded H in the stereo ordering.
            let root_index = component
                .iter()
                .find(|&&atom| !self.hydrogen_folds_into_neighbour(atom))
                .unwrap_or(&component[0]);
            let root = self.build_smiles_tree(
                *root_index,
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
                    .filter(|&target| self.hydrogen_folds_into_neighbour(target))
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

            // Process rings in order of first appearance in traversal, and an
            // atom opening several of them in the order its partners are
            // reached -- see the canonical writer above.
            let mut position = vec![usize::MAX; self.atomic_numbers().len()];
            for (place, &atom_index) in traversal_order.iter().enumerate() {
                if atom_index < position.len() {
                    position[atom_index] = place;
                }
            }
            for &atom_idx in &traversal_order {
                let mut opening: Vec<&(usize, usize, BondOrder)> = rings
                    .iter()
                    .filter(|(start, _, _)| *start == atom_idx)
                    .collect();
                opening
                    .sort_by_key(|(_, end, _)| position.get(*end).copied().unwrap_or(usize::MAX));
                for (start, end, bond_order) in opening {
                    ring_closures
                        .entry(*start)
                        .or_default()
                        .push((*end, ring_number, *bond_order));
                    ring_closures
                        .entry(*end)
                        .or_default()
                        .push((*start, ring_number, *bond_order));
                    ring_number += 1;
                }
            }

            // The digits an atom carries are written in ascending order, and
            // they are the numbers the string will actually show -- see
            // [`settle_ring_labels`]. `chiral_emission_order` reads these same
            // lists, so the neighbour ordering a tag is written against moves
            // with the digits and the stereochemistry says what it always said.
            settle_ring_labels(&root, self.atomic_numbers().len(), &mut ring_closures);

            if !smiles.is_empty() {
                smiles.push('.');
            }

            let directional_marks = self.directional_bond_marks(&root);
            let allene_orders = self.allene_emission_orders(&root, &ring_closures);
            self.construct_smiles(
                &mut smiles,
                &root,
                &ring_closures,
                &directional_marks,
                &allene_orders,
            )
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
        let molecules = SMILESParser::parse_smiles(smiles)?;
        Ok(molecules)
    }
}

fn relabel_smiles_numbers(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    // A ring-closure label is one digit, or two after a `%`, so the mapping
    // from the label as written to the label as emitted fits in a fixed table
    // indexed by the number itself. This was a `HashMap<String, String>` built
    // per molecule, with a `String` allocated for every single ring digit in
    // the input and another for every mapping stored.
    let mut mapping: Vec<u16> = vec![0; 100];
    let mut next_number: u16 = 1;
    let mut chars = input.chars().peekable();
    let mut is_bracket = false;

    // The `%` prefix belongs to the label that comes out, not the one that went
    // in. Renumbering moves labels across the one-digit boundary in both
    // directions -- a `%12` that becomes 3 must be written `3`, and a `5` that
    // becomes 10 must be written `%10` -- so carrying the prefix over from the
    // input produced `%3`, which is not a label at all, and `10`, which is two
    // labels. Molecules needing ten open ring closures at once were written as
    // SMILES no reader could take back.
    let mut push_mapped = |result: &mut String, label: u32| {
        // Every label is remapped, however it was written. Passing the high
        // ones through untouched -- which is what this did -- left them out of
        // the compaction, so a renumbering that could have fitted inside two
        // digits kept the extended form for no reason.
        if mapping.len() <= label as usize {
            mapping.resize(label as usize + 1, 0);
        }
        let slot = &mut mapping[label as usize];
        if *slot == 0 {
            *slot = next_number;
            next_number += 1;
        }
        if *slot > 9 {
            write_ring_label(result, *slot as usize);
        } else {
            let _ = write!(result, "{}", *slot);
        }
    };

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
            let mut label: u32 = 0;
            if chars.peek() == Some(&'(') {
                // The extended form `%(nnn)`, digits until the closing paren.
                chars.next();
                while let Some(&next_char) = chars.peek() {
                    let Some(digit) = next_char.to_digit(10) else {
                        break;
                    };
                    label = label * 10 + digit;
                    chars.next();
                }
                // Drop the `)`; a malformed token is left to the reader to
                // refuse rather than silently re-spelled here.
                if chars.peek() == Some(&')') {
                    chars.next();
                }
            } else {
                // Exactly two digits, which is what `%` takes. Reading greedily
                // -- which is what this did -- swallowed the *next* label too,
                // so `%12` followed by a ring `3` became label 123.
                for _ in 0..2 {
                    let Some(&next_char) = chars.peek() else {
                        break;
                    };
                    let Some(digit) = next_char.to_digit(10) else {
                        break;
                    };
                    label = label * 10 + digit;
                    chars.next();
                }
            }
            push_mapped(&mut result, label);
        } else if let Some(digit) = c.to_digit(10) {
            push_mapped(&mut result, digit);
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

    #[test]
    fn nothing_is_written_aromatic_that_cannot_be_read_back() {
        // Whatever perception marks aromatic is written in lower case, and a
        // lower-case ring with no Kekulé structure is a string this crate's own
        // reader refuses. Every canonical form must survive a round trip.
        for smiles in [
            "C1(=O)SN=S=C1",
            "c1ccc2nsnc2c1",
            "C1=CC=C2N=S=NC2=C1",
            "O=c1cc[nH]c(=O)[nH]1",
            "c1ccc2c(c1)[nH]c3ccccc23",
            "C1=CC2=NC1=Cc1ccc([nH]1)C=C1C=CC(=N1)C=c1ccc(=C2)[nH]1",
        ] {
            let written = crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                smiles, false,
            )
            .unwrap();
            let again = crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                &written, false,
            )
            .unwrap_or_else(|error| {
                panic!("{smiles} was written {written}, which fails: {error:?}")
            });
            assert_eq!(written, again, "{smiles} is not a fixed point");
        }
    }

    #[test]
    fn a_string_with_no_kekule_structure_is_refused() {
        // Five carbons cannot share out a double bond each. RDKit refuses the
        // same string; what matters here is that we refuse it rather than
        // inventing a radical on whichever atom happens to be numbered first.
        assert!(matches!(
            MolecularGraph::from_smiles_single("O=c1ccccc1"),
            Err(SMILESParseError::Kekulization(_))
        ));
        assert!(matches!(
            MolecularGraph::from_smiles_single("O=[n+]1ccccc1[O-]"),
            Err(SMILESParseError::Kekulization(_))
        ));
        // The rings that do have one are untouched, including the ones whose
        // heteroatom contributes a lone pair and demands no double bond.
        for smiles in [
            "c1ccccc1",
            "c1cc[nH]c1",
            "c1ccsc1",
            "O=c1cc[nH]c(=O)[nH]1",
            "c1ccc2[nH]ccc2c1",
            "[cH-]1cccc1",
        ] {
            assert!(
                MolecularGraph::from_smiles_single(smiles).is_ok(),
                "{smiles} should still parse"
            );
        }
        // Perception on its own stays tolerant: it leaves what it cannot
        // localize exactly as written rather than failing.
        let options = SmilesParseOptions {
            aromaticity: None,
            ..Default::default()
        };
        let mut molecule = SMILESParser::parse_smiles_single_with("O=c1ccccc1", options).unwrap();
        assert!(molecule.aromatize_opensmiles().is_ok());
    }

    #[test]
    fn an_atom_given_more_bonds_than_it_can_hold_fails_the_parse() {
        // Ring-closure labels reused across a `.` pile onto whichever atom is
        // current. The bond list is a fixed-capacity inline array, so this used
        // to abort the process inside `tinyvec` instead of failing the parse.
        assert!(matches!(
            MolecularGraph::from_smiles("c1cc2ccc3ccc4ccc5ccc1[nH]2.[nH]3.[nH]4.[nH]5"),
            Err(SMILESParseError::TooManyBonds(_))
        ));
    }

    #[test]
    fn ring_labels_carry_the_percent_of_the_label_they_become() {
        // Renumbering moves labels across the one-digit boundary both ways, and
        // the `%` has to follow the label that comes out, not the one that went
        // in. `%12` renumbered to 3 must lose its `%`, and `5` renumbered to 10
        // must gain one -- written bare, `10` is two ring closures.
        assert_eq!(relabel_smiles_numbers("C%12CC%12"), "C1CC1");
        assert_eq!(
            relabel_smiles_numbers("C1C2C3C4C5C6C7C8C9C%10C%11C1C2C3C4C5C6C7C8C9C%10C%11"),
            "C1C2C3C4C5C6C7C8C9C%10C%11C1C2C3C4C5C6C7C8C9C%10C%11"
        );
        // Bracketed atoms keep their own digits: an isotope is not a label.
        assert_eq!(relabel_smiles_numbers("[13CH3]C%12CC%12"), "[13CH3]C1CC1");
    }

    #[test]
    fn buckminsterfullerene_is_written_with_readable_ring_labels() {
        // Sixty carbons need far more than nine ring closures open at once, so
        // this is the smallest well-known molecule that shows the difference.
        // Written bare, a renumbered `10` is two ring closures rather than one,
        // and the written molecule falls apart on the way back in.
        let smiles =
            "C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C8=C2C8=C%11C%10=C1C1=C8C(=C23)C4=C1C5=C67";
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        let written = molecule.to_smiles();
        assert!(
            !written.contains("%1c") && !written.contains("%1C"),
            "a `%` must introduce two digits in {written}"
        );
        let reparsed = MolecularGraph::from_smiles_single(&written).unwrap();
        assert_eq!(reparsed.len(), molecule.len());
        assert_eq!(reparsed.to_smiles(), written);
    }

    #[test]
    fn a_bare_symbol_is_only_used_where_both_readers_infer_the_same_hydrogens() {
        // Sulfur and phosphorus have more than one normal valence, so the count
        // a reader fills in depends on the bonds the atom already carries. The
        // test used to ask only whether the hydrogen count fell within the
        // element's lowest valence, which wrote a phosphorane as a phosphine.
        for (smiles, expected) in [
            // Hydrogens that a bare symbol would drop.
            ("[PH2](C)(C)C", "[PH2](C)(C)C"),
            ("C[PH](C)C", "[PH](C)(C)C"),
            ("C[SH]C", "[SH](C)C"),
            ("O=[SH2]", "[SH2]=O"),
            // Hydrogens a bare symbol would invent elsewhere: three bonds have
            // outgrown sulfur's first valence, so every other reader gives this
            // sulfur one hydrogen that it does not have.
            ("C1(=[S]C)NCCN1", "[S](C)=C1NCCN1"),
            // And the ordinary cases stay bare.
            ("CP(C)C", "P(C)(C)C"),
            ("CS(C)(C)C", "S(C)(C)(C)C"),
            ("CS(=O)C", "S(=O)(C)C"),
            ("C[S]C", "S(C)C"),
            ("c1ccsc1", "s1cccc1"),
            ("c1ccoc1", "o1cccc1"),
            ("c1ccccc1", "c1ccccc1"),
        ] {
            assert_eq!(
                crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                    smiles, false
                )
                .unwrap(),
                expected,
                "writing {smiles}"
            );
        }
    }

    #[test]
    fn a_cumulated_ring_centre_is_not_aromatized_into_something_unreadable() {
        // Perception takes the molecule's own double bonds as proof that a ring
        // can be written back out, by checking each atom's π count against its
        // demand. A cumulated `N=P=N` phosphorus passes that -- its two double
        // bonds do meet its demand -- but writing the ring out looks for a
        // matching, which gives every atom one double bond at most. So this ring
        // was written `s1n[p+]nsn[p+]n1`, which this crate's own reader refuses.
        for smiles in [
            "N1=[P+]=NSN=[P+]=NS1",
            // Hypervalent aromatic sulfur is a different case and stays
            // aromatic: `N=S=N` becomes `n-s-n` with the π elsewhere, which is
            // a matching.
            "C1=CC=C2N=S=NC2=C1",
            "C1=NSN=C1",
        ] {
            let written = crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                smiles, false,
            )
            .unwrap();
            crate::molecule::molecular_graph::MolecularGraph::from_smiles_single(&written)
                .unwrap_or_else(|error| panic!("{smiles} was written {written}: {error:?}"));
        }
    }

    #[test]
    fn a_selenoxide_counts_as_daylight_counts_a_sulfoxide() {
        // Daylight gives `[Se](=O)` and `[Se+]([O-])` the two electrons it
        // gives `S(=O)`, so selenophene oxide is aromatic however it is
        // written, exactly as thiophene oxide is. A selenoxide outside a ring
        // stays what it is.
        for (smiles, written) in [
            ("O=[Se]1C=CC=C1", "[se]1(=O)cccc1"),
            ("O=[se]1cccc1", "[se]1(=O)cccc1"),
            ("[O-][Se+]1C=CC=C1", "[se+]1([O-])cccc1"),
            ("O=[Se]1C=CC2=CC=CC=C12", "[se]1(=O)ccc2ccccc12"),
            ("O=[Se]1CCCC1", "[Se]1(=O)CCCC1"),
        ] {
            assert_eq!(
                crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                    smiles, false,
                )
                .unwrap(),
                written,
                "{smiles}"
            );
        }
    }

    #[test]
    fn a_neutral_sulfur_is_only_aromatic_where_it_has_a_lone_pair_to_give() {
        // Thiophene's sulfur donates its lone pair, and so does the sulfoxide's
        // once its exocyclic `=O` has taken the other. A three-coordinate
        // neutral sulfur has spent one pair on its third bond and would have to
        // become positive to give up the second, and a two-coordinate sulfur
        // carrying two double bonds has no p orbital left at all -- both used
        // to be counted as thiophene's and written back as SMILES no other
        // reader accepts.
        for (smiles, aromatic) in [
            ("c1ccsc1", true),
            ("c1ccc2sccc2c1", true),
            ("C1=CC=CS1(=O)", true),
            // 1,2,5-thiadiazole, written both ways round. The hypervalent
            // sulfur is the same aromatic sulfur as the divalent one, and a
            // rule that read the valence rather than the neighbour count made
            // the two writings disagree.
            ("c1ccc2nsnc2c1", true),
            ("C1=CC=C2N=S=NC2=C1", true),
            ("C1=NSN=C1", true),
            // Three σ bonds and no exocyclic `=O`: the second lone pair can
            // only be given up by becoming positive.
            ("C1=C[S](SC)N=N1", false),
            // A ring sulfone has neither a lone pair nor a p orbital left.
            ("O=C1NS(=O)(=O)c2ccccc21", false),
            ("O=S1(=O)C=CC=C1", false),
            // Six π electrons over five atoms, and still no ring: the two
            // demands sit either side of a sulfur with none, so no assignment
            // of double bonds reproduces the aromatic form. Perception has to
            // decline what it cannot write.
            ("C1(=O)SN=S=C1", false),
        ] {
            let written = crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                smiles, false,
            )
            .unwrap();
            assert_eq!(
                written.contains('s'),
                aromatic,
                "{smiles} was written {written}"
            );
        }
    }

    #[test]
    fn an_aromatic_phosphorus_keeps_its_hydrogen() {
        // Bare `p` is read as phosphinine-like, with no hydrogen, so a phosphole
        // written `p1cccc1` has silently lost one and is not a molecule any
        // reader will take back. Only carbon may state a hydrogen by being bare.
        let molecule = MolecularGraph::from_smiles_single("c1cc[pH]c1").unwrap();
        assert!(
            molecule.to_smiles().contains("[pH]"),
            "the phosphorus lost its hydrogen: {}",
            molecule.to_smiles()
        );
        assert_eq!(
            crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles(
                "c1cc[pH]c1",
                false
            )
            .unwrap(),
            "[pH]1cccc1"
        );
    }

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
                Ok(_molecules) => {
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
    fn a_stereo_marked_ring_closure_between_aromatic_atoms_leaves_no_radical() {
        // The canonical writing of PubChem CID 175635843: the ring closes on a
        // directional bond, so the pyridinium nitrogen has one aromatic bond
        // and one marked one. Each must read as the Kekulé writing beside it.
        for (smiles, kekule) in [
            (
                "[n+]1(ccnc(c\\1=C/C=C\\CC)=C)C",
                "CC/C=C\\C=C\\1/C(=C)N=CC=[N+]1C",
            ),
            ("[n+]\\1(ccnc(c1=C/C)=C)C", "C[N+]1=CC=NC(=C)/C1=C/C"),
            ("[n+]1(ccnc(c1=CC)=C)C", "C[N+]1=CC=NC(=C)C1=CC"),
        ] {
            let molecules = SMILESParser::parse_smiles(smiles).unwrap();
            let expected = SMILESParser::parse_smiles(kekule).unwrap();
            assert!(
                molecules[0]
                    .radical_states
                    .iter()
                    .all(|&radical| radical == 0),
                "{smiles}"
            );
            assert_eq!(
                molecules[0].atomic_numbers.len(),
                expected[0].atomic_numbers.len(),
                "{smiles}"
            );
        }
    }

    #[test]
    fn an_oversized_ring_closure_or_isotope_is_refused() {
        // Five digits is the most an extended ring closure may carry.
        assert!(SMILESParser::parse_smiles("C%(99999)CC%(99999)").is_ok());
        assert!(matches!(
            SMILESParser::parse_smiles("C%(999999999999)CC%(999999999999)"),
            Err(SMILESParseError::RingIndexError)
        ));
        assert!(matches!(
            SMILESParser::parse_smiles("C%(99999999999999999999999)CC"),
            Err(SMILESParseError::RingIndexError)
        ));
        // Any mass number the field holds is kept, as RDKit keeps it.
        let molecules = SMILESParser::parse_smiles("[999C]").unwrap();
        assert_eq!(molecules[0].get_atom_isotope(0), Some(999));
        assert!(SMILESParser::parse_smiles("[65535C]").is_ok());
        assert!(SMILESParser::parse_smiles("[65536C]").is_err());
    }

    #[test]
    fn marks_to_two_separate_rings_stay_single_bonds() {
        // The phosphorus has four neighbours, so its marked bonds to two
        // phenyls are substituent bonds, not bonds of a ring of marks.
        let molecule = SMILESParser::parse_smiles_single_with(
            "N=P(/c1ccccc1)(\\c2ccccc2)C",
            SmilesParseOptions { aromaticity: None },
        )
        .unwrap();
        let phosphorus = 1;
        for bond in molecule.get_atom_bonds(phosphorus).unwrap() {
            assert_ne!(bond.bond_order(), BondOrder::Aromatic, "{bond:?}");
        }
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
