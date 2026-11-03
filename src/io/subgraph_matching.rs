/*!
# Subgraph Isomorphism for SMARTS Pattern Matching

This module implements SMARTS pattern matching using petgraph's efficient subgraph
isomorphism algorithms. It handles the special requirements of SMARTS including:

- Logical atom expressions (AND, OR, NOT)
- Wildcard atoms and bonds
- Aromatic/aliphatic constraints
- Ring membership and size constraints
- Degree, valence, and connectivity constraints
- Recursive SMARTS patterns
- Atom classes and mappings

## Algorithm Overview

Uses petgraph's `subgraph_isomorphisms_iter` with custom node and edge matching logic:
1. Convert SMARTS pattern and molecule to petgraph UnGraph
2. Define node matching function for SMARTS atom expressions
3. Define edge matching function for SMARTS bond expressions
4. Use petgraph's optimized isomorphism detection
5. Return all valid mappings

## SMARTS-Specific Features

- **Robust ring detection** without SSSR dependency
- **Aromaticity detection** using molecule's aromatize() method
- **Logical expression evaluation** for complex atom/bond constraints
- **Performance optimizations** with IntMap and FxHashSet
*/

use crate::io::cxsmarts::{AtomExpression, BondExpression, CxSMARTSPattern, LogicalOperator};
use crate::graph_algorithms::monomorphism::Monomorphism;
use crate::molecule::bond::BondOrder;
use crate::molecule::molecular_graph::MolecularGraph;
use crate::molecule::traits::*;
use bit_set::BitSet;
use nohash_hasher::IntMap;
use rustc_hash::FxHashSet;
use std::collections::{HashMap, HashSet};

/// Represents a mapping state during subgraph isomorphism
#[derive(Debug, Clone)]
pub struct MappingState {
    /// Core mapping: pattern_atom_index -> molecule_atom_index
    pub core_pattern_to_molecule: HashMap<usize, usize>,
    /// Reverse core mapping: molecule_atom_index -> pattern_atom_index
    pub core_molecule_to_pattern: HashMap<usize, usize>,
    /// Terminal set for pattern (atoms connected to core)
    pub terminal_pattern: HashSet<usize>,
    /// Terminal set for molecule (atoms connected to core)
    pub terminal_molecule: HashSet<usize>,
}

impl MappingState {
    /// Create a new empty mapping state
    pub fn new() -> Self {
        Self {
            core_pattern_to_molecule: HashMap::new(),
            core_molecule_to_pattern: HashMap::new(),
            terminal_pattern: HashSet::new(),
            terminal_molecule: HashSet::new(),
        }
    }

    /// Add a new mapping between pattern and molecule atoms
    pub fn add_mapping(
        &mut self,
        pattern_atom: usize,
        molecule_atom: usize,
        pattern: &CxSMARTSPattern,
        molecule: &MolecularGraph,
    ) {
        self.core_pattern_to_molecule
            .insert(pattern_atom, molecule_atom);
        self.core_molecule_to_pattern
            .insert(molecule_atom, pattern_atom);

        self.update_terminal_sets(pattern_atom, molecule_atom, pattern, molecule);
    }

    /// Remove a mapping (for backtracking)
    pub fn remove_mapping(
        &mut self,
        pattern_atom: usize,
        molecule_atom: usize,
        pattern: &CxSMARTSPattern,
        molecule: &MolecularGraph,
    ) {
        self.core_pattern_to_molecule.remove(&pattern_atom);
        self.core_molecule_to_pattern.remove(&molecule_atom);

        // Recompute terminal sets
        self.recompute_terminal_sets(pattern, molecule);
    }

    /// Update terminal sets when adding a mapping
    fn update_terminal_sets(
        &mut self,
        pattern_atom: usize,
        molecule_atom: usize,
        pattern: &CxSMARTSPattern,
        molecule: &MolecularGraph,
    ) {
        self.terminal_pattern.remove(&pattern_atom);
        self.terminal_molecule.remove(&molecule_atom);

        for &(atom1, atom2, _) in &pattern.bonds {
            let neighbor = if atom1 == pattern_atom {
                Some(atom2)
            } else if atom2 == pattern_atom {
                Some(atom1)
            } else {
                None
            };

            if let Some(neighbor_atom) = neighbor {
                if !self.core_pattern_to_molecule.contains_key(&neighbor_atom) {
                    self.terminal_pattern.insert(neighbor_atom);
                }
            }
        }

        if let Some(bonds) = molecule.atom_bonds().get(molecule_atom) {
            for bond in bonds {
                let neighbor_atom = bond.target();
                if !self.core_molecule_to_pattern.contains_key(&neighbor_atom) {
                    self.terminal_molecule.insert(neighbor_atom);
                }
            }
        }
    }

    /// Recompute terminal sets from scratch
    fn recompute_terminal_sets(&mut self, pattern: &CxSMARTSPattern, molecule: &MolecularGraph) {
        self.terminal_pattern.clear();
        self.terminal_molecule.clear();

        for &pattern_atom in self.core_pattern_to_molecule.keys() {
            for &(atom1, atom2, _) in &pattern.bonds {
                let neighbor = if atom1 == pattern_atom {
                    Some(atom2)
                } else if atom2 == pattern_atom {
                    Some(atom1)
                } else {
                    None
                };

                if let Some(neighbor_atom) = neighbor {
                    if !self.core_pattern_to_molecule.contains_key(&neighbor_atom) {
                        self.terminal_pattern.insert(neighbor_atom);
                    }
                }
            }
        }

        for &molecule_atom in self.core_molecule_to_pattern.keys() {
            if let Some(bonds) = molecule.atom_bonds().get(molecule_atom) {
                for bond in bonds {
                    let neighbor_atom = bond.target();
                    if !self.core_molecule_to_pattern.contains_key(&neighbor_atom) {
                        self.terminal_molecule.insert(neighbor_atom);
                    }
                }
            }
        }
    }

    /// Check if the mapping is complete
    pub fn is_complete(&self, pattern: &CxSMARTSPattern) -> bool {
        self.core_pattern_to_molecule.len() == pattern.atoms.len()
    }
}


/// What a molecule looks like to the matcher: its rings and its aromaticity.
///
/// Neither depends on the pattern, and both are expensive -- the ring analysis
/// enumerates every simple cycle up to `MAX_RING_SIZE`. Matching a molecule
/// against one pattern computes them; matching it against ten patterns used to
/// compute them ten times. Build one of these and hand it to each match
/// instead:
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::io::MoleculeMatchContext;
///
/// let molecule = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
/// let context = MoleculeMatchContext::new(&molecule);
///
/// for pattern in ["c1ccccc1", "[CX3](=O)[OX2H1]"] {
///     let pattern = parse_cxsmarts(pattern).unwrap();
///     assert!(pattern.matches_with(&molecule, &context));
/// }
/// ```
///
/// It describes one molecule and is only valid while that molecule is
/// unchanged.
#[derive(Debug, Clone)]
pub struct MoleculeMatchContext {
    pub(crate) ring_info: RingInfo,
    pub(crate) aromaticity_info: AromaticityInfo,
}

impl MoleculeMatchContext {
    pub fn new(molecule: &MolecularGraph) -> Self {
        Self {
            ring_info: compute_ring_info(molecule),
            aromaticity_info: compute_aromaticity_info(molecule),
        }
    }
}

/// Cached ring information for efficient R/r property evaluation
#[derive(Debug, Clone)]
pub struct RingInfo {
    /// Number of simple rings (up to MAX_RING_SIZE) each atom belongs to
    pub atom_ring_count: IntMap<usize, usize>,
    /// Smallest ring size for each atom
    pub atom_smallest_ring: IntMap<usize, usize>,
    /// All ring sizes for each atom
    pub atom_ring_sizes: IntMap<usize, FxHashSet<usize>>,
    /// Bonds that are part of a ring, stored as ordered (min, max) atom pairs
    pub cyclic_bonds: FxHashSet<(usize, usize)>,
    /// Atoms that lie on a ring. Exact for any ring size, unlike
    /// [`RingInfo::atom_ring_count`], which is derived from enumerating simple
    /// cycles and so stops at `MAX_RING_SIZE`.
    pub cyclic_atoms: BitSet,
}

/// Cached aromaticity information
#[derive(Debug, Clone)]
pub struct AromaticityInfo {
    /// Whether each atom is aromatic
    pub is_aromatic: IntMap<usize, bool>,
    /// Whether each bond is aromatic
    pub aromatic_bonds: FxHashSet<(usize, usize)>,
}

/// What the matcher needs to know about a molecule before it looks at a
/// pattern: which bonds and atoms are in rings, and which are aromatic.
///
/// These were associated functions of a second, unreachable matcher that has
/// been removed. Keeping two matchers in one file, each with its own copy of
/// the atom and bond evaluators, is how a stale reading of SMARTS `h` survived
/// in the copy nothing called.
/// Compute ring information for the molecule by enumerating simple cycles.
///
/// All simple cycles of size up to `MAX_RING_SIZE` are enumerated within the
/// cyclic subgraph reported by `find_cyclic_bonds_and_atoms()`. This is not
/// SSSR: an atom's ring count is the number of small simple cycles through
/// it, which agrees with SSSR membership for common fused and isolated ring
/// systems (benzene R1, naphthalene bridgeheads R2, ...) but can overcount
/// in densely bridged cages.
fn compute_ring_info(molecule: &MolecularGraph) -> RingInfo {
    let (cyclic_atoms, cyclic_bonds) = molecule.find_cyclic_bonds_and_atoms();

    let mut atom_ring_count = IntMap::default();
    let mut atom_smallest_ring = IntMap::default();
    let mut atom_ring_sizes = IntMap::default();

    for atom_index in 0..molecule.number_of_atoms() {
        atom_ring_count.insert(atom_index, 0);
        atom_smallest_ring.insert(atom_index, 0);
        atom_ring_sizes.insert(atom_index, FxHashSet::default());
    }

    // The smallest ring through each bond, which is the symmetrized ring
    // set every toolkit's `r` and `R` primitives are written against.
    //
    // This used to enumerate every simple cycle of up to eight atoms and
    // reduce them. Eight atoms: so `[C;r9]` matched nothing, ever, and
    // neither did anything else asking about a larger ring -- including the
    // whole of RDKit's macrocycle torsion rule set, whose patterns all
    // carry `r{9-}`. A thirteen-membered ring reported its atoms as being
    // in a ring, with no ring size attached to say which. The
    // smallest-ring-per-bond search has no size limit and is polynomial,
    // where enumerating simple cycles is why there was a limit at all.
    let rings = crate::molecule::topology::smallest_rings(molecule);

    for ring in &rings {
        let size = ring.len();
        for &atom in ring {
            *atom_ring_count.entry(atom).or_insert(0) += 1;
            atom_ring_sizes.entry(atom).or_default().insert(size);
            let smallest = atom_smallest_ring.entry(atom).or_insert(0);
            if *smallest == 0 || size < *smallest {
                *smallest = size;
            }
        }
    }

    RingInfo {
        atom_ring_count,
        atom_smallest_ring,
        atom_ring_sizes,
        cyclic_bonds,
        cyclic_atoms,
    }
}



/// Compute aromaticity information for the molecule using the robust aromatize() method
///
/// This implementation creates a temporary copy of the molecule and runs the full
/// aromatization algorithm on it, then extracts the aromatic information. This approach:
/// 1. Uses the robust aromatize() method which handles all complex aromaticity logic
/// 2. Avoids manual aromaticity detection and potential edge cases
/// 3. Provides reliable results that match the molecule's expected aromaticity
///
/// This is the most reliable approach as it uses the same logic that would be
/// used when the molecule is processed normally.
fn compute_aromaticity_info(molecule: &MolecularGraph) -> AromaticityInfo {
    let mut is_aromatic = IntMap::default();
    let mut aromatic_bonds = FxHashSet::default();

    // Read the molecule as it stands rather than perceiving again.
    //
    // Aromaticity perception belongs to parsing (see
    // `crate::io::SmilesParseOptions`, on by default), and re-running it
    // here made the two halves of a SMARTS test disagree: atom
    // aromaticity came from the fresh perception while bond aromaticity
    // came from the molecule's actual bond orders, so a Kekulé ring had
    // aromatic *atoms* joined by non-aromatic *bonds* and matched neither
    // `c1ccccc1` nor `C=C`. It also cloned and re-perceived the whole
    // molecule on every match call.
    let has_existing_aromatic_bonds = molecule.atom_bonds().iter().any(|bonds| {
        bonds
            .iter()
            .any(|bond| bond.bond_order() == crate::molecule::bond::BondOrder::Aromatic)
    });
    let temp_molecule = molecule.clone();

    for atom_index in 0..molecule.number_of_atoms() {
        is_aromatic.insert(atom_index, false);
    }

    // For molecules with existing aromatic bonds, mark atoms as aromatic based on their bonds
    if has_existing_aromatic_bonds {
        for (atom_index, bonds) in temp_molecule.atom_bonds().iter().enumerate() {
            let has_aromatic_bond = bonds
                .iter()
                .any(|bond| bond.bond_order() == crate::molecule::bond::BondOrder::Aromatic);
            if has_aromatic_bond {
                is_aromatic.insert(atom_index, true);
            }
        }
    } else {
        let aromatic_atom_indices = temp_molecule.get_aromatic_atoms();

        for atom_index in aromatic_atom_indices {
            is_aromatic.insert(atom_index, true);
        }
    }

    for (atom_index, bonds) in temp_molecule.atom_bonds().iter().enumerate() {
        // Only process if this atom is aromatic
        if !is_aromatic.get(&atom_index).copied().unwrap_or(false) {
            continue;
        }

        for bond in bonds {
            let target_index = bond.target();

            if bond.bond_order() == crate::molecule::bond::BondOrder::Aromatic
                && is_aromatic.get(&target_index).copied().unwrap_or(false)
            {
                // Add aromatic bond (ensure consistent ordering to avoid duplicates)
                let bond_tuple = if atom_index < target_index {
                    (atom_index, target_index)
                } else {
                    (target_index, atom_index)
                };
                aromatic_bonds.insert(bond_tuple);
            }
        }
    }

    AromaticityInfo {
        is_aromatic,
        aromatic_bonds,
    }
}

/// How many hydrogens an atom has.
///
/// Hydrogens are atoms in this crate and the parser makes every one of them
/// explicit, so SMARTS `H` -- total hydrogens, implicit and explicit together
/// -- is a neighbour count. Inferring extra ones from a valence shortfall would
/// be wrong here: `[Si]1C=CN=[Si]1` says its silicons carry no hydrogen,
/// whatever their valence leaves over.
fn total_hydrogens(molecule: &MolecularGraph, atom: usize) -> usize {
    molecule.number_of_bonded_element(atom, 1)
}

/// Simplified SMARTS matcher using petgraph
pub struct PetgraphSMARTSMatcher<'a> {
    pattern: &'a CxSMARTSPattern,
    molecule: &'a MolecularGraph,
    ring_info: std::borrow::Cow<'a, RingInfo>,
    aromaticity_info: std::borrow::Cow<'a, AromaticityInfo>,
    /// Which atoms each recursive sub-pattern can be rooted at, worked out the
    /// first time that sub-pattern is asked about.
    ///
    /// A recursive pattern is a question about one atom -- "is this atom the
    /// first atom of something that looks like *this*" -- but answering it
    /// means searching the whole molecule. Asking it afresh for every candidate
    /// atom, inside a search that is itself exponential, is what makes
    /// recursive SMARTS notoriously slow elsewhere. One search per sub-pattern
    /// per molecule answers it for every atom at once.
    recursive_roots: std::cell::RefCell<HashMap<String, Vec<bool>>>,
    /// Per pattern atom, whether its expression asks about chirality, which
    /// the search then leaves to [`PetgraphSMARTSMatcher::stereo_matches`].
    /// Worked out once here rather than for every candidate atom, where
    /// walking the expression tree was a sixth of a MACCS fingerprint.
    atom_stereo: Vec<bool>,
    /// Whether anything in the pattern asks about stereochemistry at all.
    stereo: bool,
}

thread_local! {
    /// Every recursive sub-pattern this thread has met, parsed. The source of
    /// a `[$(...)]` is all the pattern keeps, and parsing it again for every
    /// molecule it was asked of cost more than matching it.
    static RECURSIVE_PATTERNS: std::cell::RefCell<HashMap<String, Option<std::rc::Rc<CxSMARTSPattern>>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// The inside of a `[$(...)]`, parsed once per thread; `None` if it does not
/// parse.
fn parsed_recursive(source: &str) -> Option<std::rc::Rc<CxSMARTSPattern>> {
    RECURSIVE_PATTERNS.with(|cache| {
        if let Some(parsed) = cache.borrow().get(source) {
            return parsed.clone();
        }
        let parsed = crate::io::parse_cxsmarts(source).ok().map(std::rc::Rc::new);
        cache.borrow_mut().insert(source.to_string(), parsed.clone());
        parsed
    })
}

impl<'a> PetgraphSMARTSMatcher<'a> {
    /// Create a new petgraph-based SMARTS matcher
    pub fn new(pattern: &'a CxSMARTSPattern, molecule: &'a MolecularGraph) -> Self {
        Self::build(
            pattern,
            molecule,
            std::borrow::Cow::Owned(compute_ring_info(molecule)),
            std::borrow::Cow::Owned(compute_aromaticity_info(molecule)),
        )
    }

    /// [`PetgraphSMARTSMatcher::new`] against an analysis the caller already has.
    pub fn with_context(
        pattern: &'a CxSMARTSPattern,
        molecule: &'a MolecularGraph,
        context: &'a MoleculeMatchContext,
    ) -> Self {
        Self::build(
            pattern,
            molecule,
            std::borrow::Cow::Borrowed(&context.ring_info),
            std::borrow::Cow::Borrowed(&context.aromaticity_info),
        )
    }

    fn build(
        pattern: &'a CxSMARTSPattern,
        molecule: &'a MolecularGraph,
        ring_info: std::borrow::Cow<'a, RingInfo>,
        aromaticity_info: std::borrow::Cow<'a, AromaticityInfo>,
    ) -> Self {
        Self {
            pattern,
            molecule,
            ring_info,
            aromaticity_info,
            recursive_roots: std::cell::RefCell::new(HashMap::new()),
            atom_stereo: pattern.atoms.iter().map(super::smarts_stereo::has_atom_stereo).collect(),
            stereo: super::smarts_stereo::has_stereo(pattern),
        }
    }

    /// Which atoms `source` -- the inside of a `[$(...)]` -- can be rooted at.
    ///
    /// The recursive pattern's first atom is the one being asked about, which
    /// is what makes `[$(CC=O)]` a carbon next to a carbonyl rather than a
    /// three-atom fragment.
    fn recursive_roots(&self, source: &str) -> Vec<bool> {
        let Some(pattern) = parsed_recursive(source) else {
            // An unreadable sub-pattern matches nothing. Saying it matches
            // everything -- which is what leaving it unimplemented did -- turns
            // every key built on it into a bit that is always set.
            return vec![false; self.molecule.number_of_atoms()];
        };
        let inner = PetgraphSMARTSMatcher::build(
            &pattern,
            self.molecule,
            std::borrow::Cow::Borrowed(&self.ring_info),
            std::borrow::Cow::Borrowed(&self.aromaticity_info),
        );
        // One rooted search per atom, each stopping at its first match: the
        // question is only ever whether a root has one, and listing every
        // mapping to find out enumerated the symmetric ones many times over.
        inner
            .monomorphism()
            .rooted_matches(&mut |assignment| inner.stereo_matches(assignment))
    }

    /// Every way the pattern can be laid over the molecule, as
    /// `pattern atom -> molecule atom`.
    ///
    /// SMARTS matching is subgraph monomorphism; see
    /// [`crate::graph_algorithms::monomorphism`] for why that is not what
    /// petgraph's `subgraph_isomorphisms_iter` provides.
    pub fn find_all_matches(&self) -> Vec<HashMap<usize, usize>> {
        let mut matches = Vec::new();
        self.monomorphism().run(&mut |assignment| {
            if self.stereo_matches(assignment) {
                matches.push(assignment.iter().copied().enumerate().collect());
            }
            true
        });
        matches
    }

    /// Whether the pattern matches anywhere, stopping at the first complete
    /// mapping that satisfies both constitution and stereochemistry.
    pub fn has_match(&self) -> bool {
        if !self.stereo {
            return self.monomorphism().any();
        }
        let mut found = false;
        self.monomorphism().run(&mut |assignment| {
            found = self.stereo_matches(assignment);
            !found
        });
        found
    }

    fn stereo_matches(&self, assignment: &[usize]) -> bool {
        if !self.stereo {
            return true;
        }
        self.pattern.atoms.iter().enumerate().all(|(atom, expression)| {
            super::smarts_stereo::atom_matches(
                expression, self.pattern, self.molecule, atom, assignment,
                &|expr| self.evaluate_atom_expression(expr, assignment[atom]),
            )
        }) && super::smarts_stereo::bonds_match(
            self.pattern, self.molecule, assignment,
            &|expr, a, b| self.evaluate_bond_expression(expr, a, b),
        )
    }

    /// The pattern and the three questions the search asks about it.
    fn monomorphism(
        &self,
    ) -> Monomorphism<
        impl Fn(usize, &mut Vec<usize>) + '_,
        impl Fn(usize, usize) -> bool + '_,
        impl Fn(usize, usize, usize) -> bool + '_,
    > {
        Monomorphism {
            pattern_node_count: self.pattern.atoms.len(),
            pattern_edges: self.pattern.bonds.iter().map(|&(a, b, _)| (a, b)).collect(),
            target_node_count: self.molecule.number_of_atoms(),
            neighbours: |atom: usize, out: &mut Vec<usize>| {
                out.extend(self.molecule.atom_bonds()[atom].iter().map(|bond| bond.target()))
            },
            node_match: |pattern_atom: usize, molecule_atom: usize| {
                self.atom_stereo[pattern_atom]
                    || self.evaluate_atom_expression(&self.pattern.atoms[pattern_atom], molecule_atom)
            },
            edge_match: |edge_index: usize, first: usize, second: usize| {
                if self.get_bond_order(first, second).is_none() {
                    return false;
                }
                let expr = &self.pattern.bonds[edge_index].2;
                super::smarts_stereo::bond_candidate(expr, &|expr| self.evaluate_bond_expression(expr, first, second))
            },
        }
    }

    /// Evaluate an atom expression against a molecule atom
    fn evaluate_atom_expression(&self, expr: &AtomExpression, molecule_atom: usize) -> bool {
        match expr {
            AtomExpression::Wildcard => true,

            AtomExpression::Element(element) => {
                let atomic_symbol = self
                    .molecule
                    .get_atomic_symbol(molecule_atom)
                    .unwrap_or("None");

                // Handle aromatic element symbols (lowercase)
                if element.chars().next().unwrap_or(' ').is_ascii_lowercase() {
                    // For aromatic elements like "c", "n", "o", check both element type and aromaticity
                    // `eq_ignore_ascii_case` rather than uppercasing the
                    // pattern into a fresh String on every candidate atom.
                    atomic_symbol.eq_ignore_ascii_case(element)
                        && self
                            .aromaticity_info
                            .is_aromatic
                            .get(&molecule_atom)
                            .copied()
                            .unwrap_or(false)
                } else {
                    // For aliphatic elements like "C", "N", "O", check element type and non-aromatic
                    atomic_symbol == element
                        && !self
                            .aromaticity_info
                            .is_aromatic
                            .get(&molecule_atom)
                            .copied()
                            .unwrap_or(false)
                }
            }

            AtomExpression::AtomicNumber(atomic_number) => {
                self.molecule.atomic_numbers()[molecule_atom] == *atomic_number
            }

            AtomExpression::Aromatic => self
                .aromaticity_info
                .is_aromatic
                .get(&molecule_atom)
                .copied()
                .unwrap_or(false),

            AtomExpression::Aliphatic => !self
                .aromaticity_info
                .is_aromatic
                .get(&molecule_atom)
                .copied()
                .unwrap_or(false),

            AtomExpression::Degree(degree) => {
                self.molecule.atom_bonds()[molecule_atom].len() == (*degree as usize)
            }

            // SMARTS `h` and `H` ask different questions of a toolkit that
            // keeps some hydrogens implicit. This one does not -- the parser
            // makes every hydrogen an atom -- so both are the same neighbour
            // count and are answered the same way.
            AtomExpression::ImplicitHydrogens(h_count)
            | AtomExpression::TotalHydrogens(h_count) => {
                total_hydrogens(self.molecule, molecule_atom) == (*h_count as usize)
            }

            // SMARTS `X`: how many connections the atom has, hydrogens
            // included, since they are atoms here.
            AtomExpression::Connectivity(connectivity) => {
                self.molecule
                    .get_atom_bonds(molecule_atom)
                    .unwrap_or(&[])
                    .len()
                    == *connectivity as usize
            }

            // SMARTS `v`: the sum of the atom's bond orders.
            AtomExpression::Valence(valence) => {
                self.molecule.actual_valency(molecule_atom) as usize == *valence as usize
            }

            // SMARTS `^<n>`, numbered as RDKit numbers it.
            AtomExpression::Hybridization(state) => {
                use crate::molecule::featurize::Hybridization;
                match crate::molecule::featurize::hybridization(self.molecule, molecule_atom) {
                    Hybridization::S => *state == 0,
                    Hybridization::Sp => *state == 1,
                    Hybridization::Sp2 => *state == 2,
                    Hybridization::Sp3 => *state == 3,
                    Hybridization::Sp3d => *state == 4,
                    Hybridization::Sp3d2 => *state == 5,
                    Hybridization::Unspecified => false,
                }
            }

            AtomExpression::Isotope(isotope) => {
                self.molecule.get_atom_isotope(molecule_atom) == Some(*isotope)
            }

            // An atom map number in a SMARTS is a label for whatever matches,
            // not a question about it -- `[C:2]` matches any aliphatic carbon,
            // including one the molecule maps to something else. It is how a
            // reaction says which atom goes where, and constraining on it would
            // make every reaction template match nothing.
            AtomExpression::AtomClass(_) => true,

            AtomExpression::Recursive(source) => {
                if let Some(roots) = self.recursive_roots.borrow().get(source) {
                    return roots.get(molecule_atom).copied().unwrap_or(false);
                }
                let roots = self.recursive_roots(source);
                let answer = roots.get(molecule_atom).copied().unwrap_or(false);
                self.recursive_roots
                    .borrow_mut()
                    .insert(source.clone(), roots);
                answer
            }

            // Bare `R`: on a ring, however many. Exact for every ring
            // size, where the simple-cycle enumeration behind
            // `atom_ring_count` stops at `MAX_RING_SIZE`.
            AtomExpression::InRing => self.ring_info.cyclic_atoms.contains(molecule_atom),

            AtomExpression::RingMembership(ring_count) => {
                let actual_ring_count = self
                    .ring_info
                    .atom_ring_count
                    .get(&molecule_atom)
                    .copied()
                    .unwrap_or(0);
                actual_ring_count == (*ring_count as usize)
            }

            // SMARTS `r<n>`: the *smallest* ring the atom is in has that size.
            // Not "is in a ring of that size" -- an indane bridgehead is in a
            // five- and a six-membered ring and is `r5` alone.
            AtomExpression::RingSize(ring_size) => {
                self.ring_info
                    .atom_smallest_ring
                    .get(&molecule_atom)
                    .copied()
                    .unwrap_or(0)
                    == *ring_size as usize
            }

            // SMARTS `r{m-n}`: *any* ring the atom belongs to has a size in
            // range, unlike `r<n>`, which asks about the smallest one. An
            // atom shared between a six-ring and a macrocycle is both
            // `r{5-8}` and `r{9-}`.
            AtomExpression::RingSizeRange(minimum, maximum) => self
                .ring_info
                .atom_ring_sizes
                .get(&molecule_atom)
                .is_some_and(|sizes| {
                    sizes.iter().any(|&size| {
                        minimum.is_none_or(|m| size >= m as usize)
                            && maximum.is_none_or(|m| size <= m as usize)
                    })
                }),

            AtomExpression::Charge(charge) => {
                self.molecule.get_atom_charge(molecule_atom) == *charge
            }

            AtomExpression::Logical { operator, operands } => {
                self.evaluate_logical_expression(operator, operands, molecule_atom)
            }

            // Stereo needs a complete atom mapping; checked in stereo_matches.
            AtomExpression::Chirality(_) => true,
        }
    }

    /// Evaluate logical expressions (AND, OR, NOT)
    fn evaluate_logical_expression(
        &self,
        operator: &LogicalOperator,
        operands: &[AtomExpression],
        molecule_atom: usize,
    ) -> bool {
        match operator {
            LogicalOperator::Not => {
                if operands.len() != 1 {
                    return false;
                }
                !self.evaluate_atom_expression(&operands[0], molecule_atom)
            }

            LogicalOperator::HighAnd | LogicalOperator::LowAnd => operands
                .iter()
                .all(|operand| self.evaluate_atom_expression(operand, molecule_atom)),

            LogicalOperator::Or => operands
                .iter()
                .any(|operand| self.evaluate_atom_expression(operand, molecule_atom)),
        }
    }

    /// Evaluate bond expressions
    fn evaluate_bond_expression(&self, expr: &BondExpression, atom1: usize, atom2: usize) -> bool {
        match expr {
            BondExpression::Any => true,

            BondExpression::Single => {
                self.get_bond_order(atom1, atom2)
                    .unwrap_or(BondOrder::Single)
                    == BondOrder::Single
            }

            // What SMARTS means by an unwritten bond.
            BondExpression::SingleOrAromatic => matches!(
                self.get_bond_order(atom1, atom2),
                Some(BondOrder::Single) | Some(BondOrder::Aromatic)
            ),

            BondExpression::Double => {
                self.get_bond_order(atom1, atom2)
                    .unwrap_or(BondOrder::Single)
                    == BondOrder::Double
            }

            BondExpression::Triple => {
                self.get_bond_order(atom1, atom2)
                    .unwrap_or(BondOrder::Single)
                    == BondOrder::Triple
            }

            BondExpression::Aromatic => {
                self.get_bond_order(atom1, atom2)
                    .unwrap_or(BondOrder::Single)
                    == BondOrder::Aromatic
            }

            BondExpression::Ring => {
                self.is_bond_in_ring(atom1, atom2)
            }

            BondExpression::Not(inner_expr) => {
                !self.evaluate_bond_expression(inner_expr, atom1, atom2)
            }

            BondExpression::Logical { operator, operands } => match operator {
                LogicalOperator::Not => {
                    if operands.len() != 1 {
                        return false;
                    }
                    !self.evaluate_bond_expression(&operands[0], atom1, atom2)
                }
                LogicalOperator::HighAnd | LogicalOperator::LowAnd => operands
                    .iter()
                    .all(|operand| self.evaluate_bond_expression(operand, atom1, atom2)),
                LogicalOperator::Or => operands
                    .iter()
                    .any(|operand| self.evaluate_bond_expression(operand, atom1, atom2)),
            },

            BondExpression::DirectionalUp | BondExpression::DirectionalDown
            | BondExpression::DirectionalUpOrUnspec | BondExpression::DirectionalDownOrUnspec => {
                self.get_bond_order(atom1, atom2) == Some(BondOrder::Single)
            }
        }
    }

    /// Get bond order between two atoms
    fn get_bond_order(&self, atom1: usize, atom2: usize) -> Option<BondOrder> {
        for bond in &self.molecule.atom_bonds()[atom1] {
            if bond.target() == atom2 {
                return Some(bond.bond_order());
            }
        }
        None
    }

    /// Check if bond is in a ring
    /// Whether the bond between two atoms lies on a ring.
    ///
    /// The bond itself has to, which is not the same as both its atoms doing:
    /// the bond joining biphenyl's two rings is in neither of them, and reading
    /// "both ends are ring atoms" as "the bond is a ring bond" made `!@` --
    /// SMARTS for a bond that closes nothing -- false for every bond in every
    /// molecule whose rings touch.
    fn is_bond_in_ring(&self, atom1: usize, atom2: usize) -> bool {
        let pair = (atom1.min(atom2), atom1.max(atom2));
        self.ring_info.cyclic_bonds.contains(&pair)
    }
}

/// Public API functions for SMARTS matching
pub fn find_smarts_matches(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
) -> Vec<HashMap<usize, usize>> {
    let matcher = PetgraphSMARTSMatcher::new(pattern, molecule);
    matcher.find_all_matches()
}

pub fn smarts_matches(pattern: &CxSMARTSPattern, molecule: &MolecularGraph) -> bool {
    let matcher = PetgraphSMARTSMatcher::new(pattern, molecule);
    matcher.has_match()
}

/// [`find_smarts_matches`] against an analysis of the molecule the caller
/// already has -- see [`MoleculeMatchContext`].
pub fn find_smarts_matches_with(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    context: &MoleculeMatchContext,
) -> Vec<HashMap<usize, usize>> {
    PetgraphSMARTSMatcher::with_context(pattern, molecule, context).find_all_matches()
}

/// [`smarts_matches`] against an analysis the caller already has.
pub fn smarts_matches_with(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    context: &MoleculeMatchContext,
) -> bool {
    PetgraphSMARTSMatcher::with_context(pattern, molecule, context).has_match()
}

/// New petgraph-based API functions (placeholder for now)
pub fn find_smarts_matches_petgraph(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
) -> Vec<HashMap<usize, usize>> {
    // For now, fall back to the original implementation until we fix the petgraph API
    find_smarts_matches(pattern, molecule)
}

pub fn smarts_matches_petgraph(pattern: &CxSMARTSPattern, molecule: &MolecularGraph) -> bool {
    // For now, fall back to the original implementation until we fix the petgraph API
    smarts_matches(pattern, molecule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SMILESParser, parse_cxsmarts};

    #[test]
    fn test_simple_match() {
        let molecules = SMILESParser::parse_smiles("CCO").unwrap();
        let molecule = &molecules[0];

        let pattern = parse_cxsmarts("C").unwrap();
        assert!(smarts_matches(&pattern, molecule));

        let matches = find_smarts_matches(&pattern, molecule);
        assert!(!matches.is_empty());
    }

    #[test]
    fn test_logical_operators() {
        let molecules = SMILESParser::parse_smiles("CCO").unwrap();
        let molecule = &molecules[0];

        // Test OR operator
        let pattern = parse_cxsmarts("[C,O]").unwrap();
        assert!(smarts_matches(&pattern, molecule));

        // Test NOT operator
        let pattern = parse_cxsmarts("[!N]").unwrap();
        assert!(smarts_matches(&pattern, molecule));
    }

    #[test]
    fn test_no_match() {
        let molecules = SMILESParser::parse_smiles("CCO").unwrap();
        let molecule = &molecules[0];

        let pattern = parse_cxsmarts("N").unwrap();
        assert!(!smarts_matches(&pattern, molecule));
    }
    #[test]
    fn test_ring_detection() {
        let molecules = SMILESParser::parse_smiles("C1CCCCC1").unwrap();
        let molecule = &molecules[0];

        let pattern = parse_cxsmarts("[*]1[*][*][*][*][*]1").unwrap();
        assert!(smarts_matches(&pattern, molecule));
    }

}
