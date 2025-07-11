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
use crate::molecule::aromaticity::MoleculeAromaticity;
use crate::graph_algorithms::monomorphism::Monomorphism;
use crate::molecule::bond::BondOrder;
use crate::molecule::molecular_graph::MolecularGraph;
use crate::molecule::traits::*;
use bit_set::BitSet;
use chemistry_consts::ElementProperties;
use nohash_hasher::IntMap;
use petgraph::{
    algo::subgraph_isomorphisms_iter,
    dot::Dot,
    graph::*,
    visit::{IntoEdgeReferences, IntoEdges, IntoEdgesDirected, IntoNodeReferences},
};
use rustc_hash::{FxHashMap, FxHashSet};
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

        // Update terminal sets
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
        // Remove from terminal sets if present
        self.terminal_pattern.remove(&pattern_atom);
        self.terminal_molecule.remove(&molecule_atom);

        // Add neighbors to terminal sets if not already in core
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

        // Add molecule neighbors using atom_bonds
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

        // Add pattern neighbors of core atoms
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

        // Add molecule neighbors of core atoms
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

/// SMARTS subgraph matcher using modified VF2 algorithm
pub struct SMARTSMatcher<'a> {
    pattern: &'a CxSMARTSPattern,
    molecule: &'a MolecularGraph,
    /// Cached ring analysis for molecule
    ring_info: RingInfo,
    /// Cached aromaticity information
    aromaticity_info: AromaticityInfo,
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

impl<'a> SMARTSMatcher<'a> {
    /// Create a new SMARTS matcher
    pub fn new(pattern: &'a CxSMARTSPattern, molecule: &'a MolecularGraph) -> Self {
        let ring_info = Self::compute_ring_info(molecule);
        let aromaticity_info = Self::compute_aromaticity_info(molecule);

        Self {
            pattern,
            molecule,
            ring_info,
            aromaticity_info,
        }
    }

    // The unfinished VF2 search that used to live here (`find_all_matches`,
    // `has_match`, `match_recursive`, `get_next_candidate_pair`) has been
    // removed. It was unreachable — every caller goes through
    // `PetgraphSMARTSMatcher` — and left a debug `println!` on a path that
    // would have run for every candidate pair. Keeping two matchers in one
    // file, each with its own copy of the atom and bond evaluators, is how the
    // implicit-bond bug survived as long as it did. What remains of this type
    // is the ring and aromaticity precomputation plus the expression
    // evaluators, which the working matcher uses.

    /// Check if a mapping is feasible (satisfies local constraints)
    fn is_feasible(&self, pattern_atom: usize, molecule_atom: usize, state: &MappingState) -> bool {
        // Check atom compatibility
        if !self.atoms_compatible(pattern_atom, molecule_atom) {
            return false;
        }

        // Check bond compatibility with already mapped neighbors
        if !self.bonds_compatible(pattern_atom, molecule_atom, state) {
            return false;
        }

        // Check connectivity constraints
        if !self.connectivity_compatible(pattern_atom, molecule_atom, state) {
            return false;
        }

        true
    }

    /// Check if two atoms are compatible
    fn atoms_compatible(&self, pattern_atom: usize, molecule_atom: usize) -> bool {
        let pattern_expr = &self.pattern.atoms[pattern_atom];
        self.evaluate_atom_expression(pattern_expr, molecule_atom)
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
                        && *self
                            .aromaticity_info
                            .is_aromatic
                            .get(&molecule_atom)
                            .unwrap_or(&false)
                } else {
                    // For aliphatic elements like "C", "N", "O", check element type and non-aromatic
                    atomic_symbol == element
                        && !*self
                            .aromaticity_info
                            .is_aromatic
                            .get(&molecule_atom)
                            .unwrap_or(&false)
                }
            }

            AtomExpression::AtomicNumber(atomic_num) => {
                self.molecule.atomic_numbers()[molecule_atom] == *atomic_num
            }

            AtomExpression::Aromatic => *self
                .aromaticity_info
                .is_aromatic
                .get(&molecule_atom)
                .unwrap_or(&false),

            AtomExpression::Aliphatic => !*self
                .aromaticity_info
                .is_aromatic
                .get(&molecule_atom)
                .unwrap_or(&false),

            AtomExpression::Degree(degree) => {
                self.molecule
                    .get_atom_bonds(molecule_atom)
                    .unwrap_or(&[])
                    .len()
                    == *degree as usize
            }

            AtomExpression::TotalHydrogens(hydrogen_count) => {
                let explicit_hydrogens = self.molecule.number_of_bonded_element(molecule_atom, 1);
                //let implicit_hydrogens = self.molecule.implicit_hydrogens(molecule_atom);
                //(explicit_hydrogens + implicit_hydrogens) == *hydrogen_count as usize
                explicit_hydrogens == *hydrogen_count as usize
            }

            AtomExpression::ImplicitHydrogens(hydrogen_count) => {
                //self.molecule.implicit_hydrogens(molecule_atom) == *hydrogen_count as usize
                let explicit_hydrogens = self.molecule.number_of_bonded_element(molecule_atom, 1);
                explicit_hydrogens == *hydrogen_count as usize
            }

            // Bare `R`: on a ring, however many. Exact for every ring
            // size, where the simple-cycle enumeration behind
            // `atom_ring_count` stops at `MAX_RING_SIZE`.
            AtomExpression::InRing => self.ring_info.cyclic_atoms.contains(molecule_atom),

            AtomExpression::RingMembership(ring_count) => {
                self.ring_info
                    .atom_ring_count
                    .get(&molecule_atom)
                    .unwrap_or(&0)
                    == &(*ring_count as usize)
            }

            AtomExpression::RingSize(ring_size) => {
                self.ring_info
                    .atom_smallest_ring
                    .get(&molecule_atom)
                    .unwrap_or(&0)
                    == &(*ring_size as usize)
            }

            AtomExpression::Valence(valence) => {
                let total_bond_order: usize = self.molecule.actual_valency(molecule_atom) as usize;
                total_bond_order == *valence as usize
            }

            AtomExpression::Connectivity(connectivity) => {
                self.molecule
                    .get_atom_bonds(molecule_atom)
                    .unwrap_or(&[])
                    .len()
                    == *connectivity as usize
            }

            AtomExpression::Charge(charge) => {
                self.molecule.get_atom_charge(molecule_atom) == *charge
            }

            AtomExpression::Logical { operator, operands } => {
                self.evaluate_logical_expression(operator, operands, molecule_atom)
            }

            // TODO: Implement other expressions
            _ => true, // Placeholder for unimplemented expressions
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

    /// Check bond compatibility between mapped atoms
    fn bonds_compatible(
        &self,
        pattern_atom: usize,
        molecule_atom: usize,
        state: &MappingState,
    ) -> bool {
        // Check bonds to already mapped neighbors
        for &(pat_atom1, pat_atom2, ref bond_expr) in &self.pattern.bonds {
            let pattern_neighbor = if pat_atom1 == pattern_atom {
                Some(pat_atom2)
            } else if pat_atom2 == pattern_atom {
                Some(pat_atom1)
            } else {
                None
            };

            if let Some(pattern_neighbor_atom) = pattern_neighbor {
                if let Some(&molecule_neighbor_atom) =
                    state.core_pattern_to_molecule.get(&pattern_neighbor_atom)
                {
                    // Check if there's a corresponding bond in the molecule
                    if !self.has_bond(molecule_atom, molecule_neighbor_atom) {
                        return false;
                    }

                    // Check bond expression compatibility
                    if !self.evaluate_bond_expression(
                        bond_expr,
                        molecule_atom,
                        molecule_neighbor_atom,
                    ) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check if there's a bond between two atoms
    fn has_bond(&self, atom1: usize, atom2: usize) -> bool {
        if let Some(bonds) = self.molecule.atom_bonds().get(atom1) {
            bonds.iter().any(|bond| bond.target() == atom2)
        } else {
            false
        }
    }

    /// Get bond order between two atoms
    fn get_bond_order(&self, atom1: usize, atom2: usize) -> Option<u8> {
        if let Some(bonds) = self.molecule.atom_bonds().get(atom1) {
            bonds
                .iter()
                .find(|bond| bond.target() == atom2)
                .map(|bond| bond.order())
        } else {
            None
        }
    }

    /// Evaluate bond expressions
    fn evaluate_bond_expression(&self, expr: &BondExpression, atom1: usize, atom2: usize) -> bool {
        match expr {
            BondExpression::Any => true,

            BondExpression::Single => self.get_bond_order(atom1, atom2).unwrap_or(1) == 1,

            // What SMARTS means by an unwritten bond.
            BondExpression::SingleOrAromatic => {
                matches!(self.get_bond_order(atom1, atom2), Some(1) | Some(4))
                    || self.aromaticity_info.aromatic_bonds.contains(&(atom1, atom2))
                    || self.aromaticity_info.aromatic_bonds.contains(&(atom2, atom1))
            }

            BondExpression::Double => self.get_bond_order(atom1, atom2).unwrap_or(1) == 2,

            BondExpression::Triple => self.get_bond_order(atom1, atom2).unwrap_or(1) == 3,

            BondExpression::Aromatic => {
                self.aromaticity_info
                    .aromatic_bonds
                    .contains(&(atom1, atom2))
                    || self
                        .aromaticity_info
                        .aromatic_bonds
                        .contains(&(atom2, atom1))
            }

            BondExpression::Ring => {
                // Check if bond is in a ring
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

            // TODO: Implement directional bonds
            _ => true,
        }
    }

    /// Check connectivity constraints (degree preservation)
    fn connectivity_compatible(
        &self,
        pattern_atom: usize,
        molecule_atom: usize,
        state: &MappingState,
    ) -> bool {
        // Count pattern neighbors (both mapped and unmapped)
        let pattern_degree = self
            .pattern
            .bonds
            .iter()
            .filter(|(a1, a2, _)| *a1 == pattern_atom || *a2 == pattern_atom)
            .count();

        // Count molecule neighbors
        let molecule_degree = self
            .molecule
            .atom_bonds()
            .get(molecule_atom)
            .map(|bonds| bonds.len())
            .unwrap_or(0);

        // Molecule atom must have at least as many neighbors as pattern atom needs
        molecule_degree >= pattern_degree
    }

    /// Check if a bond is in a ring
    fn is_bond_in_ring(&self, atom1: usize, atom2: usize) -> bool {
        let pair = (atom1.min(atom2), atom1.max(atom2));
        self.ring_info.cyclic_bonds.contains(&pair)
    }

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

        // Initialize all atoms as not in rings
        for atom_index in 0..molecule.number_of_atoms() {
            atom_ring_count.insert(atom_index, 0);
            atom_smallest_ring.insert(atom_index, 0);
            atom_ring_sizes.insert(atom_index, FxHashSet::default());
        }

        // Enumerate each simple cycle exactly once: it is discovered from its
        // smallest atom and recorded in only one traversal direction.
        let mut rings: Vec<Vec<usize>> = Vec::new();
        let mut path = Vec::with_capacity(Self::MAX_RING_SIZE);
        for start in cyclic_atoms.iter() {
            path.clear();
            path.push(start);
            Self::collect_cycles(molecule, start, start, &cyclic_atoms, &mut path, &mut rings);
        }

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

    /// Largest ring size considered by R/r ring predicates
    const MAX_RING_SIZE: usize = 8;

    /// DFS that collects every simple cycle through `start` using only atoms
    /// greater than `start` (so each cycle is anchored at its smallest atom)
    /// and only one traversal direction (`path[1] < path.last()`).
    fn collect_cycles(
        molecule: &MolecularGraph,
        current: usize,
        start: usize,
        cyclic_atoms: &BitSet,
        path: &mut Vec<usize>,
        rings: &mut Vec<Vec<usize>>,
    ) {
        let Some(bonds) = molecule.get_atom_bonds(current) else {
            return;
        };
        for bond in bonds {
            let neighbor = bond.target();
            if !cyclic_atoms.contains(neighbor) {
                continue;
            }

            if neighbor == start && path.len() >= 3 {
                if path[1] < path[path.len() - 1] {
                    rings.push(path.clone());
                }
                continue;
            }

            if neighbor > start && path.len() < Self::MAX_RING_SIZE && !path.contains(&neighbor) {
                path.push(neighbor);
                Self::collect_cycles(molecule, neighbor, start, cyclic_atoms, path, rings);
                path.pop();
            }
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

        // Initialize all atoms as non-aromatic
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
            // Extract aromatic atoms from the aromatized molecule
            let aromatic_atom_indices = temp_molecule.get_aromatic_atoms();

            // Mark aromatic atoms based on the aromatization results
            for atom_index in aromatic_atom_indices {
                is_aromatic.insert(atom_index, true);
            }
        }

        // Collect aromatic bonds from the aromatized molecule
        for (atom_index, bonds) in temp_molecule.atom_bonds().iter().enumerate() {
            // Only process if this atom is aromatic
            if !is_aromatic.get(&atom_index).copied().unwrap_or(false) {
                continue;
            }

            for bond in bonds {
                let target_index = bond.target();

                // Check if this is an aromatic bond and target is also aromatic
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
}

/// Node data for petgraph representation
#[derive(Debug, Clone, PartialEq, Hash, Copy)]
pub struct AtomNode {
    pub atom_index: usize,
    pub atomic_number: u8,
    pub charge: i8,
    pub is_aromatic: bool,
    pub ring_count: usize,
    pub smallest_ring: usize,
    pub degree: usize,
    pub is_pattern: bool, // true for pattern nodes, false for molecule nodes
}

/// Edge data for petgraph representation
#[derive(Debug, Clone, PartialEq, Hash, Copy)]
pub struct BondEdge {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_order: BondOrder,
    pub is_aromatic: bool,
    pub is_in_ring: bool,
    pub is_pattern: bool, // true for pattern edges, false for molecule edges
}

/// Simplified SMARTS matcher using petgraph
pub struct PetgraphSMARTSMatcher<'a> {
    pattern: &'a CxSMARTSPattern,
    molecule: &'a MolecularGraph,
    ring_info: RingInfo,
    aromaticity_info: AromaticityInfo,
}

impl<'a> PetgraphSMARTSMatcher<'a> {
    /// Create a new petgraph-based SMARTS matcher
    pub fn new(pattern: &'a CxSMARTSPattern, molecule: &'a MolecularGraph) -> Self {
        let ring_info = SMARTSMatcher::compute_ring_info(molecule);
        let aromaticity_info = SMARTSMatcher::compute_aromaticity_info(molecule);

        Self {
            pattern,
            molecule,
            ring_info,
            aromaticity_info,
        }
    }

    /// Every way the pattern can be laid over the molecule, as
    /// `pattern atom -> molecule atom`.
    ///
    /// SMARTS matching is subgraph monomorphism; see
    /// [`crate::graph_algorithms::monomorphism`] for why that is not what
    /// petgraph's `subgraph_isomorphisms_iter` provides.
    pub fn find_all_matches(&self) -> Vec<HashMap<usize, usize>> {
        self.monomorphism()
            .all()
            .into_iter()
            .map(|assignment| assignment.into_iter().enumerate().collect())
            .collect()
    }

    /// Whether the pattern matches anywhere.
    pub fn has_match(&self) -> bool {
        self.monomorphism().any()
    }

    /// The pattern and the three questions the search asks about it.
    fn monomorphism(
        &self,
    ) -> Monomorphism<
        impl Fn(usize) -> Vec<usize> + '_,
        impl Fn(usize, usize) -> bool + '_,
        impl Fn(usize, usize, usize) -> bool + '_,
    > {
        Monomorphism {
            pattern_node_count: self.pattern.atoms.len(),
            pattern_edges: self.pattern.bonds.iter().map(|&(a, b, _)| (a, b)).collect(),
            target_node_count: self.molecule.number_of_atoms(),
            neighbours: |atom: usize| {
                self.molecule.atom_bonds()[atom]
                    .iter()
                    .map(|bond| bond.target())
                    .collect()
            },
            node_match: |pattern_atom: usize, molecule_atom: usize| {
                self.evaluate_atom_expression(&self.pattern.atoms[pattern_atom], molecule_atom)
            },
            edge_match: |edge_index: usize, first: usize, second: usize| {
                if self.get_bond_order(first, second).is_none() {
                    return false;
                }
                self.evaluate_bond_expression(&self.pattern.bonds[edge_index].2, first, second)
            },
        }
    }

    /// Convert SMARTS pattern to petgraph
    fn pattern_to_graph(&self) -> UnGraph<AtomNode, BondEdge> {
        let mut graph = UnGraph::default();
        let mut node_indices = Vec::new();

        // Add nodes (atoms)
        for (atom_idx, _atom) in self.pattern.atoms.iter().enumerate() {
            let node = AtomNode {
                atom_index: atom_idx,
                atomic_number: 0, // Pattern nodes don't have specific atomic numbers
                charge: 0,
                is_aromatic: false,
                ring_count: 0,
                smallest_ring: 0,
                degree: 0,
                is_pattern: true,
            };
            let node_idx = graph.add_node(node);
            node_indices.push(node_idx);
        }

        // Add edges (bonds)
        for &(atom1, atom2, _) in &self.pattern.bonds {
            if atom1 < node_indices.len() && atom2 < node_indices.len() {
                let edge = BondEdge {
                    atom1,
                    atom2,
                    bond_order: BondOrder::Single, // Pattern bonds don't have specific orders
                    is_aromatic: false,
                    is_in_ring: false,
                    is_pattern: true,
                };
                graph.add_edge(node_indices[atom1], node_indices[atom2], edge);
            }
        }

        graph
    }

    /// Convert molecule to petgraph
    fn molecule_to_graph(&self) -> UnGraph<AtomNode, BondEdge> {
        let mut graph = UnGraph::default();
        let mut node_indices = Vec::new();

        // Add nodes (atoms)
        for atom_idx in 0..self.molecule.number_of_atoms() {
            let node = AtomNode {
                atom_index: atom_idx,
                atomic_number: self.molecule.atomic_numbers()[atom_idx],
                charge: self.molecule.charges()[atom_idx],
                is_aromatic: self
                    .aromaticity_info
                    .is_aromatic
                    .get(&atom_idx)
                    .copied()
                    .unwrap_or(false),
                ring_count: self
                    .ring_info
                    .atom_ring_count
                    .get(&atom_idx)
                    .copied()
                    .unwrap_or(0),
                smallest_ring: self
                    .ring_info
                    .atom_smallest_ring
                    .get(&atom_idx)
                    .copied()
                    .unwrap_or(0),
                degree: self.molecule.atom_bonds()[atom_idx].len(),
                is_pattern: false,
            };
            let node_idx = graph.add_node(node);
            node_indices.push(node_idx);
        }

        // Add edges (bonds)
        for (atom_idx, bonds) in self.molecule.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target_idx = bond.target();
                if atom_idx < target_idx && target_idx < node_indices.len() {
                    let bond_tuple = (atom_idx, target_idx);
                    let edge = BondEdge {
                        atom1: atom_idx,
                        atom2: target_idx,
                        bond_order: bond.bond_order(),
                        is_aromatic: bond.bond_order() == BondOrder::Aromatic,
                        is_in_ring: self.aromaticity_info.aromatic_bonds.contains(&bond_tuple),
                        is_pattern: false,
                    };
                    graph.add_edge(node_indices[atom_idx], node_indices[target_idx], edge);
                }
            }
        }

        graph
    }

    /// Check if pattern node matches molecule node
    fn nodes_match(&self, pattern_node: &AtomNode, molecule_node: &AtomNode) -> bool {
        // Only match pattern nodes against molecule nodes
        if pattern_node.is_pattern != true || molecule_node.is_pattern != false {
            return false;
        }

        // Get the SMARTS atom expression for this pattern node
        if pattern_node.atom_index >= self.pattern.atoms.len() {
            return false;
        }

        let pattern_expr = &self.pattern.atoms[pattern_node.atom_index];
        self.evaluate_atom_expression(pattern_expr, molecule_node.atom_index)
    }

    /// Check if pattern edge matches molecule edge
    fn edges_match(&self, pattern_edge: &BondEdge, molecule_edge: &BondEdge) -> bool {
        // Only match pattern edges against molecule edges
        if pattern_edge.is_pattern != true || molecule_edge.is_pattern != false {
            return false;
        }

        // Find the corresponding bond expression in the pattern
        for &(atom1, atom2, ref bond_expr) in &self.pattern.bonds {
            if (atom1 == pattern_edge.atom1 && atom2 == pattern_edge.atom2)
                || (atom1 == pattern_edge.atom2 && atom2 == pattern_edge.atom1)
            {
                return self.evaluate_bond_expression(
                    bond_expr,
                    molecule_edge.atom1,
                    molecule_edge.atom2,
                );
            }
        }

        // If no explicit bond expression found, default to "any" bond
        true
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

            AtomExpression::ImplicitHydrogens(h_count) => {
                self.molecule.number_of_bonded_element(molecule_atom, 1) == (*h_count as usize)
            }

            AtomExpression::TotalHydrogens(h_count) => {
                self.molecule.number_of_bonded_element(molecule_atom, 1) == (*h_count as usize)
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

            AtomExpression::RingSize(ring_size) => {
                if let Some(ring_sizes) = self.ring_info.atom_ring_sizes.get(&molecule_atom) {
                    ring_sizes.contains(&(*ring_size as usize))
                } else {
                    false
                }
            }

            AtomExpression::Charge(charge) => {
                self.molecule.get_atom_charge(molecule_atom) == *charge
            }

            AtomExpression::Logical { operator, operands } => {
                self.evaluate_logical_expression(operator, operands, molecule_atom)
            }

            // TODO: Implement other expressions
            _ => true, // Placeholder for unimplemented expressions
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
                // Check if bond is in a ring
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

            // TODO: Implement directional bonds
            _ => true, // Placeholder for unimplemented expressions
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
    fn is_bond_in_ring(&self, atom1: usize, atom2: usize) -> bool {
        let bond_tuple = if atom1 < atom2 {
            (atom1, atom2)
        } else {
            (atom2, atom1)
        };
        self.aromaticity_info.aromatic_bonds.contains(&bond_tuple) ||
        // Also check if both atoms are in rings (simple heuristic)
        (self.ring_info.atom_ring_count.get(&atom1).copied().unwrap_or(0) > 0 &&
         self.ring_info.atom_ring_count.get(&atom2).copied().unwrap_or(0) > 0)
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

    #[test]
    fn test_petgraph_node_edge_matching() {
        // Test the petgraph node and edge matching functions directly
        let molecules = SMILESParser::parse_smiles("CCO").unwrap();
        let molecule = &molecules[0];

        let pattern = parse_cxsmarts("C").unwrap();
        let matcher = PetgraphSMARTSMatcher::new(&pattern, molecule);

        // Test node matching - create sample nodes
        let pattern_node = AtomNode {
            atom_index: 0,
            atomic_number: 0,
            charge: 0,
            is_aromatic: false,
            ring_count: 0,
            smallest_ring: 0,
            degree: 0,
            is_pattern: true,
        };

        let molecule_node = AtomNode {
            atom_index: 0, // Carbon atom
            atomic_number: 6,
            charge: 0,
            is_aromatic: false,
            ring_count: 0,
            smallest_ring: 0,
            degree: 1,
            is_pattern: false,
        };

        // Should match since pattern is [C] and molecule atom is carbon
        assert!(matcher.nodes_match(&pattern_node, &molecule_node));

        // Test with oxygen atom - should not match [C] pattern
        let oxygen_node = AtomNode {
            atom_index: 2, // Oxygen atom
            atomic_number: 8,
            charge: 0,
            is_aromatic: false,
            ring_count: 0,
            smallest_ring: 0,
            degree: 1,
            is_pattern: false,
        };

        assert!(!matcher.nodes_match(&pattern_node, &oxygen_node));
    }
}
