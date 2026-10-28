use crate::graph_algorithms::monomorphism::Monomorphism;
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::chirality::ChiralType;
use nohash_hasher::IntMap;
use petgraph::graph::{NodeIndex, UnGraph};
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct AtomInvariant {
    atomic_number: u8,
    charge: i8,
    aromaticity: bool,
    isotope: u16,
    chirality: ChiralType,
    implicit_hydrogen_count: u8,
    heavy_atom_bond_count: u8,
    /// The isotopes of the atom's hydrogens, descending, `0` where there are
    /// fewer than four of them.
    ///
    /// Hydrogens are not vertices of the canonization graph, so a deuterium is
    /// invisible to it and `[2H]CCCCC` and `CCCCC[2H]` -- the same molecule --
    /// canonicalized to different strings, the label staying wherever the input
    /// happened to put it. Counting the hydrogens is not enough to tell the two
    /// ends apart; their isotopes are what differs. Sorted, so the answer does
    /// not depend on the order the bonds are stored in.
    hydrogen_isotopes: [u16; 4],
    /// A stereocentre's tag, read against an ordering the graph itself fixes:
    /// its neighbours sorted by the orbit they fall in under the molecule's own
    /// automorphisms. `0` on an atom that is not a stereocentre, and on one
    /// whose neighbours the orbits cannot tell apart.
    ///
    /// A canonical labelling is unique only up to those automorphisms, and a
    /// tag is not part of the graph nauty sees -- so when a molecule carries
    /// several copies of the same arm, which copy lands at which canonical
    /// position is undetermined, and the tags come out in whatever order the
    /// input happened to hold. Reading each tag against an orbit ordering gives
    /// a number that is the same for every way of writing the molecule, so
    /// colouring by it tells the arms apart and removes the choice. Filled in
    /// by a second canonization pass; zero on the first, where no orbits are
    /// known yet.
    stereo_parity: u8,
    /// The atom's unpaired electrons.
    ///
    /// A radical is a property of the atom that the writer emits and the graph
    /// otherwise cannot see: a carbene `[CH]` in a ring and an ordinary ring
    /// `CH` have the same element, degree, charge and hydrogen count, so
    /// nothing else here tells them apart. Leaving it out let a molecule
    /// carrying one claim symmetries it does not have -- an imidazolylidene
    /// cage answered to its own imidazolium ring -- and which position the
    /// electron came out on was then decided by the input's atom order rather
    /// than by the molecule.
    radical_state: u8,
}

impl AtomInvariant {
    /// Records the orbit-relative tag of a stereocentre for a further pass.
    pub(crate) fn set_stereo_parity(&mut self, parity: u8) {
        self.stereo_parity = parity;
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        atomic_number: u8,
        charge: i8,
        aromaticity: bool,
        isotope: u16,
        chirality: ChiralType,
        implicit_hydrogen_count: u8,
        heavy_atom_bond_count: u8,
        hydrogen_isotopes: [u16; 4],
        radical_state: u8,
    ) -> Self {
        AtomInvariant {
            stereo_parity: 0,
            atomic_number,
            charge,
            aromaticity,
            isotope,
            chirality,
            implicit_hydrogen_count,
            heavy_atom_bond_count,
            hydrogen_isotopes,
            radical_state,
        }
    }
}

#[derive(Default, Debug)]
pub struct AdjacencyMatrix {
    pub matrix: Vec<usize>,
    pub words_per_vertex: usize,
    pub number_of_vertices: usize,
}

// impl std::fmt::Display for AdjacencyMatrix {

// }

impl AdjacencyMatrix {
    pub fn set(&mut self, source_index: usize, target_index: usize) {
        let bits_per_word = usize::BITS as usize;
        let word_index = target_index / bits_per_word + source_index * self.words_per_vertex;
        let bit_index = target_index % bits_per_word;
        self.matrix[word_index] |= 1 << (bits_per_word - 1 - bit_index);
    }
    pub fn print(&self) {
        for i in 0..self.number_of_vertices {
            for j in 0..self.words_per_vertex {
                let word = self.matrix[i * self.words_per_vertex + j];
                for i in 0..self.number_of_vertices {
                    let bit_index = 1 << (64 - 1 - i);
                    let bit = (word & bit_index) != 0;
                    print!("{:} ", bit as u8);
                }
            }
            println!();
        }
    }
}

impl AdjacencyMatrix {
    pub fn new(
        matrix: Vec<usize>,
        words_per_vertex: usize,
        number_of_vertices: usize,
    ) -> AdjacencyMatrix {
        AdjacencyMatrix {
            matrix,
            words_per_vertex,
            number_of_vertices,
        }
    }
}

/// A wrapper around BondOrder that implements Hash trait
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BondOrderWrapper(BondOrder);

impl BondOrderWrapper {
    pub fn new(bond_order: BondOrder) -> Self {
        Self(bond_order)
    }

    pub fn inner(&self) -> BondOrder {
        self.0
    }
}

impl Hash for BondOrderWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use the to_u8 method for hashing
        self.0.to_u8().hash(state);
    }
}

impl From<BondOrder> for BondOrderWrapper {
    fn from(bond_order: BondOrder) -> Self {
        Self(bond_order)
    }
}

/// Enum representing different graph types with varying node label sizes
#[derive(Debug)]
pub enum CompactGraph {
    /// Graph with u8 node labels (up to 256 unique labels)
    U8(UnGraph<u8, ()>, FxHashMap<u8, AtomLabelImplicitHydrogens>),
    /// Graph with u16 node labels (up to 65,536 unique labels)
    U16(UnGraph<u16, ()>, FxHashMap<u16, AtomLabelImplicitHydrogens>),
    /// Graph with u32 node labels (for very large molecules)
    U32(UnGraph<u32, ()>, FxHashMap<u32, AtomLabelImplicitHydrogens>),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq)]
pub struct AtomLabelImplicitHydrogensNoChirality {
    atomic_number: u8,
    charge: i8,
    aromatic: bool,
    isotope: Option<u16>,
    implicit_hydrogens: u8,
}

impl AtomLabelImplicitHydrogensNoChirality {
    pub fn new(
        atomic_number: u8,
        charge: i8,
        aromatic: bool,
        isotope: Option<u16>,
        implicit_hydrogens: u8,
    ) -> Self {
        Self {
            atomic_number,
            charge,
            aromatic,
            isotope,
            implicit_hydrogens,
        }
    }
}

impl std::cmp::Eq for AtomLabelImplicitHydrogensNoChirality {}

impl std::cmp::PartialOrd for AtomLabelImplicitHydrogensNoChirality {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for AtomLabelImplicitHydrogensNoChirality {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.atomic_number,
            self.charge,
            self.aromatic,
            self.isotope,
            self.implicit_hydrogens,
        )
            .cmp(&(
                other.atomic_number,
                other.charge,
                other.aromatic,
                other.isotope,
                other.implicit_hydrogens,
            ))
    }
}

#[derive(Clone, Copy, Hash, PartialEq)]
pub struct AtomLabelImplicitHydrogens {
    atomic_number: u8,
    charge: i8,
    aromatic: bool,
    chirality: ChiralType,
    implicit_hydrogens: u8,
}

impl std::fmt::Display for AtomLabelImplicitHydrogens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}{:?}{:?}{:?}{:?}",
            self.atomic_number, self.charge, self.aromatic, self.chirality, self.implicit_hydrogens
        )
    }
}

impl AtomLabelImplicitHydrogens {
    pub fn new(
        atomic_number: u8,
        charge: i8,
        aromatic: bool,
        chirality: ChiralType,
        implicit_hydrogens: u8,
    ) -> Self {
        Self {
            atomic_number,
            charge,
            aromatic,
            chirality,
            implicit_hydrogens,
        }
    }
}

impl std::cmp::Eq for AtomLabelImplicitHydrogens {}

impl std::cmp::PartialOrd for AtomLabelImplicitHydrogens {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.cmp(self))
    }
}

impl std::cmp::Ord for AtomLabelImplicitHydrogens {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.atomic_number,
            self.charge,
            self.aromatic,
            self.chirality,
            self.implicit_hydrogens,
        )
            .cmp(&(
                other.atomic_number,
                other.charge,
                other.aromatic,
                other.chirality,
                other.implicit_hydrogens,
            ))
    }
}

impl std::fmt::Display for AtomLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}C{:?}A{:?}C{:?}I{:?}",
            self.atomic_number, self.charge, self.aromatic, self.chiral_type, self.isotope
        )
    }
}

impl std::fmt::Debug for AtomLabelImplicitHydrogens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();
        result.push_str(&format!(
            "{:?}C{:?}A{:?}",
            self.atomic_number, self.charge, self.aromatic
        ));
        match self.chirality {
            ChiralType::None => (),
            _ => result.push_str(&format!("{:?}", self.chirality)),
        };
        match self.implicit_hydrogens {
            0 => (),
            _ => result.push_str(&format!("{:?}", self.implicit_hydrogens)),
        };
        write!(f, "{}", result)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq)]
pub struct AtomLabel {
    atomic_number: u8, // This is also used for the bonds orders which are stored as u8, so we can use the same type for both
    charge: i8,
    aromatic: bool,
    isotope: Option<u16>,
    chiral_type: ChiralType,
}

impl AtomLabel {
    pub fn new(
        atomic_number: u8,
        charge: i8,
        aromatic: bool,
        isotope: Option<u16>,
        chiral_type: ChiralType,
    ) -> Self {
        Self {
            atomic_number,
            charge,
            aromatic,
            isotope,
            chiral_type,
        }
    }
}

impl std::cmp::Eq for AtomLabel {}

impl std::cmp::PartialOrd for AtomLabel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for AtomLabel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.atomic_number,
            self.charge,
            self.aromatic,
            self.chiral_type,
            self.isotope,
        )
            .cmp(&(
                other.atomic_number,
                other.charge,
                self.aromatic,
                other.chiral_type,
                other.isotope,
            ))
    }
}
pub trait MoleculeGraph: MoleculeCore {
    /// Finds isomorphic mappings between this molecule and a query submolecule.
    ///
    /// Returns a list of maps from target atom index to pattern atom index, or `None` if no match exists.
    fn match_submolecule(&self, other: &Self) -> Option<Vec<IntMap<usize, usize>>> {
        let mut pattern_edges: Vec<(usize, usize)> = Vec::new();
        let mut pattern_bond_orders: Vec<BondOrder> = Vec::new();
        for atom in 0..other.len() {
            for bond in other.atom_bonds()[atom].iter() {
                if atom < bond.target() {
                    pattern_edges.push((atom, bond.target()));
                    pattern_bond_orders.push(bond.bond_order());
                }
            }
        }

        // Labels decide what may stand for what: element, charge, aromaticity,
        // isotope and whether the atom is a stereocentre, plus exact bond
        // order. This is a structure query, not a query language — see
        // `crate::io::find_cxsmarts_matches` for the latter.
        let search = Monomorphism {
            pattern_node_count: other.len(),
            pattern_edges,
            target_node_count: self.len(),
            neighbours: |atom: usize, out: &mut Vec<usize>| {
                out.extend(self.atom_bonds()[atom].iter().map(|bond| bond.target()))
            },
            node_match: |pattern_atom: usize, target_atom: usize| {
                other.get_atom_label(pattern_atom) == self.get_atom_label(target_atom)
            },
            edge_match: |edge_index: usize, first: usize, second: usize| {
                self.atom_bonds()[first]
                    .iter()
                    .find(|bond| bond.target() == second)
                    .map(|bond| bond.bond_order())
                    == Some(pattern_bond_orders[edge_index])
            },
        };

        let matches: Vec<IntMap<usize, usize>> = search
            .all()
            .into_iter()
            .map(|assignment| {
                assignment
                    .into_iter()
                    .enumerate()
                    .map(|(pattern_atom, target_atom)| (target_atom, pattern_atom))
                    .collect()
            })
            .collect();

        if matches.is_empty() {
            None
        } else {
            Some(matches)
        }
    }

    /// Returns the adjacency matrix of the molecule graph.
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::graph::MoleculeGraph;
    ///
    /// let mol = MolecularGraph::from_smiles_single("C1=CC=CC=C1").unwrap();
    /// let adjacency_matrix = mol.adjacency_matrix();
    /// assert_eq!(adjacency_matrix.number_of_vertices, mol.len());
    /// ```
    fn adjacency_matrix(&self) -> AdjacencyMatrix {
        let number_of_atoms = self.len();
        let bits_per_word = usize::BITS as usize;
        let words_per_atom = (number_of_atoms + bits_per_word - 1) / bits_per_word;
        let mut matrix: Vec<usize> = vec![0; words_per_atom * number_of_atoms];
        for index in 0..number_of_atoms {
            let Some(bonds) = self.get_atom_bonds(index) else {
                continue;
            };
            for bond in bonds {
                let other_index = bond.target;
                let word_index = other_index / bits_per_word + index * words_per_atom;
                let bit_index = other_index % bits_per_word;
                matrix[word_index] |= 1 << (bits_per_word - 1 - bit_index);
            }
        }
        AdjacencyMatrix::new(matrix, words_per_atom, number_of_atoms)
    }

    fn prepare_canonization(
        &self,
        labels: &[AtomInvariant],
    ) -> Option<(AdjacencyMatrix, Vec<u32>, usize)> {
        // Everything but hydrogen is a vertex, the `*` wildcard (atomic
        // number zero) included: `get_all_heavy_atom_bonds` gives it bonds,
        // and a bond to an atom with no vertex is an index past the matrix.
        let number_of_heavy_atoms = self
            .atomic_numbers()
            .iter()
            .filter(|&&atomic_number| atomic_number != 1)
            .count();

        if number_of_heavy_atoms == 0 {
            return None;
        }
        // A directional bond is a single bond that has been told which way to
        // lean, and the lean is a property of how the SMILES was written rather
        // than of the molecule: the same double bond's substituent reads `/` in
        // one traversal and `\` in the reverse one. Colouring the graph by it
        // put that choice into the canonical labelling, so the same compound
        // written two ways canonicalised two ways -- 341 of PubChem's first
        // key-drift failures. The geometry itself is carried by the cis/trans
        // chiral classes, which are canonicalised under the automorphism group
        // further down, so nothing is lost by reading these as what they are.
        let ignore_direction = |order: BondOrder| match order {
            BondOrder::Up | BondOrder::Down => BondOrder::Single,
            other => other,
        };
        let mut bonds = self.get_all_heavy_atom_bonds();
        bonds.sort_by(|a, b| ignore_direction(b.bond_order).cmp(&ignore_direction(a.bond_order)));
        let mut number_of_non_single_bonds = 0;
        for bond in &bonds {
            if ignore_direction(bond.bond_order) != BondOrder::Single {
                number_of_non_single_bonds += 1;
            }
        }
        let number_of_vertices = number_of_heavy_atoms + number_of_non_single_bonds;
        let bits_per_word = usize::BITS as usize;
        let words_per_vertex = (number_of_vertices + bits_per_word - 1) / bits_per_word;

        // nauty colouring (`ptn` format): a cell boundary (0) is placed after the
        // last vertex of each colour class; all other entries are 1. Heavy atoms
        // (already ordered by descending invariant) are coloured by invariant;
        // the virtual bond-order vertices are coloured by bond order.
        let mut partition: Vec<u32> = vec![1; number_of_vertices];

        for i in 1..number_of_heavy_atoms {
            if labels[i] != labels[i - 1] {
                partition[i - 1] = 0;
            }
        }
        // Separate the heavy-atom colours from the bond-order vertices.
        partition[number_of_heavy_atoms - 1] = 0;

        let matrix: Vec<usize> = vec![0; words_per_vertex * number_of_vertices];
        let mut matrix = AdjacencyMatrix::new(matrix, words_per_vertex, number_of_vertices);

        // Bonds are sorted by descending order, so all non-single bonds (which get
        // a virtual vertex) come first. Assign one bond vertex per non-single bond
        // and mark a colour boundary wherever the bond order changes.
        let mut bond_vertex = number_of_heavy_atoms;
        let mut last_order: Option<BondOrder> = None;
        for bond in bonds.iter() {
            let source_index = bond.atom1;
            let target_index = bond.atom2;

            let bond_order = ignore_direction(bond.bond_order);
            if bond_order == BondOrder::Single {
                matrix.set(source_index, target_index);
                matrix.set(target_index, source_index);
            } else {
                matrix.set(source_index, bond_vertex);
                matrix.set(bond_vertex, source_index);
                matrix.set(target_index, bond_vertex);
                matrix.set(bond_vertex, target_index);
                if let Some(order) = last_order {
                    if order != bond_order {
                        partition[bond_vertex - 1] = 0;
                    }
                }
                last_order = Some(bond_order);
                bond_vertex += 1;
            }
        }

        partition[number_of_vertices - 1] = 0;

        Some((matrix, partition, number_of_heavy_atoms))
    }

    /// Returns the connected components of the molecule as lists of connected atom indices.
    ///
    /// Separate components represent disconnected molecular fragments (e.g. salts or solvent adducts).
    fn get_components(&self) -> Vec<Vec<usize>> {
        let mut connected_components = vec![];
        let mut visited_atoms = vec![false; self.atomic_numbers().len()];
        let mut stack = Vec::with_capacity(self.atomic_numbers().len());
        while let Some(index) = visited_atoms.iter().position(|&a| !a) {
            let mut component = self.traverse_component(&mut stack, index, &mut visited_atoms);
            component.sort();
            connected_components.push(component);
            stack.clear();
        }
        connected_components
    }

    fn traverse_component(
        &self,
        stack: &mut Vec<usize>,
        start: usize,
        visited_atoms: &mut [bool],
    ) -> Vec<usize> {
        let mut current_component = Vec::new();
        stack.push(start);
        visited_atoms[start] = true;

        while let Some(index) = stack.pop() {
            current_component.push(index);
            let Some(bonds) = self.get_atom_bonds(index) else {
                debug_assert!(false, "no bonds for atom {index}: out of bounds access");
                continue;
            };
            for &bond in bonds {
                let target = bond.target();
                if !visited_atoms[target] {
                    stack.push(target);
                    visited_atoms[target] = true;
                }
            }
        }

        current_component
    }

    /// Returns a standardized label for an atom containing all relevant atomic information
    /// in a defined order: atomic_symbol, charge, aromaticity, isotope, chirality
    ///
    /// # Arguments
    /// * 'self' - The molecule containing the atom
    /// * `atom_index` - The index of the atom to get the label for
    ///
    /// # Returns
    /// A `AtomLabel` containing all atom information in a standardized format
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::chirality::ChiralType;
    /// use oxidane::molecule::graph::AtomLabel;
    /// let mol = MolecularGraph::from_smiles("[NH3+]").unwrap();
    /// let mol = mol.first().unwrap();
    /// // Nitrogen, charge +1; with no isotope in the SMILES the label falls
    /// // back to the most abundant isotope (14 for N)
    /// assert_eq!(mol.get_atom_label(0), AtomLabel::new(7, 1, false, Some(14), ChiralType::None));
    /// ```
    fn get_atom_label(&self, atom_index: usize) -> AtomLabel {
        let atomic_number = self.atomic_numbers()[atom_index];
        // 2. Charge (c0 for neutral, c1 for +1, c-1 for -1, etc.)
        let charge = self.get_atom_charge(atom_index);
        let isotope = self.get_atom_isotope(atom_index);
        let is_aromatic = self.is_atom_aromatic(atom_index);
        let chiral_type = self.get_chiral_type(atom_index);

        AtomLabel::new(atomic_number, charge, is_aromatic, isotope, chiral_type)
    }

    /// Returns a standardized label for an atom containing all relevant atomic information
    /// in a defined order: atomic_symbol, charge, aromaticity, isotope, chirality, implicit_hydrogen_count
    ///
    /// # Arguments
    /// * `self` - The molecule containing the atom
    /// * `atom_index` - The index of the atom to get the label for
    ///
    /// # Returns
    /// A `AtomLabelImplicitHydrogens` containing all atom information in a standardized format
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::graph::AtomLabelImplicitHydrogens;
    /// use oxidane::molecule::chirality::ChiralType;
    /// let mol = MolecularGraph::from_smiles("[NH3+]").unwrap();
    /// let mol = mol.first().unwrap();
    /// assert_eq!(mol.get_atom_label_with_hydrogens(0), AtomLabelImplicitHydrogens::new(7,1,false,ChiralType::None,3)); // Nitrogen, charge +1
    /// ```
    fn get_atom_label_with_hydrogens(&self, atom_index: usize) -> AtomLabelImplicitHydrogens {
        let atomic_number = self.atomic_numbers()[atom_index];
        let charge = self.get_atom_charge(atom_index);
        let is_aromatic = self.is_atom_aromatic(atom_index);
        let implicit_hydrogen_count = self.number_of_bonded_element(atom_index, 1);
        let chiral_type = self.get_chiral_type(atom_index);

        AtomLabelImplicitHydrogens::new(
            atomic_number,
            charge,
            is_aromatic,
            chiral_type,
            implicit_hydrogen_count as u8,
        )
    }

    /// Returns a standardized label for an atom containing all relevant atomic information
    /// in a defined order: atomic_symbol, charge, aromaticity, isotope, chirality, implicit_hydrogen_count
    ///
    /// # Arguments
    /// * `self` - The molecule containing the atom
    /// * `atom_index` - The index of the atom to get the label for
    ///
    /// # Returns
    /// A `AtomLabelImplicitHydrogens` containing all atom information in a standardized format
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::graph::AtomLabelImplicitHydrogens;
    /// use oxidane::molecule::chirality::ChiralType;
    /// let mol = MolecularGraph::from_smiles("[NH3+]").unwrap();
    /// let mol = mol.first().unwrap();
    /// assert_eq!(mol.get_atom_label_with_hydrogens(0), AtomLabelImplicitHydrogens::new(7,1,false,ChiralType::None,3)); // Nitrogen, charge +1
    /// ```
    fn get_atom_invariant(&self, atom_index: usize) -> AtomInvariant {
        let atomic_number = self.atomic_numbers()[atom_index];
        let charge = self.get_atom_charge(atom_index);
        let is_aromatic = self.is_atom_aromatic(atom_index);
        let chiral_type = self.get_chiral_type(atom_index);
        // The isotope as *written*, not the element's most abundant one. They
        // differ only where the input said `[12C]`, which the writer keeps and
        // the canonizer could not see: `C1[12CH2]C=CC=C1` and
        // `[12CH2]1CC=CC=C1` are the same molecule, and defaulting the label to
        // 12 for every carbon left nothing to tell the two ring positions
        // apart, so the bracket stayed wherever the input put it.
        let isotope = self.get_isotope(atom_index).unwrap_or(0);
        let radical_state = self.radical_states().get(atom_index).copied().unwrap_or(0);
        let Some(bonds) = self.get_atom_bonds(atom_index) else {
            return AtomInvariant::new(
                atomic_number,
                charge,
                is_aromatic,
                isotope,
                chiral_type,
                0,
                0,
                [0; 4],
                radical_state,
            );
        };
        let heavy_atom_bond_count = bonds
            .iter()
            .filter(|bond| self.atomic_numbers()[bond.target] != 1)
            .count();
        let implicit_hydrogen_count = bonds.len() - heavy_atom_bond_count;
        let mut hydrogen_isotopes = [0u16; 4];
        let mut written = 0;
        for bond in bonds.iter() {
            if self.atomic_numbers()[bond.target] != 1 || written == hydrogen_isotopes.len() {
                continue;
            }
            hydrogen_isotopes[written] = self.get_isotope(bond.target).unwrap_or(0);
            written += 1;
        }
        hydrogen_isotopes.sort_unstable_by(|first, second| second.cmp(first));

        AtomInvariant::new(
            atomic_number,
            charge,
            is_aromatic,
            isotope,
            chiral_type,
            implicit_hydrogen_count as u8,
            heavy_atom_bond_count as u8,
            hydrogen_isotopes,
            radical_state,
        )
    }
    fn to_ungraph_with_edge_nodes(&self) -> UnGraph<AtomLabel, ()> {
        let mut graph = UnGraph::<AtomLabel, ()>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut bond_indices = FxHashMap::<(usize, usize), NodeIndex<u32>>::default();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            let label = self.get_atom_label(index);
            let node_index = graph.add_node(label);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();
                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];
                    if bond_order == BondOrder::Single || bond_order == BondOrder::Aromatic {
                        graph.add_edge(source_atom, target_atom, ());
                        continue;
                    }
                    // Only process each bond once
                    let bond_label =
                        AtomLabel::new(bond.bond_order().to_u8(), 0, false, None, ChiralType::None);
                    let bond_node = graph.add_node(bond_label);
                    bond_indices.insert((index, target), bond_node);

                    // Connect atoms to bond
                    graph.add_edge(source_atom, bond_node, ());
                    graph.add_edge(target_atom, bond_node, ());
                }
            }
        }

        graph
    }

    fn to_ungraph(&self) -> UnGraph<AtomLabel, BondOrder> {
        let mut graph = UnGraph::<AtomLabel, BondOrder>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            let label = self.get_atom_label(index);
            let node_index = graph.add_node(label);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();
                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];
                    graph.add_edge(source_atom, target_atom, bond_order);
                }
            }
        }

        graph
    }

    fn to_ungraph_with_u8_edges(&self) -> UnGraph<AtomLabelImplicitHydrogens, u8> {
        let mut graph = UnGraph::<AtomLabelImplicitHydrogens, u8>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            let label = self.get_atom_label_with_hydrogens(index);
            let node_index = graph.add_node(label);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();
                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];
                    if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                        continue;
                    }
                    graph.add_edge(
                        source_atom,
                        target_atom,
                        (bond_order.to_u8() - BondOrder::Single.to_u8()) + 1,
                    );
                }
            }
        }

        graph
    }

    fn to_ungraph_with_edge_nodes_and_implicit_hydrogens(
        &self,
    ) -> UnGraph<AtomLabelImplicitHydrogens, ()> {
        let mut graph = UnGraph::<AtomLabelImplicitHydrogens, ()>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut bond_indices = FxHashMap::<(usize, usize), NodeIndex<u32>>::default();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            let node_index = graph.add_node(label);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();

                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }

                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];
                    if bond_order == BondOrder::Single || bond_order == BondOrder::Aromatic {
                        graph.add_edge(source_atom, target_atom, ());
                        continue;
                    }
                    // Only process each bond once
                    let bond_label = AtomLabelImplicitHydrogens::new(
                        bond.bond_order().to_u8(),
                        0,
                        false,
                        ChiralType::None,
                        0,
                    );
                    let bond_node = graph.add_node(bond_label);
                    bond_indices.insert((index, target), bond_node);

                    // Connect atoms to bond
                    graph.add_edge(source_atom, bond_node, ());
                    graph.add_edge(target_atom, bond_node, ());
                }
            }
        }

        graph
    }

    /// Converts the molecule to an undirected graph with the most compact representation possible
    /// based on the number of unique atom and bond types in the molecule.
    ///
    /// Analyzes the molecule to determine the minimum number of bits needed to
    /// represent all unique atom and bond labels, then creates a graph using the smallest
    /// possible integer type for node labels.
    ///
    /// # Returns
    /// An undirected graph where:
    /// - Nodes are labeled with compact integer IDs that map to the original atom/bond properties
    /// - A mapping from the compact IDs back to the original atom/bond properties is also returned
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecules = MolecularGraph::from_smiles("c1ccccc1").unwrap();
    /// let molecule = molecules.first().unwrap();
    /// let (graph, label_map) = molecule.to_ungraph_with_edge_nodes_and_implicit_hydrogens_compact();
    /// // Since benzene has only one type of atom (aromatic carbon), the label map will have just one entry
    /// assert_eq!(label_map.len(), 1);
    /// ```
    fn to_ungraph_with_edge_nodes_and_implicit_hydrogens_compact(
        &self,
    ) -> (UnGraph<u16, ()>, FxHashMap<u16, AtomLabelImplicitHydrogens>) {
        // First, collect all unique atom and bond labels
        let mut unique_labels = FxHashMap::<AtomLabelImplicitHydrogens, u16>::default();
        let mut next_id: u16 = 0;

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            if !unique_labels.contains_key(&label) {
                unique_labels.insert(label, next_id);
                next_id += 1;
            }
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();
                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }
                if index < target {
                    let bond_order = bond.bond_order();
                    if bond_order != BondOrder::Single && bond_order != BondOrder::Aromatic {
                        let bond_label = AtomLabelImplicitHydrogens::new(
                            bond.bond_order().to_u8(),
                            0,
                            false,
                            ChiralType::None,
                            0,
                        );
                        if !unique_labels.contains_key(&bond_label) {
                            unique_labels.insert(bond_label, next_id);
                            next_id += 1;
                        }
                    }
                }
            }
        }

        let label_map: FxHashMap<u16, AtomLabelImplicitHydrogens> =
            unique_labels.iter().map(|(k, v)| (*v, *k)).collect();

        let mut graph = UnGraph::<u16, ()>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut bond_indices = FxHashMap::<(usize, usize), NodeIndex<u32>>::default();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            let compact_id = unique_labels[&label];
            let node_index = graph.add_node(compact_id);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();

                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }

                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];

                    if bond_order == BondOrder::Single || bond_order == BondOrder::Aromatic {
                        graph.add_edge(source_atom, target_atom, ());
                        continue;
                    }

                    // Only process each bond once
                    let bond_label = AtomLabelImplicitHydrogens::new(
                        bond.bond_order().to_u8(),
                        0,
                        false,
                        ChiralType::None,
                        0,
                    );
                    let compact_id = unique_labels[&bond_label];
                    let bond_node = graph.add_node(compact_id);
                    bond_indices.insert((index, target), bond_node);

                    // Connect atoms to bond
                    graph.add_edge(source_atom, bond_node, ());
                    graph.add_edge(target_atom, bond_node, ());
                }
            }
        }

        (graph, label_map)
    }

    /// Converts the molecule to an undirected graph with the most compact representation possible
    /// based on the number of unique atom and bond types in the molecule.
    ///
    /// Analyzes the molecule to determine the minimum number of bits needed to
    /// represent all unique atom and bond labels, then creates a graph using the smallest
    /// possible integer type for node labels (u8, u16, or u32).
    ///
    /// # Returns
    /// A CompactGraph enum containing:
    /// - The graph with the most appropriate integer type for node labels
    /// - A mapping from the compact IDs back to the original atom/bond properties
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::graph::CompactGraph;
    ///
    /// let molecules = MolecularGraph::from_smiles("c1ccccc1").unwrap();
    /// let molecule = molecules.first().unwrap();
    /// let graph = molecule.to_ungraph_with_edge_nodes_and_implicit_hydrogens_compact_dynamic();
    ///
    /// // Since benzene has only one type of atom (aromatic carbon), we expect a u8 graph
    /// match graph {
    ///     CompactGraph::U8(_, label_map) => assert_eq!(label_map.len(), 1),
    ///     _ => panic!("Expected U8 graph for benzene"),
    /// }
    /// ```
    fn to_ungraph_with_edge_nodes_and_implicit_hydrogens_compact_dynamic(&self) -> CompactGraph {
        // First, collect all unique atom and bond labels
        let mut unique_labels = FxHashMap::<AtomLabelImplicitHydrogens, usize>::default();
        let mut next_id: usize = 0;

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            if !unique_labels.contains_key(&label) {
                unique_labels.insert(label, next_id);
                next_id += 1;
            }
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();
                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }
                if index < target {
                    let bond_order = bond.bond_order();
                    if bond_order != BondOrder::Single && bond_order != BondOrder::Aromatic {
                        let bond_label = AtomLabelImplicitHydrogens::new(
                            bond.bond_order().to_u8(),
                            0,
                            false,
                            ChiralType::None,
                            0,
                        );
                        if !unique_labels.contains_key(&bond_label) {
                            unique_labels.insert(bond_label, next_id);
                            next_id += 1;
                        }
                    }
                }
            }
        }

        // Choose the appropriate integer type based on the number of unique labels
        if next_id <= u8::MAX as usize {
            self.build_compact_graph_u8(unique_labels)
        } else if next_id <= u16::MAX as usize {
            self.build_compact_graph_u16(unique_labels)
        } else {
            self.build_compact_graph_u32(unique_labels)
        }
    }

    /// Helper method to build a graph with u8 node labels
    fn build_compact_graph_u8(
        &self,
        unique_labels: FxHashMap<AtomLabelImplicitHydrogens, usize>,
    ) -> CompactGraph {
        let mut graph = UnGraph::<u8, ()>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut bond_indices = FxHashMap::<(usize, usize), NodeIndex<u32>>::default();

        let label_map: FxHashMap<u8, AtomLabelImplicitHydrogens> =
            unique_labels.iter().map(|(k, v)| (*v as u8, *k)).collect();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            let compact_id = unique_labels[&label] as u8;
            let node_index = graph.add_node(compact_id);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();

                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }

                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];

                    if bond_order == BondOrder::Single || bond_order == BondOrder::Aromatic {
                        graph.add_edge(source_atom, target_atom, ());
                        continue;
                    }

                    // Only process each bond once
                    let bond_label = AtomLabelImplicitHydrogens::new(
                        bond.bond_order().to_u8(),
                        0,
                        false,
                        ChiralType::None,
                        0,
                    );
                    let compact_id = unique_labels[&bond_label] as u8;
                    let bond_node = graph.add_node(compact_id);
                    bond_indices.insert((index, target), bond_node);

                    // Connect atoms to bond
                    graph.add_edge(source_atom, bond_node, ());
                    graph.add_edge(target_atom, bond_node, ());
                }
            }
        }

        CompactGraph::U8(graph, label_map)
    }

    /// Helper method to build a graph with u16 node labels
    fn build_compact_graph_u16(
        &self,
        unique_labels: FxHashMap<AtomLabelImplicitHydrogens, usize>,
    ) -> CompactGraph {
        let mut graph = UnGraph::<u16, ()>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut bond_indices = FxHashMap::<(usize, usize), NodeIndex<u32>>::default();

        let label_map: FxHashMap<u16, AtomLabelImplicitHydrogens> =
            unique_labels.iter().map(|(k, v)| (*v as u16, *k)).collect();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            let compact_id = unique_labels[&label] as u16;
            let node_index = graph.add_node(compact_id);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();

                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }

                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];

                    if bond_order == BondOrder::Single || bond_order == BondOrder::Aromatic {
                        graph.add_edge(source_atom, target_atom, ());
                        continue;
                    }

                    // Only process each bond once
                    let bond_label = AtomLabelImplicitHydrogens::new(
                        bond.bond_order().to_u8(),
                        0,
                        false,
                        ChiralType::None,
                        0,
                    );
                    let compact_id = unique_labels[&bond_label] as u16;
                    let bond_node = graph.add_node(compact_id);
                    bond_indices.insert((index, target), bond_node);

                    // Connect atoms to bond
                    graph.add_edge(source_atom, bond_node, ());
                    graph.add_edge(target_atom, bond_node, ());
                }
            }
        }

        CompactGraph::U16(graph, label_map)
    }

    /// Helper method to build a graph with u32 node labels
    fn build_compact_graph_u32(
        &self,
        unique_labels: FxHashMap<AtomLabelImplicitHydrogens, usize>,
    ) -> CompactGraph {
        let mut graph = UnGraph::<u32, ()>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut bond_indices = FxHashMap::<(usize, usize), NodeIndex<u32>>::default();

        let label_map: FxHashMap<u32, AtomLabelImplicitHydrogens> =
            unique_labels.iter().map(|(k, v)| (*v as u32, *k)).collect();

        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if self.atomic_numbers()[index] == 1 {
                continue;
            }
            let label = self.get_atom_label_with_hydrogens(index);
            let compact_id = unique_labels[&label] as u32;
            let node_index = graph.add_node(compact_id);
            atom_indices.insert(index, node_index);
        }

        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                let target = bond.target();

                if self.atomic_numbers()[index] == 1 || self.atomic_numbers()[target] == 1 {
                    continue;
                }

                if index < target {
                    let bond_order = bond.bond_order();
                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];

                    if bond_order == BondOrder::Single || bond_order == BondOrder::Aromatic {
                        graph.add_edge(source_atom, target_atom, ());
                        continue;
                    }

                    // Only process each bond once
                    let bond_label = AtomLabelImplicitHydrogens::new(
                        bond.bond_order().to_u8(),
                        0,
                        false,
                        ChiralType::None,
                        0,
                    );
                    let compact_id = unique_labels[&bond_label] as u32;
                    let bond_node = graph.add_node(compact_id);
                    bond_indices.insert((index, target), bond_node);

                    // Connect atoms to bond
                    graph.add_edge(source_atom, bond_node, ());
                    graph.add_edge(target_atom, bond_node, ());
                }
            }
        }

        CompactGraph::U32(graph, label_map)
    }

    fn to_ungraph_from_slice(&self, slice: &[usize]) -> UnGraph<AtomLabel, BondOrder> {
        let mut graph = UnGraph::<AtomLabel, BondOrder>::default();
        let mut node_indices = IntMap::<usize, NodeIndex<u32>>::default();

        for &index in slice {
            let node_index = graph.add_node(self.get_atom_label(index));
            node_indices.insert(index, node_index);
        }

        for &index in slice {
            let bonds = &self.atom_bonds()[index];
            for bond in bonds {
                let source = node_indices[&index];
                let target = node_indices[&bond.target()];
                if source < target && slice.contains(&bond.target()) {
                    graph.add_edge(source, target, bond.bond_order());
                }
            }
        }
        graph
    }

    /// Converts a molecular graph to a fully indexed representation with hydrogens,
    /// where both atom and bond types are represented by u64 identifiers.
    ///
    /// Similar to `to_fully_indexed_graph` but uses detailed atom labels
    /// that include implicit hydrogen counts, skipping explicit hydrogen atoms.
    ///
    /// # Returns
    /// A tuple containing:
    /// - The converted graph with u64 node and edge labels
    /// - A mapping from node IDs to the original AtomLabelImplicitHydrogens objects
    /// - A mapping from edge IDs to the original BondOrder values
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("CC(=O)O").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let (graph, atom_map, bond_map) = molecule.to_fully_indexed_graph_with_hydrogens();
    /// // Acetic acid's four heavy atoms; the hydrogens are folded into the
    /// // labels, which is what distinguishes CH3 from CH and OH.
    /// assert_eq!(graph.node_count(), 4);
    /// assert_eq!(graph.edge_count(), 3);
    /// assert_eq!(bond_map.len(), 2, "a single and a double bond");
    /// ```
    fn to_fully_indexed_graph_with_hydrogens(
        &self,
    ) -> (
        UnGraph<u64, u64>,
        FxHashMap<u64, AtomLabelImplicitHydrogens>,
        FxHashMap<u64, BondOrder>,
    ) {
        // Use the generic function, but map the return BondOrderWrapper back to BondOrder
        let (graph, atom_map, wrapper_map) = self.to_fully_indexed_graph_generic(
            |idx| self.get_atom_label_with_hydrogens(idx),
            |idx| self.atomic_numbers()[idx] == 1, // Skip explicit hydrogens
            |bond_order| BondOrderWrapper::from(bond_order), // Use wrapper for Hash implementation
        );

        let bond_map = wrapper_map
            .into_iter()
            .map(|(id, wrapper)| (id, wrapper.inner()))
            .collect();

        (graph, atom_map, bond_map)
    }

    /// Converts a molecular graph to a fully indexed representation with generic label types
    /// for both atoms and bonds.
    ///
    /// Creates an efficient representation by:
    /// 1. Assigning unique IDs to each unique atom label type
    /// 2. Assigning unique IDs to each unique bond label type
    /// 3. Creating a graph where both nodes and edges use these numeric identifiers
    ///
    /// # Performance and Memory Benefits
    /// - Reduces memory usage by representing complex atom/bond data as simple integers
    /// - Enables faster graph operations since comparing integers is faster than comparing structs
    /// - Facilitates more efficient substructure matching and chemical fingerprinting
    /// - Provides a canonical representation useful for graph isomorphism algorithms
    ///
    /// # Use Cases
    /// - Molecular fingerprinting where molecules need to be compared efficiently
    /// - QSAR (Quantitative Structure-Activity Relationship) models
    /// - High-performance chemical substructure searching
    /// - Machine learning applications where graph features need to be encoded numerically
    ///
    /// # Type Parameters
    /// * `AF` - Function type that extracts an atom label from an atom index
    /// * `AL` - Atom label type
    /// * `BF` - Function type that extracts a bond label from a bond
    /// * `BL` - Bond label type
    ///
    /// # Arguments
    /// * `get_atom_label` - Function that extracts an atom label from an atom index
    /// * `skip_atom` - Predicate that determines which atoms to skip
    /// * `get_bond_label` - Function that extracts a bond label from a bond order
    ///
    /// # Returns
    /// A tuple containing:
    /// - The converted graph with u64 node and edge labels
    /// - A mapping from node IDs to the original atom label objects
    /// - A mapping from edge IDs to the original bond label objects
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("CC(=O)O").unwrap();
    /// let molecule = molecule.first().unwrap();
    ///
    /// // Using with AtomLabel and BondOrder
    /// let (graph, atom_map, bond_map) = molecule.to_fully_indexed_graph_generic(
    ///     |idx| molecule.get_atom_label(idx),
    ///     |idx| false,
    ///     |bond_order| bond_order
    /// );
    ///
    /// // Custom atom labeling that only considers atomic number and charge
    /// let (graph2, atom_map2, bond_map2) = molecule.to_fully_indexed_graph_generic(
    ///     |idx| (molecule.atomic_numbers()[idx], molecule.get_atom_charge(idx)),
    ///     |idx| molecule.atomic_numbers()[idx] == 1, // Skip hydrogens
    ///     |bond_order| if bond_order == BondOrder::Aromatic { 3 } else { bond_order.order() as u8 * 3}
    /// );
    ///
    /// assert_eq!(graph.node_count(), 8, "every atom, hydrogens included");
    /// assert_eq!(graph2.node_count(), 4, "heavy atoms only");
    /// ```
    fn to_fully_indexed_graph_generic<AF, AL, BF, BL>(
        &self,
        get_atom_label: AF,
        skip_atom: impl Fn(usize) -> bool,
        get_bond_label: BF,
    ) -> (UnGraph<u64, u64>, FxHashMap<u64, AL>, FxHashMap<u64, BL>)
    where
        AF: Fn(usize) -> AL,
        AL: std::hash::Hash + Eq + Copy,
        BF: Fn(BondOrder) -> BL,
        BL: std::hash::Hash + Eq + Copy,
    {
        let mut graph = UnGraph::<u64, u64>::default();
        let mut atom_indices = IntMap::<usize, NodeIndex<u32>>::default();
        let mut atom_label_to_id = FxHashMap::<AL, u64>::default();
        let mut id_to_atom_label = FxHashMap::<u64, AL>::default();
        let mut atom_next_id: u64 = 0;

        let mut bond_label_to_id = FxHashMap::<BL, u64>::default();
        let mut id_to_bond_label = FxHashMap::<u64, BL>::default();
        let mut bond_next_id: u64 = 0;

        // First pass: assign unique IDs to each unique atom label
        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if skip_atom(index) {
                continue;
            }
            let label = get_atom_label(index);
            if !atom_label_to_id.contains_key(&label) {
                atom_label_to_id.insert(label, atom_next_id);
                id_to_atom_label.insert(atom_next_id, label);
                atom_next_id += 1;
            }
        }

        // Second pass: create graph with u64 node labels
        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if skip_atom(index) {
                continue;
            }
            let label = get_atom_label(index);
            let node_id = atom_label_to_id[&label];
            let node_index = graph.add_node(node_id);
            atom_indices.insert(index, node_index);
        }

        // Third pass: collect all unique bond types and add edges
        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            if skip_atom(index) {
                continue;
            }

            for bond in bonds {
                let target = bond.target();
                if skip_atom(target) {
                    continue;
                }

                if index < target {
                    let bond_order = bond.bond_order();
                    let bond_label = get_bond_label(bond_order);

                    let bond_id = if let Some(&id) = bond_label_to_id.get(&bond_label) {
                        id
                    } else {
                        let id = bond_next_id;
                        bond_label_to_id.insert(bond_label, id);
                        id_to_bond_label.insert(id, bond_label);
                        bond_next_id += 1;
                        id
                    };

                    let source_atom = atom_indices[&index];
                    let target_atom = atom_indices[&target];
                    graph.add_edge(source_atom, target_atom, bond_id);
                }
            }
        }

        (graph, id_to_atom_label, id_to_bond_label)
    }

    /// Generic function to convert a molecular graph to a fully indexed directed graph.
    ///
    /// Provides flexible directed graph conversion by allowing the caller to:
    /// - Use custom atom and bond label types and extraction functions
    /// - Skip certain atoms (like hydrogens) if desired
    /// - Create a directed graph where each chemical bond is represented by two directed edges
    ///
    /// # Type Parameters
    /// * `AF` - Function type that extracts an atom label from an atom index
    /// * `AL` - Atom label type
    /// * `BF` - Function type that extracts a bond label from a bond
    /// * `BL` - Bond label type
    ///
    /// # Arguments
    /// * `get_atom_label` - Function that extracts an atom label from an atom index
    /// * `skip_atom` - Predicate that determines which atoms to skip
    /// * `get_bond_label` - Function that extracts a bond label from a bond order
    ///
    /// # Returns
    /// A tuple containing:
    /// - The directed graph with u64 node and edge labels
    /// - A mapping from node IDs to the original atom label objects
    /// - A mapping from edge IDs to the original bond label objects
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("CC(=O)O").unwrap();
    /// let molecule = molecule.first().unwrap();
    ///
    /// // Using with AtomLabel and BondOrder
    /// let (digraph, atom_map, bond_map) = molecule.to_fully_indexed_digraph_generic(
    ///     |idx| molecule.get_atom_label(idx),
    ///     |idx| false,
    ///     |bond_order| bond_order
    /// );
    ///
    /// // Every bond appears once in each direction.
    /// assert_eq!(digraph.node_count(), 8);
    /// assert_eq!(digraph.edge_count(), 14);
    /// ```
    fn to_fully_indexed_digraph_generic<AF, AL, BF, BL>(
        &self,
        get_atom_label: AF,
        skip_atom: impl Fn(usize) -> bool,
        get_bond_label: BF,
    ) -> (
        petgraph::graph::DiGraph<u64, u64>,
        FxHashMap<u64, AL>,
        FxHashMap<u64, BL>,
    )
    where
        AF: Fn(usize) -> AL,
        AL: std::hash::Hash + Eq + Copy,
        BF: Fn(BondOrder) -> BL,
        BL: std::hash::Hash + Eq + Copy,
    {
        let mut graph = petgraph::graph::DiGraph::<u64, u64>::default();
        let mut atom_indices = IntMap::<usize, petgraph::graph::NodeIndex<u32>>::default();
        let mut atom_label_to_id = FxHashMap::<AL, u64>::default();
        let mut id_to_atom_label = FxHashMap::<u64, AL>::default();
        let mut atom_next_id: u64 = 0;

        let mut bond_label_to_id = FxHashMap::<BL, u64>::default();
        let mut id_to_bond_label = FxHashMap::<u64, BL>::default();
        let mut bond_next_id: u64 = 0;

        // First pass: assign unique IDs to each unique atom label
        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if skip_atom(index) {
                continue;
            }
            let label = get_atom_label(index);
            if !atom_label_to_id.contains_key(&label) {
                atom_label_to_id.insert(label, atom_next_id);
                id_to_atom_label.insert(atom_next_id, label);
                atom_next_id += 1;
            }
        }

        // Second pass: create graph with u64 node labels
        for (index, _) in self.atomic_numbers().iter().enumerate() {
            if skip_atom(index) {
                continue;
            }
            let label = get_atom_label(index);
            let node_id = atom_label_to_id[&label];
            let node_index = graph.add_node(node_id);
            atom_indices.insert(index, node_index);
        }

        // Third pass: collect all unique bond types and add directed edges
        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            if skip_atom(index) {
                continue;
            }

            for bond in bonds {
                let target = bond.target();
                if skip_atom(target) {
                    continue;
                }

                let bond_order = bond.bond_order();
                let bond_label = get_bond_label(bond_order);

                let bond_id = if let Some(&id) = bond_label_to_id.get(&bond_label) {
                    id
                } else {
                    let id = bond_next_id;
                    bond_label_to_id.insert(bond_label, id);
                    id_to_bond_label.insert(id, bond_label);
                    bond_next_id += 1;
                    id
                };

                // One directed edge per visit. Every bond is visited from
                // both of its atoms, so the pair comes out on its own —
                // adding both directions here as well gave four directed
                // edges per bond instead of two.
                let source_atom = atom_indices[&index];
                let target_atom = atom_indices[&target];
                graph.add_edge(source_atom, target_atom, bond_id);
            }
        }

        (graph, id_to_atom_label, id_to_bond_label)
    }
}
