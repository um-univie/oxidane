//! Maximum common edge subgraph (MCES) search between molecules.
//!
//! Finds the largest set of bonds (and the atoms they connect) that can be
//! matched consistently between two or more molecules — the standard way to
//! measure structural overlap between chemical structures ("linking" them
//! by their shared scaffold).
//!
//! # Algorithm
//!
//! Each molecule is converted to its bond-adjacency line graph; the
//! (chemically-constrained) modular product of two line graphs reduces
//! "largest common set of bonds" to "maximum clique", found via an
//! explicit-stack Bron-Kerbosch search with branch-and-bound pruning against
//! the best size found so far. Atoms are compared by element, charge,
//! aromaticity, isotope and chirality ([`crate::molecule::graph::AtomLabel`]);
//! bonds by [`crate::molecule::bond::BondOrder`].
//!
//! # Example
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::graph_algorithms::mces::find_maximum_common_edge_subgraph;
//!
//! let ethanol = MolecularGraph::from_smiles("CCO").unwrap().remove(0);
//! let propanol = MolecularGraph::from_smiles("CCCO").unwrap().remove(0);
//!
//! let mces = find_maximum_common_edge_subgraph(&ethanol, &propanol).unwrap();
//! // The shared C-C-O scaffold: 3 atoms, 2 bonds.
//! assert_eq!(mces.num_atoms(), 3);
//! assert_eq!(mces.num_bonds(), 2);
//! ```

mod driver;
mod line_graph;
mod modular_product;
mod reconstruct;
mod search;
mod types;

use crate::molecule::base::Molecule;
use std::rc::Rc;
use types::{AlgorithmState, McesGraph};

pub use reconstruct::BondMapping;

/// One atom correspondence between two molecules' common substructure,
/// referencing each molecule's own atom index. The atomic numbers are
/// carried along (and are always equal — atom compatibility requires a
/// matching element) so the mapping is self-describing without needing to
/// keep the original molecules around to look indices back up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomMapping {
    pub atom_a: usize,
    pub atom_b: usize,
    pub atomic_number: u8,
}

/// A maximum common edge subgraph (MCES) shared between two molecules: the
/// atom and bond correspondence between them, restricted to the matched
/// substructure.
#[derive(Debug, Clone, Default)]
pub struct MaximumCommonEdgeSubgraph {
    pub atom_mapping: Vec<AtomMapping>,
    pub bond_mapping: Vec<BondMapping>,
}

impl MaximumCommonEdgeSubgraph {
    /// Number of atoms in the common substructure.
    pub fn num_atoms(&self) -> usize {
        self.atom_mapping.len()
    }

    /// Number of bonds in the common substructure (the MCES's own size
    /// metric — this is what the search maximizes).
    pub fn num_bonds(&self) -> usize {
        self.bond_mapping.len()
    }
}

/// Size of the maximum common substructure shared across several molecules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommonSubstructureSize {
    pub num_atoms: usize,
    pub num_bonds: usize,
}

/// Builds the heavy-atom-only graph a molecule is searched over: explicit
/// hydrogens are excluded (as in any standard MCS/MCES tool — matching
/// interchangeable terminal hydrogens against each other isn't a meaningful
/// structural overlap), so the common substructure reflects the heavy-atom
/// scaffold. `original_index` records, for each compact node, which atom
/// index in `mol` it came from, so results can be reported in `mol`'s own
/// atom-index coordinates.
fn to_heavy_atom_graph<T: Molecule>(mol: &T) -> McesGraph {
    let mut graph = types::MolGraphType::default();
    let mut original_index = Vec::new();
    let mut compact_of = rustc_hash::FxHashMap::default();

    for (index, &atomic_number) in mol.atomic_numbers().iter().enumerate() {
        if atomic_number == 1 {
            continue;
        }
        let node = graph.add_node(mol.get_atom_label(index));
        compact_of.insert(index, node);
        original_index.push(index);
    }

    for (index, bonds) in mol.atom_bonds().iter().enumerate() {
        let Some(&source) = compact_of.get(&index) else {
            continue;
        };
        for bond in bonds {
            let target = bond.target();
            if index < target {
                if let Some(&target_node) = compact_of.get(&target) {
                    graph.add_edge(source, target_node, bond.bond_order());
                }
            }
        }
    }

    McesGraph {
        graph,
        original_index,
    }
}

fn to_line_graph<T: Molecule>(mol: &T) -> types::LineGraph {
    let graph = Rc::new(to_heavy_atom_graph(mol));
    line_graph::molecular_graph_to_line_graph(&graph)
}

fn run_search(mut graphs: Vec<types::LineGraph>) -> AlgorithmState {
    let mut state = AlgorithmState::default();
    driver::find_maximum_common_subgraph(&mut graphs, &mut state);
    // The clique search raises `state.maximum` itself, as soon as it has a
    // bound worth pruning against, without going through the bookkeeping that
    // clears the results recorded under the old bound. So a subgraph recorded
    // while the bound was lower outlives the discovery of a bigger one, and
    // what comes back here is every *maximal* common subgraph, not the
    // *maximum* ones. Keep the largest; they are what every caller asked for.
    let largest = state
        .all_maximum_graphs
        .iter()
        .map(|resolved| resolved.bond_pairs.len())
        .max()
        .unwrap_or(0);
    state
        .all_maximum_graphs
        .retain(|resolved| resolved.bond_pairs.len() == largest);
    state
}

fn to_public<T: Molecule>(mol_a: &T, resolved: reconstruct::Resolved) -> MaximumCommonEdgeSubgraph {
    MaximumCommonEdgeSubgraph {
        atom_mapping: resolved
            .atom_pairs
            .into_iter()
            .map(|(atom_a, atom_b)| AtomMapping {
                atom_a,
                atom_b,
                atomic_number: mol_a.atomic_numbers()[atom_a],
            })
            .collect(),
        bond_mapping: resolved.bond_pairs,
    }
}

/// Finds a maximum common edge subgraph (MCES) shared between `mol_a` and
/// `mol_b`: the largest set of bonds (and their atoms) that can be matched
/// consistently between the two molecules.
///
/// Returns `None` if the molecules share no bond at all. When several
/// distinct maximum-size substructures exist, one is returned arbitrarily —
/// use [`find_all_maximum_common_edge_subgraphs`] to get every one of them.
pub fn find_maximum_common_edge_subgraph<T: Molecule>(
    mol_a: &T,
    mol_b: &T,
) -> Option<MaximumCommonEdgeSubgraph> {
    let mut state = run_search(vec![to_line_graph(mol_a), to_line_graph(mol_b)]);
    state
        .all_maximum_graphs
        .pop()
        .map(|resolved| to_public(mol_a, resolved))
}

/// Like [`find_maximum_common_edge_subgraph`], but returns every common
/// substructure tied for the maximum size instead of just one.
pub fn find_all_maximum_common_edge_subgraphs<T: Molecule>(
    mol_a: &T,
    mol_b: &T,
) -> Vec<MaximumCommonEdgeSubgraph> {
    let state = run_search(vec![to_line_graph(mol_a), to_line_graph(mol_b)]);
    state
        .all_maximum_graphs
        .into_iter()
        .map(|resolved| to_public(mol_a, resolved))
        .collect()
}

/// Finds the size (matched atom and bond count) of the maximum common edge
/// subgraph shared across all of `molecules` (2 or more).
///
/// For exactly two molecules this agrees with
/// [`find_maximum_common_edge_subgraph`]. For more than two, the common
/// substructure is found by repeated pairwise reduction (the first two
/// molecules, then their common substructure combined with the third, and so
/// on), so only the aggregate size is reported here, not a full atom-level
/// mapping back to every input molecule.
///
/// Returns `None` if fewer than 2 molecules are given or they share no bond.
pub fn common_edge_subgraph_size<T: Molecule>(molecules: &[T]) -> Option<CommonSubstructureSize> {
    if molecules.len() < 2 {
        return None;
    }
    let state = run_search(molecules.iter().map(to_line_graph).collect());
    let resolved = state.all_maximum_graphs.into_iter().next()?;
    Some(CommonSubstructureSize {
        num_atoms: resolved.pattern.graph.node_count(),
        num_bonds: resolved.bond_pairs.len(),
    })
}
