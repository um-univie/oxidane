use super::reconstruct::Resolved;
use crate::molecule::bond::BondOrder;
use crate::molecule::graph::AtomLabel;
use petgraph::graph::{NodeIndex, UnGraph};
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// A whole molecule reduced to the atom/bond information the MCES search needs:
/// atom nodes labelled with [`AtomLabel`] (element, charge, aromaticity, isotope,
/// chirality) and bond edges labelled with [`BondOrder`].
///
/// `NodeIndex(i)` always corresponds to atom index `i` in the source molecule,
/// since [`crate::molecule::graph::MoleculeGraph::to_ungraph`] adds atom nodes in
/// index order without skipping any atom.
pub(super) type MolGraphType = UnGraph<AtomLabel, BondOrder>;

/// Line graph of a [`McesGraph`]: one node per original bond (labelled with that
/// bond's [`BondOrder`]), with two line-graph nodes connected whenever the
/// corresponding bonds share an atom (edge labelled with that shared atom's
/// [`AtomLabel`]).
pub(super) type LineGraphType = UnGraph<BondOrder, AtomLabel>;

/// Modular product of two line graphs. A node pairs one bond from each input
/// molecule (compatible bond order + compatible shared-atom pattern); an edge
/// marks that the corresponding vertex pair is consistent to extend a common
/// subgraph together. Edge weight is `None` for the "type-0"/disconnected-pair
/// case used by [`super::search::find_blue_connected_components`].
pub(super) type ModularProductGraphType = UnGraph<BondOrder, Option<AtomLabel>>;

#[derive(Debug, Clone, Default)]
pub(super) struct McesGraph {
    pub graph: MolGraphType,
    /// compact node index (as used by `graph`) -> the atom's index in
    /// whatever original coordinate system this `McesGraph` was built from
    /// (an input molecule's atom indices, or — for a synthetic pattern graph
    /// built to seed the next round of an n-ary search — its own indices).
    pub original_index: Vec<usize>,
}

impl McesGraph {
    /// Wraps an already-built graph whose node indices are already the
    /// indices callers should see (identity mapping) — used for the
    /// synthetic pattern graphs built between rounds of an n-ary search.
    pub fn from_ungraph(graph: MolGraphType) -> Self {
        let original_index = (0..graph.node_count()).collect();
        Self {
            graph,
            original_index,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct LineGraph {
    pub graph: LineGraphType,
    /// line-graph node -> the two original atom indices (as `NodeIndex` into
    /// `original_graph`) making up that bond, ordered `source < target`.
    pub edge_map: FxHashMap<NodeIndex, (NodeIndex, NodeIndex)>,
    /// line-graph node -> a stable id for the original bond (its petgraph edge
    /// index), used to de-duplicate cliques already covered by a larger one.
    pub clique_map: FxHashMap<NodeIndex, usize>,
    pub original_graph: Rc<McesGraph>,
    /// Index of this graph within the current n-ary search's `clique_vectors`
    /// bookkeeping (see [`AlgorithmState`]).
    pub num: usize,
}

#[derive(Debug, Clone)]
pub(super) struct ModularProductGraph {
    pub graph: ModularProductGraphType,
    /// product node -> (g1 line-graph node, g2 line-graph node)
    pub node_mapping: FxHashMap<NodeIndex, (NodeIndex, NodeIndex)>,
    pub g1: Rc<LineGraph>,
    pub g2: Rc<LineGraph>,
    /// product node -> g1's bond id (see `LineGraph::clique_map`)
    pub clique_mapping: FxHashMap<NodeIndex, usize>,
}

impl ModularProductGraph {
    pub fn new(g1: Rc<LineGraph>, g2: Rc<LineGraph>) -> Self {
        Self {
            graph: ModularProductGraphType::default(),
            node_mapping: FxHashMap::default(),
            g1,
            g2,
            clique_mapping: FxHashMap::default(),
        }
    }
}

pub(super) type Clique = Vec<NodeIndex>;

/// Search-wide state threaded through the recursive MCES search, mirroring the
/// global state used by the original C++ algorithm: the running maximum size,
/// every subgraph tied for that maximum, and per-molecule "clique vectors"
/// used to skip cliques that are subsets of an already-found maximal one.
#[derive(Default)]
pub(super) struct AlgorithmState {
    pub maximum: usize,
    pub num_cliques_found: usize,
    pub clique_vectors: Vec<Vec<Vec<usize>>>,
    pub not_first: bool,
    pub all_maximum_graphs: Vec<Resolved>,
}
