use super::graph::{Graph, StereoLabel, stereo_label_key};
use crate::molecule::bond::BondOrder;
use rustc_hash::FxHashMap;

/// A bond's node label for line-graph matching: its bond order plus the
/// (canonically ordered, so orientation-agnostic) pair of
/// [`StereoLabel`]s at its two endpoints. Bond order alone isn't enough —
/// two `Single` bonds are only actually interchangeable if their endpoint
/// atoms match too (a C-C bond is not the same thing as a C-O bond just
/// because both are single bonds, and a bond to an *R* centre is not the
/// same as a bond to an *S* one); this is the line-graph-vertex equivalent
/// of the endpoint atom-label check the clique-based
/// [`super::super::mces`] reduction does when building modular-product
/// *vertices*.
type BondSignature = (BondOrder, StereoLabel, StereoLabel);

/// The line graph of a [`Graph`]: one vertex per bond (labelled with its
/// [`BondSignature`]), with two vertices connected whenever the
/// corresponding bonds share an atom (labelled with that shared atom's
/// [`StereoLabel`]).
///
/// Searching for a maximum common *induced* subgraph of two line graphs
/// (via [`super::search`]) is exactly searching for a maximum common *edge*
/// subgraph of the original molecules: a line-graph vertex is a bond, so
/// maximizing matched vertices there directly maximizes matched bonds. Two
/// bond-pairs that share no atom in *either* molecule are simply
/// non-adjacent on both sides, which already satisfies the induced
/// constraint trivially — so independently-matching fragments combine for
/// free, with no separate mechanism needed for it (contrast with `mces`'s
/// "type-0" edges, which exist to give the modular-product/clique reduction
/// that same freedom by construction).
pub(super) struct LineGraph {
    pub bond_signature: Vec<BondSignature>,
    /// line-graph vertex -> the two (compact) atom indices of that bond.
    pub endpoints: Vec<(usize, usize)>,
    adjacent_atom: FxHashMap<(usize, usize), StereoLabel>,
    /// How many other bonds each bond shares an atom with. Precomputed
    /// because the search consults it on every branch.
    degree: Vec<usize>,
}

impl LineGraph {
    pub fn from_graph(g: &Graph) -> Self {
        let mut bond_signature = Vec::new();
        let mut endpoints = Vec::new();
        for ((a, b), order) in g.bonds() {
            let (label_a, label_b) = (g.labels[a], g.labels[b]);
            let (lo, hi) = if stereo_label_key(&label_a) <= stereo_label_key(&label_b) {
                (label_a, label_b)
            } else {
                (label_b, label_a)
            };
            bond_signature.push((order, lo, hi));
            endpoints.push((a, b));
        }

        let n = endpoints.len();
        let mut adjacent_atom = FxHashMap::default();
        let mut degree = vec![0usize; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let (a1, a2) = endpoints[i];
                let (b1, b2) = endpoints[j];
                let shared = if a1 == b1 || a1 == b2 {
                    Some(a1)
                } else if a2 == b1 || a2 == b2 {
                    Some(a2)
                } else {
                    None
                };
                if let Some(atom) = shared {
                    let label = g.labels[atom];
                    adjacent_atom.insert((i, j), label);
                    adjacent_atom.insert((j, i), label);
                    degree[i] += 1;
                    degree[j] += 1;
                }
            }
        }

        Self { bond_signature, endpoints, adjacent_atom, degree }
    }

    pub fn shared_atom_label(&self, i: usize, j: usize) -> Option<StereoLabel> {
        self.adjacent_atom.get(&(i, j)).copied()
    }
}

impl super::search::McsplitGraph for LineGraph {
    type NodeLabel = BondSignature;
    type EdgeLabel = StereoLabel;

    fn len(&self) -> usize {
        self.bond_signature.len()
    }

    fn node_label(&self, index: usize) -> BondSignature {
        self.bond_signature[index]
    }

    fn edge_label(&self, a: usize, b: usize) -> Option<StereoLabel> {
        self.shared_atom_label(a, b)
    }

    fn degree(&self, index: usize) -> usize {
        self.degree[index]
    }
}
