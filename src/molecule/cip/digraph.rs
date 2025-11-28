//! The CIP hierarchical digraph: the molecule unrolled into a tree rooted at
//! the centre being labelled, with duplicate ("phantom") nodes standing in for
//! ring closures and for the extra bonds of double and triple bonds.
//!
//! Ported from RDKit's `Code/GraphMol/CIPLabeler/{Digraph,Node,Edge}.cpp`.
//! See the module docs in [`super`] for provenance and licensing.
//!
//! Nodes and edges live in arenas on [`Digraph`] and are referred to by index,
//! because the graph is cyclic in the ownership sense: every edge names two
//! nodes and every node names its edges. Expansion is *lazy* — a node's edges
//! are created the first time they are asked for — which is what keeps a fused
//! polycycle from materialising its (exponentially many) simple paths before
//! the rules have had a chance to prune.

use std::rc::Rc;

use super::mol::{CipMol, Frac};

pub(super) type NodeId = usize;
pub(super) type EdgeId = usize;

/// Upper bound on digraph size, so a pathological structure fails loudly
/// instead of exhausting memory. Matches RDKit's limit.
const MAX_NODE_COUNT: usize = 100_000;

/// Node was created to stand in for a ring closure.
pub(super) const RING_DUPLICATE: u8 = 0x2;
/// Node was created to stand in for the second/third bond of a multiple bond.
pub(super) const BOND_DUPLICATE: u8 = 0x4;
const DUPLICATE: u8 = RING_DUPLICATE | BOND_DUPLICATE;
const EXPANDED: u8 = 0x1;

pub(super) struct Node {
    /// The molecule atom this node stands for.
    pub(super) atom: usize,
    /// Number of bonds from the root.
    pub(super) distance: i32,
    /// Averaged atomic number (see [`super::mol`]); duplicates inherit it.
    pub(super) atomic_number_fraction: Frac,
    pub(super) atomic_number: u8,
    pub(super) mass_number: u16,
    pub(super) atomic_mass: f64,
    /// Descriptor assigned to this node by the auxiliary-descriptor pass; what
    /// rules 3, 4a, 4b, 4c and 5 read.
    pub(super) aux: super::descriptor::Descriptor,
    flags: u8,
    pub(super) edges: Vec<EdgeId>,
    /// Per-atom "distance at which this atom was first visited on the path to
    /// here", or 0 for not-yet-visited. Empty for duplicate nodes, which are
    /// terminal and never expand.
    visit: Vec<u8>,
}

impl Node {
    pub(super) fn is_set(&self, mask: u8) -> bool {
        self.flags & mask != 0
    }

    pub(super) fn is_duplicate(&self) -> bool {
        self.flags & DUPLICATE != 0
    }

    fn is_expanded(&self) -> bool {
        self.flags & EXPANDED != 0
    }

    /// A node nothing further hangs off: a duplicate, or an expanded node
    /// whose only edge is the one it was reached by.
    pub(super) fn is_terminal(&self) -> bool {
        self.visit.is_empty() || (self.is_expanded() && self.edges.len() == 1)
    }
}

pub(super) struct Edge {
    pub(super) beg: NodeId,
    pub(super) end: NodeId,
    /// Descriptor carried by the bond itself (E/Z), read by rules 4a and 4c.
    pub(super) aux: super::descriptor::Descriptor,
}

impl Edge {
    pub(super) fn other(&self, node: NodeId) -> NodeId {
        if self.beg == node { self.end } else { self.beg }
    }

    pub(super) fn is_beg(&self, node: NodeId) -> bool {
        self.beg == node
    }

    pub(super) fn is_end(&self, node: NodeId) -> bool {
        self.end == node
    }
}

pub(super) struct Digraph {
    /// Shared with every other centre's digraph in the same molecule: the
    /// Kekulisation and Mancude averaging are molecule-wide and cost far more
    /// than one digraph does.
    pub(super) mol: Rc<CipMol>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    /// The node the digraph was built from — the atom being labelled.
    origin: NodeId,
    /// The node edges currently point away from. Auxiliary-descriptor
    /// calculation re-roots the same digraph at inner nodes rather than
    /// rebuilding it.
    root: NodeId,
    /// Reference atom for rule 6, when one is set.
    pub(super) rule6_ref: Option<usize>,
    /// Set once the node budget is blown; the caller abandons the centre.
    pub(super) overflowed: bool,
}

impl Digraph {
    pub(super) fn new(mol: Rc<CipMol>, atom: usize) -> Self {
        let n = mol.len();
        let mut visit = vec![0u8; n];
        visit[atom] = 1;

        let root_node = make_node(&mol, visit, atom, Frac::whole(mol.atomic_number(atom) as i32), 1, 0);
        Digraph {
            mol,
            nodes: vec![root_node],
            edges: Vec::new(),
            origin: 0,
            root: 0,
            rule6_ref: None,
            overflowed: false,
        }
    }

    pub(super) fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    pub(super) fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }

    pub(super) fn edge(&self, id: EdgeId) -> &Edge {
        &self.edges[id]
    }

    pub(super) fn origin(&self) -> NodeId {
        self.origin
    }

    pub(super) fn root(&self) -> NodeId {
        self.root
    }

    /// The node's edges, expanding it first if it has not been expanded.
    ///
    /// Returns a copy: every caller sorts what it gets back, and sorting is
    /// per-comparison scratch work that must not disturb the node.
    pub(super) fn edges_of(&mut self, node: NodeId) -> Vec<EdgeId> {
        if !self.nodes[node].is_expanded() {
            self.nodes[node].flags |= EXPANDED;
            self.expand(node);
        }
        self.nodes[node].edges.clone()
    }

    /// Edges leading away from `node` to something that has further structure
    /// — the branches rule 4b walks when it looks for a reference descriptor.
    pub(super) fn non_terminal_out_edges(&mut self, node: NodeId) -> Vec<EdgeId> {
        self.edges_of(node)
            .into_iter()
            .filter(|&e| self.edges[e].is_beg(node) && !self.nodes[self.edges[e].end].is_terminal())
            .collect()
    }

    /// Non-duplicate edges of `node` that touch molecule atom `atom`.
    pub(super) fn edges_to_atom(&mut self, node: NodeId, atom: usize) -> Vec<EdgeId> {
        self.edges_of(node)
            .into_iter()
            .filter(|&e| {
                let edge = &self.edges[e];
                if self.nodes[edge.end].is_duplicate() {
                    return false;
                }
                self.nodes[edge.beg].atom == atom || self.nodes[edge.end].atom == atom
            })
            .collect()
    }

    /// Every node standing for molecule atom `atom`, in order of distance from
    /// the current root.
    pub(super) fn nodes_for_atom(&mut self, atom: usize) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut queue = vec![self.root];
        let mut position = 0;
        while position < queue.len() {
            let node = queue[position];
            position += 1;
            if self.nodes[node].atom == atom {
                result.push(node);
            }
            for edge in self.edges_of(node) {
                if !self.edges[edge].is_beg(node) {
                    continue;
                }
                queue.push(self.edges[edge].end);
            }
        }
        result
    }

    /// Has any node been created for this molecule atom yet?
    pub(super) fn seen_atom(&self, atom: usize) -> bool {
        self.nodes.iter().any(|n| n.atom == atom && !n.is_duplicate())
    }

    /// Re-roots the digraph at `new_root` by flipping the edges on the path
    /// back to the old root, rather than rebuilding — which is what makes the
    /// auxiliary-descriptor pass affordable, since it labels every stereocentre
    /// reachable inside one centre's digraph.
    pub(super) fn change_root(&mut self, new_root: NodeId) {
        let mut to_flip: Vec<EdgeId> = Vec::new();
        let mut queue = vec![new_root];
        let mut position = 0;
        while position < queue.len() {
            let node = queue[position];
            position += 1;
            for edge in self.edges_of(node) {
                if self.edges[edge].is_end(node) {
                    to_flip.push(edge);
                    queue.push(self.edges[edge].beg);
                }
            }
        }
        for edge in to_flip {
            let e = &mut self.edges[edge];
            std::mem::swap(&mut e.beg, &mut e.end);
        }
        self.root = new_root;
    }

    fn add_node(&mut self, node: Node) -> NodeId {
        if self.nodes.len() >= MAX_NODE_COUNT {
            self.overflowed = true;
        }
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn add_edge(&mut self, beg: NodeId, end: NodeId) {
        let id = self.edges.len();
        self.edges.push(Edge { beg, end, aux: super::descriptor::Descriptor::None });
        self.nodes[beg].edges.push(id);
        self.nodes[end].edges.push(id);
    }

    /// Creates the children of `beg`: one ordinary node per not-yet-visited
    /// neighbour, a ring-duplicate for each neighbour already on the path, and
    /// bond-duplicates for the second and third bonds of multiple bonds.
    fn expand(&mut self, beg: NodeId) {
        if self.overflowed {
            return;
        }
        let atom = self.nodes[beg].atom;

        // The bond this node was reached by, if any — its extra orders are
        // duplicated backwards rather than forwards.
        let previous_atom = self.nodes[beg]
            .edges
            .first()
            .filter(|&&e| !self.edges[e].is_beg(beg))
            .map(|&e| self.nodes[self.edges[e].beg].atom);

        let origin_atom = self.nodes[self.origin].atom;
        let bonds: Vec<_> = self.mol.bonds(atom).to_vec();

        for bond in bonds {
            let neighbour = bond.other;
            let virtual_nodes = (bond.order - 1).max(0);
            // A negatively charged atom in a delocalised system spreads its
            // extra bond over the whole system, so it contributes exactly one
            // duplicate however the Kekulé structure placed the double bond.
            let charge_delocalised = self.mol.charge(atom) < 0
                && self.mol.fractional_atomic_number(atom).denominator() > 1;

            if self.nodes[beg].visit.get(neighbour).copied().unwrap_or(0) == 0 {
                let child = self.new_child(beg, neighbour);
                self.add_edge(beg, child);

                // The root itself does not duplicate its own multiple bonds:
                // the centre is not a substituent of itself.
                if self.origin != beg {
                    if charge_delocalised {
                        let dup = self.new_terminal_child(beg, neighbour, BOND_DUPLICATE);
                        self.add_edge(beg, dup);
                    } else {
                        for _ in 0..virtual_nodes {
                            let dup = self.new_terminal_child(beg, neighbour, BOND_DUPLICATE);
                            self.add_edge(beg, dup);
                        }
                    }
                }
            } else if Some(neighbour) == previous_atom {
                // Bond order expansion backwards along the bond we came in on.
                if origin_atom != neighbour {
                    for _ in 0..virtual_nodes {
                        let dup = self.new_terminal_child(beg, neighbour, BOND_DUPLICATE);
                        self.add_edge(beg, dup);
                    }
                }
            } else {
                // Ring closure.
                let dup = self.new_terminal_child(beg, neighbour, RING_DUPLICATE);
                self.add_edge(beg, dup);

                if charge_delocalised {
                    let dup = self.new_terminal_child(beg, neighbour, BOND_DUPLICATE);
                    self.add_edge(beg, dup);
                } else {
                    for _ in 0..virtual_nodes {
                        let dup = self.new_terminal_child(beg, neighbour, BOND_DUPLICATE);
                        self.add_edge(beg, dup);
                    }
                }
            }
            if self.overflowed {
                return;
            }
        }
    }

    fn new_child(&mut self, parent: NodeId, atom: usize) -> NodeId {
        let distance = self.nodes[parent].distance + 1;
        let mut visit = self.nodes[parent].visit.clone();
        visit[atom] = distance.min(u8::MAX as i32) as u8;
        let frac = Frac::whole(self.mol.atomic_number(atom) as i32);
        let node = make_node(&self.mol, visit, atom, frac, distance, 0);
        self.add_node(node)
    }

    /// A duplicate node: no children, and its distance is that of the atom it
    /// duplicates rather than one further out — which is exactly what rule 1b
    /// ranks by.
    fn new_terminal_child(&mut self, parent: NodeId, atom: usize, flags: u8) -> NodeId {
        let distance = if flags & DUPLICATE != 0 {
            self.nodes[parent].visit.get(atom).copied().unwrap_or(0) as i32
        } else {
            self.nodes[parent].distance + 1
        };

        // A bond duplicate of a delocalised atom inherits the averaged atomic
        // number, so the phantom cannot betray which Kekulé form was picked.
        let frac = if flags & BOND_DUPLICATE != 0 {
            let parent_frac = self.mol.fractional_atomic_number(self.nodes[parent].atom);
            if parent_frac.denominator() > 1 {
                parent_frac
            } else {
                Frac::whole(self.mol.atomic_number(atom) as i32)
            }
        } else {
            Frac::whole(self.mol.atomic_number(atom) as i32)
        };

        let node = make_node(&self.mol, Vec::new(), atom, frac, distance, flags);
        self.add_node(node)
    }
}

fn make_node(
    mol: &CipMol,
    visit: Vec<u8>,
    atom: usize,
    atomic_number_fraction: Frac,
    distance: i32,
    mut flags: u8,
) -> Node {
    let duplicate = flags & DUPLICATE != 0;
    if visit.is_empty() || duplicate {
        flags |= EXPANDED;
    }
    // A duplicate has no mass of its own: rule 2 must not see a phantom as a
    // heavier or lighter version of the atom it stands in for.
    let (mass_number, atomic_mass) =
        if duplicate { (0, 0.0) } else { (mol.isotope(atom), mol.atomic_mass(atom)) };

    Node {
        atom,
        distance,
        atomic_number_fraction,
        atomic_number: mol.atomic_number(atom),
        mass_number,
        atomic_mass,
        aux: super::descriptor::Descriptor::None,
        flags,
        edges: Vec::new(),
        visit,
    }
}
