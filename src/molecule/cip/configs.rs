//! The two kinds of stereo configuration this labeller resolves: a
//! tetrahedral centre and a double bond.
//!
//! Ported from RDKit's `Code/GraphMol/CIPLabeler/configs/*`.
//! See the module docs in [`super`] for provenance and licensing.

use super::descriptor::Descriptor;
use super::digraph::{BOND_DUPLICATE, Digraph, EdgeId, NodeId};
use super::rules::{Ctx, Priority, Sorter, get_groups, prioritize, rank_independent_edges};

/// Stands for the fourth carrier of a three-coordinate centre — a lone pair or
/// the phantom atom that occupies the vacant vertex. It must be distinct from
/// every real atom index and from the focus itself.
const PHANTOM: usize = usize::MAX;

/// Parity of the permutation taking `reference` to `target`: 2 for even, 1 for
/// odd, 0 if they are not permutations of one another.
///
/// A tetrahedral tag is a statement about one particular ordering of the four
/// neighbours. Re-expressing it against the CIP priority ordering is exactly
/// asking whether that reordering was an even or an odd permutation.
fn parity4(target: &[usize; 4], reference: &[usize; 4]) -> i32 {
    let mut positions = [0usize; 4];
    for (i, &value) in reference.iter().enumerate() {
        match target.iter().position(|&t| t == value) {
            Some(position) => positions[i] = position,
            None => return 0,
        }
    }
    // Must be a genuine permutation, not merely all-present (repeats would
    // make the parity meaningless).
    let mut seen = [false; 4];
    for &position in &positions {
        if seen[position] {
            return 0;
        }
        seen[position] = true;
    }

    let mut inversions = 0;
    for i in 0..4 {
        for j in (i + 1)..4 {
            if positions[i] > positions[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 { 2 } else { 1 }
}

/// A tetrahedral centre, and the neighbour ordering its stored tag is defined
/// against.
pub(super) struct Tetrahedral {
    pub(super) focus: usize,
    /// The four neighbours in the reference order, with [`PHANTOM`] standing
    /// in for a missing fourth.
    carriers: [usize; 4],
    /// True when the tag says the carriers wind clockwise.
    clockwise: bool,
}

/// A stereogenic double bond, and the two neighbours whose relative sides
/// define it.
pub(super) struct Sp2Bond {
    pub(super) focus: usize,
    pub(super) other: usize,
    carriers: [usize; 2],
    /// True when the two carriers are on the same side.
    cis: bool,
}

/// An allene: `C(A)(B)=C=C(D)(E)`.
///
/// The two ends are held at right angles, so the four substituents sit at the
/// corners of an elongated tetrahedron and the centre is chiral in the same
/// way an sp3 carbon is. OpenSMILES puts it as the allene being "conceptually
/// collapsed into a single tetrahedral chiral center", which is exactly how it
/// is labelled here: the same parity calculation, with the carriers gathered
/// from the two ends rather than from one atom.
pub(super) struct Allene {
    pub(super) centre: usize,
    terminals: [usize; 2],
    /// The four substituents in the reference order the stored tag is read
    /// against — see
    /// [`crate::molecule::chirality::index_sorted_allene_carriers`].
    carriers: [usize; 4],
    clockwise: bool,
}

pub(super) enum Configuration {
    Tetrahedral(Tetrahedral),
    Sp2Bond(Sp2Bond),
    Allene(Allene),
}

impl Configuration {
    /// The atoms that carry the resulting label — one for a centre, two for a
    /// double bond.
    pub(super) fn foci(&self) -> Vec<usize> {
        match self {
            Configuration::Tetrahedral(t) => vec![t.focus],
            Configuration::Sp2Bond(b) => vec![b.focus, b.other],
            Configuration::Allene(a) => vec![a.centre],
        }
    }

    pub(super) fn root_atom(&self) -> usize {
        self.foci()[0]
    }

    pub(super) fn new_tetrahedral(focus: usize, neighbours: &[usize], clockwise: bool) -> Option<Self> {
        let mut carriers = [PHANTOM; 4];
        for (slot, &neighbour) in carriers.iter_mut().zip(neighbours.iter()) {
            *slot = neighbour;
        }
        match neighbours.len() {
            4 => {}
            // Three-coordinate centre: the vacant vertex is represented by the
            // focus itself, matching how the digraph reports a centre that
            // ran out of substituents.
            3 => carriers[3] = focus,
            _ => return None,
        }
        Some(Configuration::Tetrahedral(Tetrahedral { focus, carriers, clockwise }))
    }

    pub(super) fn new_sp2(focus: usize, other: usize, carriers: [usize; 2], cis: bool) -> Self {
        Configuration::Sp2Bond(Sp2Bond { focus, other, carriers, cis })
    }


    pub(super) fn new_allene(
        centre: usize,
        terminals: [usize; 2],
        carriers: [usize; 4],
        clockwise: bool,
    ) -> Self {
        Configuration::Allene(Allene { centre, terminals, carriers, clockwise })
    }

    /// Labels this configuration from the digraph's original root.
    pub(super) fn label(&self, g: &mut Digraph, ctx: &mut Ctx) -> Descriptor {
        let root = g.origin();
        if g.root() != root {
            g.change_root(root);
        }
        self.label_at(g, ctx, root)
    }

    /// Labels this configuration as seen from `node`, which becomes the
    /// digraph's root. Used by the auxiliary pass, which needs the descriptor
    /// of a centre *inside* another centre's digraph.
    pub(super) fn label_from(&self, g: &mut Digraph, ctx: &mut Ctx, node: NodeId) -> Descriptor {
        g.change_root(node);
        self.label_at(g, ctx, node)
    }

    fn label_at(&self, g: &mut Digraph, ctx: &mut Ctx, node: NodeId) -> Descriptor {
        match self {
            Configuration::Tetrahedral(t) => t.label_at(g, ctx, node),
            Configuration::Sp2Bond(b) => b.label_at(g, ctx, node),
            Configuration::Allene(a) => a.label_at(g, ctx, node),
        }
    }
}

impl Tetrahedral {
    fn label_at(&self, g: &mut Digraph, ctx: &mut Ctx, node: NodeId) -> Descriptor {
        let mut edges = g.edges_of(node);
        if edges.len() < 3 {
            return Descriptor::Ns;
        }

        let sorter = Sorter::Composite;
        let mut priority = prioritize(g, ctx, &sorter, node, &mut edges, true);

        if !priority.unique {
            if edges.len() != 4 {
                return Descriptor::Unknown;
            }
            // With only the constitutional rules in play there is nothing left
            // to try; the caller will come back with the full set.
            if ctx.master.len() == 3 {
                return Descriptor::Unknown;
            }
            priority = match self.break_symmetry(g, ctx, &sorter, node, &mut edges) {
                Some(priority) => priority,
                None => return Descriptor::Unknown,
            };
            if !priority.unique {
                return Descriptor::Unknown;
            }
        }

        // The substituent atoms in priority order. Bond duplicates are phantom
        // copies of an atom that is already present, so they never occupy a
        // carrier slot.
        let mut ordered = [PHANTOM; 4];
        let mut index = 0;
        for &edge in &edges {
            let end = g.edge(edge).end;
            if g.node(end).is_set(BOND_DUPLICATE) {
                continue;
            }
            if index >= 4 {
                break;
            }
            ordered[index] = g.node(end).atom;
            index += 1;
        }
        // A three-coordinate centre's vacant vertex ranks last, and is the
        // focus itself, exactly as in `carriers`.
        if index < 4 {
            ordered[index] = self.focus;
        }

        let parity = parity4(&ordered, &self.carriers);
        if parity == 0 {
            return Descriptor::Unknown;
        }

        // An odd permutation between the stored reference order and the CIP
        // priority order reverses the apparent winding.
        let clockwise = self.clockwise ^ (parity == 1);

        match (clockwise, priority.pseudo_asymmetric) {
            (true, false) => Descriptor::R,
            (true, true) => Descriptor::LowerR,
            (false, false) => Descriptor::S,
            (false, true) => Descriptor::LowerS,
        }
    }

    /// Last resort for a centre whose substituents the rules could not order:
    /// nominate one of them as a reference for rule 6 and sort again.
    ///
    /// Two equivalence classes means a simple mirror pair, and nominating one
    /// member is enough. One class means an S4-symmetric centre, where the two
    /// possible nominations must agree — if they give opposite windings, the
    /// centre genuinely has no descriptor.
    fn break_symmetry(
        &self,
        g: &mut Digraph,
        ctx: &mut Ctx,
        sorter: &Sorter,
        node: NodeId,
        edges: &mut Vec<EdgeId>,
    ) -> Option<Priority> {
        let partition = get_groups(g, ctx, sorter, node, edges);
        match partition.len() {
            2 => {
                let reference = g.node(g.edge(edges[1]).end).atom;
                g.rule6_ref = Some(reference);
                let priority = prioritize(g, ctx, sorter, node, edges, true);
                g.rule6_ref = None;
                Some(priority)
            }
            1 => {
                let reference = g.node(g.edge(edges[0]).end).atom;
                g.rule6_ref = Some(reference);
                prioritize(g, ctx, sorter, node, edges, true);
                let first: Vec<EdgeId> = edges.clone();

                let reference = g.node(g.edge(edges[1]).end).atom;
                g.rule6_ref = Some(reference);
                let priority = prioritize(g, ctx, sorter, node, edges, true);
                g.rule6_ref = None;

                if parity4(&first_four(&first), &first_four(edges)) == 1 {
                    return None;
                }
                Some(priority)
            }
            _ => Some(prioritize(g, ctx, sorter, node, edges, true)),
        }
    }
}

/// The first four edges as a fixed-size array, so two orderings of the same
/// edge set can be compared by [`parity4`].
fn first_four(edges: &[EdgeId]) -> [usize; 4] {
    let mut result = [PHANTOM; 4];
    for (slot, &edge) in result.iter_mut().zip(edges.iter()) {
        *slot = edge;
    }
    result
}

impl Sp2Bond {
    fn label_at(&self, g: &mut Digraph, ctx: &mut Ctx, root1: NodeId) -> Descriptor {
        let constitutional = ctx.master.len() == 3;

        let edges = g.edges_of(root1);
        let Some(internal) = self.find_internal_edge(g, &edges) else {
            return Descriptor::Unknown;
        };
        let root2 = g.edge(internal).other(root1);

        let mut edges1: Vec<EdgeId> = g
            .edges_of(root1)
            .into_iter()
            .filter(|&e| !self.is_internal_edge(g, e))
            .collect();
        let mut edges2: Vec<EdgeId> = g
            .edges_of(root2)
            .into_iter()
            .filter(|&e| !self.is_internal_edge(g, e))
            .collect();

        let mut carriers = self.carriers;
        let mut cis = self.cis;
        // The digraph may be rooted at the far end of the bond, in which case
        // the two carriers swap roles.
        if g.node(root1).atom == self.other {
            carriers.swap(0, 1);
        }

        let sorter = Sorter::Composite;

        g.change_root(root1);
        let priority1 = prioritize(g, ctx, &sorter, root1, &mut edges1, true);
        if !priority1.unique && !constitutional {
            return Descriptor::Unknown;
        }
        // If the higher-priority substituent on this end is not the one the
        // stored geometry was expressed against, the sense flips.
        if edges1.len() > 1 && carriers[0] != g.node(g.edge(edges1[0]).end).atom {
            cis = !cis;
        }

        g.change_root(root2);
        let priority2 = prioritize(g, ctx, &sorter, root2, &mut edges2, true);
        if !priority2.unique || !priority1.unique {
            return Descriptor::Unknown;
        }
        if edges2.len() > 1 && carriers[1] != g.node(g.edge(edges2[0]).end).atom {
            cis = !cis;
        }

        // Exactly one end being pseudo-asymmetric makes the bond itself only
        // pseudo-stereogenic, which is the lowercase seqCis/seqTrans case.
        let pseudo = priority1.pseudo_asymmetric != priority2.pseudo_asymmetric;
        match (cis, pseudo) {
            (true, true) => Descriptor::SeqCis,
            (true, false) => Descriptor::Z,
            (false, true) => Descriptor::SeqTrans,
            (false, false) => Descriptor::E,
        }
    }

    fn is_internal_edge(&self, g: &Digraph, edge: EdgeId) -> bool {
        let beg = g.node(g.edge(edge).beg).atom;
        let end = g.node(g.edge(edge).end).atom;
        (beg == self.focus && end == self.other) || (beg == self.other && end == self.focus)
    }

    fn find_internal_edge(&self, g: &Digraph, edges: &[EdgeId]) -> Option<EdgeId> {
        edges
            .iter()
            .copied()
            .find(|&e| {
                !g.node(g.edge(e).beg).is_duplicate()
                    && !g.node(g.edge(e).end).is_duplicate()
                    && self.is_internal_edge(g, e)
            })
    }
}



impl Allene {
    fn label_at(&self, g: &mut Digraph, ctx: &mut Ctx, node: NodeId) -> Descriptor {
        // The two ends' substituents, kept apart. They are ranked *within*
        // each end rather than all four together: an axially chiral system is
        // read along its axis, so the pair nearest the viewer takes the first
        // two places whatever their own priority, and only their order between
        // themselves is a CIP question. Ranking all four together would also
        // stall on 2,3-pentadiene, where the two methyls are indistinguishable
        // from one another and so are the two hydrogens, even though the
        // molecule is perfectly chiral.
        //
        // Which end is "near" does not matter: reading from the other side
        // swaps both pairs at once, an even permutation, so the descriptor is
        // the same either way. The ends are taken in the order of
        // `self.terminals`, which is sorted, so the result is deterministic.
        let mut ordered = [PHANTOM; 4];
        let mut filled = 0;

        for &terminal_atom in &self.terminals {
            let mut pair: Vec<EdgeId> = Vec::with_capacity(2);
            for edge in g.edges_of(node) {
                let terminal = g.edge(edge).end;
                if g.node(terminal).is_duplicate() || g.node(terminal).atom != terminal_atom {
                    continue;
                }
                for outward in g.edges_of(terminal) {
                    if !g.edge(outward).is_beg(terminal) {
                        continue;
                    }
                    let end = g.edge(outward).end;
                    if g.node(end).is_duplicate() || g.node(end).atom == self.centre {
                        continue;
                    }
                    pair.push(outward);
                }
            }
            if pair.len() != 2 {
                return Descriptor::Ns;
            }
            // Two substituents the rules cannot separate leave that end with
            // no orientation, and the axis with no configuration.
            if rank_independent_edges(g, ctx, &mut pair).is_none() {
                return Descriptor::Unknown;
            }
            for &edge in &pair {
                if filled >= 4 {
                    return Descriptor::Unknown;
                }
                ordered[filled] = g.node(g.edge(edge).end).atom;
                filled += 1;
            }
        }
        if filled != 4 {
            return Descriptor::Ns;
        }

        let parity = parity4(&ordered, &self.carriers);
        if parity == 0 {
            return Descriptor::Unknown;
        }
        // As for a tetrahedral centre, an odd permutation between the stored
        // reference order and the priority order reverses the winding.
        if self.clockwise ^ (parity == 1) { Descriptor::R } else { Descriptor::S }
    }
}
