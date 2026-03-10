//! Subgraph **monomorphism** search — every way a pattern can be laid over a
//! target.
//!
//! Monomorphism, not isomorphism: each bond the pattern asks for must exist in
//! the target, but the target is free to have *extra* bonds between the
//! matched atoms. That is what "does this substructure occur" means to a
//! chemist. A benzene ring occurs in naphthalene; a three-atom chain occurs in
//! cyclopropane six times over, even though its two ends happen to be bonded
//! there and the pattern says nothing about that.
//!
//! The distinction is invisible for acyclic patterns and decisive for rings
//! smaller than the pattern path, which is why an induced search passes most
//! tests and then silently returns nothing for `CCC` in cyclopropane.
//! `petgraph::algo::subgraph_isomorphisms_iter` is documented as
//! node-*induced* and offers no monomorphic variant, so the search is written
//! out here.
//!
//! Everything about what *matches* is left to the caller's closures, so the
//! same search serves exact structure queries (label equality) and SMARTS
//! (expression evaluation).

/// A pattern to look for, plus how to recognise it in the target.
///
/// - `neighbours(target_node)` lists the target nodes adjacent to one.
/// - `node_match(pattern_node, target_node)` decides whether a node may stand
///   for another.
/// - `edge_match(pattern_edge, target_a, target_b)` decides whether the
///   pattern's edge is satisfied between two target nodes. It is responsible
///   for rejecting the case where they are not joined at all.
pub(crate) struct Monomorphism<Neighbours, NodeMatch, EdgeMatch> {
    pub pattern_node_count: usize,
    /// Owned so that callers can build it on the fly; one small allocation per
    /// search, against a search that visits the whole target.
    pub pattern_edges: Vec<(usize, usize)>,
    pub target_node_count: usize,
    pub neighbours: Neighbours,
    pub node_match: NodeMatch,
    pub edge_match: EdgeMatch,
}

impl<Neighbours, NodeMatch, EdgeMatch> Monomorphism<Neighbours, NodeMatch, EdgeMatch>
where
    Neighbours: Fn(usize) -> Vec<usize>,
    NodeMatch: Fn(usize, usize) -> bool,
    EdgeMatch: Fn(usize, usize, usize) -> bool,
{
    /// Every mapping, each indexed by pattern node and holding the target node
    /// it was assigned.
    pub(crate) fn all(&self) -> Vec<Vec<usize>> {
        let mut matches = Vec::new();
        self.run(&mut |assignment| {
            matches.push(assignment.to_vec());
            true
        });
        matches
    }

    /// Whether the pattern occurs at all. Stops at the first mapping.
    pub(crate) fn any(&self) -> bool {
        let mut found = false;
        self.run(&mut |_| {
            found = true;
            false
        });
        found
    }

    /// Drives the search, reporting each mapping to `found`, which returns
    /// whether to keep looking.
    pub(crate) fn run(&self, found: &mut impl FnMut(&[usize]) -> bool) {
        if self.pattern_node_count == 0 || self.pattern_node_count > self.target_node_count {
            return;
        }
        let order = self.search_order();
        let mut assignment = vec![usize::MAX; self.pattern_node_count];
        let mut used = vec![false; self.target_node_count];
        self.extend(&order, 0, &mut assignment, &mut used, found);
    }

    /// The order pattern nodes are assigned in: each one after the first is
    /// adjacent to something already placed, so candidates come from a mapped
    /// neighbour's neighbours rather than from the whole target. A pattern in
    /// several pieces simply starts a new run for each.
    fn search_order(&self) -> Vec<usize> {
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); self.pattern_node_count];
        for &(first, second) in &self.pattern_edges {
            if first < self.pattern_node_count && second < self.pattern_node_count {
                adjacency[first].push(second);
                adjacency[second].push(first);
            }
        }

        let mut order = Vec::with_capacity(self.pattern_node_count);
        let mut placed = vec![false; self.pattern_node_count];
        for seed in 0..self.pattern_node_count {
            if placed[seed] {
                continue;
            }
            placed[seed] = true;
            order.push(seed);
            let mut position = order.len() - 1;
            while position < order.len() {
                let current = order[position];
                position += 1;
                for &neighbour in &adjacency[current] {
                    if !placed[neighbour] {
                        placed[neighbour] = true;
                        order.push(neighbour);
                    }
                }
            }
        }
        order
    }

    /// Assigns `order[depth]` every way it can be, then recurses. Returns
    /// whether the search should continue.
    fn extend(
        &self,
        order: &[usize],
        depth: usize,
        assignment: &mut Vec<usize>,
        used: &mut Vec<bool>,
        found: &mut impl FnMut(&[usize]) -> bool,
    ) -> bool {
        if depth == order.len() {
            return found(assignment);
        }
        let pattern_node = order[depth];

        for target_node in self.candidates(pattern_node, assignment) {
            if used[target_node] || !(self.node_match)(pattern_node, target_node) {
                continue;
            }
            if !self.edges_satisfied(pattern_node, target_node, assignment) {
                continue;
            }
            assignment[pattern_node] = target_node;
            used[target_node] = true;
            let keep_going = self.extend(order, depth + 1, assignment, used, found);
            used[target_node] = false;
            assignment[pattern_node] = usize::MAX;
            if !keep_going {
                return false;
            }
        }
        true
    }

    /// Target nodes worth trying: the neighbours of an already-assigned
    /// pattern neighbour if there is one, otherwise the whole target.
    fn candidates(&self, pattern_node: usize, assignment: &[usize]) -> Vec<usize> {
        for &(first, second) in &self.pattern_edges {
            let anchor = if first == pattern_node {
                second
            } else if second == pattern_node {
                first
            } else {
                continue;
            };
            if anchor < assignment.len() && assignment[anchor] != usize::MAX {
                return (self.neighbours)(assignment[anchor]);
            }
        }
        (0..self.target_node_count).collect()
    }

    /// Every pattern edge from `pattern_node` to an already-assigned node must
    /// be satisfied. Extra target edges are not checked — that is exactly what
    /// makes this a monomorphism rather than an induced isomorphism.
    fn edges_satisfied(
        &self,
        pattern_node: usize,
        target_node: usize,
        assignment: &[usize],
    ) -> bool {
        for (edge_index, &(first, second)) in self.pattern_edges.iter().enumerate() {
            let other = if first == pattern_node {
                second
            } else if second == pattern_node {
                first
            } else {
                continue;
            };
            if other >= assignment.len() {
                continue;
            }
            let mapped = assignment[other];
            if mapped == usize::MAX {
                continue;
            }
            if !(self.edge_match)(edge_index, target_node, mapped) {
                return false;
            }
        }
        true
    }
}
