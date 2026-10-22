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
/// - `neighbours(target_node, out)` appends the target nodes adjacent to one
///   to `out`, which the search reuses rather than allocating per step.
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
    Neighbours: Fn(usize, &mut Vec<usize>),
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
        let plan = self.plan();
        let mut search = Search::new(self, &plan);
        search.extend(0, found);
    }

    /// For every target node, whether some mapping with pattern node 0 held
    /// there passes `accept` -- the question a recursive SMARTS asks.
    ///
    /// Each root stops at its first accepted mapping, which is never more work
    /// than listing every mapping and reading the roots off, and on a
    /// symmetric pattern it is far less.
    pub(crate) fn rooted_matches(&self, accept: &mut impl FnMut(&[usize]) -> bool) -> Vec<bool> {
        let mut roots = vec![false; self.target_node_count];
        if self.pattern_node_count == 0 || self.pattern_node_count > self.target_node_count {
            return roots;
        }
        // The search order always starts from pattern node 0, so holding it
        // fixed is just starting one level down.
        let plan = self.plan();
        debug_assert_eq!(plan.first().map(|step| step.node), Some(0));
        let mut search = Search::new(self, &plan);
        for (target, root) in roots.iter_mut().enumerate() {
            if !(self.node_match)(0, target) {
                continue;
            }
            search.assignment[0] = target;
            search.used[target] = true;
            search.extend(1, &mut |assignment| {
                *root = accept(assignment);
                !*root
            });
            search.used[target] = false;
            search.assignment[0] = usize::MAX;
        }
        roots
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

    /// What each depth of the search does, worked out once from the order.
    ///
    /// The nodes assigned before depth `d` are always `order[..d]`, so which
    /// neighbour a node's candidates come from and which of its edges reach an
    /// already-placed node are the same every time that depth is entered.
    /// Both are taken from the edges in their given order, so the candidates
    /// and the checks come in the order a search that worked them out afresh
    /// would use.
    fn plan(&self) -> Vec<Step> {
        let mut placed = vec![false; self.pattern_node_count];
        self.search_order()
            .into_iter()
            .map(|node| {
                let mut anchor = None;
                let mut back_edges = Vec::new();
                for (edge_index, &(first, second)) in self.pattern_edges.iter().enumerate() {
                    let other = if first == node {
                        second
                    } else if second == node {
                        first
                    } else {
                        continue;
                    };
                    if other >= self.pattern_node_count || !placed[other] {
                        continue;
                    }
                    anchor.get_or_insert(other);
                    back_edges.push((edge_index, other));
                }
                placed[node] = true;
                Step { node, anchor, back_edges }
            })
            .collect()
    }
}

/// One depth of the search: the pattern node it assigns, the placed node its
/// candidates are the neighbours of (or `None` to try the whole target), and
/// the edges back to placed nodes a candidate has to satisfy.
struct Step {
    node: usize,
    anchor: Option<usize>,
    back_edges: Vec<(usize, usize)>,
}

/// The state of one search: the mapping so far, which target nodes it uses,
/// and a candidate buffer per depth so that no step allocates.
struct Search<'s, Neighbours, NodeMatch, EdgeMatch> {
    problem: &'s Monomorphism<Neighbours, NodeMatch, EdgeMatch>,
    plan: &'s [Step],
    assignment: Vec<usize>,
    used: Vec<bool>,
    candidates: Vec<Vec<usize>>,
}

impl<'s, Neighbours, NodeMatch, EdgeMatch> Search<'s, Neighbours, NodeMatch, EdgeMatch>
where
    Neighbours: Fn(usize, &mut Vec<usize>),
    NodeMatch: Fn(usize, usize) -> bool,
    EdgeMatch: Fn(usize, usize, usize) -> bool,
{
    fn new(problem: &'s Monomorphism<Neighbours, NodeMatch, EdgeMatch>, plan: &'s [Step]) -> Self {
        Self {
            problem,
            plan,
            assignment: vec![usize::MAX; problem.pattern_node_count],
            used: vec![false; problem.target_node_count],
            candidates: vec![Vec::new(); plan.len()],
        }
    }

    /// Assigns `plan[depth]`'s node every way it can be, then recurses.
    /// Returns whether the search should continue.
    ///
    /// Extra target edges are not checked -- that is exactly what makes this
    /// a monomorphism rather than an induced isomorphism.
    fn extend(&mut self, depth: usize, found: &mut impl FnMut(&[usize]) -> bool) -> bool {
        if depth == self.plan.len() {
            return found(&self.assignment);
        }
        let step = &self.plan[depth];
        let mut candidates = std::mem::take(&mut self.candidates[depth]);
        candidates.clear();
        match step.anchor {
            Some(anchor) => (self.problem.neighbours)(self.assignment[anchor], &mut candidates),
            None => candidates.extend(0..self.problem.target_node_count),
        }

        let mut keep_going = true;
        for &target in &candidates {
            if self.used[target] || !(self.problem.node_match)(step.node, target) {
                continue;
            }
            if !step.back_edges.iter().all(|&(edge_index, other)| {
                (self.problem.edge_match)(edge_index, target, self.assignment[other])
            }) {
                continue;
            }
            self.assignment[step.node] = target;
            self.used[target] = true;
            keep_going = self.extend(depth + 1, found);
            self.used[target] = false;
            self.assignment[step.node] = usize::MAX;
            if !keep_going {
                break;
            }
        }
        self.candidates[depth] = candidates;
        keep_going
    }
}
