//! Maximum matching in a general graph, by Edmonds' blossom algorithm.
//!
//! Kekulization is a matching problem and nothing less will do. A greedy search
//! that gives up when an atom has no free partner does not find the largest set
//! of π bonds, and the difference is not cosmetic: every atom it fails to pair
//! becomes an unpaired electron, so a greedy answer invents radicals that a
//! maximum matching would have paired elsewhere. Seventeen thousand molecules
//! in the test corpus grew radicals that way.
//!
//! The size of a maximum matching is the same for every maximum matching, so
//! the *number* of unpaired electrons a molecule is left with is a property of
//! the molecule rather than of the search. That is what makes radical counts
//! canonical, and it is why this has to be exact rather than nearly right.
//!
//! Odd cycles are what makes general-graph matching hard, and rings are exactly
//! that. Blossom contraction is the standard answer: an odd cycle reachable by
//! an alternating path is contracted to a single vertex, the search continues
//! on the smaller graph, and the path is expanded again on the way out.

/// A maximum matching of `vertex_count` vertices with the given edges.
///
/// Returns, for each vertex, the vertex it is matched to, or `None`.
/// Deterministic: vertices are scanned in index order, so the caller decides
/// the outcome among equally large matchings by choosing that order.
pub fn maximum_matching(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    let mut adjacency = vec![Vec::new(); vertex_count];
    for &(a, b) in edges {
        if a < vertex_count && b < vertex_count && a != b {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }
    for neighbours in adjacency.iter_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    let mut matched: Vec<Option<usize>> = vec![None; vertex_count];

    // A greedy start costs nothing and saves augmenting for the easy pairs.
    for vertex in 0..vertex_count {
        if matched[vertex].is_some() {
            continue;
        }
        if let Some(&free) = adjacency[vertex]
            .iter()
            .find(|&&neighbour| matched[neighbour].is_none())
        {
            matched[vertex] = Some(free);
            matched[free] = Some(vertex);
        }
    }

    // A vertex with no edge has no augmenting path, and searching from one
    // still costs a full sweep of the graph. Callers hand this the whole
    // molecule while the edges cover only its aromatic bonds, so almost every
    // vertex is isolated and almost every search was wasted. Greedy leaves
    // nothing to search for in a ring that alternates cleanly, which is most of
    // them, and then the search state is never built at all.
    let roots: Vec<usize> = (0..vertex_count)
        .filter(|&vertex| matched[vertex].is_none() && !adjacency[vertex].is_empty())
        .collect();
    if roots.is_empty() {
        return matched;
    }

    let mut state = Search {
        vertex_count,
        adjacency: &adjacency,
        parent: vec![usize::MAX; vertex_count],
        base: (0..vertex_count).collect(),
        in_queue: vec![false; vertex_count],
        blossom: vec![false; vertex_count],
        queue: Vec::with_capacity(vertex_count),
    };

    for root in roots {
        if matched[root].is_some() {
            continue;
        }
        if let Some(last) = state.find_augmenting_path(root, &matched) {
            augment(&mut matched, &state.parent, last);
        }
    }
    matched
}

struct Search<'a> {
    vertex_count: usize,
    adjacency: &'a [Vec<usize>],
    parent: Vec<usize>,
    base: Vec<usize>,
    in_queue: Vec<bool>,
    blossom: Vec<bool>,
    queue: Vec<usize>,
}

impl Search<'_> {
    /// The lowest common ancestor of two vertices in the alternating forest.
    fn lowest_common_ancestor(
        &self,
        matched: &[Option<usize>],
        mut a: usize,
        mut b: usize,
    ) -> usize {
        let mut seen = vec![false; self.vertex_count];
        loop {
            a = self.base[a];
            seen[a] = true;
            match matched[a] {
                Some(partner) => a = self.parent[partner],
                None => break,
            }
        }
        loop {
            b = self.base[b];
            if seen[b] {
                return b;
            }
            b = self.parent[matched[b].expect("only matched vertices continue the path")];
        }
    }

    /// Marks every vertex on the odd cycle between `v` and the blossom's base.
    fn mark_path(
        &mut self,
        matched: &[Option<usize>],
        mut v: usize,
        base: usize,
        mut child: usize,
    ) {
        while self.base[v] != base {
            let partner = matched[v].expect("a blossom alternates, so this is matched");
            self.blossom[self.base[v]] = true;
            self.blossom[self.base[partner]] = true;
            self.parent[v] = child;
            child = partner;
            v = self.parent[partner];
        }
    }

    /// Grows the alternating tree from `root`, returning the far end of an
    /// augmenting path if one is found.
    fn find_augmenting_path(&mut self, root: usize, matched: &[Option<usize>]) -> Option<usize> {
        self.parent.iter_mut().for_each(|p| *p = usize::MAX);
        self.in_queue.iter_mut().for_each(|q| *q = false);
        for (vertex, base) in self.base.iter_mut().enumerate() {
            *base = vertex;
        }
        self.queue.clear();
        self.queue.push(root);
        self.in_queue[root] = true;

        let mut head = 0;
        while head < self.queue.len() {
            let v = self.queue[head];
            head += 1;

            for index in 0..self.adjacency[v].len() {
                let to = self.adjacency[v][index];

                // Same blossom, or the edge we came in on: nothing to learn.
                if self.base[v] == self.base[to] || matched[v] == Some(to) {
                    continue;
                }

                if to == root || (matched[to].is_some() && self.parent[matched[to].unwrap()] != usize::MAX)
                {
                    // An odd cycle: contract it and carry on.
                    let base = self.lowest_common_ancestor(matched, v, to);
                    self.blossom.iter_mut().for_each(|b| *b = false);
                    self.mark_path(matched, v, base, to);
                    self.mark_path(matched, to, base, v);
                    for vertex in 0..self.vertex_count {
                        if self.blossom[self.base[vertex]] {
                            self.base[vertex] = base;
                            if !self.in_queue[vertex] {
                                self.in_queue[vertex] = true;
                                self.queue.push(vertex);
                            }
                        }
                    }
                } else if self.parent[to] == usize::MAX {
                    self.parent[to] = v;
                    match matched[to] {
                        None => return Some(to), // augmenting path found
                        Some(partner) => {
                            if !self.in_queue[partner] {
                                self.in_queue[partner] = true;
                                self.queue.push(partner);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Flips the matching along the augmenting path ending at `last`.
fn augment(matched: &mut [Option<usize>], parent: &[usize], last: usize) {
    let mut v = last;
    while v != usize::MAX {
        let previous = parent[v];
        let next = matched[previous];
        matched[v] = Some(previous);
        matched[previous] = Some(v);
        v = match next {
            Some(vertex) => vertex,
            None => break,
        };
    }
}
