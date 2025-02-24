use rustc_hash::FxHashMap;
use std::hash::Hash;

/// The minimal interface [`search`] needs: a labelled graph on compact
/// `0..len()` vertex indices. Implemented once for the plain atom graph
/// ([`super::graph::Graph`] — vertex = atom, edge label = bond order) and
/// once for its line graph ([`super::line_graph::LineGraph`] — vertex =
/// bond, edge label = shared atom), so the same search (and its
/// branch-and-bound bound) drives both maximum-common-*induced*-subgraph and
/// maximum-common-*edge*-subgraph search without duplicating the algorithm.
pub(super) trait McsplitGraph {
    type NodeLabel: Copy + Eq + Hash;
    type EdgeLabel: Copy + Eq + Hash;

    fn len(&self) -> usize;
    fn node_label(&self, index: usize) -> Self::NodeLabel;
    /// The label of the edge between `a` and `b`, or `None` if they aren't
    /// adjacent.
    fn edge_label(&self, a: usize, b: usize) -> Option<Self::EdgeLabel>;
    /// How many vertices `index` is adjacent to. Used only to order the
    /// search (see [`select_bidomain`]); it never affects which mappings are
    /// considered valid.
    fn degree(&self, index: usize) -> usize;
}

/// A "bidomain": a set of not-yet-mapped vertices in each graph that are,
/// given the vertex pairs already chosen, still structurally
/// indistinguishable — every vertex in `left` has the same relationship
/// (unrelated, or related with the same edge label) to every already-mapped
/// vertex as every other vertex in `left` does, and likewise for `right`.
/// Any vertex in `left` could still be validly paired with any vertex in
/// `right`.
#[derive(Clone)]
struct Bidomain {
    left: Vec<usize>,
    right: Vec<usize>,
}

/// Searches for a maximum common induced subgraph of `g1` and `g2`,
/// returning the chosen vertex correspondence as compact-index pairs.
///
/// At each step, one not-yet-decided vertex `v` (from some bidomain) is
/// either mapped to a candidate `w` from that same bidomain's `right` side,
/// or left unmapped — for every candidate tried, every remaining bidomain is
/// refined by splitting on matching-edge-label-to-`v`/`w`, which is what
/// lets branch-and-bound prune on the *sum of bidomain sizes* (a much
/// tighter bound than a flat candidate count) instead of exploring every
/// possible pairing blindly.
pub(super) fn search<G: McsplitGraph>(g1: &G, g2: &G) -> Vec<(usize, usize)> {
    let domains = initial_bidomains(g1, g2);
    let mut current = Vec::new();
    let mut best = Vec::new();
    recurse(g1, g2, domains, &mut current, &mut best);
    best
}

/// Groups every vertex of each graph by its node label — the coarsest
/// possible bidomains, before any adjacency information is known.
fn initial_bidomains<G: McsplitGraph>(g1: &G, g2: &G) -> Vec<Bidomain> {
    let mut by_label_1: FxHashMap<G::NodeLabel, Vec<usize>> = FxHashMap::default();
    for i in 0..g1.len() {
        by_label_1.entry(g1.node_label(i)).or_default().push(i);
    }
    let mut by_label_2: FxHashMap<G::NodeLabel, Vec<usize>> = FxHashMap::default();
    for i in 0..g2.len() {
        by_label_2.entry(g2.node_label(i)).or_default().push(i);
    }

    by_label_1
        .into_iter()
        .filter_map(|(label, left)| {
            by_label_2
                .get(&label)
                .map(|right| Bidomain { left, right: right.clone() })
        })
        .collect()
}

/// Picks which bidomain to branch on next: the one with the smallest
/// `max(|left|, |right|)`.
///
/// Branching on the most-constrained bidomain first is the standard McSplit
/// selection rule, and it is *not* optional in practice — with arbitrary
/// selection (branch on whichever bidomain comes first), molecules
/// containing long unbranched alkyl chains blow up combinatorially, because
/// every C-C bond along such a chain carries an identical label and so sits
/// in one large bidomain that gets explored exhaustively before anything
/// forces a decision. Picking the smallest first instead resolves the
/// genuinely constrained parts of the structure early, which tightens the
/// bound (and so prunes the chain's permutations) long before that happens.
///
/// This affects only search *order*: every mapping remains reachable, so the
/// result is unchanged.
fn select_bidomain<G: McsplitGraph>(g1: &G, domains: &[Bidomain]) -> Option<usize> {
    domains
        .iter()
        .enumerate()
        .filter(|(_, d)| !d.left.is_empty() && !d.right.is_empty())
        .min_by_key(|(_, d)| {
            let size = d.left.len().max(d.right.len());
            // Tie-break towards the bidomain holding the highest-degree
            // vertex: mapping a hub vertex constrains more of what's left
            // than mapping a peripheral one.
            let max_degree = d.left.iter().map(|&v| g1.degree(v)).max().unwrap_or(0);
            (size, std::cmp::Reverse(max_degree))
        })
        .map(|(i, _)| i)
}

fn recurse<G: McsplitGraph>(
    g1: &G,
    g2: &G,
    mut domains: Vec<Bidomain>,
    current: &mut Vec<(usize, usize)>,
    best: &mut Vec<(usize, usize)>,
) {
    let bound: usize =
        current.len() + domains.iter().map(|d| d.left.len().min(d.right.len())).sum::<usize>();
    if bound <= best.len() {
        return;
    }

    let Some(bd_index) = select_bidomain(g1, &domains) else {
        if current.len() > best.len() {
            *best = current.clone();
        }
        return;
    };

    // Within the chosen bidomain, branch on its highest-degree vertex, for
    // the same reason the tie-break above prefers one.
    let v_pos = domains[bd_index]
        .left
        .iter()
        .enumerate()
        .max_by_key(|&(_, &v)| g1.degree(v))
        .map(|(i, _)| i)
        .expect("select_bidomain only returns non-empty bidomains");
    let v = domains[bd_index].left[v_pos];

    let mut remaining_left = domains[bd_index].left.clone();
    remaining_left.swap_remove(v_pos);

    let candidates = domains[bd_index].right.clone();

    for &w in &candidates {
        let mut new_domains = Vec::with_capacity(domains.len());
        for (i, d) in domains.iter().enumerate() {
            let left = if i == bd_index { &remaining_left[..] } else { &d.left[..] };
            split_into(g1, g2, left, &d.right, v, w, i == bd_index, &mut new_domains);
        }
        current.push((v, w));
        recurse(g1, g2, new_domains, current, best);
        current.pop();
    }

    // Leave v unmapped: it's no longer a candidate, but everything else
    // about the current bidomains is unchanged.
    domains[bd_index].left = remaining_left;
    if domains[bd_index].left.is_empty() {
        domains.remove(bd_index);
    }
    recurse(g1, g2, domains, current, best);
}

/// Splits one bidomain's `left`/`right` sets on their relationship to the
/// just-chosen pair `(v, w)`: vertices unrelated to `v`/`w` in the same way
/// (same edge label, or both unrelated) stay grouped together as a new
/// (smaller) bidomain; vertices that relate differently on the two sides
/// (e.g. adjacent to `v` in `g1` but not to `w` in `g2`) can never be
/// validly paired given this choice, so they're simply dropped from further
/// consideration.
fn split_into<G: McsplitGraph>(
    g1: &G,
    g2: &G,
    left: &[usize],
    right: &[usize],
    v: usize,
    w: usize,
    exclude_w: bool,
    out: &mut Vec<Bidomain>,
) {
    let mut left_buckets: FxHashMap<Option<G::EdgeLabel>, Vec<usize>> = FxHashMap::default();
    for &x in left {
        left_buckets.entry(g1.edge_label(v, x)).or_default().push(x);
    }

    let mut right_buckets: FxHashMap<Option<G::EdgeLabel>, Vec<usize>> = FxHashMap::default();
    for &y in right {
        if exclude_w && y == w {
            continue;
        }
        right_buckets.entry(g2.edge_label(w, y)).or_default().push(y);
    }

    for (key, left_vertices) in left_buckets {
        if let Some(right_vertices) = right_buckets.get(&key) {
            out.push(Bidomain { left: left_vertices, right: right_vertices.clone() });
        }
    }
}
