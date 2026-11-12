use super::types::{AlgorithmState, Clique, ModularProductGraph};
use bit_set::BitSet;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Dfs, Walker};
use std::rc::Rc;

/// Connected components of a modular product graph (only counting the
/// "positive" edges added by [`super::modular_product::modular_graph_product`]).
fn find_connected_components(graph: &ModularProductGraph) -> Vec<Vec<NodeIndex>> {
    let mut visited = vec![false; graph.graph.node_count()];
    let mut components = Vec::new();

    for start in graph.graph.node_indices() {
        if visited[start.index()] {
            continue;
        }
        let component: Vec<NodeIndex> = Dfs::new(&graph.graph, start)
            .iter(&graph.graph)
            .inspect(|node| visited[node.index()] = true)
            .collect();
        if !component.is_empty() {
            components.push(component);
        }
    }

    components
}

/// Splits a modular product graph into its connected components, discarding
/// components too small to possibly beat `maximum`, and re-derives each
/// component's edges directly from the two line graphs. This adds back
/// "type-0" edges between vertex pairs that are mutually non-adjacent in
/// *both* line graphs (labelled `None`) — pairs of bonds independent of each
/// other in both molecules, which can still coexist in the same common
/// subgraph (e.g. two separate matching fragments) even though they weren't
/// connected by the plain modular product. Components are returned largest
/// first.
pub(super) fn find_blue_connected_components(
    graph: &ModularProductGraph,
    maximum: usize,
) -> Vec<ModularProductGraph> {
    let components = find_connected_components(graph);
    let g1 = &graph.g1;
    let g2 = &graph.g2;

    let mut results: Vec<ModularProductGraph> = Vec::new();

    for curr_vertices in &components {
        if curr_vertices.len() < maximum {
            continue;
        }

        let mut new_product = ModularProductGraph::new(Rc::clone(g1), Rc::clone(g2));

        for &vertex in curr_vertices {
            let bond_order = *graph.graph.node_weight(vertex).unwrap();
            let v = new_product.graph.add_node(bond_order);
            new_product.node_mapping.insert(v, graph.node_mapping[&vertex]);
            if let Some(&clique_val) = graph.clique_mapping.get(&vertex) {
                new_product.clique_mapping.insert(v, clique_val);
            }
        }

        let new_nodes: Vec<NodeIndex> = new_product.graph.node_indices().collect();
        for node1 in new_nodes.iter().copied() {
            for node2 in new_nodes.iter().copied() {
                if node1 <= node2 {
                    continue;
                }

                let (n1_g1, n1_g2) = new_product.node_mapping[&node1];
                let (n2_g1, n2_g2) = new_product.node_mapping[&node2];

                if n1_g1 == n2_g1 || n1_g2 == n2_g2 {
                    continue;
                }

                let edge_g1 = g1.graph.find_edge(n1_g1, n2_g1);
                let edge_g2 = g2.graph.find_edge(n1_g2, n2_g2);

                if edge_g1.is_none() && edge_g2.is_none() {
                    // "Type-0" pair: independent bonds in both molecules, still
                    // free to coexist in the same common subgraph.
                    new_product.graph.add_edge(node1, node2, None);
                } else if let (Some(e1), Some(e2)) = (edge_g1, edge_g2) {
                    let atom1 = g1.graph.edge_weight(e1).unwrap();
                    let atom2 = g2.graph.edge_weight(e2).unwrap();
                    if atom1 == atom2 {
                        new_product.graph.add_edge(node1, node2, Some(*atom1));
                    }
                }
            }
        }

        results.push(new_product);
    }

    results.sort_by(|a, b| b.graph.node_count().cmp(&a.graph.node_count()));
    results
}

/// R/P/X triple for the Bron-Kerbosch search: current clique, candidates, and
/// already-excluded vertices.
struct Triple {
    clique: BitSet,
    candidates: BitSet,
    excluded: BitSet,
}

/// Explicit-stack (non-recursive) Bron-Kerbosch-with-pivoting maximum-clique
/// search over a [`ModularProductGraph`], branch-and-bound pruned against the
/// running best (`state.maximum`). Manual stack (rather than plain recursion)
/// to avoid the recursion-depth blowups large modular products can cause (see
/// cmces's `COROUTINE_SOLUTIONS.md`).
///
/// Adjacency for clique purposes is "any edge, `Some`- or `None`-labelled" —
/// a `None` edge (see [`find_blue_connected_components`]) still means two
/// vertices are compatible to sit in the same clique, it just doesn't itself
/// correspond to a matched bond. At each branch a pivot vertex `u` is chosen
/// from `candidates ∪ excluded` maximizing `|candidates ∩ N(u)|`, and only
/// `candidates \ N(u)` are branched on — the standard refinement that keeps
/// the search from blowing up on the large, near-complete, highly symmetric
/// modular products that fused-ring/heavily-symmetric molecules produce
/// (every maximal clique still gets found; a plain unpivoted search just
/// re-explores far more equivalent branches to find them).
///
/// `next()` returns (and raises `state.maximum` for) the first qualifying
/// maximal clique it finds rather than exhaustively enumerating every
/// maximal clique before returning any of them — the branch-and-bound
/// pruning above only helps later calls if the bound is tightened as soon as
/// a better clique is found, not just once at the very end.
pub(super) struct CliqueIterator {
    stack: Vec<Triple>,
    /// `adjacency[v]` = bitset of `v`'s neighbors, precomputed once so that
    /// `neighbors_within` — called on every branch, the hot path of the whole
    /// search — is a single word-level `BitSet::intersect_with` instead of
    /// `set.len()` individual `find_edge` graph lookups.
    adjacency: Vec<BitSet>,
}

impl CliqueIterator {
    pub fn new(graph: Rc<ModularProductGraph>) -> Self {
        let n = graph.graph.node_count();
        let mut candidates = BitSet::new();
        for node in 0..n {
            candidates.insert(node);
        }

        let mut adjacency = vec![BitSet::new(); n];
        for edge in graph.graph.edge_indices() {
            let (a, b) = graph.graph.edge_endpoints(edge).unwrap();
            adjacency[a.index()].insert(b.index());
            adjacency[b.index()].insert(a.index());
        }

        Self {
            stack: vec![Triple {
                clique: BitSet::new(),
                candidates,
                excluded: BitSet::new(),
            }],
            adjacency,
        }
    }

    pub fn next(&mut self, state: &mut AlgorithmState) -> Option<Clique> {
        while let Some(Triple {
            clique,
            mut candidates,
            mut excluded,
        }) = self.stack.pop()
        {
            if candidates.count() + clique.count() < state.maximum {
                continue;
            }

            if candidates.is_empty() && excluded.is_empty() {
                if clique.count() < state.maximum {
                    continue;
                }
                // A 3-vertex clique's *actual* matched-bond count can end up
                // smaller after `induced_subgraph`'s "bad triangle" splitting
                // (see `super::reconstruct`) — so it isn't a safe live bound
                // to prune on; leave `state.maximum` for the caller to raise
                // once it knows the real, post-split size. Any other size is
                // never affected by that ambiguity and is safe to use
                // immediately, which is what lets pruning kick in during the
                // same search pass instead of only on the next one.
                if !state.enumerate_all && clique.count() > state.maximum && clique.count() != 3 {
                    state.maximum = clique.count();
                }
                return Some(clique.iter().map(NodeIndex::new).collect());
            }

            let pivot = self.choose_pivot(&candidates, &excluded);
            let pivot_neighbors = pivot
                .map(|p| self.neighbors_within(&candidates, p))
                .unwrap_or_default();
            let branch: Vec<usize> = candidates
                .iter()
                .filter(|idx| !pivot_neighbors.contains(*idx))
                .collect();

            for candidate_idx in branch {
                let candidate = NodeIndex::new(candidate_idx);

                let mut new_clique = clique.clone();
                new_clique.insert(candidate_idx);

                let new_candidates = self.neighbors_within(&candidates, candidate);
                let new_excluded = self.neighbors_within(&excluded, candidate);

                self.push_stack(Triple {
                    clique: new_clique,
                    candidates: new_candidates,
                    excluded: new_excluded,
                });

                candidates.remove(candidate_idx);
                excluded.insert(candidate_idx);
            }
        }

        None
    }

    /// Picks `u ∈ candidates ∪ excluded` maximizing `|candidates ∩ N(u)|`.
    fn choose_pivot(&self, candidates: &BitSet, excluded: &BitSet) -> Option<NodeIndex> {
        candidates
            .iter()
            .chain(excluded.iter())
            .map(NodeIndex::new)
            .max_by_key(|&u| self.neighbors_within(candidates, u).count())
    }

    /// Inserts a branch keeping the stack sorted by clique size (ascending),
    /// so `stack.pop()` explores the largest partial clique first — better
    /// branch-and-bound pruning against `state.maximum`.
    fn push_stack(&mut self, value: Triple) {
        let pos = self
            .stack
            .binary_search_by(|x| x.clique.count().cmp(&value.clique.count()))
            .unwrap_or_else(|pos| pos);
        self.stack.insert(pos, value);
    }

    fn neighbors_within(&self, set: &BitSet, of: NodeIndex) -> BitSet {
        let mut result = set.clone();
        result.intersect_with(&self.adjacency[of.index()]);
        result
    }
}

/// `true` iff every element of `subset` is present in `superset`.
pub(super) fn is_subset(subset: &[usize], superset: &[usize]) -> bool {
    subset.iter().all(|item| superset.contains(item))
}

/// A fast (non-exhaustive) lower bound on the maximum clique size: from each
/// of a handful of high-degree starting vertices, greedily grow a clique by
/// repeatedly adding whichever remaining candidate has the most neighbors
/// still among the remaining candidates, and keep the best of these.
///
/// Used to seed `state.maximum` before the exhaustive search starts. Without
/// it, the very first search explores with `state.maximum == 0`, so the
/// `candidates.len() + clique.len() < state.maximum` branch-and-bound check
/// never fires — on the large, near-complete modular products that
/// symmetric/fused-ring molecules produce (see `find_blue_connected_components`'s
/// "type-0" edges), that means enumerating every maximal clique of every
/// size before any pruning kicks in, which is exponentially slower than
/// starting from a decent bound. A fixed-order (non-adaptive) greedy pass
/// tends to land on a small, mediocre clique on these graphs — recomputing
/// "most-connected-among-what's-left" at every step gets much closer to the
/// true maximum, which is what makes the seeded bound actually effective.
pub(super) fn greedy_clique_lower_bound(graph: &ModularProductGraph) -> usize {
    let n = graph.graph.node_count();
    if n == 0 {
        return 0;
    }

    let mut adjacency = vec![BitSet::new(); n];
    for edge in graph.graph.edge_indices() {
        let (a, b) = graph.graph.edge_endpoints(edge).unwrap();
        adjacency[a.index()].insert(b.index());
        adjacency[b.index()].insert(a.index());
    }

    let mut by_degree: Vec<usize> = (0..n).collect();
    by_degree.sort_by_key(|&v| std::cmp::Reverse(adjacency[v].count()));

    let mut best = 0;
    for &start in by_degree.iter().take(n.min(8)) {
        let mut candidates: BitSet = (0..n).collect();
        let mut size = 0;
        let mut next = Some(start);
        while let Some(v) = next {
            size += 1;
            candidates.intersect_with(&adjacency[v]);
            candidates.remove(v);
            next = candidates.iter().max_by_key(|&idx| {
                let mut degree_within = candidates.clone();
                degree_within.intersect_with(&adjacency[idx]);
                degree_within.count()
            });
        }
        best = best.max(size);
    }
    best
}
