use super::line_graph::molecular_graph_to_line_graph;
use super::modular_product::modular_graph_product;
use super::reconstruct::{induced_subgraph, resolve, Resolved};
use super::search::{find_blue_connected_components, greedy_clique_lower_bound, is_subset, CliqueIterator};
use super::types::{AlgorithmState, Clique, LineGraph, ModularProductGraph};
use std::rc::Rc;

/// Drives the maximum-common-edge-subgraph search over `graphs`: for exactly
/// two graphs, enumerates every maximal common subgraph and keeps every one
/// tied for the largest seen so far in `state.all_maximum_graphs`; for more
/// than two, recursively reduces pairwise (first two graphs -> their MCES ->
/// combine with the third, and so on), following the same approach the
/// original algorithm uses for combining more than two structures.
pub(super) fn find_maximum_common_subgraph(graphs: &mut [LineGraph], state: &mut AlgorithmState) {
    if graphs.len() < 2 {
        return;
    }
    state.enumerate_all = graphs.len() > 2;
    if !state.not_first {
        state.maximum = 0;
        state.num_cliques_found = 0;
        for (index, graph) in graphs.iter_mut().enumerate() {
            state.clique_vectors.push(Vec::new());
            graph.num = index;
        }
        state.not_first = true;
    }

    if graphs[0].graph.node_count() < state.maximum {
        return;
    }

    let g1 = Rc::new(graphs[0].clone());
    let g2 = Rc::new(graphs[1].clone());
    let Some(mut iter) = MaximalCommonSubgraphIterator::new(g1, g2, state) else {
        return;
    };

    if graphs.len() == 2 {
        while let Some(resolved) = iter.next(state) {
            record_if_maximum(state, resolved);
        }
        return;
    }

    let mut best = AlgorithmState::default();
    while let Some(resolved) = iter.next(state) {
        let pattern_line = molecular_graph_to_line_graph(&Rc::new(resolved.pattern));
        let mut remaining = vec![pattern_line];
        remaining.extend_from_slice(&graphs[2..]);
        // Bond ids and coverage belong to this new pair, and a bound for the
        // previous pair says nothing about what the remaining molecules share.
        let mut branch = AlgorithmState::default();
        find_maximum_common_subgraph(&mut remaining, &mut branch);
        for result in branch.all_maximum_graphs {
            record_if_maximum(&mut best, result);
        }
    }
    state.maximum = best.maximum;
    state.all_maximum_graphs = best.all_maximum_graphs;
}

fn record_if_maximum(state: &mut AlgorithmState, resolved: Resolved) {
    let edge_count = resolved.bond_pairs.len();
    if edge_count == state.maximum {
        state.all_maximum_graphs.push(resolved);
    } else if edge_count > state.maximum {
        state.maximum = edge_count;
        state.all_maximum_graphs.clear();
        state.all_maximum_graphs.push(resolved);
    }
}

/// Iterates every maximal common subgraph of two line graphs across all
/// (size-eligible) connected components of their modular product, in
/// descending component size.
struct MaximalCommonSubgraphIterator {
    g1: Rc<LineGraph>,
    clique_iterator: CliqueIterator,
    mcs_iterator: MCSIterator,
    subgraphs: Vec<Rc<ModularProductGraph>>,
    current_subgraph: usize,
}

impl MaximalCommonSubgraphIterator {
    fn new(g1: Rc<LineGraph>, g2: Rc<LineGraph>, state: &mut AlgorithmState) -> Option<Self> {
        let modular_product = modular_graph_product(&g1, &g2);

        // Seed a lower bound so branch-and-bound pruning has something to
        // work with from the very first search (see `greedy_clique_lower_bound`).
        // The `-1` margin covers the (rare, only possible at clique size 3)
        // case where the greedy clique turns out to be a "bad triangle" that
        // shrinks once resolved, so the true bound is never overshot.
        let greedy_bound = greedy_clique_lower_bound(&modular_product).saturating_sub(1);
        if !state.enumerate_all && greedy_bound > state.maximum {
            state.maximum = greedy_bound;
        }

        let subgraphs: Vec<Rc<ModularProductGraph>> =
            find_blue_connected_components(&modular_product, state.maximum)
                .into_iter()
                .map(Rc::new)
                .collect();

        let mut subgraph_index = 0;
        while let Some(subgraph) = subgraphs.get(subgraph_index) {
            if subgraph.graph.node_count() < state.maximum {
                subgraph_index += 1;
                continue;
            }
            if g1.graph.node_count() < state.maximum {
                return None;
            }

            let mut clique_iterator = CliqueIterator::new(Rc::clone(subgraph));
            let mut found = None;
            while let Some(clique) = clique_iterator.next(state) {
                if let Some(mcs_iterator) = MCSIterator::new(subgraph, clique, state) {
                    found = Some(mcs_iterator);
                    break;
                }
            }

            if let Some(mcs_iterator) = found {
                return Some(Self {
                    g1,
                    clique_iterator,
                    mcs_iterator,
                    subgraphs,
                    current_subgraph: subgraph_index,
                });
            }

            subgraph_index += 1;
        }
        None
    }

    fn next(&mut self, state: &mut AlgorithmState) -> Option<Resolved> {
        if let Some(resolved) = self.mcs_iterator.next(state) {
            return Some(resolved);
        }

        while let Some(clique) = self.clique_iterator.next(state) {
            if let Some(mcs_iterator) =
                MCSIterator::new(&self.subgraphs[self.current_subgraph], clique, state)
            {
                self.mcs_iterator = mcs_iterator;
                return self.mcs_iterator.next(state);
            }
        }

        while self.current_subgraph + 1 < self.subgraphs.len() {
            self.current_subgraph += 1;
            let subgraph = &self.subgraphs[self.current_subgraph];

            if subgraph.graph.node_count() < state.maximum {
                continue;
            }
            if self.g1.graph.node_count() < state.maximum {
                return None;
            }

            self.clique_iterator = CliqueIterator::new(Rc::clone(subgraph));
            while let Some(clique) = self.clique_iterator.next(state) {
                if let Some(mcs_iterator) = MCSIterator::new(subgraph, clique, state) {
                    self.mcs_iterator = mcs_iterator;
                    return self.mcs_iterator.next(state);
                }
            }
        }

        None
    }
}

/// Resolves the cliques found in one connected component into concrete
/// common-substructure results, applying the same "already covered by a
/// bigger clique" de-duplication as the original algorithm via
/// `state.clique_vectors`.
struct MCSIterator {
    resolved_subgraphs: Vec<ModularProductGraph>,
    current: usize,
    clique_set: Vec<usize>,
    clique_set2: Vec<usize>,
    g2_num: usize,
}

impl MCSIterator {
    fn new(subgraph: &ModularProductGraph, clique: Clique, state: &mut AlgorithmState) -> Option<Self> {
        state.num_cliques_found += 1;

        if clique.len() < state.maximum {
            return None;
        }

        let mut clique_set: Vec<usize> = Vec::with_capacity(clique.len());
        let mut clique_set2: Vec<usize> = Vec::with_capacity(clique.len());
        for &vertex in &clique {
            if let Some(&id1) = subgraph.clique_mapping.get(&vertex) {
                clique_set.push(id1);
            }
            if let Some(&(_, g2_node)) = subgraph.node_mapping.get(&vertex) {
                if let Some(&id2) = subgraph.g2.clique_map.get(&g2_node) {
                    clique_set2.push(id2);
                }
            }
        }
        clique_set.sort_unstable();
        clique_set2.sort_unstable();

        let g2_num = subgraph.g2.num;

        // The two sides must be covered by the *same* previous result.
        // Independent membership tests discard unseen combinations when a
        // component on either side has already matched a different partner.
        let covered_pair = state.clique_vectors[0]
            .iter()
            .zip(&state.clique_vectors[g2_num])
            .any(|(left, right)| {
                is_subset(&clique_set, left) && is_subset(&clique_set2, right)
            });
        if !state.enumerate_all && covered_pair {
            return None;
        }

        let resolved_subgraphs = induced_subgraph(&clique, subgraph);
        Some(Self {
            resolved_subgraphs,
            current: 0,
            clique_set,
            clique_set2,
            g2_num,
        })
    }

    fn next(&mut self, state: &mut AlgorithmState) -> Option<Resolved> {
        if self.current >= self.resolved_subgraphs.len() {
            state.clique_vectors[0].push(std::mem::take(&mut self.clique_set));
            state.clique_vectors[self.g2_num].push(std::mem::take(&mut self.clique_set2));
            return None;
        }
        let candidate = &self.resolved_subgraphs[self.current];
        self.current += 1;
        if candidate.graph.node_count() >= state.maximum {
            return Some(resolve(candidate));
        }
        None
    }
}
