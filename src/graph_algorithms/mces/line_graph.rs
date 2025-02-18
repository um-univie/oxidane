use super::types::{LineGraph, LineGraphType, McesGraph};
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// Converts a molecule to its line graph: each bond becomes a node (labelled
/// with its `BondOrder`), and two bond-nodes are connected whenever the
/// corresponding bonds share an atom (edge labelled with that atom's
/// `AtomLabel`). This turns the edge-based MCES search into a vertex-based
/// maximum-clique search on the line graph's modular product.
pub(super) fn molecular_graph_to_line_graph(mol: &Rc<McesGraph>) -> LineGraph {
    let mut graph = LineGraphType::default();
    let mut edge_map = FxHashMap::default();
    let mut clique_map = FxHashMap::default();

    for edge_idx in mol.graph.edge_indices() {
        let (source, target) = mol.graph.edge_endpoints(edge_idx).unwrap();
        let (source, target) = if source < target {
            (source, target)
        } else {
            (target, source)
        };
        let bond_order = *mol.graph.edge_weight(edge_idx).unwrap();

        let line_node = graph.add_node(bond_order);
        edge_map.insert(line_node, (source, target));
        clique_map.insert(line_node, edge_idx.index());
    }

    let line_nodes: Vec<NodeIndex> = graph.node_indices().collect();
    for i in 0..line_nodes.len() {
        for j in (i + 1)..line_nodes.len() {
            let node1 = line_nodes[i];
            let node2 = line_nodes[j];
            let &(s1, t1) = &edge_map[&node1];
            let &(s2, t2) = &edge_map[&node2];

            let shared_atom = if s1 == s2 || s1 == t2 {
                Some(s1)
            } else if t1 == s2 || t1 == t2 {
                Some(t1)
            } else {
                None
            };

            let Some(shared_atom) = shared_atom else {
                continue;
            };

            let atom_label = *mol.graph.node_weight(shared_atom).unwrap();
            graph.add_edge(node1, node2, atom_label);
        }
    }

    LineGraph {
        graph,
        edge_map,
        clique_map,
        original_graph: Rc::clone(mol),
        num: 0,
    }
}
