use super::types::{LineGraph, ModularProductGraph};
use petgraph::graph::NodeIndex;
use std::rc::Rc;

/// Builds the (chemically-constrained) modular product of two line graphs.
///
/// A vertex `(node1, node2)` is kept iff `node1`/`node2` have the same bond
/// order, and the underlying original-graph bond endpoints have matching atom
/// types in either orientation (so the two bonds could plausibly be "the same
/// bond" in a common substructure). An edge is added between two kept
/// vertices iff the corresponding bond-pairs are simultaneously adjacent (or
/// simultaneously non-adjacent, in the disabled "type-0" case from the
/// original algorithm — kept disabled here too) in both line graphs with a
/// matching shared-atom label.
///
/// A maximum clique in this graph corresponds to a maximum common edge
/// subgraph of the two original molecules (Levi/Koch-style reduction).
pub(super) fn modular_graph_product(g1: &Rc<LineGraph>, g2: &Rc<LineGraph>) -> ModularProductGraph {
    let mut product = ModularProductGraph::new(Rc::clone(g1), Rc::clone(g2));

    for node1 in g1.graph.node_indices() {
        for node2 in g2.graph.node_indices() {
            if g1.graph.node_weight(node1) != g2.graph.node_weight(node2) {
                continue;
            }

            let Some(&edge1) = g1.edge_map.get(&node1) else {
                continue;
            };
            let Some(&edge2) = g2.edge_map.get(&node2) else {
                continue;
            };

            let orig1 = &g1.original_graph;
            let orig2 = &g2.original_graph;
            let a1 = orig1.graph.node_weight(edge1.0);
            let a2 = orig1.graph.node_weight(edge1.1);
            let b1 = orig2.graph.node_weight(edge2.0);
            let b2 = orig2.graph.node_weight(edge2.1);

            let same_orientation = a1 == b1 && a2 == b2;
            let flipped_orientation = a1 == b2 && a2 == b1;
            if !(same_orientation || flipped_orientation) {
                continue;
            }

            let bond_order = *g1.graph.node_weight(node1).unwrap();
            let v = product.graph.add_node(bond_order);
            product.node_mapping.insert(v, (node1, node2));
            if let Some(&clique_id) = g1.clique_map.get(&node1) {
                product.clique_mapping.insert(v, clique_id);
            }
        }
    }

    let product_nodes: Vec<NodeIndex> = product.graph.node_indices().collect();
    for i in 0..product_nodes.len() {
        for j in 0..i {
            let node1 = product_nodes[i];
            let node2 = product_nodes[j];

            let (n1_g1, n1_g2) = product.node_mapping[&node1];
            let (n2_g1, n2_g2) = product.node_mapping[&node2];

            // Same original bond used twice: not a valid pairing.
            if n1_g1 == n2_g1 || n1_g2 == n2_g2 {
                continue;
            }

            let Some(e1) = g1.graph.find_edge(n1_g1, n2_g1) else {
                continue;
            };
            let Some(e2) = g2.graph.find_edge(n1_g2, n2_g2) else {
                continue;
            };

            let atom1 = g1.graph.edge_weight(e1).unwrap();
            let atom2 = g2.graph.edge_weight(e2).unwrap();
            if atom1 == atom2 {
                product.graph.add_edge(node1, node2, Some(*atom1));
            }
        }
    }

    product
}
