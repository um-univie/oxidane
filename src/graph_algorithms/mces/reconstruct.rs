use super::types::{Clique, McesGraph, ModularProductGraph, MolGraphType};
use crate::molecule::bond::BondOrder;
use crate::molecule::graph::AtomLabel;
use bit_set::BitSet;
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

/// One matched bond of the common subgraph, in each input molecule's own atom
/// indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BondMapping {
    pub bond_a: (usize, usize),
    pub bond_b: (usize, usize),
}

/// A resolved (internally-consistent) common substructure: the atom
/// correspondence between the two input molecules, the matched bonds, and a
/// standalone reconstruction of the substructure's atoms/bonds (in molecule
/// A's labelling) usable to seed a further round of pairwise search when
/// combining more than two molecules.
pub(super) struct Resolved {
    pub pattern: McesGraph,
    pub atom_pairs: Vec<(usize, usize)>,
    pub bond_pairs: Vec<BondMapping>,
}

/// Returns the induced subgraph of `graph` containing just the vertices in
/// `clique`. For 3-vertex cliques this also checks for a "bad triangle": a
/// clique whose bonds are mutually consistent pairwise but admit two
/// different global atom mappings (a known ambiguity of the line-graph +
/// modular-product reduction, see cmces's `graph_theory_explanation.md`). When
/// found, the ambiguous vertex combinations are split out into separate
/// (each internally-consistent) subgraphs rather than reporting one
/// self-contradictory mapping.
pub(super) fn induced_subgraph(clique: &Clique, graph: &ModularProductGraph) -> Vec<ModularProductGraph> {
    let mut subgraph = ModularProductGraph::new(graph.g1.clone(), graph.g2.clone());
    let max_nodes = graph.graph.node_count();
    let mut vertex_map: Vec<Option<NodeIndex>> = vec![None; max_nodes];

    for &vertex in clique {
        let bond_order = *graph.graph.node_weight(vertex).unwrap();
        let v = subgraph.graph.add_node(bond_order);
        vertex_map[vertex.index()] = Some(v);
        subgraph.node_mapping.insert(v, graph.node_mapping[&vertex]);
        if let Some(&clique_val) = graph.clique_mapping.get(&vertex) {
            subgraph.clique_mapping.insert(v, clique_val);
        }
    }

    for edge_idx in graph.graph.edge_indices() {
        let (source, target) = graph.graph.edge_endpoints(edge_idx).unwrap();
        if !(clique.contains(&source) && clique.contains(&target)) {
            continue;
        }
        let atom_type = *graph.graph.edge_weight(edge_idx).unwrap();
        if atom_type.is_none() {
            continue;
        }
        let new_source = vertex_map[source.index()].unwrap();
        let new_target = vertex_map[target.index()].unwrap();
        subgraph.graph.add_edge(new_source, new_target, atom_type);
    }

    if clique.len() != 3 {
        return vec![subgraph];
    }

    let mut bad_triangles: Vec<Vec<NodeIndex>> = Vec::new();
    let mut seen_triangles: Vec<BitSet> = Vec::new();
    let max_nodes = subgraph.graph.node_count();

    for edge_idx in subgraph.graph.edge_indices() {
        let (source, target) = subgraph.graph.edge_endpoints(edge_idx).unwrap();
        if subgraph.graph.edge_weight(edge_idx).unwrap().is_none() {
            continue;
        }

        for vertex in subgraph.graph.node_indices() {
            let Some(e1) = subgraph.graph.find_edge(vertex, target) else {
                continue;
            };
            let Some(e2) = subgraph.graph.find_edge(vertex, source) else {
                continue;
            };
            if subgraph.graph.edge_weight(e1).unwrap().is_none()
                || subgraph.graph.edge_weight(e2).unwrap().is_none()
            {
                continue;
            }

            let triangle1 = original_atoms_of(&subgraph, true, &[vertex, source, target]);
            let triangle2 = original_atoms_of(&subgraph, false, &[vertex, source, target]);

            if triangle1.len() != triangle2.len() {
                let bad_triangle = vec![vertex, target, source];
                let mut triangle_bits = BitSet::with_capacity(max_nodes);
                for &v in &bad_triangle {
                    triangle_bits.insert(v.index());
                }
                if seen_triangles.contains(&triangle_bits) {
                    continue;
                }
                bad_triangles.push(bad_triangle);
                seen_triangles.push(triangle_bits);
            }
        }
    }

    if bad_triangles.is_empty() {
        return vec![subgraph];
    }

    let mut bad_triangles_mut = bad_triangles;
    let remove_vertex_sets = cartesian(&mut bad_triangles_mut);

    let mut keep = vec![true; remove_vertex_sets.len()];
    for i in 0..remove_vertex_sets.len() {
        for j in 0..remove_vertex_sets.len() {
            if i == j {
                continue;
            }
            if remove_vertex_sets[j].is_subset(&remove_vertex_sets[i]) {
                keep[j] = false;
            }
        }
    }

    remove_vertex_sets
        .into_iter()
        .zip(keep)
        .filter(|(_, keep)| *keep)
        .map(|(remove_set, _)| delete_set_of_vertices(&subgraph, &remove_set))
        .collect()
}

/// Collects the distinct original-molecule atom indices spanned by `vertices`
/// (line-graph bond endpoints), from molecule A's line graph if `from_g1` else
/// molecule B's. Used only to compare how many distinct atoms a candidate
/// "bad triangle" touches on each side.
fn original_atoms_of(subgraph: &ModularProductGraph, from_g1: bool, vertices: &[NodeIndex]) -> Vec<usize> {
    let mut atoms = Vec::with_capacity(vertices.len() * 2);
    for &vertex in vertices {
        let Some(&(g1_node, g2_node)) = subgraph.node_mapping.get(&vertex) else {
            continue;
        };
        let line_node = if from_g1 { g1_node } else { g2_node };
        let line_graph = if from_g1 { &subgraph.g1 } else { &subgraph.g2 };
        if let Some(&(a, b)) = line_graph.edge_map.get(&line_node) {
            atoms.push(a.index());
            atoms.push(b.index());
        }
    }
    atoms.sort_unstable();
    atoms.dedup();
    atoms
}

/// Cartesian product of the given vertex sets, returned as bitsets over node
/// index (one bitset per combination), used to enumerate every way of
/// removing one vertex from each bad triangle.
fn cartesian(sets: &mut [Vec<NodeIndex>]) -> Vec<BitSet> {
    let max_index = sets.iter().flatten().map(|x| x.index()).max().unwrap_or(0);
    let set_size = max_index + 1;

    let mut combinations: Vec<BitSet> = vec![BitSet::with_capacity(set_size)];
    for set in sets.iter() {
        let mut next = Vec::new();
        for combo in &combinations {
            for &element in set {
                let mut extended = combo.clone();
                extended.insert(element.index());
                next.push(extended);
            }
        }
        combinations = next;
    }
    combinations
}

fn delete_set_of_vertices(graph: &ModularProductGraph, delete_vertices: &BitSet) -> ModularProductGraph {
    let mut subgraph = ModularProductGraph::new(graph.g1.clone(), graph.g2.clone());
    let mut vertex_map: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();

    for vertex in graph.graph.node_indices() {
        let deleted = delete_vertices.contains(vertex.index());
        if deleted {
            continue;
        }
        let bond_order = *graph.graph.node_weight(vertex).unwrap();
        let v = subgraph.graph.add_node(bond_order);
        vertex_map.insert(vertex, v);
        subgraph.node_mapping.insert(v, graph.node_mapping[&vertex]);
        if let Some(&clique_val) = graph.clique_mapping.get(&vertex) {
            subgraph.clique_mapping.insert(v, clique_val);
        }
    }

    for edge_idx in graph.graph.edge_indices() {
        let (source, target) = graph.graph.edge_endpoints(edge_idx).unwrap();
        let (Some(&new_source), Some(&new_target)) =
            (vertex_map.get(&source), vertex_map.get(&target))
        else {
            continue;
        };
        let atom_type = *graph.graph.edge_weight(edge_idx).unwrap();
        if atom_type.is_none() {
            continue;
        }
        subgraph.graph.add_edge(new_source, new_target, atom_type);
    }

    subgraph
}

/// Resolves a (triangle-disambiguated) modular-product subgraph into a
/// concrete common-substructure result: the atom correspondence between the
/// two input molecules and a standalone pattern graph.
///
/// The atom correspondence is derived structurally (via the bond endpoints
/// each product vertex maps back to), not by matching element/charge alone,
/// so atoms of the same element are only paired when the bond structure
/// actually forces it.
///
/// Orientation for a bond-pair `(a1,a2)<->(b1,b2)` (does `a1` map to `b1` or
/// to `b2`?) is **not** guessed or propagated from a neighbor's already-fixed
/// mapping — that path-dependent approach could silently commit to a wrong
/// choice early and then produce a self-contradictory result (the same
/// target atom claimed twice) once a later bond-pair's constraint disagreed
/// with it. Instead, whenever this bond-pair shares an atom with another
/// bond-pair in the clique (a "real", non-`None`, edge — the two must
/// therefore agree on where that shared atom goes), the orientation is
/// *computed directly* from that structural relationship, independent of
/// traversal order: exactly one atom is common to both bond-pairs' `mol_a`
/// endpoints, and exactly one is common to both `mol_b` endpoints (that's
/// what a real edge between them means), so the orientation that sends the
/// former to the latter is fully determined, not chosen. Only a bond-pair
/// with no such neighbor (an independently-matched fragment, joined to the
/// rest only via `None`/"type-0" edges, if at all) has nothing to derive
/// from, and picks an arbitrary — but then unconstrained, so always safe —
/// starting orientation via atom-label comparison.
pub(super) fn resolve(subgraph: &ModularProductGraph) -> Resolved {
    let g1 = &subgraph.g1;
    let g2 = &subgraph.g2;
    let orig_a = &g1.original_graph;
    let orig_b = &g2.original_graph;

    struct VertexInfo {
        vertex: NodeIndex,
        a1: NodeIndex,
        a2: NodeIndex,
        b1: NodeIndex,
        b2: NodeIndex,
    }

    let vertex_info: Vec<VertexInfo> = subgraph
        .graph
        .node_indices()
        .filter_map(|vertex| {
            let &(line_g1, line_g2) = subgraph.node_mapping.get(&vertex)?;
            let &(a1, a2) = g1.edge_map.get(&line_g1)?;
            let &(b1, b2) = g2.edge_map.get(&line_g2)?;
            Some(VertexInfo { vertex, a1, a2, b1, b2 })
        })
        .collect();

    let info_of: FxHashMap<NodeIndex, usize> = vertex_info
        .iter()
        .enumerate()
        .map(|(i, info)| (info.vertex, i))
        .collect();

    // `orientation[i] == true` means vertex_info[i] maps a1->b1, a2->b2;
    // `false` means a1->b2, a2->b1.
    let mut orientation = vec![false; vertex_info.len()];

    for (idx, info) in vertex_info.iter().enumerate() {
        let mut derived = None;

        for neighbor in subgraph.graph.neighbors(info.vertex) {
            let Some(&neighbor_idx) = info_of.get(&neighbor) else {
                continue;
            };
            let Some(edge) = subgraph.graph.find_edge(info.vertex, neighbor) else {
                continue;
            };
            if subgraph.graph.edge_weight(edge).unwrap().is_none() {
                continue;
            }
            let other = &vertex_info[neighbor_idx];

            // The atom common to both bond-pairs' mol_a endpoints, and the
            // one common to both mol_b endpoints — a real edge between two
            // bond-pairs means exactly one of each exists.
            let shared_a = if info.a1 == other.a1 || info.a1 == other.a2 {
                info.a1
            } else {
                info.a2
            };
            let shared_b = if info.b1 == other.b1 || info.b1 == other.b2 {
                info.b1
            } else {
                info.b2
            };

            let required = if shared_a == info.a1 {
                shared_b == info.b1
            } else {
                shared_b == info.b2
            };
            derived = Some(required);
            break;
        }

        orientation[idx] = derived.unwrap_or_else(|| {
            orig_a.graph.node_weight(info.a1) == orig_b.graph.node_weight(info.b1)
        });
    }

    let mut atom_map: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for (idx, info) in vertex_info.iter().enumerate() {
        let (mapped_a1, mapped_a2) = if orientation[idx] {
            (info.b1, info.b2)
        } else {
            (info.b2, info.b1)
        };
        atom_map.insert(info.a1, mapped_a1);
        atom_map.insert(info.a2, mapped_a2);
    }

    // Bond and pattern edges are read back from the now-fully-committed
    // `atom_map` (never from a vertex's local, possibly-superseded guess),
    // so they can never disagree with the final atom correspondence.
    let mut bond_pairs = Vec::with_capacity(vertex_info.len());
    let mut pattern_edges: Vec<(NodeIndex, NodeIndex, AtomLabel, AtomLabel, BondOrder)> =
        Vec::with_capacity(vertex_info.len());

    for info in &vertex_info {
        let mapped_a1 = atom_map[&info.a1];
        let mapped_a2 = atom_map[&info.a2];

        bond_pairs.push(BondMapping {
            bond_a: (
                orig_a.original_index[info.a1.index()],
                orig_a.original_index[info.a2.index()],
            ),
            bond_b: (
                orig_b.original_index[mapped_a1.index()],
                orig_b.original_index[mapped_a2.index()],
            ),
        });

        let label_a1 = *orig_a.graph.node_weight(info.a1).unwrap();
        let label_a2 = *orig_a.graph.node_weight(info.a2).unwrap();
        let bond_order = *orig_a
            .graph
            .edge_weight(orig_a.graph.find_edge(info.a1, info.a2).unwrap())
            .unwrap();
        pattern_edges.push((info.a1, info.a2, label_a1, label_a2, bond_order));
    }

    let mut atom_pairs: Vec<(usize, usize)> = atom_map
        .iter()
        .map(|(&a, &b)| (orig_a.original_index[a.index()], orig_b.original_index[b.index()]))
        .collect();
    atom_pairs.sort_unstable();

    let mut pattern_graph = MolGraphType::default();
    let mut pattern_nodes: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &(a1, a2, label_a1, label_a2, _) in &pattern_edges {
        pattern_nodes
            .entry(a1)
            .or_insert_with(|| pattern_graph.add_node(label_a1));
        pattern_nodes
            .entry(a2)
            .or_insert_with(|| pattern_graph.add_node(label_a2));
    }
    for &(a1, a2, _, _, bond_order) in &pattern_edges {
        let n1 = pattern_nodes[&a1];
        let n2 = pattern_nodes[&a2];
        if pattern_graph.find_edge(n1, n2).is_none() {
            pattern_graph.add_edge(n1, n2, bond_order);
        }
    }

    Resolved {
        pattern: McesGraph::from_ungraph(pattern_graph),
        atom_pairs,
        bond_pairs,
    }
}
