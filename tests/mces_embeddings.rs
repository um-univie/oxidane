use oxidane::graph_algorithms::mces::find_all_maximum_common_edge_subgraphs;
use oxidane::prelude::*;
use std::collections::BTreeSet;

fn disconnected(smiles: &str) -> Molecule3D {
    let mut result = Molecule3D::default();
    for graph in MolecularGraph::from_smiles(smiles).unwrap() {
        let offset = result.len();
        result
            .atomic_numbers
            .extend_from_slice(graph.atomic_numbers());
        result.charges.extend_from_slice(graph.charges());
        result
            .radical_states
            .extend_from_slice(graph.radical_states());
        result
            .positions
            .extend(vec![Vector3D::default(); graph.len()]);
        for bonds in graph.atom_bonds() {
            let mut bonds = *bonds;
            for bond in &mut bonds {
                bond.target += offset;
            }
            result.atom_bonds.push(bonds);
        }
    }
    result
}

#[test]
fn every_pair_of_component_placements_is_returned() {
    for (left, right, expected) in [
        ("CCC.CCC", "CCCC.CCCC", 8),
        ("CCC", "CCCC", 2),
        ("CCC.CCC.CCC", "CCCC.CCCC", 12),
    ] {
        let a = disconnected(left);
        let b = disconnected(right);
        for (a, b) in [(&a, &b), (&b, &a)] {
            let results = find_all_maximum_common_edge_subgraphs(a, b);
            let mut pairs = BTreeSet::new();
            let mut left_sets = BTreeSet::new();
            let mut right_sets = BTreeSet::new();
            for result in &results {
                assert_eq!(result.num_bonds(), 2);
                let mut left: Vec<_> = result.bond_mapping.iter().map(|m| m.bond_a).collect();
                let mut right: Vec<_> = result.bond_mapping.iter().map(|m| m.bond_b).collect();
                for (i, j) in left.iter_mut().chain(&mut right) {
                    if *i > *j {
                        std::mem::swap(i, j);
                    }
                }
                left.sort();
                right.sort();
                left_sets.insert(left.clone());
                right_sets.insert(right.clone());
                pairs.insert((left, right));
            }
            assert_eq!(pairs.len(), expected, "{left} / {right}");
            for left in &left_sets {
                for right in &right_sets {
                    assert!(pairs.contains(&(left.clone(), right.clone())));
                }
            }
            assert_eq!(results.len(), pairs.len(), "duplicate placements");
        }
    }
}
