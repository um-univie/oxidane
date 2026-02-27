use oxidane::graph_algorithms::mces::common_edge_subgraph_size;
use oxidane::prelude::*;

fn bonds(smiles: &[&str]) -> Option<usize> {
    let molecules: Vec<_> = smiles
        .iter()
        .map(|s| MolecularGraph::from_smiles_single(s).unwrap())
        .collect();
    common_edge_subgraph_size(&molecules).map(|size| size.num_bonds)
}

#[test]
fn a_later_molecule_can_shrink_the_common_subgraph() {
    for inputs in [
        ["CCCC", "CCCC", "CC"],
        ["CC", "CCCC", "CCCC"],
        ["CCCC", "CC", "CCCC"],
    ] {
        assert_eq!(bonds(&inputs), Some(1), "{inputs:?}");
    }
    assert_eq!(bonds(&["CCCCC", "CCCC", "CCC", "CC"]), Some(1));
    assert_eq!(bonds(&["CCCC", "CCC", "CCCC"]), Some(2));
}

#[test]
fn one_bond_and_no_bond_are_distinct_outcomes() {
    assert_eq!(bonds(&["CC", "CC"]), Some(1));
    assert_eq!(bonds(&["CC", "CC", "CC"]), Some(1));
    assert_eq!(bonds(&["CO", "CC", "CC"]), None);
    assert_eq!(bonds(&["C", "CC", "CC"]), None);
    assert_eq!(bonds(&["C=C", "CC", "CC"]), None);
    assert_eq!(bonds(&["CC"]), None);
    assert_eq!(bonds(&[]), None);
}

#[test]
fn the_pairwise_maximum_need_not_survive_the_third_molecule() {
    // The shared long carbon path is larger than the shared N-N bond, but
    // only N-N survives the third input. The connecting bonds disagree.
    for inputs in [
        ["NN=CCCCCC", "NNCCCCCC", "NN"],
        ["NN", "NN=CCCCCC", "NNCCCCCC"],
        ["NNCCCCCC", "NN", "NN=CCCCCC"],
    ] {
        assert_eq!(bonds(&inputs), Some(1), "{inputs:?}");
    }
}
