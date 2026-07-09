use oxidane::prelude::*;
use oxidane::molecule::descriptors::MolecularDescriptors;

#[test]
fn acceptor_counts_match_typed_reference_cases() {
    let mut failures = Vec::new();
    for line in include_str!("hbond_acceptors.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let (smiles, count) = line.split_once('\t').unwrap();
        let expected: usize = count.parse().unwrap();
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        let actual = molecule.number_of_hydrogen_bond_acceptors();
        if actual != expected {
            failures.push(format!("{smiles}: {actual} != {expected}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn hydrogen_representations_and_aromatic_writings_agree() {
    for pair in [
        ["CO", "[H]OC"],
        ["CC(=O)O", "CC(=O)O[H]"],
        ["c1cc[nH]c1", "C1=CNC=C1"],
        ["c1ccncc1", "C1=CC=NC=C1"],
    ] {
        let counts = pair.map(|s| {
            MolecularGraph::from_smiles_single(s)
                .unwrap()
                .number_of_hydrogen_bond_acceptors()
        });
        assert_eq!(counts[0], counts[1], "{pair:?}");
    }
}
