//! Regressions discovered by the frozen September 2026 publication evaluation.
use crate::graph_algorithms::mol_canonization::{canonize_mol, reorder_molecule};
use crate::molecular_formula::MolecularFormula;
use crate::molecule::tautomer::{TautomerRules, tautomer_key, try_normalize};
use crate::prelude::*;

#[test]
fn publication_hydrogen_writer_preserves_atoms_under_relabeling() {
    for smiles in ["[2HH]", "[H][2H]", "[H][H]", "[2H][3H]", "[H+]", "[H-]"] {
        let original = MolecularGraph::from_smiles_single(smiles).unwrap();
        for reverse in [false, true] {
            let mut mol = original.clone();
            let mut order: Vec<usize> = (0..mol.len()).collect();
            if reverse {
                order.reverse();
            }
            reorder_molecule(&mut mol, &order);
            let written = mol.to_smiles();
            let reparsed = MolecularGraph::from_smiles_single(&written)
                .unwrap_or_else(|e| panic!("{smiles} -> {written}: {e:?}"));
            assert_eq!(
                MolecularFormula::from_molecule(&original),
                MolecularFormula::from_molecule(&reparsed),
                "{smiles} -> {written}"
            );
            let labels = |m: &MolecularGraph| {
                let mut atoms: Vec<_> = (0..m.len())
                    .map(|i| (m.atomic_numbers()[i], m.get_isotope(i), m.charges()[i]))
                    .collect();
                atoms.sort_unstable();
                atoms
            };
            assert_eq!(
                labels(&original),
                labels(&reparsed),
                "{smiles} -> {written}"
            );
        }
    }
}

#[test]
fn publication_bridged_nitrogen_preserves_identity_across_writings() {
    let writings = [
        "CCON=C1C[N@]2CC[C@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC)cc12",
        "C(C)ON=C1[C@@H]2C[C@@H]([C@@H](c3ccnc4ccc(OC)cc34)O)[N@](C1)CC2",
    ];
    let rules = TautomerRules::default();
    let signatures: Vec<_> = writings
        .iter()
        .map(|s| {
            let mol = MolecularGraph::from_smiles_single(s).unwrap();
            (
                canonize_mol(&mol).to_smiles(),
                tautomer_key(&mol, &rules),
                try_normalize(&mol, &rules).unwrap().to_smiles(),
            )
        })
        .collect();
    assert_eq!(
        signatures[0], signatures[1],
        "CHEMBL4448224 equivalent inputs diverged"
    );
    let inverted =
        MolecularGraph::from_smiles_single(&writings[0].replace("[N@]", "[N@@]")).unwrap();
    assert_ne!(
        signatures[0].0,
        canonize_mol(&inverted).to_smiles(),
        "nitrogen stereoisomers collapsed"
    );
    assert_ne!(
        signatures[0].1,
        tautomer_key(&inverted, &rules),
        "nitrogen stereoisomer keys collapsed"
    );
}
