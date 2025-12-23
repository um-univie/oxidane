//! Bemis--Murcko scaffolds, checked against RDKit.
//!
//! The cases came from `MurckoScaffold.GetScaffoldForMol`, verified on 4996
//! molecules by `examples/scaffold_diff.rs` against `tests/scaffold_oracle.py`.
//! The three that differ there are hydrogen counts the writer places
//! differently on hypervalent phosphorus and on a charged aromatic nitrogen,
//! not scaffolds; the cases below are ones where the two agree exactly.

use crate::graph_algorithms::mol_canonization::canonize_mol;
use crate::io::ToSMILES;
use crate::molecule::base::MoleculeCore;
use crate::molecule::scaffold::{
    framework_key, murcko_scaffold, murcko_scaffold_atoms, scaffold_key,
};
use crate::prelude::*;

fn parse(smiles: &str) -> MolecularGraph {
    MolecularGraph::from_smiles_single(smiles).unwrap()
}

/// `(input, scaffold, framework)`, every expectation from RDKit.
const SCAFFOLDS: &[(&str, &str, &str)] = &[
    // No rings, no scaffold.
    ("CCCC", "", ""),
    ("O", "", ""),
    ("C(=O)(O)C", "", ""),
    // The ring itself.
    ("c1ccccc1", "c1ccccc1", "C1CCCCC1"),
    ("c1ccc2ccccc2c1", "c1ccc2ccccc2c1", "C1CCC2CCCCC2C1"),
    ("C1CC2CCC1CC2", "C1CC2CCC1CC2", "C1CC2CCC1CC2"),
    // Side chains come off, however long.
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccccc1", "C1CCCCC1"),
    ("CCOC(=O)c1ccc(N)cc1", "c1ccccc1", "C1CCCCC1"),
    ("CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "C1CCCCC1"),
    ("N#Cc1ccccc1", "c1ccccc1", "C1CCCCC1"),
    ("C1CC1C(=O)NC", "C1CC1", "C1CC1"),
    // An exocyclic multiple bond holds its atom on.
    ("O=C1CCCN1C", "O=C1CCCN1", "CC1CCCC1"),
    ("C1CCC(=O)CC1", "O=C1CCCCC1", "CC1CCCCC1"),
    ("C1CCCCC1=NC", "N=C1CCCCC1", "CC1CCCCC1"),
    ("C1CCCCC1=CC(=O)O", "C=C1CCCCC1", "CC1CCCCC1"),
    // ...but only what the bond itself holds. The acetyl on the ring nitrogen
    // goes entirely; the ring's own carbonyl stays.
    ("CC(=O)N1CCCC1=O", "O=C1CCCN1", "CC1CCCC1"),
    // Linkers between two ring systems are the scaffold too.
    (
        "c1ccc(CCc2ccncc2)cc1",
        "c1ccc(CCc2ccncc2)cc1",
        "C1CCC(CCC2CCCCC2)CC1",
    ),
    (
        "c1ccccc1C(=O)Nc1ccccc1",
        "O=C(Nc1ccccc1)c1ccccc1",
        "CC(CC1CCCCC1)C1CCCCC1",
    ),
];

#[test]
fn scaffolds_match_rdkit() {
    for &(smiles, expected_scaffold, expected_framework) in SCAFFOLDS {
        let molecule = parse(smiles);
        let scaffold = scaffold_key(&molecule);
        let expected = if expected_scaffold.is_empty() {
            String::new()
        } else {
            canonize_mol(&parse(expected_scaffold)).to_smiles()
        };
        assert_eq!(scaffold, expected, "scaffold of {smiles}");

        let framework = framework_key(&molecule);
        let expected = if expected_framework.is_empty() {
            String::new()
        } else {
            canonize_mol(&parse(expected_framework)).to_smiles()
        };
        assert_eq!(framework, expected, "framework of {smiles}");
    }
}

#[test]
fn a_substituted_aromatic_nitrogen_gets_its_hydrogen_back() {
    // Cutting the aromatic form leaves an `n` with two ring bonds and no
    // hydrogen, which is not a molecule and which RDKit refuses to read. The
    // cut is made on a Kekule copy for exactly this.
    for smiles in [
        "c1ccc2[nH]ccc2c1",
        "c1ccc2[nH]c(C)cc2c1",
        "CCC(Cn1ccc2c1cc(CNCCC)cc2)CC",
    ] {
        assert_eq!(
            scaffold_key(&parse(smiles)),
            canonize_mol(&parse("c1ccc2[nH]ccc2c1")).to_smiles(),
            "indole scaffold of {smiles}"
        );
    }
}

#[test]
fn a_scaffold_is_a_whole_molecule() {
    // Every atom has the hydrogens its bonds leave room for, rather than the
    // ones the parent gave it.
    let scaffold = murcko_scaffold(&parse("CC(C)Cc1ccc(cc1)C(C)C(=O)O"));
    assert_eq!(scaffold.molecular_formula().to_string(), "C6H6");
}

#[test]
fn a_scaffold_does_not_depend_on_how_the_molecule_was_written() {
    let writings = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "O=C(C)Oc1ccccc1C(O)=O",
        "OC(=O)c1ccccc1OC(C)=O",
    ];
    let keys: Vec<String> = writings.iter().map(|s| scaffold_key(&parse(s))).collect();
    assert!(keys.windows(2).all(|pair| pair[0] == pair[1]), "{keys:?}");
}

#[test]
fn scaffold_atoms_are_ascending_and_within_range() {
    let molecule = parse("CC(=O)Nc1ccc(O)cc1");
    let atoms = murcko_scaffold_atoms(&molecule);
    assert!(atoms.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(atoms.iter().all(|&atom| atom < molecule.len()));
    assert!(
        atoms
            .iter()
            .all(|&atom| molecule.get_atomic_number(atom) != 1)
    );
}
