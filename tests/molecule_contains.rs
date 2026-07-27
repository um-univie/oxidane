use oxidane::io::{matches_cxsmarts, parse_cxsmarts};
use oxidane::molecule::{MolecularGraph, Molecule3D};

#[test]
fn inline_and_reusable_queries_are_inherent_methods() {
    // No extension traits or prelude needed for either method.
    let ethanol = MolecularGraph::from_smiles_single("CCO").unwrap();
    let acetone = MolecularGraph::from_smiles_single("CC(=O)C").unwrap();
    let alcohol = parse_cxsmarts("[OX2H1][CX4]").unwrap();
    assert!(ethanol.contains_smarts("[OX2H1][CX4]").unwrap());
    assert!(ethanol.contains(&alcohol));
    assert!(!acetone.contains_smarts("[OX2H1][CX4]").unwrap());
    assert!(!acetone.contains(&alcohol));
}

#[test]
fn convenience_methods_preserve_matching_semantics() {
    for (smiles, query, expected) in [
        ("C1CC1", "CCC", true),
        ("c1ccc2ccccc2c1", "c1ccccc1", true),
        ("CC(=O)O", "[$([CX3](=O)[OX2H1])]", true),
        ("CC(=O)OC", "[$([CX3](=O)[OX2H1])]", false),
        ("[13CH3][OH:7]", "[13C][O;H1]", true),
        ("[13CH3][OH:7]", "[12C]", false),
        ("F[C@H](Cl)Br", "F[C@H](Cl)Br", true),
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br", false),
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        let before = format!("{molecule:?}");
        let pattern = parse_cxsmarts(query).unwrap();
        assert_eq!(
            molecule.contains_smarts(query).unwrap(),
            expected,
            "{smiles}: {query}"
        );
        assert_eq!(molecule.contains(&pattern), expected, "{smiles}: {query}");
        assert_eq!(matches_cxsmarts(&molecule, query).unwrap(), expected);
        assert_eq!(format!("{molecule:?}"), before);
    }
}

#[test]
fn disconnected_queries_match_a_combined_graph() {
    let ammonium = MolecularGraph::from_smiles_single("[NH4+]").unwrap();
    let chloride = MolecularGraph::from_smiles_single("[Cl-]").unwrap();
    let salt = MolecularGraph::combine(&[&ammonium, &chloride]);
    assert!(salt.contains_smarts("[N+].[Cl-]").unwrap());
    assert!(!ammonium.contains_smarts("[N+].[Cl-]").unwrap());
}

#[test]
fn invalid_queries_are_errors_not_absent_patterns() {
    let graph = MolecularGraph::from_smiles_single("CCO").unwrap();
    let spatial = Molecule3D::default();
    for query in ["", " ", "[", "C1CC", "C(C", "C=", "[C@TB21]"] {
        let expected = parse_cxsmarts(query).unwrap_err().to_string();
        assert_eq!(
            graph.contains_smarts(query).unwrap_err().to_string(),
            expected
        );
        assert_eq!(
            spatial.contains_smarts(query).unwrap_err().to_string(),
            expected
        );
    }
    assert!(!graph.contains_smarts("N").unwrap());
    assert!(!MolecularGraph::default().contains_smarts("C").unwrap());
    assert!(!spatial.contains_smarts("C").unwrap());
}

#[test]
fn spatial_queries_preserve_metadata_and_ignore_coordinates() {
    use oxidane::molecule::base::MoleculeCore;
    use oxidane::vector::Vector3D;

    let graph = MolecularGraph::from_smiles_single("[13CH3:7][C@H](F)[NH3+]").unwrap();
    let mut spatial = Molecule3D {
        atomic_numbers: graph.atomic_numbers.clone(),
        atom_classes: graph.atom_classes.clone(),
        charges: graph.charges.clone(),
        chiral_classes: graph.chiral_classes.clone(),
        isotopes: graph.isotopes.clone(),
        radical_states: graph.radical_states.clone(),
        atom_bonds: graph.atom_bonds().to_vec(),
        positions: vec![Vector3D::default(); graph.len()],
    };
    for (query, expected) in [
        ("[13C][C@H](F)[N+]", true),
        ("[13C][C@@H](F)[N+]", false),
        ("[14C]", false),
        ("[N+]", true),
        ("[N+0]", false),
    ] {
        let before = format!("{spatial:?}");
        let pattern = parse_cxsmarts(query).unwrap();
        assert_eq!(spatial.contains_smarts(query).unwrap(), expected, "{query}");
        assert_eq!(spatial.contains(&pattern), expected, "{query}");
        assert_eq!(format!("{spatial:?}"), before);
        spatial.positions.fill(Vector3D::new(100.0, -50.0, 2.0));
        assert_eq!(spatial.contains(&pattern), expected);
    }
}
