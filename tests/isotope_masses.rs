use oxidane::prelude::*;
use oxidane::molecule::descriptors::MolecularDescriptors;

#[test]
fn explicit_isotopes_contribute_their_own_masses() {
    for (smiles, expected) in [
        ("[13CH4]", 13.00335483507 + 4.0 * 1.00782503223),
        ("[2H]O[2H]", 2.0 * 2.01410177812 + 15.99491461957),
        ("[18OH2]", 17.99915961286 + 2.0 * 1.00782503223),
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        assert!(
            (MoleculeCore::monoisotopic_mass(&molecule) - expected).abs() < 1e-8,
            "{smiles}"
        );
        assert!(
            (MolecularDescriptors::monoisotopic_mass(&molecule) - expected).abs() < 1e-8,
            "{smiles}"
        );
    }
    let ordinary = MolecularGraph::from_smiles_single("C").unwrap();
    let labelled = MolecularGraph::from_smiles_single("[13CH4]").unwrap();
    assert!(
        (labelled.molecular_weight() - ordinary.molecular_weight() - (13.00335483507 - 12.0106))
            .abs()
            < 1e-8
    );
    let mut atom = Atom::new(6);
    atom.isotope = Some(13);
    assert!((atom.monoisotopic_mass().unwrap() - 13.00335483507).abs() < 1e-8);
}

#[test]
fn unspecified_and_wildcard_isotopes_keep_their_meaning() {
    let molecule = MolecularGraph::from_smiles_single("C").unwrap();
    assert!((MoleculeCore::monoisotopic_mass(&molecule) - 16.03130012892).abs() < 1e-8);
    let labelled_dummy = MolecularGraph::from_smiles_single("[13*]").unwrap();
    assert_eq!(labelled_dummy.molecular_weight(), 0.0);
    assert_eq!(MoleculeCore::monoisotopic_mass(&labelled_dummy), 0.0);
}

#[test]
fn unavailable_isotope_masses_do_not_silently_fall_back() {
    let mut molecule = MolecularGraph::from_smiles_single("C").unwrap();
    *molecule.isotopes_mut() = Some(vec![999]);
    assert!(molecule.molecular_weight().is_nan());
    assert!(MoleculeCore::monoisotopic_mass(&molecule).is_nan());
    assert!(MolecularDescriptors::monoisotopic_mass(&molecule).is_nan());
    let mut atom = Atom::new(6);
    atom.isotope = Some(999);
    assert_eq!(atom.monoisotopic_mass(), None);
    atom.isotope = Some(0);
    assert_eq!(atom.monoisotopic_mass(), Some(12.0));
}
