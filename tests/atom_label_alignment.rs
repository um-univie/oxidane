use oxidane::prelude::*;

#[test]
fn adding_hydrogens_keeps_maps_aligned_and_is_idempotent() {
    for smiles in ["[CH4:7]", "[13CH3:7][OH:8]", "N[C@@H:9](C)C(=O)[O-]"] {
        let mut molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        let classes = molecule.atom_classes().as_ref().unwrap();
        assert_eq!(classes.len(), molecule.len(), "{smiles}");
        for (atom, &number) in molecule.atomic_numbers().iter().enumerate() {
            if number == 1 {
                assert_eq!(classes[atom], 0, "generated H must be unmapped");
            }
        }
        let before = molecule.to_smiles();
        let labels = classes.clone();
        let count = molecule.len();
        molecule.add_hydrogens();
        molecule.add_hydrogens();
        assert_eq!(molecule.len(), count);
        assert_eq!(molecule.atom_classes().as_ref().unwrap(), &labels);
        assert_eq!(molecule.to_smiles(), before);
    }
}

#[test]
fn adding_the_first_mapped_atom_backfills_unmapped_atoms() {
    let mut molecule = MolecularGraph::from_smiles_single("C").unwrap();
    assert!(molecule.atom_classes().is_none());
    let initial = molecule.len();
    molecule.add_atom(Atom::new(10).with_atom_class(Some(42)));
    let mut expected = vec![0; initial];
    expected.push(42);
    assert_eq!(molecule.atom_classes().as_ref(), Some(&expected));
    molecule.add_atom(Atom::new(18));
    expected.push(0);
    assert_eq!(molecule.atom_classes().as_ref(), Some(&expected));
    assert_eq!(
        molecule.atom_classes().as_ref().unwrap().len(),
        molecule.len()
    );
}

#[test]
fn atom_addition_does_not_create_maps_when_none_are_present() {
    let mut molecule = MolecularGraph::from_smiles_single("C").unwrap();
    molecule.add_atom(Atom::new(10));
    molecule.add_hydrogens();
    assert!(molecule.atom_classes().is_none());
}
