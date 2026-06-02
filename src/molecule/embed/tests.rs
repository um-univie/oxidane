use super::*;
use crate::io::ToSMILES;
use crate::molecule::molecular_graph::MolecularGraph;

#[test]
fn extracted_components_keep_their_own_isotopes_and_atom_labels() {
    let originals =
        MolecularGraph::from_smiles("[Ne].[13CH3:7][18OH:8].N[C@@H:9](C)C(=O)[O-]").unwrap();
    let combined = MolecularGraph::combine(&originals.iter().collect::<Vec<_>>());
    let molecule = to_molecule3d_topology(&combined);
    let components = molecule.get_components();
    assert_eq!(components.len(), originals.len());

    for (atoms, original) in components.iter().zip(&originals) {
        let fragment = extract_component(&molecule, atoms);
        assert_eq!(fragment.len(), original.len());
        assert_eq!(
            fragment.isotopes.as_ref().unwrap(),
            &original
                .isotopes()
                .cloned()
                .unwrap_or(vec![0; original.len()]),
        );
        assert_eq!(
            fragment.atom_classes.as_ref().unwrap(),
            &original
                .atom_classes()
                .clone()
                .unwrap_or(vec![0; original.len()]),
        );
        assert_eq!(fragment.to_smiles(), original.to_smiles());
        assert!((fragment.monoisotopic_mass() - original.monoisotopic_mass()).abs() < 1e-10);
        assert_eq!(fragment.atom_bonds(), original.atom_bonds());
        assert_eq!(fragment.charges(), original.charges());
        assert_eq!(fragment.positions.len(), fragment.len());
    }
}

#[test]
fn extracted_components_leave_absent_metadata_absent() {
    let originals = MolecularGraph::from_smiles("CO.CC").unwrap();
    let combined = MolecularGraph::combine(&originals.iter().collect::<Vec<_>>());
    let molecule = to_molecule3d_topology(&combined);
    assert!(molecule.isotopes.is_none());
    assert!(molecule.atom_classes.is_none());

    for atoms in molecule.get_components() {
        let fragment = extract_component(&molecule, &atoms);
        assert!(fragment.isotopes.is_none());
        assert!(fragment.atom_classes.is_none());
        assert!(fragment.chiral_classes.is_none());
        assert!(
            fragment
                .atom_bonds()
                .iter()
                .flatten()
                .all(|bond| bond.target() < fragment.len())
        );
    }
}
