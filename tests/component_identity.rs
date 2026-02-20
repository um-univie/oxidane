use oxidane::prelude::*;
use oxidane::vector::Vector3D;

fn disconnected(smiles: &str) -> Molecule3D {
    let mut result = Molecule3D {
        isotopes: Some(vec![]),
        chiral_classes: Some(vec![]),
        atom_classes: Some(vec![]),
        ..Default::default()
    };
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
            .extend((offset..offset + graph.len()).map(|i| Vector3D::new(i as f64, 0., 0.)));
        result
            .atom_classes
            .as_mut()
            .unwrap()
            .extend(offset..offset + graph.len());
        result
            .isotopes
            .as_mut()
            .unwrap()
            .extend(graph.isotopes().cloned().unwrap_or(vec![0; graph.len()]));
        result.chiral_classes.as_mut().unwrap().extend(
            graph
                .chiral_classes()
                .map(|v| v.to_vec())
                .unwrap_or(vec![oxidane::io::ChiralClass::None; graph.len()]),
        );
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
fn splitting_preserves_every_atom_and_its_identity() {
    for smiles in [
        "[13CH4].[Na+]",
        "N[C@@H](C)C(=O)O.[Na+]",
        "N#N.O",
        "[O-].[15NH4+]",
        "F/C=C/F.Cl",
    ] {
        let molecule = disconnected(smiles);
        let fragments = molecule.split_components();
        assert_eq!(
            fragments.iter().map(|f| f.len()).sum::<usize>(),
            molecule.len(),
            "{smiles}"
        );
        let expected = MolecularGraph::from_smiles(smiles).unwrap();
        assert_eq!(fragments.len(), expected.len());
        for (fragment, original) in fragments.iter().zip(expected) {
            // Maps are synthetic labels here, not part of the SMILES control.
            let mut unlabelled = fragment.clone();
            unlabelled.atom_classes = None;
            assert_eq!(unlabelled.to_smiles(), original.to_smiles(), "{smiles}");
            for (local, &global) in fragment.atom_classes.as_ref().unwrap().iter().enumerate() {
                assert_eq!(fragment.positions[local].x, global as f64);
                assert_eq!(
                    fragment.isotopes.as_ref().unwrap()[local],
                    molecule.isotopes.as_ref().unwrap()[global]
                );
                assert_eq!(
                    fragment.chiral_classes.as_ref().unwrap()[local],
                    molecule.chiral_classes.as_ref().unwrap()[global]
                );
                let remapped: Vec<_> = fragment.atom_bonds[local]
                    .iter()
                    .map(|b| fragment.atom_classes.as_ref().unwrap()[b.target()])
                    .collect();
                assert_eq!(
                    remapped,
                    molecule.atom_bonds[global]
                        .iter()
                        .map(|b| b.target())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn empty_and_single_component_splits_are_lossless() {
    assert!(Molecule3D::default().split_components().is_empty());
    let molecule = disconnected("[13CH4]");
    let fragments = molecule.split_components();
    assert_eq!(fragments.len(), 1);
    assert_eq!(fragments[0].to_smiles(), molecule.to_smiles());
}
