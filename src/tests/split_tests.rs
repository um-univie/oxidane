#[cfg(test)]
mod tests {
    use crate::atom::Atom;
    use crate::molecule::{
        SplitError,
        bond::{BondOrder, BondTarget},
        molecular_graph::MolecularGraph,
        traits::*,
    };
    

    /// Helper function to create a simple test molecule
    /// Creates a linear chain: 0-1-2-3
    fn create_test_molecule_chain() -> MolecularGraph {
        let mut atoms = Vec::new();

        // Create atoms with bonds
        // Atom 0: connected to 1
        let mut atom0 = Atom::new(6); // Carbon
        atom0.bonds.push(BondTarget::single(1));
        atoms.push(atom0);

        // Atom 1: connected to 0 and 2
        let mut atom1 = Atom::new(6);
        atom1.bonds.push(BondTarget::single(0));
        atom1.bonds.push(BondTarget::single(2));
        atoms.push(atom1);

        // Atom 2: connected to 1 and 3
        let mut atom2 = Atom::new(6);
        atom2.bonds.push(BondTarget::single(1));
        atom2.bonds.push(BondTarget::single(3));
        atoms.push(atom2);

        // Atom 3: connected to 2
        let mut atom3 = Atom::new(6);
        atom3.bonds.push(BondTarget::single(2));
        atoms.push(atom3);

        MolecularGraph::from_atoms(&atoms)
    }

    /// Helper function to create a disconnected molecule
    /// Creates: 0-1  2-3 (two separate pairs)
    fn create_test_molecule_disconnected() -> MolecularGraph {
        let mut atoms = Vec::new();

        // Atom 0: connected to 1
        let mut atom0 = Atom::new(6);
        atom0.bonds.push(BondTarget::single(1));
        atoms.push(atom0);

        // Atom 1: connected to 0
        let mut atom1 = Atom::new(6);
        atom1.bonds.push(BondTarget::single(0));
        atoms.push(atom1);

        // Atom 2: connected to 3
        let mut atom2 = Atom::new(6);
        atom2.bonds.push(BondTarget::single(3));
        atoms.push(atom2);

        // Atom 3: connected to 2
        let mut atom3 = Atom::new(6);
        atom3.bonds.push(BondTarget::single(2));
        atoms.push(atom3);

        MolecularGraph::from_atoms(&atoms)
    }

    /// Helper function to create a cyclic molecule
    /// Creates a 4-membered ring: 0-1-2-3-0
    fn create_test_molecule_ring() -> MolecularGraph {
        let mut atoms = Vec::new();

        // Atom 0: connected to 1 and 3
        let mut atom0 = Atom::new(6);
        atom0.bonds.push(BondTarget::single(1));
        atom0.bonds.push(BondTarget::single(3));
        atoms.push(atom0);

        // Atom 1: connected to 0 and 2
        let mut atom1 = Atom::new(6);
        atom1.bonds.push(BondTarget::single(0));
        atom1.bonds.push(BondTarget::single(2));
        atoms.push(atom1);

        // Atom 2: connected to 1 and 3
        let mut atom2 = Atom::new(6);
        atom2.bonds.push(BondTarget::single(1));
        atom2.bonds.push(BondTarget::single(3));
        atoms.push(atom2);

        // Atom 3: connected to 2 and 0
        let mut atom3 = Atom::new(6);
        atom3.bonds.push(BondTarget::single(2));
        atom3.bonds.push(BondTarget::single(0));
        atoms.push(atom3);

        MolecularGraph::from_atoms(&atoms)
    }

    #[test]
    fn test_split_disconnected_molecule_success() {
        let molecule = create_test_molecule_disconnected();

        // Split into two groups: [0,1] and [2,3]
        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = molecule.split(groups);

        assert!(result.is_ok());
        let molecules = result.unwrap();
        assert_eq!(molecules.len(), 2);

        // Check first molecule
        assert_eq!(molecules[0].atomic_numbers().len(), 2);
        assert_eq!(molecules[0].atomic_numbers, vec![6, 6]);
        assert_eq!(molecules[0].atom_bonds().len(), 2);
        assert_eq!(molecules[0].atom_bonds()[0].len(), 1); // First atom has one bond
        assert_eq!(molecules[0].atom_bonds()[1].len(), 1); // Second atom has one bond
        assert_eq!(molecules[0].atom_bonds()[0][0].target(), 1); // Bond to remapped index 1
        assert_eq!(molecules[0].atom_bonds()[1][0].target(), 0); // Bond to remapped index 0

        // Check second molecule
        assert_eq!(molecules[1].atomic_numbers().len(), 2);
        assert_eq!(molecules[1].atomic_numbers, vec![6, 6]);
        assert_eq!(molecules[1].atom_bonds().len(), 2);
        assert_eq!(molecules[1].atom_bonds()[0].len(), 1); // First atom has one bond
        assert_eq!(molecules[1].atom_bonds()[1].len(), 1); // Second atom has one bond
        assert_eq!(molecules[1].atom_bonds()[0][0].target(), 1); // Bond to remapped index 1
        assert_eq!(molecules[1].atom_bonds()[1][0].target(), 0); // Bond to remapped index 0
    }

    #[test]
    fn test_split_connected_molecule_fails() {
        let molecule = create_test_molecule_chain();

        // Try to split connected molecule: [0,1] and [2,3]
        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = molecule.split(groups);

        assert!(result.is_err());
        match result.unwrap_err() {
            SplitError::BondWouldBreak {
                atom1,
                atom2,
                bond_order,
            } => {
                // Should detect the bond between atoms 1 and 2
                assert!((atom1 == 1 && atom2 == 2) || (atom1 == 2 && atom2 == 1));
                assert_eq!(bond_order, BondOrder::Single);
            }
            _ => panic!("Expected BondWouldBreak error"),
        }
    }

    #[test]
    fn test_split_single_atom_groups() {
        let molecule = create_test_molecule_disconnected();

        // Split into individual atoms: [0], [1], [2], [3]
        let groups = vec![vec![0], vec![1], vec![2], vec![3]];
        let result = molecule.split(groups);

        // This should fail because atoms 0-1 and 2-3 are bonded
        assert!(result.is_err());
    }

    #[test]
    fn test_split_invalid_atom_index() {
        let molecule = create_test_molecule_chain();

        // Try to include non-existent atom index
        let groups = vec![vec![0, 1], vec![2, 10]]; // atom 10 doesn't exist
        let result = molecule.split(groups);

        assert!(result.is_err());
        match result.unwrap_err() {
            SplitError::InvalidAtomIndex(idx) => {
                assert_eq!(idx, 10);
            }
            _ => panic!("Expected InvalidAtomIndex error"),
        }
    }

    #[test]
    fn test_split_overlapping_groups() {
        let molecule = create_test_molecule_chain();

        // Create overlapping groups: atom 1 appears in both groups
        let groups = vec![vec![0, 1], vec![1, 2]];
        let result = molecule.split(groups);

        assert!(result.is_err());
        match result.unwrap_err() {
            SplitError::OverlappingGroups(atom) => {
                assert_eq!(atom, 1);
            }
            _ => panic!("Expected OverlappingGroups error"),
        }
    }

    #[test]
    fn test_split_empty_groups() {
        let molecule = create_test_molecule_chain();

        // Test with empty groups
        let groups = vec![vec![], vec![]];
        let result = molecule.split(groups);

        assert!(result.is_err());
        match result.unwrap_err() {
            SplitError::EmptyGroups => {}
            _ => panic!("Expected EmptyGroups error"),
        }
    }

    #[test]
    fn test_split_ring_molecule_fails() {
        let molecule = create_test_molecule_ring();

        // Try to split a ring molecule
        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = molecule.split(groups);

        // This should fail because the ring has bonds crossing group boundaries
        assert!(result.is_err());
        match result.unwrap_err() {
            SplitError::BondWouldBreak { .. } => {}
            _ => panic!("Expected BondWouldBreak error"),
        }
    }

    #[test]
    fn test_split_preserves_properties() {
        // Create a molecule with some properties
        let mut molecule = create_test_molecule_disconnected();
        molecule.charges = vec![1, -1, 0, 0];
        molecule.radical_states = vec![0, 1, 0, 0];
        molecule.atom_classes = Some(vec![1, 2, 3, 4]);
        molecule.isotopes = Some(vec![12, 13, 12, 12]);
        molecule.partial_charges = Some(vec![0.1, -0.1, 0.0, 0.0]);

        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = molecule.split(groups);

        assert!(result.is_ok());
        let molecules = result.unwrap();

        // Check that properties are preserved in first molecule
        assert_eq!(molecules[0].charges, vec![1, -1]);
        assert_eq!(molecules[0].radical_states, vec![0, 1]);
        assert_eq!(molecules[0].atom_classes, Some(vec![1, 2]));
        assert_eq!(molecules[0].isotopes, Some(vec![12, 13]));
        assert_eq!(molecules[0].partial_charges, Some(vec![0.1, -0.1]));

        // Check that properties are preserved in second molecule
        assert_eq!(molecules[1].charges, vec![0, 0]);
        assert_eq!(molecules[1].radical_states, vec![0, 0]);
        assert_eq!(molecules[1].atom_classes, Some(vec![3, 4]));
        assert_eq!(molecules[1].isotopes, Some(vec![12, 12]));
        assert_eq!(molecules[1].partial_charges, Some(vec![0.0, 0.0]));
    }

    #[test]
    fn test_split_single_group_returns_copy() {
        let molecule = create_test_molecule_chain();

        // Split with single group containing all atoms
        let groups = vec![vec![0, 1, 2, 3]];
        let result = molecule.split(groups);

        assert!(result.is_ok());
        let molecules = result.unwrap();
        assert_eq!(molecules.len(), 1);

        // The result should be equivalent to the original
        let split_molecule = &molecules[0];
        assert_eq!(
            split_molecule.atomic_numbers().len(),
            molecule.atomic_numbers().len()
        );
        assert_eq!(split_molecule.atomic_numbers, molecule.atomic_numbers);
        assert_eq!(split_molecule.charges, molecule.charges);
        assert_eq!(split_molecule.radical_states, molecule.radical_states);

        // Check that bonds are preserved
        for (i, bonds) in split_molecule.atom_bonds().iter().enumerate() {
            assert_eq!(bonds.len(), molecule.atom_bonds()[i].len());
            for (j, bond) in bonds.iter().enumerate() {
                assert_eq!(bond.target(), molecule.atom_bonds()[i][j].target());
                assert_eq!(bond.bond_order(), molecule.atom_bonds()[i][j].bond_order());
            }
        }
    }

    #[test]
    fn test_split_handles_none_optional_fields() {
        // Create a simple molecule without any optional properties
        let mut atoms = Vec::new();

        // Create atoms without isotopes, atom classes, or partial charges
        let mut atom0 = Atom::new(6);
        atom0.bonds.push(BondTarget::single(1));
        atoms.push(atom0);

        let mut atom1 = Atom::new(6);
        atom1.bonds.push(BondTarget::single(0));
        atoms.push(atom1);

        let mut atom2 = Atom::new(8);
        atom2.bonds.push(BondTarget::single(3));
        atoms.push(atom2);

        let mut atom3 = Atom::new(8);
        atom3.bonds.push(BondTarget::single(2));
        atoms.push(atom3);

        let molecule = MolecularGraph::from_atoms(&atoms);

        // Verify original molecule has None for optional fields
        assert!(molecule.atom_classes.is_none());
        assert!(molecule.isotopes.is_none());
        assert!(molecule.partial_charges.is_none());
        assert!(molecule.chiral_classes.is_none());

        // Split into two groups
        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = molecule.split(groups);

        assert!(result.is_ok());
        let molecules = result.unwrap();
        assert_eq!(molecules.len(), 2);

        // Verify that split molecules also have None for optional fields
        for mol in &molecules {
            assert!(
                mol.atom_classes.is_none(),
                "Split molecule should have None atom_classes"
            );
            assert!(
                mol.isotopes.is_none(),
                "Split molecule should have None isotopes"
            );
            assert!(
                mol.partial_charges.is_none(),
                "Split molecule should have None partial_charges"
            );
            assert!(
                mol.chiral_classes.is_none(),
                "Split molecule should have None chiral_classes"
            );
        }
    }

    #[test]
    fn test_split_handles_some_optional_fields() {
        // Create a molecule with some optional properties to ensure they're preserved
        let mut atoms = Vec::new();

        // Create atoms with optional properties
        let mut atom0 = Atom::new(6)
            .with_isotope(13)
            .with_atom_class(Some(1))
            .with_charge(1);
        atom0.partial_charge = Some(0.5);
        atom0.bonds.push(BondTarget::single(1));
        atoms.push(atom0);

        let mut atom1 = Atom::new(6)
            .with_isotope(12)
            .with_atom_class(Some(2))
            .with_charge(-1);
        atom1.partial_charge = Some(-0.5);
        atom1.bonds.push(BondTarget::single(0));
        atoms.push(atom1);

        let mut atom2 = Atom::new(8).with_isotope(16).with_atom_class(Some(3));
        atom2.partial_charge = Some(0.0);
        atom2.bonds.push(BondTarget::single(3));
        atoms.push(atom2);

        let mut atom3 = Atom::new(8).with_isotope(18).with_atom_class(Some(4));
        atom3.partial_charge = Some(0.0);
        atom3.bonds.push(BondTarget::single(2));
        atoms.push(atom3);

        let molecule = MolecularGraph::from_atoms(&atoms);

        // Verify original molecule has Some for optional fields
        assert!(molecule.atom_classes.is_some());
        assert!(molecule.isotopes.is_some());
        assert!(molecule.partial_charges.is_some());
        // chiral_classes should still be None since no atoms have chirality
        assert!(molecule.chiral_classes.is_none());

        // Split into two groups
        let groups = vec![vec![0, 1], vec![2, 3]];
        let result = molecule.split(groups);

        assert!(result.is_ok());
        let molecules = result.unwrap();
        assert_eq!(molecules.len(), 2);

        // Verify that split molecules preserve the optional fields
        for mol in &molecules {
            assert!(
                mol.atom_classes.is_some(),
                "Split molecule should preserve atom_classes"
            );
            assert!(
                mol.isotopes.is_some(),
                "Split molecule should preserve isotopes"
            );
            assert!(
                mol.partial_charges.is_some(),
                "Split molecule should preserve partial_charges"
            );
            assert!(
                mol.chiral_classes.is_none(),
                "Split molecule should keep None chiral_classes"
            );
        }

        // Check specific values are correctly mapped
        assert_eq!(molecules[0].atom_classes.as_ref().unwrap(), &vec![1, 2]);
        assert_eq!(molecules[0].isotopes.as_ref().unwrap(), &vec![13, 12]);
        assert_eq!(
            molecules[0].partial_charges.as_ref().unwrap(),
            &vec![0.5, -0.5]
        );

        assert_eq!(molecules[1].atom_classes.as_ref().unwrap(), &vec![3, 4]);
        assert_eq!(molecules[1].isotopes.as_ref().unwrap(), &vec![16, 18]);
        assert_eq!(
            molecules[1].partial_charges.as_ref().unwrap(),
            &vec![0.0, 0.0]
        );
    }

    #[test]
    fn test_split_performance_with_large_molecule() {
        // Create a larger disconnected molecule for performance testing
        let mut atoms = Vec::new();
        let num_pairs = 50; // 100 atoms in 50 pairs

        // Create disconnected pairs: 0-1, 2-3, 4-5, etc.
        for i in 0..num_pairs {
            let atom1_idx = i * 2;
            let atom2_idx = i * 2 + 1;

            // First atom of pair
            let mut atom1 = Atom::new(6);
            atom1.bonds.push(BondTarget::single(atom2_idx));
            atoms.push(atom1);

            // Second atom of pair
            let mut atom2 = Atom::new(6);
            atom2.bonds.push(BondTarget::single(atom1_idx));
            atoms.push(atom2);
        }

        let molecule = MolecularGraph::from_atoms(&atoms);

        // Create groups for each pair
        let groups: Vec<Vec<usize>> = (0..num_pairs).map(|i| vec![i * 2, i * 2 + 1]).collect();

        // This should succeed quickly with optimized data structures
        let start = std::time::Instant::now();
        let result = molecule.split(groups);
        let duration = start.elapsed();

        assert!(result.is_ok());
        let molecules = result.unwrap();
        assert_eq!(molecules.len(), num_pairs);

        // Should complete quickly (within reasonable time)
        assert!(
            duration.as_millis() < 100,
            "Split took too long: {:?}",
            duration
        );

        // Verify each resulting molecule has 2 atoms
        for mol in &molecules {
            assert_eq!(mol.atomic_numbers().len(), 2);
            assert_eq!(mol.atom_bonds()[0].len(), 1);
            assert_eq!(mol.atom_bonds()[1].len(), 1);
        }
    }
}
