use oxidane::prelude::*;

#[test]
fn ordinary_hypervalent_and_charged_atoms_are_plausible() {
    for smiles in [
        "OS(=O)(=O)O",
        "OP(=O)(O)O",
        "CS(=O)C",
        "FS(F)(F)(F)(F)F",
        "P(Cl)(Cl)(Cl)(Cl)Cl",
        "[NH4+]",
        "[OH-]",
        "[BH4-]",
        "[CH3]",
        "[CH2]",
        "[He]",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        for atom in 0..molecule.len() {
            assert!(
                molecule.has_plausible_valency(atom),
                "{smiles}, atom {atom}"
            );
        }
    }
}

#[test]
fn excess_bonds_and_radicals_are_rejected() {
    // Construct connectivity directly so this checks the validator, not the parser.
    for (element, bonds, radicals) in [
        (6, 5, 0),
        (7, 4, 0),
        (8, 3, 0),
        (15, 6, 0),
        (15, 8, 0),
        (16, 7, 0),
        (16, 8, 0),
        (6, 4, 1),
        (6, 3, 2),
    ] {
        let mut molecule = MolecularGraph::from_smiles_single("C").unwrap();
        molecule.atomic_numbers_mut()[0] = element;
        let count = molecule.len();
        molecule.radical_states_mut().resize(count, 0);
        molecule.radical_states_mut()[0] = radicals;
        molecule.atom_bonds_mut()[0].clear();
        for _ in 0..bonds {
            molecule.atom_bonds_mut()[0].push(oxidane::molecule::bond::BondTarget::new(
                1,
                BondOrder::Single,
            ));
        }
        assert!(
            !molecule.has_plausible_valency(0),
            "Z={element}, bonds={bonds}, radicals={radicals}"
        );
    }
}
