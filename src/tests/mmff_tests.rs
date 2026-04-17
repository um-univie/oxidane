//! MMFF94 atom typing and charges, against what RDKit assigns the same
//! molecules.
//!
//! Every number here was taken from RDKit's
//! `MMFFGetMoleculeProperties`, and `tests/mmff_oracle.py` is the harness
//! that compares the two sides atom by atom over a corpus. What these tests add is that the corpus run
//! needs RDKit installed, and `cargo test` does not: they pin one molecule
//! from each family the typing rules distinguish, so a rule that quietly
//! stops firing is caught here first.

#[cfg(test)]
mod tests {
    use crate::molecule::base::MoleculeCore;
    use crate::molecule::forcefield::mmff;
    use crate::prelude::*;

    /// The MMFF type number of every atom, hydrogens included and added.
    fn types(smiles: &str) -> Vec<u8> {
        let mut molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        molecule.add_hydrogens();
        mmff::atom_types(&molecule)
            .expect("should type")
            .iter()
            .map(|kind| kind.number())
            .collect()
    }

    /// Every atom's MMFF partial charge, hydrogens included and added.
    fn charges(smiles: &str) -> Vec<f64> {
        let mut molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        molecule.add_hydrogens();
        mmff::partial_charges(&molecule).expect("should charge")
    }

    fn assert_charges(smiles: &str, expected: &[f64]) {
        let ours = charges(smiles);
        assert_eq!(ours.len(), expected.len(), "{smiles} has {} atoms", ours.len());
        for (atom, (ours, theirs)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (ours - theirs).abs() < 1e-4,
                "{smiles} atom {atom}: {ours:+.4} against RDKit's {theirs:+.4}"
            );
        }
    }

    /// The heavy-atom types only, in SMILES order.
    fn heavy_types(smiles: &str) -> Vec<u8> {
        let mut molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        molecule.add_hydrogens();
        let assigned = mmff::atom_types(&molecule).expect("should type");
        (0..molecule.len())
            .filter(|&atom| molecule.atomic_numbers()[atom] != 1)
            .map(|atom| assigned[atom].number())
            .collect()
    }

    #[test]
    fn aliphatic_carbon_nitrogen_and_oxygen_types() {
        // Alkyl carbon, carbonyl carbon, carbonyl oxygen, amide nitrogen.
        assert_eq!(heavy_types("CC(=O)N"), vec![1, 3, 7, 10]);
        // Carboxylate: the carbon is CO2M and both oxygens share the charge.
        assert_eq!(heavy_types("CC(=O)[O-]"), vec![1, 41, 32, 32]);
        // Acid: a carbonyl oxygen and a divalent one, typed apart.
        assert_eq!(heavy_types("CC(=O)O"), vec![1, 3, 7, 6]);
        // Nitrile carbon and its triply bonded nitrogen.
        assert_eq!(heavy_types("CC#N"), vec![1, 4, 42]);
        // Nitro nitrogen, with both oxygens again equivalent.
        assert_eq!(heavy_types("C[N+](=O)[O-]"), vec![1, 45, 32, 32]);
        // Water, which has a type of its own rather than being an ether.
        assert_eq!(types("O"), vec![70, 31, 31]);
    }

    /// Ring size is part of the type for saturated carbon, and a double bond
    /// in a four-ring gets its own type again.
    #[test]
    fn small_ring_carbon_types() {
        assert_eq!(heavy_types("C1CC1"), vec![22, 22, 22]);
        assert_eq!(heavy_types("C1CCC1"), vec![20, 20, 20, 20]);
        assert_eq!(heavy_types("C1=CC1"), vec![2, 2, 22]);
        assert_eq!(heavy_types("C1=CCC1"), vec![30, 30, 20, 20]);
        assert_eq!(heavy_types("C1CCCCC1"), vec![1; 6]);
    }

    /// The five-ring aromatics, where a carbon's type says whether the
    /// heteroatom is next to it or one further round.
    #[test]
    fn five_membered_aromatic_types() {
        // Thiophene: two carbons alpha to sulfur, two beta.
        assert_eq!(heavy_types("c1ccsc1"), vec![64, 64, 63, 44, 63]);
        // Pyrrole, whose nitrogen holds its lone pair in the ring.
        assert_eq!(heavy_types("c1cc[nH]c1"), vec![64, 64, 63, 39, 63]);
        // Furan.
        assert_eq!(heavy_types("c1ccoc1"), vec![64, 64, 63, 59, 63]);
        // Imidazole: one pyrrole-like nitrogen, one pyridine-like.
        assert_eq!(heavy_types("c1cnc[nH]1"), vec![63, 64, 66, 63, 39]);
    }

    /// Six-ring aromatics, and the two nitrogens that are neither pyridine's
    /// nor pyrrole's.
    #[test]
    fn six_membered_aromatic_types() {
        assert_eq!(heavy_types("c1ccccc1"), vec![37; 6]);
        assert_eq!(heavy_types("c1ccncc1"), vec![37, 37, 37, 38, 37, 37]);
        assert_eq!(heavy_types("c1cc[n+](C)cc1"), vec![37, 37, 37, 58, 1, 37, 37]);
        // Pyridine N-oxide: the nitrogen is NPOX and the oxygen carries the
        // charge, not a carbonyl's type.
        assert_eq!(heavy_types("[O-][n+]1ccccc1"), vec![32, 69, 37, 37, 37, 37, 37]);
    }

    /// The delocalized cations, which are typed as a group rather than atom
    /// by atom: every nitrogen of a guanidinium is NGD+, whichever one the
    /// SMILES wrote the charge on.
    #[test]
    fn delocalized_cation_types() {
        assert_eq!(heavy_types("NC(N)=[NH2+]"), vec![56, 57, 56, 56]);
        assert_eq!(heavy_types("CC(=[NH2+])N"), vec![1, 57, 55, 55]);
        // Imidazolium fused to benzene: CIM+ between two NIM+ nitrogens.
        assert_eq!(
            heavy_types("C[n+]1cn(C)c2ccccc21"),
            vec![1, 81, 80, 81, 1, 78, 37, 37, 37, 37, 78]
        );
    }

    /// Sulfur, where the oxidation state decides the type.
    #[test]
    fn sulfur_types() {
        assert_eq!(heavy_types("CSC"), vec![1, 15, 1]);
        assert_eq!(heavy_types("CS(=O)C"), vec![1, 17, 7, 1]);
        assert_eq!(heavy_types("CS(=O)(=O)C"), vec![1, 18, 32, 32, 1]);
        assert_eq!(heavy_types("CC(C)=S"), vec![1, 3, 1, 16]);
    }

    /// Hydroxyl hydrogens are four different types depending on what the
    /// oxygen is attached to, which is the sort of distinction MMFF makes
    /// and UFF does not.
    #[test]
    fn hydroxyl_hydrogen_types() {
        // Alcohol, acid, phenol, water.
        assert_eq!(*types("CCO").last().unwrap(), 21);
        assert_eq!(*types("CC(=O)O").last().unwrap(), 24);
        assert_eq!(types("Oc1ccccc1")[7], 29);
        assert_eq!(types("O")[1], 31);
    }

    /// MMFF perceives aromaticity itself, and its rules exclude a ring whose
    /// atom carries a double bond leaving the ring. Quinone is written
    /// aromatic by many toolkits; to MMFF it is two carbonyls and four
    /// vinylic carbons, and typing it as benzene would be wrong by whole
    /// families of parameters.
    #[test]
    fn mmff_aromaticity_is_not_this_crate_s_aromaticity() {
        assert_eq!(heavy_types("O=C1C=CC(=O)C=C1"), vec![7, 3, 2, 2, 3, 7, 2, 2]);
    }

    /// The same molecule written two ways types the same, which is the
    /// property a rule keyed on neighbour order would break.
    #[test]
    fn typing_does_not_depend_on_how_the_smiles_was_written() {
        for (first, second) in [
            ("Oc1ccccc1", "c1ccccc1O"),
            ("CC(=O)Nc1ccccc1", "c1ccccc1NC(C)=O"),
            ("c1cc2ccccc2[nH]1", "c1ccc2[nH]ccc2c1"),
        ] {
            let mut ours = types(first);
            let mut theirs = types(second);
            ours.sort_unstable();
            theirs.sort_unstable();
            assert_eq!(ours, theirs, "{first} and {second} are the same molecule");
        }
    }

    /// MMFF94 has no type for carbonate's oxygens -- it is outside what the
    /// published rules enumerate, and RDKit refuses it too. Refusing is the
    /// point: a force field that types this molecule at all is inventing.
    #[test]
    fn a_molecule_outside_mmff94_is_refused() {
        let mut molecule = MolecularGraph::from_smiles_single("C(=O)([O-])[O-]").expect("parse");
        molecule.add_hydrogens();
        let error = mmff::atom_types(&molecule).expect_err("carbonate has no MMFF94 types");
        assert_eq!(error.scheme(), "MMFF94");
        assert_eq!(error.atomic_number, 8);
    }

    /// Charges on the molecules whose types the tests above pin. A charge is
    /// a sum of tabulated increments over shared formal charges, so it is
    /// either exactly RDKit's or the model behind it is wrong somewhere.
    #[test]
    fn partial_charges_match_rdkit() {
        assert_charges("CCO", &[0.0, 0.28, -0.68, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]);
        assert_charges("c1ccccc1", &[-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]);
        assert_charges(
            "CC(=O)Nc1ccccc1O",
            &[
                0.061, 0.569, -0.57, -0.547, 0.117, -0.15, -0.15, -0.15, -0.15, 0.0825, -0.5325,
                0.0, 0.0, 0.0, 0.37, 0.15, 0.15, 0.15, 0.15, 0.45,
            ],
        );
    }

    /// A carboxylate's two oxygens carry the same charge, because MMFF
    /// spreads the formal charge over both before any increment is added.
    /// Which oxygen the SMILES wrote the minus on does not survive that.
    #[test]
    fn a_shared_formal_charge_is_shared_equally() {
        assert_charges("CC(=O)[O-]", &[-0.106, 0.906, -0.9, -0.9, 0.0, 0.0, 0.0]);
        let nitrate = charges("O=[N+]([O-])[O-]");
        assert!((nitrate[0] - nitrate[2]).abs() < 1e-9);
        assert!((nitrate[2] - nitrate[3]).abs() < 1e-9);
        // Imidazolium's +1 sits on both nitrogens, not on the written one.
        let imidazolium = charges("C[n+]1cn(C)cc1");
        assert!((imidazolium[1] - imidazolium[3]).abs() < 1e-9);
    }

    /// Charges sum to the molecule's total charge, which is the one thing
    /// the increments cannot be allowed to break.
    #[test]
    fn charges_sum_to_the_molecular_charge() {
        for (smiles, total) in [
            ("CCO", 0.0),
            ("c1ccc(cc1)C(=O)N", 0.0),
            ("CC(=O)[O-]", -1.0),
            ("O=[N+]([O-])[O-]", -1.0),
            ("C[N+](C)(C)C", 1.0),
            ("NC(N)=[NH2+]", 1.0),
            ("[O-]C(=O)CC(=O)[O-]", -2.0),
        ] {
            let sum: f64 = charges(smiles).iter().sum();
            assert!((sum - total).abs() < 1e-9, "{smiles} sums to {sum}, not {total}");
        }
    }

    /// Every analytic gradient against a central difference of the energy it
    /// differentiates. The molecules between them exercise all seven terms:
    /// a strained ring for the angle and stretch-bend forms, an amide for
    /// out-of-plane, a conjugated system for the torsion, and charged
    /// species for the electrostatic one.
    #[test]
    fn mmff_gradients_match_finite_differences() {
        use crate::molecule::minimize::Energy;
        use crate::vector::Vector3D;

        const STEP: f64 = 1e-6;
        for smiles in [
            "CCO", "CC(C)C", "C=CC", "c1ccccc1", "CC(=O)NC", "C1CC1",
            "CS(=O)(=O)C", "CC(=O)[O-]", "C[NH3+]", "CC#N", "c1ccsc1",
        ] {
            let molecule = embedded(smiles);
            let field = mmff::ForceField::new(&molecule).expect("should type");
            let mut analytic = vec![Vector3D::default(); molecule.len()];
            field.energy_and_gradient(&molecule.positions, &mut analytic);

            for atom in 0..molecule.len() {
                for axis in 0..3 {
                    let mut moved = molecule.positions.clone();
                    let shift = |position: &mut Vector3D, by: f64| match axis {
                        0 => position.x += by,
                        1 => position.y += by,
                        _ => position.z += by,
                    };
                    shift(&mut moved[atom], STEP);
                    let forward = field.energy(&moved);
                    shift(&mut moved[atom], -2.0 * STEP);
                    let backward = field.energy(&moved);

                    let numeric = (forward - backward) / (2.0 * STEP);
                    let ours = match axis {
                        0 => analytic[atom].x,
                        1 => analytic[atom].y,
                        _ => analytic[atom].z,
                    };
                    let tolerance = 1e-3 * (1.0 + ours.abs());
                    assert!(
                        (ours - numeric).abs() < tolerance,
                        "{smiles} atom {atom} axis {axis}: {ours} against {numeric}"
                    );
                }
            }
        }
    }

    /// Minimisation lowers the energy and stops somewhere flat.
    #[test]
    fn mmff_minimisation_lowers_the_energy() {
        for smiles in ["CCO", "c1ccccc1", "CC(=O)NC", "C1CCCCC1"] {
            let mut molecule = embedded(smiles);
            let field = mmff::ForceField::new(&molecule).expect("should type");
            let before = field.energy(&molecule.positions);
            let after = field.minimize(&mut molecule.positions, 400);
            assert!(after <= before, "{smiles}: {after} is not below {before}");
        }
    }

    /// Distance geometry only, with the relaxation `embed_3d` normally
    /// finishes with switched off: what these tests want is a structure the
    /// force field has not already had its say in.
    fn embedded(smiles: &str) -> crate::molecule::molecule3d::Molecule3D {
        let config = crate::molecule::embed::EmbedConfig { relax: false, ..Default::default() };
        MolecularGraph::from_smiles_single(smiles)
            .expect("should parse")
            .embed_3d_with_config(&config)
            .expect("should embed")
    }
}
