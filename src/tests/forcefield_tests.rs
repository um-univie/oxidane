//! UFF, checked against RDKit's implementation of the same force field and
//! against finite differences.
//!
//! A force field is a large pile of arithmetic where every term produces a
//! plausible-looking number whether or not it is right, so neither check is
//! optional: the energies pin the parameters, the typing and the term
//! selection against an independent implementation, and the gradients pin
//! the derivatives against the energies they claim to differentiate.

#[cfg(test)]
mod tests {
    use crate::molecule::base::MoleculeCore;
    use crate::molecule::embed::Embed3D;
    use crate::molecule::forcefield::ForceField;
    use crate::molecule::minimize::Energy;
    use crate::prelude::*;

    /// Distance geometry only, with the relaxation `embed_3d` normally
    /// finishes with switched off -- what these tests want is a structure
    /// the force field has *not* seen, since a relaxed one is already at
    /// the minimum they are checking the minimiser can find.
    fn embedded(smiles: &str) -> crate::molecule::molecule3d::Molecule3D {
        let config = crate::molecule::embed::EmbedConfig {
            relax: false,
            ..Default::default()
        };
        MolecularGraph::from_smiles_single(smiles)
            .expect("should parse")
            .embed_3d_with_config(&config)
            .expect("should embed")
    }

    /// Every analytic gradient against a central difference of the energy it
    /// differentiates.
    ///
    /// Covers each term at least once: sp3 chains for bond/angle/torsion and
    /// non-bonded, a double bond and an aromatic ring for the conjugated
    /// torsion forms, a sulfone for the higher oxidation states.
    #[test]
    fn uff_gradients_match_finite_differences() {
        const STEP: f64 = 1e-6;
        for smiles in [
            "CCO",
            "CC(C)C",
            "C=CC",
            "c1ccccc1",
            "CS(=O)(=O)C",
            "CC(=O)OC",
            "CC(=O)N",
            "CP(C)C",
            "C[P+](C)(C)C",
        ] {
            let molecule = embedded(smiles);
            let field = ForceField::new(&molecule).expect("should type");

            // Nudged off the embedding first. A planar ring leaves its
            // torsions at exactly 0 or pi, which is where `acos` has
            // infinite slope: the analytic and numeric derivatives there
            // disagree about a singularity rather than about the formula.
            // Every real geometry a minimiser walks through is off it.
            let mut positions = molecule.positions.clone();
            let mut wobble = 0u64;
            for position in &mut positions {
                let mut next = || {
                    wobble = wobble.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (wobble >> 33) as f64 / (1u64 << 31) as f64 * 0.02 - 0.01
                };
                *position = Vector3D::new(
                    position.x + next(),
                    position.y + next(),
                    position.z + next(),
                );
            }

            let mut analytic = vec![Vector3D::default(); positions.len()];
            field.energy_and_gradient(&positions, &mut analytic);

            for atom in 0..positions.len() {
                for axis in 0..3 {
                    let mut shifted = positions.clone();
                    let nudge = |p: Vector3D, delta: f64| match axis {
                        0 => Vector3D::new(p.x + delta, p.y, p.z),
                        1 => Vector3D::new(p.x, p.y + delta, p.z),
                        _ => Vector3D::new(p.x, p.y, p.z + delta),
                    };
                    shifted[atom] = nudge(positions[atom], STEP);
                    let plus = field.energy(&shifted);
                    shifted[atom] = nudge(positions[atom], -STEP);
                    let minus = field.energy(&shifted);

                    let numeric = (plus - minus) / (2.0 * STEP);
                    let exact = match axis {
                        0 => analytic[atom].x,
                        1 => analytic[atom].y,
                        _ => analytic[atom].z,
                    };
                    // Relative, since a strained embedding carries gradients
                    // of a few hundred kcal/mol/A. Every term agrees to
                    // about 1e-9 on its own; the bound allows for the
                    // central difference's own truncation error on the
                    // Lennard-Jones term, which goes as `r^-12`.
                    let tolerance = 1e-5 * exact.abs().max(1.0);
                    assert!(
                        (exact - numeric).abs() < tolerance,
                        "{smiles} atom {atom} axis {axis}: analytic {exact}, \
                         finite difference {numeric}"
                    );
                }
            }
        }
    }

    /// Minimisation has to lower the energy and stop somewhere flat.
    #[test]
    fn minimisation_lowers_the_energy_and_settles() {
        for smiles in ["CCO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "C1CCCCC1"] {
            let mut molecule = embedded(smiles);
            let field = ForceField::new(&molecule).expect("should type");
            let before = field.energy(&molecule.positions);
            let after = field.minimize(&mut molecule.positions, 500);
            assert!(after < before, "{smiles}: {before} -> {after}");

            let mut gradient = vec![Vector3D::default(); molecule.len()];
            field.energy_and_gradient(&molecule.positions, &mut gradient);
            let largest = gradient.iter().map(|g| g.length()).fold(0.0_f64, f64::max);
            assert!(
                largest < 5.0,
                "{smiles}: still pulling at {largest} kcal/mol/A"
            );
        }
    }

    /// Relaxation must not break the structure it was given: bonds stay
    /// bonds, and nothing collapses onto anything else.
    #[test]
    fn minimisation_keeps_the_structure_intact() {
        for smiles in ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"] {
            let mut molecule = embedded(smiles);
            let field = ForceField::new(&molecule).expect("should type");
            field.minimize(&mut molecule.positions, 500);

            for atom in 0..molecule.len() {
                for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
                    let length = molecule
                        .get_atom_position(atom)
                        .unwrap()
                        .distance(&molecule.get_atom_position(bond.target()).unwrap());
                    assert!(
                        (0.85..=2.2).contains(&length),
                        "{smiles}: bond {atom}-{} relaxed to {length} A",
                        bond.target()
                    );
                }
            }
        }
    }

    /// An element outside the typing rules is refused rather than guessed
    /// at: UFF's table spans the periodic table, but a metal's type depends
    /// on a coordination geometry this crate does not perceive.
    #[test]
    fn an_untypable_element_is_refused() {
        let molecule = MolecularGraph::from_smiles_single("[U][U]").expect("should parse");
        let Err(failure) = ForceField::new(&molecule) else {
            panic!("uranium should have no typing rule");
        };
        assert_eq!(failure.atomic_number, 92);
    }

    /// The group-15 inversion barrier is 22 kcal/mol, which is RDKit's
    /// reading (after Towhee's `ffuff`) of a number the UFF paper leaves
    /// open. The term's coefficients make flattening the pyramid cost
    /// `K * (c0 + c1 + c2)` per permutation, so the force constant is
    /// divided by that sum: without the division the barrier grows with
    /// the element's own `omega_0` and a phosphine came out at 36 kcal/mol
    /// and a bismuthine at 44 rather than 22.
    #[test]
    fn group_fifteen_inversion_barrier_is_twenty_two() {
        use crate::molecule::forcefield::uff::inversion_parameters;

        let mut phosphine = MolecularGraph::from_smiles_single("CP(C)C").expect("should parse");
        phosphine.add_hydrogens();
        let phosphorus = phosphine
            .atomic_numbers()
            .iter()
            .position(|&atomic_number| atomic_number == 15)
            .expect("a phosphine");
        let Some((force_constant, c0, c1, c2)) = inversion_parameters(&phosphine, phosphorus)
        else {
            panic!("a trivalent phosphorus should carry an inversion term");
        };

        // Flattened: omega = 0, so cos(omega) = cos(2 omega) = 1, and the
        // three permutations of which bond is out of the plane each pay
        // K * (c0 + c1 + c2).
        let barrier = 3.0 * force_constant * (c0 + c1 + c2);
        assert!(
            (barrier - 22.0).abs() < 1e-9,
            "barrier is {barrier}, not 22"
        );

        // And the minimum sits at the omega_0 the paper gives phosphorus.
        let omega0 = 84.4339_f64.to_radians();
        let at_minimum = c0 + c1 * omega0.cos() + c2 * (2.0 * omega0).cos();
        assert!(at_minimum.abs() < 1e-9, "minimum is off by {at_minimum}");
    }

    /// The amide C-N bond carries the paper's 1.41 order, and nothing else
    /// does: not an aniline's C-N, whose resonance ends inside the ring, not
    /// a thioamide's, whose carbonyl is a sulfur's, and not the plain C-C
    /// beside the carbonyl. Urea's two C-N bonds both qualify.
    #[test]
    fn the_amide_bond_order_is_the_paper_s() {
        use crate::molecule::forcefield::uff::bond_order_value;

        let order = |molecule: &MolecularGraph, from: usize, to: usize| {
            let bond = molecule
                .get_atom_bonds(from)
                .unwrap()
                .iter()
                .find(|b| b.target() == to)
                .unwrap();
            bond_order_value(molecule, from, bond)
        };

        let mut acetamide = MolecularGraph::from_smiles_single("CC(=O)N").expect("should parse");
        acetamide.add_hydrogens();
        // Carbonyl carbon 1, amide nitrogen 3.
        assert!((order(&acetamide, 1, 3) - 1.41).abs() < 1e-9);
        assert!((order(&acetamide, 0, 1) - 1.0).abs() < 1e-9);
        assert!((order(&acetamide, 1, 2) - 2.0).abs() < 1e-9);

        let mut urea = MolecularGraph::from_smiles_single("NC(=O)N").expect("should parse");
        urea.add_hydrogens();
        assert!((order(&urea, 1, 0) - 1.41).abs() < 1e-9);
        assert!((order(&urea, 1, 3) - 1.41).abs() < 1e-9);

        let mut aniline = MolecularGraph::from_smiles_single("Nc1ccccc1").expect("should parse");
        aniline.add_hydrogens();
        assert!((order(&aniline, 0, 1) - 1.0).abs() < 1e-9);

        let mut thioamide = MolecularGraph::from_smiles_single("CC(=S)N").expect("should parse");
        thioamide.add_hydrogens();
        assert!((order(&thioamide, 1, 3) - 1.0).abs() < 1e-9);
    }

    /// A phosphonium's phosphorus is the paper's `P_3+q` -- tetrahedral --
    /// where a phosphine's is the pyramidal `P_3+3` and a phosphate's is
    /// `P_3+5`. RDKit cannot type the phosphonium at all: its valence rules
    /// know +3 and +5 and nothing between.
    #[test]
    fn quaternary_phosphorus_gets_its_own_type() {
        use crate::molecule::forcefield::uff::typing::label_for;

        let mut phosphonium =
            MolecularGraph::from_smiles_single("C[P+](C)(C)C").expect("should parse");
        phosphonium.add_hydrogens();
        assert_eq!(label_for(&phosphonium, 1).as_deref(), Some("P_3+q"));

        let mut phosphine = MolecularGraph::from_smiles_single("CP(C)C").expect("should parse");
        phosphine.add_hydrogens();
        assert_eq!(label_for(&phosphine, 1).as_deref(), Some("P_3+3"));

        let mut oxide = MolecularGraph::from_smiles_single("O=P(C)(C)C").expect("should parse");
        oxide.add_hydrogens();
        assert_eq!(label_for(&oxide, 1).as_deref(), Some("P_3+5"));
    }
}
