//! Two independently-sourced partial-charge methods, and the properties that
//! matter most about them: each conserves the molecule's total formal
//! charge exactly, both agree with each other -- and with chemical
//! intuition -- about which way a bond is polarised, and QEq reproduces the
//! reference implementation it was ported from to five decimal places on
//! that implementation's own test molecules.
//!
//! That last one is the load-bearing test, and it is deliberately tight. An
//! earlier version of this file checked two of `cheq`'s four molecules with
//! a tolerance of 0.2 -- wider than the whole charge being checked, since
//! CH4's hydrogen carries 0.13 e. Every wrong answer that version could
//! have produced would have passed, and one did: a QEq built on the
//! Gaussian-approximated integral with hydrogen's charge-dependent hardness
//! skipped, which put CH4's hydrogen at 0.04 instead of 0.13.

#[cfg(test)]
mod charges_tests {
    use crate::molecule::base::MoleculeCore;
    use crate::molecule::charges::slater::coulomb_integral;
    use crate::molecule::charges::{ChargeError, gasteiger_charges, qeq_charges};
    use crate::molecule::descriptors::MolecularDescriptors;
    use crate::prelude::*;

    fn parse(smiles: &str) -> MolecularGraph {
        MolecularGraph::from_smiles_single(smiles).expect("should parse")
    }

    fn embedded_positions(molecule: &MolecularGraph) -> Vec<Vector3D> {
        molecule.embed_3d().expect("should embed").positions
    }

    fn sum(charges: &[f64]) -> f64 {
        charges.iter().sum()
    }

    /// Principal quantum number pairs spanning everything
    /// `QEQ_PARAMETERS` can ask for: hydrogen's 1s through iodine's 5s.
    const ORBITAL_PAIRS: &[(u8, u8)] = &[(1, 1), (1, 2), (2, 2), (2, 5), (3, 4), (5, 5)];

    /// The one case with a closed form simple enough to write down: two 1s
    /// orbitals of equal exponent. Everything else in the kernel is checked
    /// against identities rather than a formula, so this is the single point
    /// where the absolute value is pinned by algebra alone.
    #[test]
    fn slater_coulomb_matches_the_closed_form_for_two_1s_orbitals() {
        let (r, zeta) = (2.0f64, 1.0f64);
        let rho = r * zeta;
        let expected = 1.0 / r
            - (-2.0 * rho).exp() / r * (1.0 + 1.375 * rho + 0.75 * rho.powi(2) + rho.powi(3) / 6.0);
        assert!((coulomb_integral(r, 1, zeta, 1, zeta) - expected).abs() < 1e-12);
    }

    /// Far enough apart, neither cloud sees the other's shape and the
    /// integral has to collapse to what two point charges would give.
    #[test]
    fn slater_coulomb_becomes_a_point_charge_at_long_range() {
        for &(n_a, n_b) in ORBITAL_PAIRS {
            let j = coulomb_integral(100.0, n_a, 1.3, n_b, 0.7);
            assert!(
                (j - 0.01).abs() < 1e-10,
                "n = ({n_a}, {n_b}): {j} is not 1/100 at 100 Bohr"
            );
        }
    }

    /// `J_AB` is the interaction of two charge distributions, so which one
    /// is written first cannot matter.
    #[test]
    fn slater_coulomb_is_symmetric_under_swapping_the_orbitals() {
        for &(n_a, n_b) in ORBITAL_PAIRS {
            for r in [0.05, 0.8, 2.5, 7.0] {
                let forward = coulomb_integral(r, n_a, 1.4, n_b, 0.6);
                let backward = coulomb_integral(r, n_b, 0.6, n_a, 1.4);
                assert!(
                    (forward - backward).abs() < 1e-12 * forward.abs().max(1.0),
                    "n = ({n_a}, {n_b}) at r = {r}: {forward} != {backward}"
                );
            }
        }
    }

    /// Two atoms of the same element bonded together give the auxiliary
    /// integral `B_k` two identical exponents, and its recurrence divides by
    /// their difference. The kernel takes a series instead across the whole
    /// band where that division costs more digits than `f64` has.
    ///
    /// `J_AB` is smooth in the exponents, so moving one of them by `gap`
    /// may move the integral by about `gap` and no more. Checking that
    /// against the exactly-equal value, over six decades of `gap`, is what
    /// distinguishes a kernel that merely returns a number near the
    /// degeneracy from one that is actually continuous there: the
    /// recurrence-only version this replaced returned `-3.7e6` here, where
    /// the true value is `0.46`.
    #[test]
    fn slater_coulomb_stays_smooth_as_two_exponents_become_equal() {
        for &(n_a, n_b) in ORBITAL_PAIRS {
            let equal = coulomb_integral(1.9, n_a, 1.5, n_b, 1.5);
            for gap in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9] {
                let nearly = coulomb_integral(1.9, n_a, 1.5, n_b, 1.5 + gap);
                assert!(
                    (nearly - equal).abs() < 0.5 * gap,
                    "n = ({n_a}, {n_b}): {equal} at equal exponents but {nearly} \
                     a gap of {gap} away"
                );
            }
        }
    }

    /// The two-center expansion divides by the separation, so coincident
    /// atoms take a separate one-center branch. It has to agree with what
    /// the two-center path approaches, or the shielding would step
    /// discontinuously for atoms that drift very close.
    #[test]
    fn slater_coulomb_is_continuous_into_the_one_center_limit() {
        for &(n_a, n_b) in ORBITAL_PAIRS {
            let coincident = coulomb_integral(0.0, n_a, 1.2, n_b, 0.9);
            let nearly = coulomb_integral(1e-6, n_a, 1.2, n_b, 0.9);
            assert!(
                (coincident - nearly).abs() < 1e-5 * coincident.abs(),
                "n = ({n_a}, {n_b}): {coincident} at zero, {nearly} just off it"
            );
        }
    }

    /// `cheq`'s own `tests/hydrogen.rs` group, rebuilt here: the SMILES this
    /// crate parses it from, the index of the hydrogen `cheq` checks, the
    /// geometry `cheq` builds (experimental bond lengths and angles, in
    /// angstrom and degrees), the charge `cheq` computes there with its
    /// default options, and the reference charge `cheq`'s test asserts
    /// against.
    ///
    /// Two separate numbers, because they are two separate claims. Matching
    /// `computed` says this module is the same method as `cheq`, to the
    /// precision the SCF converges to. Matching `published` -- which `cheq`
    /// itself only manages to ~0.03 -- says that method lands where the QEq
    /// literature says it should.
    struct ReferenceCase {
        smiles: &'static str,
        hydrogen: usize,
        geometry: Vec<Vector3D>,
        computed: f64,
        published: f64,
    }

    /// Reproduces `cheq`'s `make_diatomic` / `make_water_like` /
    /// `make_pyramidal_xy3` / `make_tetrahedral_xy4`, with this crate's atom
    /// ordering (SMILES `F` is fluorine first, hydrogen second; `cheq`'s
    /// diatomic is the other way round).
    fn reference_cases() -> Vec<ReferenceCase> {
        let position = Vector3D::new;

        let (r_oh, half_hoh) = (0.958, 104.5f64.to_radians() / 2.0);
        let water = vec![
            position(0.0, 0.0, 0.0),
            position(r_oh * half_hoh.cos(), r_oh * half_hoh.sin(), 0.0),
            position(r_oh * half_hoh.cos(), -r_oh * half_hoh.sin(), 0.0),
        ];

        // Pyramidal NH3: the three hydrogens sit on a circle of radius
        // `spread` below the nitrogen, at the depth that keeps every N-H
        // bond 1.012 A long with an H-N-H angle of 106.7 degrees.
        let (r_nh, hnh) = (1.012, 106.7f64.to_radians());
        let spread = 2.0 * r_nh * (hnh / 2.0).sin() / 3.0f64.sqrt();
        let depth = (r_nh * r_nh - spread * spread).sqrt();
        let ammonia = vec![
            position(0.0, 0.0, 0.0),
            position(spread, 0.0, -depth),
            position(-spread * 0.5, spread * 3.0f64.sqrt() * 0.5, -depth),
            position(-spread * 0.5, -spread * 3.0f64.sqrt() * 0.5, -depth),
        ];

        // Tetrahedral CH4: alternating corners of a cube whose half-diagonal
        // is the 1.087 A C-H bond.
        let corner = 1.087 / 3.0f64.sqrt();
        let methane = vec![
            position(0.0, 0.0, 0.0),
            position(corner, corner, corner),
            position(corner, -corner, -corner),
            position(-corner, corner, -corner),
            position(-corner, -corner, corner),
        ];

        vec![
            ReferenceCase {
                smiles: "F",
                hydrogen: 1,
                geometry: vec![position(0.917, 0.0, 0.0), position(0.0, 0.0, 0.0)],
                computed: 0.440_693_637,
                published: 0.46,
            },
            ReferenceCase {
                smiles: "O",
                hydrogen: 1,
                geometry: water,
                computed: 0.325_146_681,
                published: 0.35,
            },
            ReferenceCase {
                smiles: "N",
                hydrogen: 1,
                geometry: ammonia,
                computed: 0.218_869_177,
                published: 0.24,
            },
            ReferenceCase {
                smiles: "C",
                hydrogen: 1,
                geometry: methane,
                computed: 0.130_984_794,
                published: 0.15,
            },
        ]
    }

    /// The tight half: same method, same geometry, same answer. The bound is
    /// an order of magnitude above the SCF's own convergence tolerance and
    /// four orders below the charges being compared, so anything that
    /// changes the method -- a different integral, a dropped hardness
    /// refinement, a mistyped parameter -- fails here rather than hiding.
    #[test]
    fn qeq_reproduces_cheq_on_cheq_s_own_geometries() {
        for case in reference_cases() {
            let molecule = parse(case.smiles);
            let charges = qeq_charges(&molecule, &case.geometry).expect("should solve");
            let error = (charges[case.hydrogen] - case.computed).abs();
            assert!(
                error < 1e-5,
                "{}: hydrogen {} = {}, cheq computes {} (off by {error:e})",
                case.smiles,
                case.hydrogen,
                charges[case.hydrogen],
                case.computed
            );
        }
    }

    /// The loose half, and loose for a stated reason: `cheq`'s published
    /// reference charges are two-decimal literature values that `cheq`'s own
    /// solver reproduces to about 0.03, so that is the bound here too. It
    /// catches a method that has drifted somewhere else entirely without
    /// pretending to a precision neither implementation claims.
    #[test]
    fn qeq_lands_where_the_qeq_literature_puts_these_hydrogens() {
        for case in reference_cases() {
            let molecule = parse(case.smiles);
            let charges = qeq_charges(&molecule, &case.geometry).expect("should solve");
            assert!(
                (charges[case.hydrogen] - case.published).abs() < 0.03,
                "{}: hydrogen {} = {}, published {}",
                case.smiles,
                case.hydrogen,
                charges[case.hydrogen],
                case.published
            );
        }
    }

    /// What swapping the experimental geometry for one this crate generates
    /// actually costs. Small -- which is why it cannot be an excuse for a
    /// method-sized disagreement in either test above.
    ///
    /// The bound is measured, not derived, and it is set by hydrogen
    /// fluoride: `embed_3d` now finishes against UFF, whose `H_`-`F_` rest
    /// length works out to 0.986 A against an experimental 0.917, and QEq
    /// reads 0.0085 e off that extra 0.07 A. The other three cases here sit
    /// between 0.0006 and 0.0031. All four stay well inside the 0.03 the
    /// test above allows against the published charges.
    #[test]
    fn embedded_geometry_moves_qeq_charges_only_slightly() {
        for case in reference_cases() {
            let molecule = parse(case.smiles);
            let embedded =
                qeq_charges(&molecule, &embedded_positions(&molecule)).expect("should solve");
            let experimental = qeq_charges(&molecule, &case.geometry).expect("should solve");
            let shift = (embedded[case.hydrogen] - experimental[case.hydrogen]).abs();
            assert!(
                shift < 0.01,
                "{}: embed_3d moved hydrogen {} by {shift}",
                case.smiles,
                case.hydrogen
            );
        }
    }

    #[test]
    fn gasteiger_conserves_total_formal_charge() {
        for smiles in ["CCO", "c1ccccc1", "CC(=O)O", "[NH4+]", "CC(=O)[O-]"] {
            let molecule = parse(smiles);
            let charges = gasteiger_charges(&molecule);
            let expected = molecule.total_formal_charge() as f64;
            assert!(
                (sum(&charges) - expected).abs() < 1e-6,
                "{smiles}: sum {} != total formal charge {expected}",
                sum(&charges)
            );
        }
    }

    #[test]
    fn qeq_conserves_total_formal_charge() {
        for smiles in ["CCO", "c1ccccc1", "CC(=O)O", "[NH4+]", "CC(=O)[O-]"] {
            let molecule = parse(smiles);
            let positions = embedded_positions(&molecule);
            let charges = qeq_charges(&molecule, &positions).expect("should solve");
            let expected = molecule.total_formal_charge() as f64;
            assert!(
                (sum(&charges) - expected).abs() < 1e-6,
                "{smiles}: sum {} != total formal charge {expected}",
                sum(&charges)
            );
        }
    }

    /// `(molecule, negative atom index, positive atom index)` for a handful
    /// of textbook-polar bonds -- the electronegative atom should be
    /// negative and its partner positive, independent of which method
    /// computed it.
    const POLARITY_CASES: &[(&str, usize, usize)] = &[
        ("O", 0, 1),   // water: O negative, H positive
        ("Cl", 0, 1),  // HCl written as `Cl`: Cl negative, H positive
        ("C=O", 1, 0), // formaldehyde: O negative, carbonyl C positive
    ];

    #[test]
    fn gasteiger_gets_the_polarity_right() {
        for &(smiles, negative, positive) in POLARITY_CASES {
            let molecule = parse(smiles);
            let charges = gasteiger_charges(&molecule);
            assert!(
                charges[negative] < 0.0,
                "{smiles}: atom {negative} should be negative, got {:?}",
                charges
            );
            assert!(
                charges[positive] > 0.0,
                "{smiles}: atom {positive} should be positive, got {:?}",
                charges
            );
        }
    }

    #[test]
    fn qeq_gets_the_polarity_right() {
        for &(smiles, negative, positive) in POLARITY_CASES {
            let molecule = parse(smiles);
            let positions = embedded_positions(&molecule);
            let charges = qeq_charges(&molecule, &positions).expect("should solve");
            assert!(
                charges[negative] < 0.0,
                "{smiles}: atom {negative} should be negative, got {:?}",
                charges
            );
            assert!(
                charges[positive] > 0.0,
                "{smiles}: atom {positive} should be positive, got {:?}",
                charges
            );
        }
    }

    #[test]
    fn gasteiger_and_qeq_agree_on_sign_even_when_they_disagree_on_size() {
        for smiles in ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccncc1"] {
            let molecule = parse(smiles);
            let positions = embedded_positions(&molecule);
            let gasteiger = gasteiger_charges(&molecule);
            let qeq = qeq_charges(&molecule, &positions).expect("should solve");
            for atom in 0..molecule.len() {
                // A charge near zero on one side and a small opposite-sign
                // wobble on the other isn't a real disagreement -- only
                // flag atoms both methods consider meaningfully charged.
                if gasteiger[atom].abs() < 0.05 || qeq[atom].abs() < 0.05 {
                    continue;
                }
                assert_eq!(
                    gasteiger[atom].signum(),
                    qeq[atom].signum(),
                    "{smiles} atom {atom}: gasteiger {} vs qeq {}",
                    gasteiger[atom],
                    qeq[atom]
                );
            }
        }
    }

    #[test]
    fn qeq_reports_a_missing_element_instead_of_guessing() {
        // Uranium: not in FEATURED_ELEMENTS, not in QEQ_PARAMETERS. Bonded to
        // a second atom so `embed_3d` (which needs at least two) succeeds.
        let molecule = MolecularGraph::from_smiles_single("[U][U]").expect("should parse");
        let positions = embedded_positions(&molecule);
        assert_eq!(
            qeq_charges(&molecule, &positions),
            Err(ChargeError::MissingParameters(92))
        );
    }

    #[test]
    fn an_atom_with_no_neighbours_keeps_its_formal_charge() {
        let molecule = MolecularGraph::from_smiles_single("[Na+]").expect("should parse");
        let charges = gasteiger_charges(&molecule);
        assert_eq!(charges, vec![1.0]);
    }
}
