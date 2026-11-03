//! CIP assignment checked against RDKit.
//!
//! Every expected descriptor here came from `rdkit.Chem.rdCIPLabeler`, the
//! implementation validated against Hanson's reference suite — *not* the
//! legacy `AssignStereochemistry`, which is approximate and
//! disagrees with it. The second block of [`CASES`] is RDKit's own CIP
//! regression suite (`Code/GraphMol/CIPLabeler/catch_tests.cpp`), which is
//! where the pseudo-asymmetric centres, bridged cages and ring-closure
//! duplicates that a partial rule set gets wrong actually live.
//!
//! # Deviations from RDKit (not in [`CASES`])
//!
//! - **A three-coordinate atom whose hydrogen is written inside the
//!   brackets**, e.g. `C[P@@H]C1CCCCC1`.
//!
//!   OpenSMILES defines the bracketed hydrogen's place only for a centre with
//!   four neighbours: it "is considered to be the first atom in the clockwise
//!   or anticlockwise accounting", i.e. it takes the slot of the first
//!   neighbour listed after the preceding atom. We and RDKit both follow that,
//!   so `N[C@H](O)C` and `N[C@]([H])(O)C` agree.
//!
//!   For a centre with only *three* neighbours and a lone pair the
//!   specification says nothing at all — it never mentions lone pairs, and its
//!   tetrahedral section describes four neighbours throughout. We apply the
//!   four-neighbour rule unchanged; RDKit instead orders the bracketed
//!   hydrogen last, so it reads `C[P@@H]X` and `C[P@@]([H])X` as opposite
//!   configurations. Neither behaviour is more conformant than the other,
//!   because there is nothing to conform to; ours is preferred here only
//!   because it keeps a descriptor from depending on how the hydrogen was
//!   typed. See [`implicit_and_explicit_hydrogen_agree`].
//!
//! - **A centre carrying both a wildcard `*` and a hydrogen.** `*` is atomic
//!   number 0, the only thing that ranks below hydrogen, so hydrogen is not
//!   the lowest-priority substituent there. RDKit assumes it always is and
//!   gets these inverted; see [`a_wildcard_ranks_below_hydrogen`] for the
//!   derivation. With four heavy substituents the two agree, and those cases
//!   are in [`CASES`].

use crate::graph_algorithms::mol_canonization::{canonize_mol, reorder_molecule};
use crate::molecule::cip::CipConfig;
use crate::prelude::*;

/// `(name, smiles, [(atom index, expected descriptor)])`
const CASES: &[(&str, &str, &[(usize, CipConfig)])] = &[
    ("alanine_L", "N[C@@H](C)C(=O)O", &[(1, CipConfig::S)]),
    ("alanine_D", "N[C@H](C)C(=O)O", &[(1, CipConfig::R)]),
    ("serine", "N[C@@H](CO)C(=O)O", &[(1, CipConfig::S)]),
    ("serine_phosphate", "N[C@@H](COP(=O)(O)O)C(=O)O", &[(1, CipConfig::S)]),
    ("cysteine", "N[C@@H](CS)C(=O)O", &[(1, CipConfig::R)]),
    ("threonine", "C[C@@H](O)[C@H](N)C(=O)O", &[(1, CipConfig::R), (3, CipConfig::S)]),
    ("valine", "CC(C)[C@@H](N)C(=O)O", &[(3, CipConfig::R)]),
    ("leucine", "CC(C)C[C@@H](N)C(=O)O", &[(4, CipConfig::R)]),
    ("isoleucine", "CC[C@H](C)[C@@H](N)C(=O)O", &[(2, CipConfig::S), (4, CipConfig::R)]),
    ("methionine", "CSCC[C@@H](N)C(=O)O", &[(4, CipConfig::R)]),
    ("phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O", &[(1, CipConfig::S)]),
    ("tyrosine", "N[C@@H](Cc1ccc(O)cc1)C(=O)O", &[(1, CipConfig::S)]),
    ("tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", &[(1, CipConfig::S)]),
    ("proline", "OC(=O)[C@@H]1CCCN1", &[(3, CipConfig::S)]),
    ("aspartate", "N[C@@H](CC(=O)O)C(=O)O", &[(1, CipConfig::S)]),
    ("glutamate", "N[C@@H](CCC(=O)O)C(=O)O", &[(1, CipConfig::S)]),
    ("lysine", "NCCCC[C@@H](N)C(=O)O", &[(5, CipConfig::R)]),
    ("arginine", "NC(=N)NCCC[C@@H](N)C(=O)O", &[(7, CipConfig::R)]),
    ("histidine", "N[C@@H](Cc1c[nH]cn1)C(=O)O", &[(1, CipConfig::S)]),
    ("glyceraldehyde_R", "OC[C@@H](O)C=O", &[(2, CipConfig::R)]),
    ("glyceraldehyde_S", "OC[C@H](O)C=O", &[(2, CipConfig::S)]),
    ("glycerol_1_phosphate", "OC[C@H](O)COP(=O)(O)O", &[(2, CipConfig::S)]),
    ("lactate", "C[C@H](O)C(=O)O", &[(1, CipConfig::S)]),
    ("malate", "OC(=O)C[C@H](O)C(=O)O", &[(4, CipConfig::S)]),
    ("tartrate", "O[C@H](C(=O)O)[C@@H](O)C(=O)O", &[(1, CipConfig::S), (5, CipConfig::R)]),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", &[(2, CipConfig::R), (6, CipConfig::R), (8, CipConfig::S), (10, CipConfig::S)]),
    ("galactose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O", &[(2, CipConfig::R), (6, CipConfig::R), (8, CipConfig::S), (10, CipConfig::R)]),
    ("mannose", "OC[C@H]1OC(O)[C@@H](O)[C@@H](O)[C@@H]1O", &[(2, CipConfig::R), (6, CipConfig::S), (8, CipConfig::S), (10, CipConfig::S)]),
    ("ribose", "OC[C@H]1OC(O)[C@H](O)[C@H]1O", &[(2, CipConfig::R), (6, CipConfig::R), (8, CipConfig::R)]),
    ("chlorofluoro", "F[C@H](Cl)Br", &[(1, CipConfig::R)]),
    ("chlorofluoro2", "F[C@@H](Cl)I", &[(1, CipConfig::S)]),
    ("bromochloro_C", "[C@H](F)(Cl)Br", &[(0, CipConfig::S)]),
    ("isotope_center", "[13CH3][C@H](C)O", &[(1, CipConfig::S)]),
    ("adenosine", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", &[(10, CipConfig::R), (12, CipConfig::R), (15, CipConfig::S), (17, CipConfig::R)]),
    ("amp", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O", &[(10, CipConfig::R), (12, CipConfig::R), (19, CipConfig::S), (21, CipConfig::R)]),
    ("cholesterol", "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C", &[(6, CipConfig::R), (8, CipConfig::R), (11, CipConfig::S), (12, CipConfig::S), (17, CipConfig::S), (21, CipConfig::R), (23, CipConfig::S), (26, CipConfig::R)]),
    ("testosterone", "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O", &[(1, CipConfig::S), (4, CipConfig::S), (5, CipConfig::R), (8, CipConfig::R), (14, CipConfig::S), (16, CipConfig::S), (19, CipConfig::S)]),
    ("camphor", "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2", &[(3, CipConfig::R), (6, CipConfig::R)]),
    ("menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", &[(3, CipConfig::S), (6, CipConfig::R), (9, CipConfig::R)]),
    ("nicotine", "CN1CCC[C@H]1c1cccnc1", &[(5, CipConfig::S)]),
    ("thalidomide", "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1", &[]),
    ("ibuprofen_S", "C[C@H](C(=O)O)c1ccc(CC(C)C)cc1", &[(1, CipConfig::S)]),
    ("naproxen_S", "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1", &[(1, CipConfig::S)]),
    ("deep_S_branch", "C[C@H](CCS)CCO", &[(1, CipConfig::S)]),
    ("deep_P_branch", "C[C@H](CCOP(=O)(O)O)CCO", &[(1, CipConfig::S)]),
    ("deep_Br_branch", "C[C@H](CCCBr)CCO", &[(1, CipConfig::R)]),

    // RDKit's own CIP regression suite (`Code/GraphMol/CIPLabeler/catch_tests.cpp`),
    // which is where the pseudo-asymmetric centres, the bridged cages and the
    // ring-closure duplicates live.
    ("rdkit_suite_00", "F[C@@H](Cl)Br", &[(1, CipConfig::S)]),
    ("rdkit_suite_01", "C[P@@](c1ccccc1)C1CCCC1", &[(1, CipConfig::R)]),
    ("rdkit_suite_02", "C[As@](c1ccccc1)C1CCCC1", &[(1, CipConfig::S)]),
    ("rdkit_suite_03", "C[C@H](Cl)CC[C@@H](C)Cl", &[(1, CipConfig::S), (5, CipConfig::R)]),
    ("rdkit_suite_04", r"C/C=C\[C@H](/C=C/O)[C@@H](C)[C@@H](/C=C\C)/C=C/O", &[(1, CipConfig::Z), (2, CipConfig::Z), (3, CipConfig::S), (4, CipConfig::E), (5, CipConfig::E), (7, CipConfig::PseudoR), (9, CipConfig::R), (10, CipConfig::Z), (11, CipConfig::Z), (13, CipConfig::E), (14, CipConfig::E)]),
    ("rdkit_suite_05", r"C/C=C\[C@H](/C=C/C)[C@@H](C)[C@@H](/C=C\C)/C=C/C", &[(1, CipConfig::Z), (2, CipConfig::Z), (3, CipConfig::S), (4, CipConfig::E), (5, CipConfig::E), (7, CipConfig::PseudoR), (9, CipConfig::R), (10, CipConfig::Z), (11, CipConfig::Z), (13, CipConfig::E), (14, CipConfig::E)]),
    ("rdkit_suite_06", "CC[C@H](C)CCCCC[C@H]1CC[C@@H](C)CC1", &[(2, CipConfig::S), (9, CipConfig::PseudoS), (12, CipConfig::S)]),
    ("rdkit_suite_07", "O=C(O)[C@H]1CC[C@H](O[C@@H](F)Cl)CC1", &[(3, CipConfig::S), (6, CipConfig::PseudoR), (8, CipConfig::S)]),
    ("rdkit_suite_08", "O=C(O)[C@H]1CC[C@H](OC(Cl)Cl)CC1", &[(3, CipConfig::PseudoR), (6, CipConfig::PseudoR)]),
    ("rdkit_suite_09", "*[C@](F)(Cl)Br", &[(1, CipConfig::S)]),
    ("rdkit_suite_10", "*C(*)(Cl)Br", &[]),
    ("rdkit_suite_11", "*[C@]12CCCC[C@H]1CCCC2", &[(1, CipConfig::PseudoS), (6, CipConfig::PseudoS)]),
    ("rdkit_suite_12", "*C1C[C@H](CCC)[C@@H](C)[C@H](C)C1", &[(3, CipConfig::R), (7, CipConfig::S), (9, CipConfig::R)]),
    ("rdkit_suite_13", r"C1CC[C@H]2C/C(=C3\C[C@H]4CCCC[C@H]4C3)C[C@H]2C1", &[(3, CipConfig::S), (5, CipConfig::E), (6, CipConfig::E), (8, CipConfig::R), (13, CipConfig::S), (16, CipConfig::R)]),
    ("rdkit_suite_14", "Cc1c(-c2c(F)cc(C(N)=O)c3[nH]c4c(c23)CC[C@H](C(C)(C)O)C4)cccc1-n1c(=O)c2cccc(F)c2n(C)c1=O", &[(18, CipConfig::S)]),
    ("rdkit_suite_15", "C=C(n1c(=O)cc[nH]c1=O)[C@@]([C@H](C)F)([C@H](C)Br)[C@@H](C)Cl", &[(10, CipConfig::S), (11, CipConfig::S), (14, CipConfig::S), (17, CipConfig::R)]),
    ("rdkit_suite_16", "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)[C@H](C)C1", &[(38, CipConfig::R)]),
    ("rdkit_suite_17", "CCc1c(C#N)cc2ccccc2c1C(=O)N(C)C[C@H](CCN1CCC(c2ccccc2S(C)=O)CC1)c1ccc(Cl)c(Cl)c1", &[(19, CipConfig::R)]),
    ("rdkit_suite_18", "Cc1ccc(Br)n1N1C(=O)[C@@H](C)[C@@H](C)C1=O", &[(10, CipConfig::S), (12, CipConfig::R)]),
];

#[test]
fn cip_matches_rdkit() {
    let mut failures = Vec::new();

    for (name, smiles, expected) in CASES {
        let mol = MolecularGraph::from_smiles_single(smiles)
            .unwrap_or_else(|e| panic!("{name}: failed to parse {smiles}: {e:?}"));
        let configs = mol.assign_cip_configs();

        for &(index, want) in *expected {
            let Some(&got) = configs.get(index) else {
                failures.push(format!(
                    "{name} ({smiles}) atom {index}: expected {want}, but the molecule \
                     only parsed to {} atoms",
                    configs.len()
                ));
                continue;
            };
            if got != want {
                failures.push(format!("{name} ({smiles}) atom {index}: expected {want}, got {got}"));
            }
        }

        // Nothing outside the expected set may claim a descriptor.
        for (index, &got) in configs.iter().enumerate() {
            if got != CipConfig::None && !expected.iter().any(|&(i, _)| i == index) {
                failures.push(format!("{name} ({smiles}) atom {index}: unexpected {got}"));
            }
        }
    }

    assert!(failures.is_empty(), "CIP disagrees with RDKit:\n  {}", failures.join("\n  "));
}

/// The exact defect the sphere-by-sphere comparison fixed, kept as its own
/// test because it is a one-line reproducer: the two branches at the
/// stereocentre are both carbon and differ in the *first* sphere
/// (`{O,H,H}` for `CH2-O-P` vs `{O,O,O}` for `COOH`), so `COOH` outranks and
/// the centre is *S*. Comparing branches depth-first instead reached the
/// phosphorus before that first-sphere difference and reported *R*.
#[test]
fn phosphate_deep_in_a_branch_does_not_outrank_a_nearer_difference() {
    let serine = MolecularGraph::from_smiles_single("N[C@@H](CO)C(=O)O").unwrap();
    let phosphoserine = MolecularGraph::from_smiles_single("N[C@@H](COP(=O)(O)O)C(=O)O").unwrap();

    assert_eq!(serine.assign_cip_configs()[1], CipConfig::S);
    assert_eq!(phosphoserine.assign_cip_configs()[1], CipConfig::S);
}

/// A carbon whose substituents are not all different is not a stereocentre,
/// however the SMILES decorates it. Ranking such a centre by breaking the tie
/// on atom index gave the *same molecule* opposite descriptors depending on
/// which way round it was written, which in turn made it fail to match itself
/// under [`crate::graph_algorithms::mcsplit`] (isopropanol against
/// isopropanol found no common substructure at all).
#[test]
fn a_centre_with_two_identical_substituents_is_not_a_stereocentre() {
    for smiles in [
        "C[C@H](C)O",      // isopropanol: two methyls
        "C[C@@H](C)O",     // ... written the other way round
        "C[C@H](CC)CC",    // 3-methylpentane: two ethyls
        "OC[C@H](CO)N",    // serinol: two hydroxymethyls
        "C[C@H](C)C(C)C",  // two isopropyls
    ] {
        let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
        let configs = mol.assign_cip_configs();
        assert!(
            configs.iter().all(|&c| c == CipConfig::None),
            "{smiles}: expected no descriptor anywhere, got {configs:?}"
        );
    }
}

/// The same for double bonds: an end carrying two identical substituents has
/// no distinguishable sides, so there is no E/Z.
#[test]
fn a_double_bond_end_with_two_identical_substituents_has_no_ez() {
    for smiles in ["C/C=C(\\C)C", "ClC(Cl)=C(Cl)Cl", "C/C=C/C"] {
        let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
        let configs = mol.assign_cip_configs();
        let stereogenic = smiles == "C/C=C/C";
        let has_ez = configs
            .iter()
            .any(|&c| c == CipConfig::E || c == CipConfig::Z);
        assert_eq!(
            has_ez, stereogenic,
            "{smiles}: expected E/Z present = {stereogenic}, got {configs:?}"
        );
    }
}

/// A descriptor is a property of the molecule, not of the atom numbering it
/// happens to arrive in, so permuting the atom indices must carry every
/// descriptor along with its atom and change nothing else.
///
/// This is a self-check: it needs no external ground truth, so it stays
/// meaningful for structures where the CIP rules implemented here are known
/// to be incomplete — a *wrong* descriptor still has to be wrong the same way
/// under every numbering. It is the cheapest guard against the class of bug
/// where a tie in the priority ranking gets resolved by atom index, which is
/// invisible to a fixed-numbering test suite.
#[test]
fn descriptors_are_invariant_under_atom_renumbering() {
    // Deterministic, so a failure is reproducible from the seed alone.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let molecules = [
        "N[C@@H](C)C(=O)O",
        "N[C@@H](COP(=O)(O)O)C(=O)O",
        "O[C@H](C(=O)O)[C@@H](O)C(=O)O",
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2",
        "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
        "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
        "C[C@H](C)O",
        "C/C=C/C",
        // A centre that is only stereogenic because of an isotope. This one
        // caught `reorder_molecule` leaving the isotope vector unpermuted, so
        // the `13C` label stayed at its old index and rule 2 ranked the wrong
        // branch.
        "C[C@@H]([13CH3])O",
        // Pseudo-asymmetric: the descriptor depends on the handedness of the
        // branches, which is where the descriptor-reading rules run.
        "CC[C@H](C)CCCCC[C@H]1CC[C@@H](C)CC1",
        // Three-coordinate, lone pair as the fourth vertex — both spellings.
        "CC[S@](C)=O",
        // Allenes, whose carriers sit one atom out from the centre.
        "NC(Br)=[C@]=C(O)C",
        "IC(Br)=[C@@]=C(Cl)F",
        "CC=[C@]=CC",
        "C[P@@H]C1CCCCC1",
        "C[P@@]([H])C1CCCCC1",
        "C[As@H]C1CCCC1",
    ];

    for smiles in molecules {
        let original = MolecularGraph::from_smiles_single(smiles).unwrap();
        let reference = original.assign_cip_configs();
        let n = reference.len();

        for round in 0..16 {
            // `reorder_molecule` reads its argument as `labeling[new] = old`:
            // the atom that ends up at index `i` is the one that was at
            // `labeling[i]`.
            let mut labeling: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                labeling.swap(i, (next() % (i as u64 + 1)) as usize);
            }

            let mut permuted = original.clone();
            reorder_molecule(&mut permuted, &labeling);
            let got = permuted.assign_cip_configs();

            for (new, &old) in labeling.iter().enumerate() {
                assert_eq!(
                    got[new], reference[old],
                    "{smiles} round {round}: atom {old} -> {new} was {} \
                     before renumbering and {} after (labeling {labeling:?})",
                    reference[old], got[new]
                );
            }
        }
    }
}

/// A centre's descriptor cannot depend on whether its hydrogen was written
/// inside the brackets or as an explicit atom — the two SMILES denote the same
/// molecule.
///
/// This is one place the port departs from RDKit, for this reason: for a *three*-coordinate
/// centre RDKit orders the bracketed hydrogen last rather than immediately
/// after the preceding atom, so it labels the two spellings *S* and *R*.
/// OpenSMILES does not cover the case, so neither reading is more conformant;
/// what decides it is that one of them is self-consistent and the other is
/// not. See the module docs.
#[test]
fn implicit_and_explicit_hydrogen_agree() {
    for (implicit, explicit, atom) in [
        // Three-coordinate, a lone pair on the fourth vertex — where RDKit
        // reads the two spellings as opposite configurations.
        ("C[P@@H]C1CCCCC1", "C[P@@]([H])C1CCCCC1", 1),
        ("C[P@H]C1CCCCC1", "C[P@]([H])C1CCCCC1", 1),
        ("C[As@@H]C1CCCC1", "C[As@@]([H])C1CCCC1", 1),
        ("C[As@H]C1CCCC1", "C[As@]([H])C1CCCC1", 1),
        ("CC[S@H]C", "CC[S@]([H])C", 2),
        ("C[N@@H]CC", "C[N@@]([H])CC", 1),
        // Four-coordinate, where we and RDKit already agree.
        ("N[C@@H](C)C(=O)O", "N[C@@]([H])(C)C(=O)O", 1),
        ("N[C@H](C)C(=O)O", "N[C@]([H])(C)C(=O)O", 1),
    ] {
        let a = MolecularGraph::from_smiles_single(implicit).unwrap();
        let b = MolecularGraph::from_smiles_single(explicit).unwrap();
        let (x, y) = (a.assign_cip_configs()[atom], b.assign_cip_configs()[atom]);
        assert_eq!(x, y, "{implicit} gave {x} but {explicit} gave {y} for the same atom");
        assert_ne!(x, CipConfig::None, "{implicit}: expected a descriptor");

        // The same thing said structurally: one molecule, one canonical form.
        assert_eq!(
            canonize_mol(&a).to_smiles(),
            canonize_mol(&b).to_smiles(),
            "{implicit} and {explicit} are the same molecule"
        );
    }
}

/// Three-coordinate stereocentres — a lone pair occupying the fourth vertex —
/// get descriptors, not silence. These are the cases the previous
/// implementation could not represent at all: the parser classified a
/// sulfoxide's `@` as double-bond geometry (it has one double bond and three
/// connections, like an alkene carbon) and rejected a phosphine's outright.
#[test]
fn three_coordinate_centres_with_a_lone_pair_are_labelled() {
    for (smiles, atom, want) in [
        ("CC[S@](C)=O", 2, CipConfig::S),
        ("C[S@@](=O)CC", 1, CipConfig::R),
        ("C[P@@](c1ccccc1)C1CCCC1", 1, CipConfig::R),
        ("C[As@](c1ccccc1)C1CCCC1", 1, CipConfig::S),
    ] {
        let configs = MolecularGraph::from_smiles_single(smiles).unwrap().assign_cip_configs();
        assert_eq!(configs[atom], want, "{smiles} atom {atom}");
    }
}

/// A wildcard atom ranks below hydrogen, so at a centre carrying both, the
/// hydrogen is *not* the lowest-priority substituent.
///
/// This is the second deviation from RDKit, and unlike the
/// three-coordinate one it is not a matter of convention. RDKit's
/// `Tetrahedral::label` builds its priority-ordered carrier list by skipping
/// the hydrogen and then appending it in the last slot unconditionally — which
/// silently assumes hydrogen always ranks last. `*` (atomic number 0) is the
/// only thing that can rank below it, so the assumption holds everywhere else
/// and fails exactly here.
///
/// Derivation for the first case below. Priorities are Cl(17) > F(9) > H(1) >
/// `*`(0), so the lowest is `*`. `@` says that looking *from* `*`, the
/// sequence (H, F, Cl) runs anticlockwise. Reordering that to priority order
/// (Cl, F, H) swaps two of the three, so it runs clockwise from `*`. CIP reads
/// with the lowest priority pointing *away*, the opposite viewpoint, which
/// reverses the sense again: anticlockwise, i.e. *S*. RDKit says *R*.
///
/// Note the disagreement appears only when the centre carries a hydrogen as
/// well as the wildcard; with four heavy substituents the two agree, and those
/// cases are in [`CASES`].
#[test]
fn a_wildcard_ranks_below_hydrogen() {
    for (smiles, atom, want) in [
        ("*[C@H](F)Cl", 1, CipConfig::S),
        ("*[C@@H](F)Cl", 1, CipConfig::R),
        ("*[C@H](C)CC", 1, CipConfig::S),
        // The same molecule with the hydrogen written out must agree.
        ("*[C@]([H])(F)Cl", 1, CipConfig::S),
        ("*[C@@]([H])(F)Cl", 1, CipConfig::R),
    ] {
        let configs = MolecularGraph::from_smiles_single(smiles).unwrap().assign_cip_configs();
        assert_eq!(configs[atom], want, "{smiles} atom {atom}");
    }
}

/// An allene is chiral about its axis, and CIP labels it as though the axis
/// collapsed to a single tetrahedral centre whose four ligands are the two
/// ends' substituents.
///
/// The ranking is done *within* each end rather than over all four: an axially
/// chiral system is read along its axis, so the pair nearest the viewer takes
/// the first two places whatever their own priority. That is not a shortcut —
/// without it 2,3-pentadiene has no descriptor at all, since its two methyls
/// are indistinguishable from each other and so are its two hydrogens, and it
/// is a textbook chiral allene.
///
/// RDKit is no help here: `MolToSmiles` drops allene stereochemistry entirely,
/// so these were derived by hand from the OpenSMILES rule and the values below
/// are pinned against that derivation.
#[test]
fn allenes_are_labelled() {
    for (smiles, centre, want) in [
        ("NC(Br)=[C@]=C(O)C", 3, CipConfig::R),
        ("NC(Br)=[C@@]=C(O)C", 3, CipConfig::S),
        ("IC(Br)=[C@]=C(Cl)F", 3, CipConfig::S),
        ("IC(Br)=[C@@]=C(Cl)F", 3, CipConfig::R),
        ("C(F)(Cl)=[C@]=C(Br)I", 3, CipConfig::S),
        // 2,3-pentadiene: only assignable under the proximity rule.
        ("CC=[C@]=CC", 2, CipConfig::R),
        ("CC=[C@@]=CC", 2, CipConfig::S),
    ] {
        let configs = MolecularGraph::from_smiles_single(smiles).unwrap().assign_cip_configs();
        assert_eq!(configs[centre], want, "{smiles} atom {centre}");
    }
}

/// The same allene written from either end has to give the same descriptor —
/// reading along the axis from the other side swaps both pairs at once, which
/// is an even permutation.
#[test]
fn an_allene_reads_the_same_from_either_end() {
    for (one, two) in [
        ("NC(Br)=[C@]=C(O)C", "OC(C)=[C@@]=C(Br)N"),
        ("NC(Br)=[C@@]=C(O)C", "OC(C)=[C@]=C(Br)N"),
    ] {
        let a = MolecularGraph::from_smiles_single(one).unwrap().assign_cip_configs();
        let b = MolecularGraph::from_smiles_single(two).unwrap().assign_cip_configs();
        let first = a.iter().find(|c| **c != CipConfig::None).copied();
        let second = b.iter().find(|c| **c != CipConfig::None).copied();
        assert_eq!(first, second, "{one} vs {two}");
        assert!(first.is_some(), "{one}: expected a descriptor");
    }
}

/// An end whose two substituents the rules cannot separate leaves the axis
/// with no configuration.
#[test]
fn an_allene_with_a_symmetric_end_has_no_descriptor() {
    for smiles in ["CC(C)=[C@]=C(O)C", "C=[C@]=C"] {
        let configs = MolecularGraph::from_smiles_single(smiles).unwrap().assign_cip_configs();
        assert!(
            configs.iter().all(|&c| c == CipConfig::None),
            "{smiles}: {configs:?}"
        );
    }
}

/// Enantiomers must receive opposite descriptors at every centre.
#[test]
fn enantiomers_get_opposite_descriptors() {
    for (left, right) in [
        ("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"),
        ("OC[C@@H](O)C=O", "OC[C@H](O)C=O"),
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br"),
    ] {
        let a = MolecularGraph::from_smiles_single(left).unwrap().assign_cip_configs();
        let b = MolecularGraph::from_smiles_single(right).unwrap().assign_cip_configs();
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let opposite = match (x, y) {
                (CipConfig::None, CipConfig::None) => true,
                (CipConfig::R, CipConfig::S) | (CipConfig::S, CipConfig::R) => true,
                _ => false,
            };
            assert!(opposite, "{left} vs {right}: atom {i} gave {x} and {y}");
        }
    }
}

/// A functionalised fullerene (PubChem CID 178493059) with six quaternary
/// stereocentres that all see each other. Each labelling had its own budget, and
/// so did every auxiliary descriptor inside every centre's digraph, so the call
/// never finished; one budget for the whole call bounds it at under two seconds
/// in a release build. Ignored by default because a debug build takes far
/// longer -- run with `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn a_cage_of_mutually_visible_centres_finishes() {
    let smiles = "CC[C@@]12C3=C4C5=C1[C@@]6(C7=C8C=CC9=C7[C@]5(C1=C9C=CC5=C1[C@]4(C1=C5C=CC4=C1[C@]3(C1=C4C=CC3=C1[C@]2(C1=C3C=CC8=C16)C1=CC=C(C=C1)C(CC(=O)O)OC=O)C1=CC=C(C=C1)C(CC(=O)O)OC=O)C1=CC=C(C=C1)C(CC(=O)O)C(=O)O)C1=CC=C(C=C1)C(COC=O)C(=O)O)C1=CC=C(C=C1)C(COC=O)OC=O";
    let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
    let start = std::time::Instant::now();
    let labels = molecule.assign_cip_configs();
    assert_eq!(labels.len(), molecule.len());
    assert!(start.elapsed().as_secs() < 30, "took {:?}", start.elapsed());
}
