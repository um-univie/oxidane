//! Stress tests for stereochemistry-aware SMILES canonization.
//!
//! Two properties are exercised:
//!
//! * **Invariance** — every way of writing the *same* molecule must produce the
//!   *same* canonical SMILES. Each `writings` list below holds several distinct
//!   SMILES (generated with RDKit's random writer and confirmed by RDKit to be
//!   the same molecule); they must all collapse to one canonical string.
//! * **Discrimination** — stereoisomers (enantiomers / diastereomers) are
//!   different molecules and must produce *different* canonical strings.
//!
//! Invariance assertions do not pin the exact canonical string
//! (that would bake in whatever the canonizer currently emits); they only
//! require agreement across writings, which is the property that matters.
//!
//! Known gaps are captured as `#[ignore]`d tests at the bottom with the
//! root cause, so `cargo test` stays green while the regressions stay visible.

#[cfg(test)]
mod chiral_canonization_tests {
    use crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;

    fn canon(smiles: &str) -> String {
        canonicalize_and_sort_smiles(smiles, false)
            .unwrap_or_else(|e| panic!("canonization failed for {smiles}: {e:?}"))
    }

    /// All `writings` must canonicalize to a single string.
    fn assert_invariant(name: &str, writings: &[&str]) {
        let mut canons: Vec<(String, String)> =
            writings.iter().map(|w| (w.to_string(), canon(w))).collect();
        canons.sort_by(|a, b| a.1.cmp(&b.1));
        let distinct: std::collections::BTreeSet<&String> =
            canons.iter().map(|(_, c)| c).collect();
        assert_eq!(
            distinct.len(),
            1,
            "INVARIANCE FAILED for {name}: {} distinct canonical forms:\n{}",
            distinct.len(),
            canons
                .iter()
                .map(|(w, c)| format!("    {w:45} -> {c}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    /// Each SMILES in `stereoisomers` must canonicalize to a distinct string.
    fn assert_distinct(name: &str, stereoisomers: &[&str]) {
        let canons: Vec<String> = stereoisomers.iter().map(|s| canon(s)).collect();
        let distinct: std::collections::BTreeSet<&String> = canons.iter().collect();
        assert_eq!(
            distinct.len(),
            stereoisomers.len(),
            "DISCRIMINATION FAILED for {name}: {} isomers collapsed to {}:\n{}",
            stereoisomers.len(),
            distinct.len(),
            stereoisomers
                .iter()
                .zip(&canons)
                .map(|(s, c)| format!("    {s:45} -> {c}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    // ---- INVARIANCE: single tetrahedral centre ---------------------------

    #[test]
    fn invariance_alanine_l() {
        assert_invariant(
            "L-alanine",
            &[
                "[C@H](C)(C(=O)O)N",
                "C(O)(=O)[C@@H](N)C",
                "[C@@H](C)(N)C(=O)O",
                "OC(=O)[C@@H](N)C",
                "O=C([C@H](C)N)O",
                "N[C@H](C(=O)O)C",
            ],
        );
    }

    #[test]
    fn invariance_alanine_d() {
        assert_invariant(
            "D-alanine",
            &[
                "O=C(O)[C@H](N)C",
                "N[C@H](C)C(=O)O",
                "C[C@@H](N)C(O)=O",
                "N[C@@H](C(O)=O)C",
                "OC([C@@H](C)N)=O",
            ],
        );
    }

    #[test]
    fn invariance_2_butanol() {
        assert_invariant(
            "2-butanol",
            &[
                "C[C@@H](CC)O",
                "[C@@H](CC)(C)O",
                "CC[C@@H](O)C",
                "O[C@@H](C)CC",
                "O[C@H](CC)C",
            ],
        );
    }

    #[test]
    fn invariance_bromochlorofluoromethane() {
        assert_invariant(
            "CHFClBr",
            &[
                "[C@@H](Br)(F)Cl",
                "Cl[C@H](Br)F",
                "F[C@H](Cl)Br",
                "Br[C@@H](Cl)F",
                "F[C@@H](Br)Cl",
            ],
        );
    }

    #[test]
    fn invariance_quaternary_center() {
        // No hydrogen on the stereocentre — the four substituents are the only
        // neighbors, exercising the root-atom (no-parent) listing case.
        assert_invariant(
            "C[C@](N)(O)F",
            &[
                "C[C@](F)(N)O",
                "C[C@](N)(O)F",
                "F[C@](N)(C)O",
                "O[C@@](N)(C)F",
                "O[C@](N)(F)C",
                "O[C@](F)(C)N",
            ],
        );
    }

    #[test]
    fn invariance_serine() {
        assert_invariant(
            "serine",
            &[
                "[C@H](N)(C(=O)O)CO",
                "C([C@H](N)CO)(O)=O",
                "C(=O)(O)[C@@H](CO)N",
                "[C@@H](C(O)=O)(N)CO",
                "OC[C@H](C(O)=O)N",
            ],
        );
    }

    // ---- INVARIANCE: multiple stereocentres ------------------------------

    #[test]
    fn invariance_threonine() {
        assert_invariant(
            "threonine",
            &[
                "O=C(O)[C@H](N)[C@@H](C)O",
                "N[C@@H](C(=O)O)[C@@H](C)O",
                "C([C@@H]([C@@H](C)O)N)(O)=O",
                "O[C@@H]([C@H](C(O)=O)N)C",
                "O=C([C@H](N)[C@@H](C)O)O",
            ],
        );
    }

    #[test]
    fn invariance_tartaric_acid() {
        assert_invariant(
            "tartaric acid (R,R)",
            &[
                "OC([C@@H]([C@H](C(=O)O)O)O)=O",
                "C([C@H](O)[C@H](C(=O)O)O)(=O)O",
                "O=C(O)[C@H](O)[C@@H](O)C(O)=O",
                "[C@@H]([C@@H](O)C(=O)O)(C(=O)O)O",
                "O[C@H]([C@H](C(O)=O)O)C(=O)O",
            ],
        );
    }

    #[test]
    fn invariance_cysteine() {
        assert_invariant(
            "L-cysteine",
            &[
                "[C@H](CS)(C(O)=O)N",
                "SC[C@@H](C(O)=O)N",
                "C(S)[C@@H](C(O)=O)N",
                "N[C@H](C(O)=O)CS",
                "OC(=O)[C@H](CS)N",
            ],
        );
    }

    #[test]
    fn invariance_open_chain_sugar() {
        assert_invariant(
            "acyclic polyol",
            &[
                "O[C@@H](C(O)[C@@H](CO)O)CO",
                "[C@H](C([C@H](O)CO)O)(CO)O",
                "OC[C@H](C(O)[C@@H](CO)O)O",
                "OC[C@@H](O)C(O)[C@@H](CO)O",
                "[C@H](C(O)[C@@H](CO)O)(CO)O",
            ],
        );
    }

    // ---- INVARIANCE: single ring + aromatic ------------------------------

    #[test]
    fn invariance_epoxide() {
        assert_invariant(
            "2,3-epoxybutane",
            &[
                "[C@@H]1(O[C@@H]1C)C",
                "O1[C@H](C)[C@H]1C",
                "C[C@@H]1[C@@H](C)O1",
                "O1[C@@H]([C@H]1C)C",
                "C[C@H]1O[C@@H]1C",
            ],
        );
    }

    #[test]
    fn invariance_proline() {
        assert_invariant(
            "proline",
            &[
                "C1CN[C@@H](C1)C(=O)O",
                "N1[C@@H](CCC1)C(O)=O",
                "C1CC[C@@H](C(O)=O)N1",
                "C([C@H]1NCCC1)(=O)O",
                "C1[C@@H](C(O)=O)NCC1",
            ],
        );
    }

    #[test]
    fn invariance_menthol() {
        assert_invariant(
            "menthol",
            &[
                "[C@@H]1(C(C)C)CC[C@@H](C)C[C@H]1O",
                "C1C[C@@H](C)C[C@@H](O)[C@@H]1C(C)C",
                "C[C@@H]1CC[C@@H](C(C)C)[C@@H](C1)O",
                "[C@H]1(C(C)C)[C@H](O)C[C@@H](CC1)C",
                "C[C@H]1C[C@@H](O)[C@@H](CC1)C(C)C",
            ],
        );
    }

    #[test]
    fn invariance_pyranose_sugar() {
        assert_invariant(
            "pyranose",
            &[
                "[C@H]1([C@@H](O)[C@@H]([C@H](C(O)O1)O)O)CO",
                "OC[C@H]1OC(O)[C@@H]([C@H]([C@@H]1O)O)O",
                "O[C@H]1C(O)O[C@@H]([C@@H](O)[C@@H]1O)CO",
                "OC1O[C@H](CO)[C@H]([C@@H]([C@H]1O)O)O",
            ],
        );
    }

    #[test]
    fn invariance_phenylglycine() {
        assert_invariant(
            "phenylglycine",
            &[
                "c1cc(ccc1)[C@@H](C(O)=O)N",
                "N[C@H](C(=O)O)c1ccccc1",
                "OC(=O)[C@@H](N)c1ccccc1",
                "O=C(O)[C@H](c1ccccc1)N",
                "c1ccccc1[C@@H](C(O)=O)N",
            ],
        );
    }

    #[test]
    fn invariance_aromatic_with_chiral_sidechain() {
        assert_invariant(
            "ibuprofen skeleton",
            &[
                "c1([C@H](C)C(=O)O)ccc(cc1)CC(C)C",
                "[C@H](C)(c1ccc(CC(C)C)cc1)C(=O)O",
                "[C@H](C(O)=O)(C)c1ccc(cc1)CC(C)C",
                "O=C([C@H](c1ccc(cc1)CC(C)C)C)O",
                "O=C(O)[C@H](c1ccc(cc1)CC(C)C)C",
            ],
        );
    }

    // ---- DISCRIMINATION --------------------------------------------------

    #[test]
    fn discriminate_alanine_enantiomers() {
        assert_distinct("alanine", &["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"]);
    }

    #[test]
    fn discriminate_butanol_enantiomers() {
        assert_distinct("2-butanol", &["C[C@H](O)CC", "C[C@@H](O)CC"]);
    }

    #[test]
    fn discriminate_halomethane_enantiomers() {
        assert_distinct("CHFClBr", &["F[C@H](Cl)Br", "F[C@@H](Cl)Br"]);
    }

    #[test]
    fn discriminate_quaternary_enantiomers() {
        assert_distinct("C[C@](N)(O)F", &["C[C@](N)(O)F", "C[C@@](N)(O)F"]);
    }

    #[test]
    fn discriminate_threonine_diastereomers() {
        assert_distinct(
            "threonine diastereomers",
            &["C[C@@H](O)[C@@H](N)C(=O)O", "C[C@H](O)[C@@H](N)C(=O)O"],
        );
    }

    #[test]
    fn discriminate_tartaric_chiral_vs_meso() {
        assert_distinct(
            "tartaric (R,R) vs meso",
            &["OC(=O)[C@H](O)[C@@H](O)C(=O)O", "OC(=O)[C@H](O)[C@H](O)C(=O)O"],
        );
    }

    #[test]
    fn discriminate_epoxide_cis_trans() {
        assert_distinct("epoxybutane", &["C[C@H]1O[C@@H]1C", "C[C@H]1O[C@H]1C"]);
    }

    #[test]
    fn discriminate_all_four_halomethane_writings_two_groups() {
        // Two enantiomers, three writings each: exactly two distinct outputs.
        let group_r = ["F[C@H](Cl)Br", "Cl[C@H](Br)F", "Br[C@@H](Cl)F"];
        let group_s = ["F[C@@H](Cl)Br", "Cl[C@@H](Br)F", "Br[C@H](Cl)F"];
        for w in &group_r {
            assert_eq!(canon(w), canon(group_r[0]), "R writing {w} disagreed");
        }
        for w in &group_s {
            assert_eq!(canon(w), canon(group_s[0]), "S writing {w} disagreed");
        }
        assert_ne!(canon(group_r[0]), canon(group_s[0]), "enantiomers collapsed");
    }

    // ---- KNOWN GAPS (documented, currently failing) ----------------------
    //
    // These reproduce open bugs. They are ignored so the suite stays green
    // while keeping the regressions visible; remove `#[ignore]` when fixed.

    // Fused / bridged polycyclic stereocentres (atoms that open more than one
    // ring bond). These exercise the parse-time capture of the true written
    // ring-digit order feeding the index-sorted chirality reference.

    #[test]
    fn invariance_fused_bicyclic() {
        assert_invariant(
            "fused bicyclic",
            &[
                "[C@@H]1(C)C[C@H]2CCCC[C@H]2CC1",
                "[C@@H]1(CC[C@@H]2CCCC[C@@H]2C1)C",
                "C1C[C@@H]2C[C@H](CC[C@@H]2CC1)C",
                "C1[C@@H](C[C@H]2CCCC[C@H]2C1)C",
                "[C@H]12CCCC[C@H]1CC[C@@H](C2)C",
            ],
        );
    }

    #[test]
    fn invariance_camphor() {
        assert_invariant(
            "camphor",
            &[
                "CC1([C@@H]2CC[C@@]1(C(=O)C2)C)C",
                "C1[C@@H]2CC([C@@](C1)(C2(C)C)C)=O",
                "C1[C@@H]2CC([C@](C)(C2(C)C)C1)=O",
                "CC1(C)[C@@H]2CC[C@]1(C)C(=O)C2",
                "C1[C@@]2(C(=O)C[C@@H](C1)C2(C)C)C",
            ],
        );
    }

    #[test]
    fn invariance_bridged_multiring() {
        assert_invariant(
            "bridged multi-ring",
            &[
                "C1CCC[C@@]2([C@H]1C[C@@H](O)CC2)C",
                "C1C[C@@]2(C)[C@@H](C[C@H]1O)CCCC2",
                "C1[C@H](CC[C@]2(C)CCCC[C@@H]21)O",
                "C1[C@@H](C[C@@H]2[C@@](C1)(C)CCCC2)O",
                "C1CCC[C@]2(CC[C@H](O)C[C@H]21)C",
            ],
        );
    }

    // ---- cis/trans double bonds (emitted as directional `/`,`\` bonds) ------

    #[test]
    fn invariance_cis_trans_alkene() {
        assert_invariant("2-butene (E)", &["C/C=C/C", "C(\\C)=C/C", "C(=C/C)\\C"]);
    }

    #[test]
    fn discriminate_cis_trans_alkene() {
        assert_distinct("2-butene E vs Z", &["C/C=C/C", "C/C=C\\C"]);
    }

    #[test]
    fn discriminate_difluoroethene() {
        assert_distinct("difluoroethene E vs Z", &["F/C=C/F", "F/C=C\\F"]);
    }

    #[test]
    fn invariance_maleic_fumaric() {
        // Fumaric acid (E). Different writings must agree.
        assert_invariant(
            "fumaric acid",
            &[
                "OC(=O)/C=C/C(=O)O",
                "O=C(O)/C=C/C(O)=O",
                "OC(=O)/C=C/C(O)=O",
            ],
        );
    }

    /// Writing an allene from the other end reverses the order of all four
    /// carriers, which is an odd permutation, so the equivalent spelling
    /// carries the opposite tag.
    ///
    /// The two spellings this test used to list alongside the first --
    /// `CC(Br)=[C@]=C(C)O` and `OC(C)=[C@@]=C(C)Br` -- are its *enantiomer*:
    /// each transposes one end's two substituents while keeping the tag. They
    /// only passed as invariants while an allene tag meant nothing, which is
    /// what `index_sorted_allene_carriers` fixed.
    #[test]
    fn invariance_allene() {
        assert_invariant(
            "chiral allene",
            &["CC(Br)=[C@]=C(O)C", "OC(C)=[C@@]=C(Br)C"],
        );
        assert_invariant(
            "chiral allene, the other hand",
            &["CC(Br)=[C@@]=C(O)C", "OC(C)=[C@]=C(Br)C"],
        );
    }

    #[test]
    fn discriminate_allene_enantiomers() {
        assert_distinct(
            "allene enantiomers",
            &["CC(Br)=[C@]=C(O)C", "CC(Br)=[C@@]=C(O)C"],
        );
        // Transposing one end's substituents inverts the centre just as
        // surely as flipping the tag does.
        assert_distinct(
            "allene, one end transposed",
            &["CC(Br)=[C@]=C(O)C", "CC(Br)=[C@]=C(C)O"],
        );
    }

    #[test]
    fn discriminate_two_stereogenic_double_bonds() {
        // Two independent E/Z centers separated by a saturated linker.
        assert_distinct(
            "diene with two E/Z centers",
            &[
                "C/C=C/C/C=C/F",
                "C/C=C/C/C=C\\F",
                "C/C=C\\C/C=C/F",
                "C/C=C\\C/C=C\\F",
            ],
        );
    }

    #[test]
    fn invariance_conjugated_diene() {
        assert_invariant(
            "conjugated diene (E,E)",
            &[
                "C/C=C/C=C/C",
                "C(=C/C)\\C=C\\C",
                "C(=C/C=C/C)\\C",
                "C(/C=C/C)=C\\C",
            ],
        );
    }

    #[test]
    fn discriminate_conjugated_diene() {
        // Non-symmetric conjugated diene: four distinct stereoisomers (RDKit).
        assert_distinct(
            "1-fluoro-hexadiene isomers",
            &[
                "C/C=C/C=C/F",
                "C/C=C/C=C\\F",
                "C/C=C\\C=C/F",
                "C/C=C\\C=C\\F",
            ],
        );
    }

    #[test]
    fn invariance_conjugated_triene() {
        assert_invariant(
            "conjugated triene",
            &[
                "OC(=O)/C=C/C=C/C=C/C(O)=O",
                "C(=C/C=C/C(=O)O)\\C=C\\C(O)=O",
                "C(=C/C(=O)O)\\C=C\\C=C\\C(=O)O",
            ],
        );
    }

    #[test]
    fn discriminate_two_stereogenic_double_bonds_linker() {
        // Two E/Z centers separated by one carbon, RDKit-verified as 4 isomers.
        assert_distinct(
            "F/C=C/CC=CC isomers",
            &[
                "C/C=C/C/C=C/F",
                "C/C=C/C/C=C\\F",
                "C/C=C\\C/C=C/F",
                "C/C=C\\C/C=C\\F",
            ],
        );
    }

    #[test]
    fn invariance_base_canonization_symmetric() {
        // Achiral graph-symmetric skeleton (palindromic chain, N one end / O the
        // other) — nauty must use the atom colouring to pin the labeling.
        assert_invariant(
            "symmetric backbone",
            &["NC(C)CCCCCCC(O)C", "OC(C)CCCCCCC(N)C", "CC(N)CCCCCCC(C)O"],
        );
    }

    #[test]
    fn invariance_ring_bridge_automorphism() {
        // Para-disubstituted cyclohexane: the two -CH2CH2- bridges swap under a
        // graph automorphism that flips both stereocentre tags. Canonicalizing the
        // tag assignment under the automorphism group makes it deterministic.
        assert_invariant(
            "para-halocyclohexane",
            &[
                "C1C[C@H](CC[C@H]1Br)Cl",
                "C1[C@H](CC[C@H](C1)Br)Cl",
                "Cl[C@H]1CC[C@H](CC1)Br",
                "Br[C@H]1CC[C@@H](Cl)CC1",
            ],
        );
    }

    // ---- Higher-order stereo: octahedral / trigonal-bipyramidal / square-planar

    #[test]
    fn invariance_octahedral() {
        assert_invariant(
            "octahedral cobalt",
            &[
                "[Co@OH18]([SH])([Br])([I])([CH3])([F])[Cl]",
                "[I][Co@OH28]([Cl])([Br])([F])([CH3])[SH]",
                "[Br][Co@OH2]([CH3])([I])([SH])([Cl])[F]",
                "[CH3][Co@OH5]([F])([Cl])([I])([SH])[Br]",
            ],
        );
    }

    #[test]
    fn invariance_trigonal_bipyramidal() {
        assert_invariant(
            "trigonal bipyramidal arsenic",
            &[
                "S[As@TB1](Br)(F)(Cl)N",
                "N[As@TB4](F)(Cl)(S)Br",
                "[As@TB16](Cl)(F)(N)(S)Br",
                "F[As@TB17](Br)(Cl)(N)S",
            ],
        );
    }

    #[test]
    fn invariance_square_planar() {
        assert_invariant(
            "square planar platinum",
            &[
                "[Pt@SP2]([NH2])([Cl])([Br])[F]",
                "[NH2][Pt@SP2]([Cl])([F])[Br]",
                "[NH2][Pt@SP1]([Br])([Cl])[F]",
                "[NH2][Pt@SP3]([Br])([F])[Cl]",
            ],
        );
    }

    #[test]
    fn discriminate_octahedral_isomers() {
        assert_distinct(
            "octahedral isomers",
            &[
                "S[Co@OH1](F)(I)(Cl)(C)Br",
                "S[Co@OH2](F)(I)(Cl)(C)Br",
                "S[Co@OH5](F)(I)(Cl)(C)Br",
            ],
        );
    }

    #[test]
    fn discriminate_trigonal_bipyramidal_isomers() {
        assert_distinct(
            "TB isomers",
            &["S[As@TB1](F)(Cl)(Br)N", "S[As@TB2](F)(Cl)(Br)N"],
        );
    }
}
