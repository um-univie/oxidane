//! Large-scale, self-contained checks that SMILES canonization is invariant
//! to how a molecule's atoms happen to be ordered.
//!
//! Unlike `benchmark_canonization_from_file` (which replays a 425MB corpus of
//! RDKit-generated alternate writings and is `#[ignore]`d because of its
//! size), the tests here generate their own permutations in Rust via
//! [`reorder_molecule`] and hard-`assert!` on every mismatch, so they run by
//! default under plain `cargo test` and fail loudly on any regression.
//!
//! For every molecule in the corpus we check three properties:
//! * **Invariance** — canonizing a randomly relabeled copy gives back the
//!   same canonical SMILES as canonizing the original.
//! * **Idempotence** — re-parsing and re-canonizing the canonical SMILES
//!   reproduces it exactly.
//! * **Composition** — the molecular formula survives relabeling, reparsing
//!   and canonization unchanged.

#[cfg(test)]
mod canonization_stress_tests {
    use crate::graph_algorithms::mol_canonization::{
        canonicalize_and_sort_smiles, canonize_mol, reorder_molecule,
    };
    use crate::molecular_formula::MolecularFormula;
    use crate::prelude::*;
    use rand::prelude::*;
    use std::collections::BTreeSet;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    /// The 66 drug-like molecules already checked into `tests/smiles.txt`,
    /// plus hand-picked structures exercising charges, isotopes, disconnected
    /// salts, fused/bridged rings, heteroaromatics and triple bonds.
    fn curated_corpus() -> Vec<String> {
        let base = BufReader::new(
            File::open("tests/smiles.txt").expect("tests/smiles.txt should exist"),
        )
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty());

        let extra = [
            "[Na+].[Cl-]",
            "CC(=O)[O-].[NH4+]",
            "c1ccccc1",
            "C1=CC=CC=C1",
            "N#Cc1ccccc1",
            "[13CH4]",
            "O=[N+]([O-])c1ccccc1",
            "C1CC2CCC1CC2",
            "c1ccc2ccccc2c1",
            "c1ccc2c(c1)ccc1ccccc12",
            "c1ccc2[nH]ccc2c1",
            "[Fe+3].[O-]C(=O)C(=O)[O-].[O-]C(=O)C(=O)[O-].[O-]C(=O)C(=O)[O-]",
            "CC(C)(C)c1ccc(O)cc1",
            "OCC1OC(O)C(O)C(O)C1O",
            "c1cc[nH]c1",
            "c1ccsc1",
            "c1ncncn1",
            "C#CC#CC#C",
            "c1ccc(-c2ccccc2)cc1",
            "[O-]S(=O)(=O)[O-].[Na+].[Na+]",
            "CC[N+](CC)(CC)CC.[Br-]",
            "C(F)(F)(F)C(Cl)(Br)I",
            "O=C1CCCCC1",
            "N1C=CC=C1",
            // More than nine ring closures open at once, which is where a
            // renumbered label crosses the one-digit boundary and has to gain
            // or lose its `%`.
            "C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C8=C2C8=C%11C%10=C1C1=C8C(=C23)C4=C1C5=C67",
            // An aromatic heteroatom whose hydrogen a bare symbol would drop.
            "c1cc[pH]c1",
            "c1cc[asH]c1",
        ];

        base.chain(extra.into_iter().map(String::from)).collect()
    }

    fn random_permutation(n: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut labeling: Vec<usize> = (0..n).collect();
        labeling.shuffle(rng);
        labeling
    }

    /// Parses `smiles`, canonizes every fragment, then checks idempotence and
    /// `permutations_per_molecule` random relabelings against it, appending a
    /// description of every failure to `issues`.
    fn check_permutation_invariance(
        smiles: &str,
        permutations_per_molecule: usize,
        seed: u64,
        issues: &mut Vec<String>,
    ) {
        let mols = match MolecularGraph::from_smiles(smiles) {
            Ok(mols) => mols,
            Err(e) => {
                issues.push(format!("{smiles}: failed to parse ({e:?})"));
                return;
            }
        };

        for (fragment_index, mol) in mols.iter().enumerate() {
            let reference_formula = MolecularFormula::from_molecule(mol);
            let reference_canonical = canonize_mol(mol).to_smiles();

            match MolecularGraph::from_smiles_single(&reference_canonical) {
                Ok(reparsed) => {
                    if MolecularFormula::from_molecule(&reparsed) != reference_formula {
                        issues.push(format!(
                            "{smiles} (fragment {fragment_index}): canonical SMILES \
                             {reference_canonical} has a different formula on re-parse"
                        ));
                    }
                    let recanonicalized = canonize_mol(&reparsed).to_smiles();
                    if recanonicalized != reference_canonical {
                        issues.push(format!(
                            "{smiles} (fragment {fragment_index}): canonization is not \
                             idempotent: {reference_canonical} -> {recanonicalized}"
                        ));
                    }
                }
                Err(e) => issues.push(format!(
                    "{smiles} (fragment {fragment_index}): canonical SMILES \
                     {reference_canonical} failed to re-parse ({e:?})"
                )),
            }

            let mut rng =
                StdRng::seed_from_u64(seed ^ (fragment_index as u64).wrapping_mul(0x9E37_79B9));
            for trial in 0..permutations_per_molecule {
                let labeling = random_permutation(mol.len(), &mut rng);
                let mut permuted = mol.clone();
                reorder_molecule(&mut permuted, &labeling);

                let permuted_smiles = permuted.to_smiles();
                let reparsed = match MolecularGraph::from_smiles_single(&permuted_smiles) {
                    Ok(m) => m,
                    Err(e) => {
                        issues.push(format!(
                            "{smiles} (fragment {fragment_index}, trial {trial}): permuted \
                             SMILES {permuted_smiles} (labeling {labeling:?}) failed to \
                             re-parse ({e:?})"
                        ));
                        continue;
                    }
                };

                let permuted_formula = MolecularFormula::from_molecule(&reparsed);
                if permuted_formula != reference_formula {
                    issues.push(format!(
                        "{smiles} (fragment {fragment_index}, trial {trial}): permuted SMILES \
                         {permuted_smiles} has formula {permuted_formula}, expected \
                         {reference_formula}"
                    ));
                    continue;
                }

                let permuted_canonical = canonize_mol(&permuted).to_smiles();
                if permuted_canonical != reference_canonical {
                    issues.push(format!(
                        "{smiles} (fragment {fragment_index}, trial {trial}): {permuted_smiles} \
                         (labeling {labeling:?}) -> {permuted_canonical}, expected \
                         {reference_canonical}"
                    ));
                }
            }
        }
    }

    #[test]
    fn permutation_invariance_curated_corpus() {
        let corpus = curated_corpus();
        assert!(
            corpus.len() >= 80,
            "curated corpus unexpectedly small ({} entries)",
            corpus.len()
        );

        let mut issues = Vec::new();
        for smiles in &corpus {
            check_permutation_invariance(smiles, 20, 0x5EED_1234, &mut issues);
        }

        assert!(
            issues.is_empty(),
            "{} permutation-invariance issue(s) found across {} molecules:\n{}",
            issues.len(),
            corpus.len(),
            issues.join("\n")
        );
    }

    fn canon_sorted(smiles: &str) -> String {
        canonicalize_and_sort_smiles(smiles, true)
            .unwrap_or_else(|e| panic!("canonization failed for {smiles}: {e:?}"))
    }

    /// All `writings` (distinct SMILES for the same molecule) must
    /// canonicalize to a single string.
    fn assert_all_equal(name: &str, writings: &[&str]) {
        let canons: Vec<(String, String)> = writings
            .iter()
            .map(|w| (w.to_string(), canon_sorted(w)))
            .collect();
        let distinct: BTreeSet<&String> = canons.iter().map(|(_, c)| c).collect();
        assert_eq!(
            distinct.len(),
            1,
            "INVARIANCE FAILED for {name}: {} distinct canonical forms:\n{}",
            distinct.len(),
            canons
                .iter()
                .map(|(w, c)| format!("    {w:60} -> {c}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    /// Skipping perception for a molecule that has already been perceived must
    /// change nothing but the time it takes. Every molecule here comes from
    /// `from_smiles`, which perceives, so the two entry points have to agree --
    /// including on the aromatic/Kekule pairs, which is exactly where they
    /// would diverge if the skip were unsound.
    #[test]
    fn skipping_perception_agrees_when_the_molecule_was_perceived() {
        use crate::graph_algorithms::mol_canonization::canonize_mol_perceived;

        for smiles in [
            "c1ccccc1",
            "C1=CC=CC=C1",
            "c1ccc2ccccc2c1",
            "C1=CC2=CC=CC=C2C=C1",
            "c1cc[nH]c1",
            "CC(=O)Oc1ccccc1C(=O)O",
            "O=C(C)Oc1ccccc1C(O)=O",
            "c1ccc(-c2ccccc2)cc1",
            "[Na+].[Cl-]",
            "C[Si](C)CC",
        ] {
            let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(
                canonize_mol_perceived(&molecule).to_smiles(),
                canonize_mol(&molecule).to_smiles(),
                "{smiles}"
            );
        }
    }

    // ---- Regression: radical states survive canonization -----------------
    //
    // `reorder_molecule` permuted atomic numbers, charges, isotopes, atom
    // classes, bonds and chirality tags, and left `radical_states` alone. The
    // unpaired electron therefore stayed at the index its atom used to
    // occupy and was written out against whatever atom landed there -- as a
    // hydrogen count short of the valence, which is how SMILES spells a
    // radical. `C[Si](C)CC` came out as `[Si]([CH3])(C)CC`: a methyl radical
    // that is not in the molecule, on a different methyl for each way of
    // writing the same compound.
    //
    // These use a bracketed atom with a spare valence, since that is where
    // radicals come from on the way in. There is nothing silicon-specific
    // about it; silicon is just the common case in the corpus.
    #[test]
    fn regression_radicals_do_not_move_during_canonization() {
        assert_all_equal(
            "dimethylethylsilyl radical",
            &["C[Si](C)CC", "CC[Si](C)C", "[Si](C)(C)CC"],
        );
        assert_all_equal(
            "dimethylphenylsilyl radical",
            &[
                "C[Si](C)c1ccccc1",
                "c1ccccc1[Si](C)C",
                "[Si](C)(C)c1ccccc1",
                "c1cc([Si](C)C)ccc1",
            ],
        );
        // A radical must still be written where there really is one, and only
        // there: the methyl radical keeps its brackets, methane has none.
        assert_eq!(canon_sorted("[CH3]"), "[CH3]");
        assert_eq!(canon_sorted("C"), "C");
    }

    // ---- Regression: every fragment gets its radicals assigned ------------
    //
    // The pass that turns a bracketed atom's unaccounted valence into unpaired
    // electrons ran once, over whatever was left in the parser when the string
    // ended -- which is only the last fragment. In `C(C)[Ge](CC)CC.[K]` the
    // potassium was marked and the germanium was not, so the germanium had a
    // hydrogen invented for it; writing the fragments the other way round
    // moved the invented hydrogen to the potassium.
    #[test]
    fn regression_radicals_are_assigned_in_every_fragment() {
        assert_all_equal(
            "triethylgermyl radical with potassium",
            &[
                "C(C)[Ge](CC)CC.[K]",
                "[K].C(C)[Ge](CC)CC",
                "[K].CC[Ge](CC)CC",
                "CC[Ge](CC)CC.[K]",
                "[Ge](CC)(CC)CC.[K]",
            ],
        );
        assert_all_equal("two bare fragments", &["[C].[N]", "[N].[C]"]);
    }

    // ---- Brackets are not what makes an atom a radical -------------------
    //
    // The rule is the valence left after the bonds *and* the stated hydrogens,
    // so a bracketed atom is usually not a radical at all. Pinned here because
    // both bugs above looked like stray brackets, and the obvious wrong fix --
    // treating any bracketed atom as a radical, or none of them -- passes the
    // invariance tests while making the chemistry wrong in one direction or
    // the other. Every count below agrees with RDKit's
    // GetNumRadicalElectrons().
    #[test]
    fn radicals_are_the_valence_left_after_hydrogens() {
        let radicals = |smiles: &str| -> Vec<u8> {
            let mols = MolecularGraph::from_smiles(smiles).expect("should parse");
            mols.iter()
                .flat_map(|mol| {
                    mol.atomic_numbers()
                        .iter()
                        .enumerate()
                        .filter(|&(_, &z)| z != 1)
                        .map(|(index, _)| mol.radical_states()[index])
                        .collect::<Vec<_>>()
                })
                .collect()
        };

        // Ethane written with one carbon bracketed: the stated hydrogens and
        // the C-C bond fill both valences, so neither atom is a radical.
        assert_eq!(radicals("C[CH3]"), vec![0, 0]);
        assert_eq!(radicals("CC"), vec![0, 0]);
        assert_eq!(radicals("[NH2]C"), vec![0, 0]);
        // Bracketed and fully substituted: still not a radical.
        assert_eq!(radicals("C[Si](C)(C)C"), vec![0, 0, 0, 0, 0]);
        assert_eq!(radicals("C[SiH](C)CC"), vec![0, 0, 0, 0, 0]);

        // One short of the valence, in each case by one stated hydrogen.
        assert_eq!(radicals("[CH3]"), vec![1]);
        assert_eq!(radicals("C[Si](C)CC"), vec![0, 1, 0, 0, 0]);
        assert_eq!(radicals("[K]"), vec![1]);
        // Further short: a carbene has two, a bare carbon four.
        assert_eq!(radicals("[CH2]"), vec![2]);
        assert_eq!(radicals("[C]"), vec![4]);
    }

    // ---- Regression: fused aromatic sulfonium electron count -------------
    //
    // `handle_sulfur_aromaticity`'s formal_charge==+1 arm had no fallback for
    // a tricoordinate sulfonium (S+, valency=3, bond_count=3, no exocyclic
    // O-): it fell through to lookup_index 0 (NotAromatic), so a fused-ring
    // sulfonium like dibenzothiophenium was only perceived aromatic when the
    // input SMILES was already lowercase (bond order pre-tagged Aromatic at
    // parse time) and not when Kekulized. Fixed by giving that case the same
    // 2-electron (lone-pair-donor) classification as the S-O(-) case, and by
    // fixing `find_aromatic_cycles`'s ring-growth loop, which was resetting
    // `visited` for every atom every pass instead of only the atoms it had
    // just processed, corrupting the growth frontier for fused ring systems.
    #[test]
    fn regression_aromatic_sulfonium_fused_ring() {
        assert_all_equal(
            "diiodo-tert-butyl dibenzothiophenium",
            &[
                "c12[s+](-c3ccc(C(C)(C)C)cc3)c3c(cc(I)cc3)c1cc(I)cc2",
                "Ic1ccc2[s+](-c3ccc(cc3)C(C)(C)C)c4ccc(I)cc4c2c1",
                "CC(C)(C)C1=CC=C([S+]2C3=CC=C(I)C=C3C3=C2C=CC(I)=C3)C=C1",
                "CC(C)(C)C1=CC=C([S+]2C3=C(C=C(I)C=C3)C3=C2C=CC(I)=C3)C=C1",
                "CC(C)(C)C1=CC=C([S+]2C3=C(C=C(I)C=C3)C3=CC(I)=CC=C32)C=C1",
            ],
        );
        assert_all_equal(
            "difluoro dibenzothiophenium",
            &[
                "c1c(F)cc2c(c1)[s+](-c1ccccc1)c1ccc(F)cc21",
                "[s+]1(-c2ccccc2)c3ccc(F)cc3c4cc(F)ccc14",
                "FC1=CC2=C(C=C1)[S+](C1=CC=CC=C1)C1=C2C=C(F)C=C1",
                "FC1=CC=C2C(=C1)C1=C(C=CC(F)=C1)[S+]2C1=CC=CC=C1",
                "FC1=CC2=C(C=C1)[S+](C1=CC=CC=C1)C1=CC=C(F)C=C21",
            ],
        );
    }

    // ---- Regression: aromatic ring written with a triple bond -------------
    //
    // The last two mismatches in the permutation corpus, and an instructive
    // pair. RDKit reports the triple-bonded ring atoms of `c1cc#cc(C)n1` as
    // both `SP`-hybridized and aromatic, which cannot both hold: a linear sp
    // carbon has no p orbital lying in the ring plane to overlap. The Kekulé
    // writing disagreed with the lower-case ones here because perception read
    // the bond orders it was handed and the lower-case writings were never
    // re-derived. Localizing the input before perceiving removes the
    // difference: all three writings now agree, and they agree on the reading
    // that a triple bond in the ring is not aromatic.
    #[test]
    fn aromatic_ring_written_with_a_triple_bond() {
        // A pyridine ring written with an explicit triple bond (`c#c`) to a
        // ring carbon: the Kekule writing disagrees with the aromatic ones.
        assert_all_equal(
            "macrocycle with alkynyl-pyridine",
            &[
                "CC1=CCC(C(C)=Cc2cc#cc(C)n2)OC(=O)CCC(C)(O)C(=O)C(C)C(O)C(C)CCC1",
                "OC1C(C)C(=O)C(O)(C)CCC(OC(CC=C(C)CCCC1C)C(C)=Cc2nc(c#cc2)C)=O",
                "CC1=CCC(C(C)=CC2=CC#CC(C)=N2)OC(=O)CCC(C)(O)C(=O)C(C)C(O)C(C)CCC1",
            ],
        );
    }
}
