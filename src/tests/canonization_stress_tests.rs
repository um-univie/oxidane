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

    #[test]
    fn hydrogen_first_reordering_preserves_stereochemistry() {
        let molecule = MolecularGraph::from_smiles_single("CC(C)[C@@H](C(=O)O)N").unwrap();
        let expected = canonize_mol(&molecule).to_smiles();
        for first in 0..molecule.len() {
            let mut order: Vec<_> = (0..molecule.len()).collect();
            order.swap(0, first);
            let mut permuted = molecule.clone();
            reorder_molecule(&mut permuted, &order);
            // Emission must list every neighbor exactly once, even when atom
            // zero is a hydrogen folded into a tetrahedral centre.
            let emission = permuted.smiles_emission_orders();
            for atom in 0..permuted.len() {
                if !matches!(permuted.get_chiral_class(atom), ChiralClass::TH(_)) {
                    continue;
                }
                let mut emitted = emission[atom].clone();
                emitted.sort_unstable();
                let neighbors = crate::molecule::chirality::index_sorted_neighbors(&permuted, atom);
                assert_eq!(emitted, neighbors, "root {first}, centre {atom}");
            }
            assert_eq!(
                canonize_mol(&permuted).to_smiles(),
                expected,
                "root {first}"
            );
            let written = permuted.to_smiles();
            let reparsed = MolecularGraph::from_smiles_single(&written).unwrap();
            assert_eq!(
                canonize_mol(&reparsed).to_smiles(),
                expected,
                "root {first}: {written}"
            );
            reorder_molecule(&mut permuted, &order); // a swap is its own inverse
            assert_eq!(
                permuted.chiral_classes(),
                molecule.chiral_classes(),
                "root {first}"
            );
        }
    }

    /// A hydrogen on no heavy atom is fixed by every automorphism. Left out of
    /// the full-atom permutation it voided every generator, the tags were never
    /// canonized, and the E and Z forms of one alkene beside `[H+]` came out in
    /// whichever order the input gave them.
    #[test]
    fn a_bare_hydrogen_leaves_the_symmetry_whole() {
        for smiles in [
            "[H+].C/C=C/C.C/C=C\\C",
            "[H][H].F/C=C/F.F/C=C\\F",
            "[H-].[H+].Cl/C=C/C(F)(F)F.Cl/C=C\\C(F)(F)F",
        ] {
            let parts = MolecularGraph::from_smiles(smiles).unwrap();
            let parts: Vec<&MolecularGraph> = parts.iter().collect();
            let molecule = MolecularGraph::combine(&parts);
            let expected = canonize_mol(&molecule).to_smiles();
            let mut rng = StdRng::seed_from_u64(0xBEEF);
            for trial in 0..32 {
                let order = random_permutation(molecule.len(), &mut rng);
                let mut permuted = molecule.clone();
                reorder_molecule(&mut permuted, &order);
                assert_eq!(
                    canonize_mol(&permuted).to_smiles(),
                    expected,
                    "{smiles}, trial {trial}"
                );
            }
        }
    }

    /// How many copies of a component a record holds is no limit on telling
    /// their stereo apart. Six butenes, E and Z mixed, beside four butynes are
    /// 6! * 2^6 * 4! * 2^4 automorphisms, and walking them, bounded at 4096,
    /// left which isomer came first to the input's order. Read against the
    /// orbits an E and a Z are no longer interchangeable, and nothing is left
    /// to walk. A tag the orbits cannot read but one automorphism turns over
    /// alone -- isopropanol's, seven times -- is free, and is written the
    /// smaller way without a walk; and copies that carry such a tag are told
    /// apart from copies that carry none.
    #[test]
    fn many_copies_of_a_stereo_component_canonize_alike() {
        for smiles in [
            "C/C=C/C.C/C=C/C.C/C=C/C.C/C=C\\C.C/C=C\\C.C/C=C\\C.CC#CC.CC#CC.CC#CC.CC#CC",
            "C[C@H](C)O.C[C@@H](C)O.C[C@H](C)O.C[C@@H](C)O.C[C@H](C)O.C[C@@H](C)O.C[C@H](C)O",
            "C[C@H](C)O.C[C@H](C)O.C[C@H](C)O.C[C@H](C)O.CC(C)O.CC(C)O.CC(C)O.CC(C)O",
        ] {
            let parts = MolecularGraph::from_smiles(smiles).unwrap();
            let parts: Vec<&MolecularGraph> = parts.iter().collect();
            let molecule = MolecularGraph::combine(&parts);
            let expected = canonize_mol(&molecule).to_smiles();
            let mut rng = StdRng::seed_from_u64(0xBEEF);
            for trial in 0..32 {
                let order = random_permutation(molecule.len(), &mut rng);
                let mut permuted = molecule.clone();
                reorder_molecule(&mut permuted, &order);
                assert_eq!(
                    canonize_mol(&permuted).to_smiles(),
                    expected,
                    "{smiles}, trial {trial}"
                );
            }
        }
    }

    /// A record's components are canonized one at a time. Cycloalkanes of
    /// different sizes side by side are a graph colour refinement cannot split,
    /// and searched whole, eight of each of four sizes did not finish: every
    /// extra set of four multiplied the time by ten. PubChem's CID 157779877
    /// holds 57 such rings and ran for hours. The record's tautomer key goes
    /// through the same labelling, so it is checked too.
    #[test]
    fn a_record_of_many_small_rings_canonizes_alike() {
        use crate::molecule::tautomer::{TautomerRules, tautomer_key};
        let rules = TautomerRules::new();
        let rings = [
            "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1", "C1CCOC1", "C1CCSCC1",
        ];
        let smiles = rings.repeat(8).join(".");
        let parts = MolecularGraph::from_smiles(&smiles).unwrap();
        let parts: Vec<&MolecularGraph> = parts.iter().collect();
        let molecule = MolecularGraph::combine(&parts);
        let expected = canonize_mol(&molecule).to_smiles();
        let key = tautomer_key(&molecule, &rules);
        let mut rng = StdRng::seed_from_u64(157779877);
        for trial in 0..8 {
            let order = random_permutation(molecule.len(), &mut rng);
            let mut permuted = molecule.clone();
            reorder_molecule(&mut permuted, &order);
            assert_eq!(
                canonize_mol(&permuted).to_smiles(),
                expected,
                "trial {trial}"
            );
            assert_eq!(tautomer_key(&permuted, &rules), key, "trial {trial}");
        }
    }

    #[test]
    fn pubchem_stereo_permutations_preserve_both_canonical_forms() {
        use crate::molecule::tautomer::{TautomerRules, normalize, tautomer_key};
        let rules = TautomerRules::new();
        for line in include_str!("../../tests/pubchem_stereo_permutation_regressions.tsv").lines() {
            let (cid, smiles) = line.split_once('\t').unwrap();
            for molecule in MolecularGraph::from_smiles(smiles).unwrap() {
                let expected = canonize_mol(&molecule).to_smiles();
                let normal = normalize(&molecule, &rules).unwrap().to_smiles();
                let key = tautomer_key(&molecule, &rules);
                let mut rng = StdRng::seed_from_u64(cid.parse::<u64>().unwrap() ^ 0xBEEF);
                for trial in 0..12 {
                    let order = random_permutation(molecule.len(), &mut rng);
                    let mut permuted = molecule.clone();
                    reorder_molecule(&mut permuted, &order);
                    assert_eq!(
                        canonize_mol(&permuted).to_smiles(),
                        expected,
                        "CID {cid}, trial {trial}"
                    );
                    assert_eq!(
                        normalize(&permuted, &rules).unwrap().to_smiles(),
                        normal,
                        "CID {cid}, trial {trial}"
                    );
                    assert_eq!(
                        tautomer_key(&permuted, &rules),
                        key,
                        "CID {cid}, trial {trial}"
                    );
                }
            }
        }
    }

    #[test]
    fn cage_symmetry_group_is_complete_after_relabeling() {
        use crate::graph_algorithms::mol_canonization::Canonizer;
        use crate::molecule::tautomer::{TautomerRules, erased_skeleton};
        let record = include_str!("../../tests/pubchem_stereo_permutation_regressions.tsv")
            .lines()
            .find(|line| line.starts_with("4048341\t"))
            .unwrap();
        let molecule =
            MolecularGraph::from_smiles_single(record.split_once('\t').unwrap().1).unwrap();
        let (skeleton, _) = erased_skeleton(&molecule, &TautomerRules::new());
        let mut rng = StdRng::seed_from_u64(4048341 ^ 0xBEEF);
        for graph in [molecule, skeleton] {
            for trial in 0..32 {
                let mut permuted = graph.clone();
                let order = random_permutation(graph.len(), &mut rng);
                reorder_molecule(&mut permuted, &order);
                let mut canonizer = Canonizer::new();
                let canonical = canonizer.canonize(&permuted);
                let heavy = canonical
                    .atomic_numbers()
                    .iter()
                    .filter(|&&z| z != 1)
                    .count();
                let identity: Vec<_> = (0..heavy).collect();
                let mut group = BTreeSet::from([identity.clone()]);
                let mut pending = vec![identity];
                while let Some(element) = pending.pop() {
                    for generator in canonizer.last_graph_automorphisms() {
                        let composed: Vec<_> =
                            element.iter().map(|&atom| generator[atom]).collect();
                        if group.insert(composed.clone()) {
                            pending.push(composed);
                            assert!(group.len() <= 4, "spurious cage symmetry, trial {trial}");
                        }
                    }
                }
                // Four heavy-atom symmetries, independently checked against
                // graph self-isomorphisms; canonical-label equality alone did
                // not catch the backend pruning away a symmetry generator.
                assert_eq!(group.len(), 4, "incomplete cage symmetry, trial {trial}");
            }
        }
    }

    /// The 66 drug-like molecules already checked into `tests/smiles.txt`,
    /// plus hand-picked structures exercising charges, isotopes, disconnected
    /// salts, fused/bridged rings, heteroaromatics and triple bonds.
    fn curated_corpus() -> Vec<String> {
        let base =
            BufReader::new(File::open("tests/smiles.txt").expect("tests/smiles.txt should exist"))
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

    /// The digits an atom writes are the ones the finished string shows, in
    /// ascending order.
    ///
    /// The writer numbers rings as it meets them and a string pass renumbers
    /// them afterwards by order of first appearance. Sorting before that pass
    /// sorted the wrong numbers: this tetralithium carbanion macrocycle wrote
    /// `\5=2` from its deposited form and `=2\5` from its own canonical one,
    /// the same two closures read out in opposite orders, so canonization took
    /// a second pass to settle. PubChem CID 134866860.
    #[test]
    fn ring_closure_digits_are_the_ones_the_string_ends_up_with() {
        assert_all_equal(
            "tetralithium carbanion macrocycle",
            &[
                "CC(C1=C/C/2=C/3\\C(=C\\4/C(=C\\5/C(=C2/[C-]=C1)/[C-]=CC(=C5)C(C)(C)C)/C=C(C=[C-]4)C(C)(C)C)\\[C-]=CC(=C3)C(C)(C)C)(C)C",
                "c1c(c[c-]c=2\\c1=c3/cc(c[c-]/c/3=c4/c(cc(c[c-]4)C(C)(C)C)=c5\\cc(c[c-]\\c=2\\5)C(C)(C)C)C(C)(C)C)C(C)(C)C",
            ],
        );
    }

    /// A ring whose every atom brings a lone pair and none brings half of a π
    /// bond has no π system to delocalize, whatever Hückel's count says.
    ///
    /// `O1S[Te+]1` is three such atoms and six electrons, so `4n + 2` held and
    /// the ring was perceived aromatic -- and the writer then emitted
    /// `[te]1s[o+]1`, which no reader takes back, this crate's own included,
    /// because a three-membered ring of aromatic bonds has no Kekulé structure.
    /// RDKit reads the same molecule as three single bonds. PubChem CID
    /// 173332364.
    #[test]
    fn a_ring_of_lone_pairs_is_not_aromatic() {
        let molecule = MolecularGraph::from_smiles_single("O1S[Te+]1").expect("the SMILES parses");
        let written = canonize_mol(&molecule).to_smiles();
        assert!(
            !written.contains("[te]") && !written.contains("[o+]"),
            "{written} should not be aromatic"
        );
        MolecularGraph::from_smiles_single(&written)
            .unwrap_or_else(|_| panic!("{written} should parse back"));

        // The cyclopropenyl cation has a π bond and two electrons, and stays
        // aromatic.
        let cation = MolecularGraph::from_smiles_single("C1=C[CH+]1").expect("the SMILES parses");
        let written = canonize_mol(&cation).to_smiles();
        assert!(written.contains("[cH+]"), "{written} should stay aromatic");
    }

    /// An atom that opens two ring closures writes their digits in one order,
    /// whichever way the molecule was handed to the writer.
    ///
    /// The two ends of a ring are reached at different moments, so an atom's
    /// digits were pushed in one order when it opened them and in another when
    /// it closed them -- and which happened depended on the bond lists, which
    /// are a fact about the input rather than about the molecule. A bis-indolyl
    /// pigment from PubChem (CID 10817494) canonicalized to `...ccccc32...`
    /// from its deposited form and to `...ccccc23...` from that answer, so
    /// canonization was not idempotent and the second pass was needed to settle
    /// it. Found by the corpus sweep, 10.8 million records in.
    #[test]
    fn ring_closure_digits_do_not_depend_on_the_writing() {
        assert_all_equal(
            "bis-indolyl pigment with fused ring closures",
            &[
                "C1=CC=C2C(=C1)C(=CN2)C3=[13CH][13C](=C(N3)/C=C\\4/[13C](=[13CH]/C(=C\\5/C=NC6=CC=CC=C65)/N4)C7=CNC8=CC=CC=C87)C9=CNC1=CC=CC=C19",
                "[nH]1/c([13cH][13c](-c2c[nH]c3ccccc32)/c1=C/c4[nH]c([13cH][13c]4-c5c[nH]c6ccccc65)-c7c[nH]c8ccccc87)=C9/c%10ccccc%10N=C9",
            ],
        );
    }
}
