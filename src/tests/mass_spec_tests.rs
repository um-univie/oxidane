//! Parent compounds, as a spectral library records them.
//!
//! Each case here is a way a spectral library or registry record really writes
//! a compound -- a salt, a sodium drawn as a bond, a zwitterion, the other
//! spelling of a sulfoxide -- and the property is that every writing of one
//! compound reduces to the same parent, while compounds that differ do not.

use crate::graph_algorithms::mol_canonization::{canonicalize_and_sort_smiles, reorder_molecule};
use crate::molecule::mass_spec::{COUNTERIONS, ParentOptions, parent_components, parent_smiles};
use crate::molecule::tautomer::{TautomerRules, normalized_smiles};
use crate::prelude::*;
use rand::{SeedableRng, seq::SliceRandom};

fn parent(smiles: &str) -> String {
    parent_smiles(smiles, &ParentOptions::default()).expect("the SMILES parses")
}

fn canonical(smiles: &str) -> String {
    canonicalize_and_sort_smiles(smiles, false).expect("the SMILES parses")
}

fn net_charge(smiles: &str) -> i32 {
    MolecularGraph::from_smiles(smiles)
        .expect("the SMILES parses")
        .iter()
        .map(|component| component.formal_charge())
        .sum()
}

fn options(disconnect_metals: bool, neutralize: bool, strip_salts: bool) -> ParentOptions {
    let mut options = ParentOptions::default();
    options.disconnect_metals = disconnect_metals;
    options.neutralize = neutralize;
    options.strip_salts = strip_salts;
    options
}

/// Every group here is one compound, and all its writings give one parent.
#[test]
fn writings_of_one_compound_share_a_parent() {
    for writings in [
        // Salts and the free acid or base.
        &[
            "CCN",
            "CCN.Cl",
            "CC[NH3+].[Cl-]",
            "Cl.NCC",
            "CCN.OC(=O)C(F)(F)F",
        ][..],
        &[
            "CC(=O)O",
            "CC(=O)[O-].[Na+]",
            "[Na+].CC([O-])=O",
            "CC(=O)O[Na]",
        ],
        &["c1ccncc1", "c1cc[nH+]cc1.[Br-]", "c1ccncc1.O"],
        &["CS(=O)(=O)O", "CS(=O)(=O)[O-].[K+]", "CS(=O)(=O)O[K]"],
        &[
            "CN1CCC[C@H]1c1cccnc1",
            "OC(=O)/C=C\\C(=O)O.CN1CCC[C@H]1c1cccnc1",
            "C[NH+]1CCC[C@H]1c1cccnc1.OC(=O)C(O)C(O)C(=O)[O-]",
            "CCCCCCCCCCCCCCCCCC(=O)O.CN1CCC[C@H]1c1cccnc1",
            "CN1CCC[C@H]1c1cccnc1.CN1CCC[C@H]1c1cccnc1.OS(=O)(=O)O",
        ],
        // Dimenhydrinate, aminophylline and a methyl sulfate: the salt
        // formers the GNPS library deposits most beside a drug.
        &[
            "CN(C)CCOC(c1ccccc1)c1ccccc1",
            "CN(C)CCOC(c1ccccc1)c1ccccc1.Cn1c(=O)c2[nH]c(Cl)nc2n(C)c1=O",
        ],
        &[
            "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
            "Cn1c(=O)c2[nH]cnc2n(C)c1=O.Cn1c(=O)c2[nH]cnc2n(C)c1=O.NCCN",
        ],
        &[
            "C[N+](C)(C)CCOC(C)=O",
            "C[N+](C)(C)CCOC(C)=O.COS(=O)(=O)[O-]",
        ],
        // Phosphate esters: any degree of ionization.
        &[
            "COP(=O)(O)O",
            "COP(=O)([O-])O",
            "COP(=O)([O-])[O-].[Na+].[Na+]",
        ],
        // An amino acid is its neutral form, not its zwitterion.
        &["NCC(=O)O", "[NH3+]CC([O-])=O", "[NH3+]CC(=O)O.[Cl-]"],
        &[
            "C[C@H](N)C(=O)O",
            "C[C@H]([NH3+])C(=O)[O-]",
            "C[C@H](N)C(=O)O.Cl",
        ],
        // A fixed cation keeps the anion that balances it, from either side.
        &[
            "C[N+](C)(C)CC(=O)[O-]",
            "C[N+](C)(C)CC(=O)O",
            "C[N+](C)(C)CC(=O)O.[Cl-]",
        ],
        &[
            "C[N+](C)(C)CC(O)CC(=O)[O-]",
            "OC(C[N+](C)(C)C)CC(O)=O",
            "C[N+](C)(C)CC([O-])CC(=O)O",
        ],
        &[
            "C[N+](C)(C)CCOP(=O)([O-])OCC(O)CO",
            "C[N+](C)(C)CCOP(=O)(O)OCC(O)CO",
        ],
        // A fixed anion keeps the ammonium that balances it.
        &[
            "[NH3+]CC[B-](F)(F)F",
            "NCC[B-](F)(F)F",
            "NCC[B-](F)(F)F.[K+]",
        ],
        // A cation with nothing to balance it stays a cation, salt or not.
        &[
            "OCC[N+](C)(C)C",
            "OCC[N+](C)(C)C.[Cl-]",
            "[Cl-].OCC[N+](C)(C)C",
            "C[N+](C)(C)CC[O-]",
        ],
        &["C[n+]1ccccc1", "C[n+]1ccccc1.[I-]"],
        // A sodium drawn into an amide nitrogen as the [M+Na]+ ion was.
        &["CC(=O)N(C)CCO", "CC(=O)[N+](C)([Na])CCO"],
        // One spelling per group.
        &["CS(C)=O", "C[S+](C)[O-]"],
        &["CS(C)(=O)=O", "C[S+2](C)([O-])[O-]", "C[S+](C)([O-])=O"],
        &["CP(C)(C)=O", "C[P+](C)(C)[O-]"],
        &["CN=[N+]=[N-]", "CN=N#N"],
        &["CC(=O)C=[N+]=[N-]", "CC(=O)[CH-][N+]#N"],
        &["C[N+](=O)[O-]", "CN(=O)=O"],
        // A pyridone drawn as its betaine is the pyridone, and so is its
        // conjugate acid, protonated on the oxygen: losing that proton leaves
        // no charge and no ring less aromatic.
        &[
            "CN1C=CC=CC1=O",
            "[O-]c1cccc[n+]1C",
            "C[n+]1ccccc1[O-]",
            "OC1=CC=CC=[N+]1C",
            "Oc1cccc[n+]1C.[Cl-]",
        ],
        // A protonated heteroaromatic gives its proton back wherever the
        // canonical tautomer writes the charge: on the methylated nitrogen of
        // an imidazolium, or on the sulfur of a benzothiadiazole.
        &[
            "Cn1ccnc1",
            "C[n+]1cc[nH]c1",
            "Cn1cc[nH+]c1",
            "C[n+]1cc[nH]c1.[Cl-]",
        ],
        &[
            "Cc1ccc2[nH]c(=O)c(CN(CC3CCCO3)S(=O)(=O)c3cccc4nsnc34)cc2c1",
            "Cc1ccc2[nH+]c(O)c(CN(CC3CCCO3)[S+](=O)([O-])c3cccc4nsnc34)cc2c1",
        ],
        // A protonated nitro group beside a carboxylate, as GNPS deposits
        // p-nitrobenzoyl amino acids, is the nitro acid.
        &[
            "CC(C)C[C@H](NC(=O)c1ccc(cc1)[N+](=O)[O-])C(=O)O",
            "O[N+](=O)c1ccc(cc1)C(=O)N[C@@H](CC(C)C)C(=O)[O-]",
        ],
        // A λ5-phosphinine oxide, which only the default tautomer rules write.
        &["C1=CC=P(=O)C=C1", "[O-][p+]1ccccc1", "C1=CC=CC=[P+]1[O-]"],
        // An N-hydroxypyridinium is the protonated N-oxide.
        &["[O-][n+]1ccccc1", "O[n+]1ccccc1", "O[n+]1ccccc1.[Cl-]"],
        // Which of two terminal anions on a cationic centre takes the double
        // bond is not up to the order they were written in.
        &[
            "C[P+](C)([O-])[S-]",
            "C[P+](C)([S-])[O-]",
            "CP(C)(=S)O",
            "CP(C)(=O)S",
        ],
        &[
            "C[P+](C)([18O-])[O-]",
            "C[P+](C)([O-])[18O-]",
            "CP(C)(=[18O])O",
            "CP(C)(=O)[18OH]",
        ],
        // PubChem's three-coordinate phosphinic cation, its oxide pair and
        // the neutral form the tautomer writes for that pair.
        &["C[P+](=O)O", "C[P+](=O)[O-]", "CP(=O)=O"],
    ] {
        let first = parent(writings[0]);
        for writing in &writings[1..] {
            assert_eq!(parent(writing), first, "{writing} and {}", writings[0]);
        }
    }
}

/// Every pair here is two compounds, and keeps two parents.
#[test]
fn different_compounds_keep_different_parents() {
    for (first, second) in [
        // Stereoisomers.
        ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"),
        ("C/C=C/C(=O)O", "C/C=C\\C(=O)O"),
        ("C/C=[P+](/C)[O-]", "C/C=[P+](\\C)[O-]"),
        // Isotopes.
        ("[13CH3]C(=O)O", "CC(=O)O"),
        // A quaternary ammonium is not the tertiary amine it was made from.
        ("C[N+](C)(C)CC(=O)[O-]", "CN(C)CC(=O)O"),
        // Choline is not its dehydration product or its acid.
        ("OCC[N+](C)(C)C", "C=C[N+](C)(C)C"),
        // A carbocation has no proton to give: tropylium is not
        // cycloheptatriene, and the isopropyl cation is not propene.
        ("c1cc[cH+]ccc1", "C1=CC=CCC=C1"),
        ("C[CH+]C", "CC=C"),
        // A phenol is not acid enough to balance a quaternary ammonium, so
        // the cation keeps its proton; and a phenolate betaine, already
        // neutral, does not take one.
        ("C[N+](C)(C)CCc1ccc(O)cc1", "C[N+](C)(C)CCc1ccc([O-])cc1"),
        // A flavylium is a cation, not its quinoidal base.
        (
            "Oc1cc(O)c2cc(O)c(-c3ccc(O)c(O)c3)[o+]c2c1",
            "Oc1cc(O)c2cc(O)c(=C3C=CC(=O)C(O)=C3)oc2c1",
        ),
        // Two compounds, neither a known counterion, are not reduced to one.
        ("CCCO.c1ccccc1N", "c1ccccc1N"),
    ] {
        assert_ne!(parent(first), parent(second), "{first} and {second}");
    }
}

/// The charge rules: as close to neutral as the fixed charges allow.
#[test]
fn net_charge_is_as_low_as_the_fixed_charges_allow() {
    for (smiles, expected) in [
        ("CC(=O)[O-]", 0),
        ("CC[NH3+]", 0),
        ("COP(=O)([O-])[O-]", 0),
        ("[O-]S(=O)(=O)c1ccccc1", 0),
        ("c1cc[nH+]cc1", 0),
        // Nitrate: one oxygen pairs with the nitrogen, the other is an anion.
        ("O=[N+]([O-])[O-].CCN", 0),
        // A protonated tertiary amine has a proton to give back.
        ("CC[NH+](C)C", 0),
        // A quaternary ammonium has none and nothing to pair with.
        ("OCC[N+](C)(C)C", 1),
        ("C[n+]1ccccc1", 1),
        // A phenol is not acid enough to be the balancing anion, and an NH
        // of a dye is no acid either.
        ("C[N+](C)(C)Cc1ccc(O)cc1", 1),
        ("Oc1cc(O)c2cc(O)c(-c3ccc(O)c(O)c3)[o+]c2c1", 1),
        ("Cc1cc2nc3cc(C)c(N)cc3[n+](-c3ccccc3)c2cc1N", 1),
        (
            "Cc1cc2nc3c(C)c(N)ccc3[n+](-c3ccccc3)c2cc1NC(=O)c1ccc([N+](=O)[O-])s1",
            1,
        ),
        // A neutral zwitterion stays neutral: a phenolate betaine, resazurin
        // and a mesoionic sydnone imine.
        ("C[N+](C)(C)CCc1ccc([O-])cc1", 0),
        ("C1=CC2=C(C=C1O)OC3=CC(=O)C=CC3=[N+]2[O-]", 0),
        ("CC1CCCC(C)N1[n+]1cc(=N)o[n-]1", 0),
        // Two fixed cations take two acids' protons.
        ("C[N+](C)(C)CCC(=O)O.C[N+](C)(C)CCC(=O)O", 0),
        ("OC(=O)CC[N+](C)(C)CC[N+](C)(C)CCC(=O)O", 0),
        // A carbocation keeps its charge and its protons.
        ("c1cc[cH+]ccc1", 1),
        ("C[CH+]C", 1),
        // A fixed anion with a base takes its proton; without one it stays.
        ("NCC[B-](F)(F)F", 0),
        ("CNC(=N)CC[B-](F)(F)F", 0),
        ("CC[B-](F)(F)F", -1),
        ("c1ccc(N)cc1C[B-](F)(F)F", -1),
    ] {
        let out = parent(smiles);
        assert_eq!(net_charge(&out), expected, "{smiles} -> {out}");
    }
}

/// Charge pairs that are how a group is written are not ions to neutralize,
/// and their oxygen is not a base: the proton of an acid stays on the acid.
#[test]
fn charge_separated_groups_are_left_alone() {
    for smiles in [
        "OC(=O)c1ccc[n+]([O-])c1",
        "OC(=O)CCN=[N+]=[N-]",
        "O=[N+]([O-])c1ccc(O)cc1",
        "C[N+](C)(C)[O-]",
        "OC(=O)C/C=[P+](/C)[O-]",
    ] {
        let out = parent(smiles);
        assert_eq!(parent(&out), out, "{smiles}");
        assert_eq!(net_charge(&out), 0, "{smiles} -> {out}");
        assert!(
            !out.contains("C(=O)[O-]") && !out.contains("[O-]C(=O)"),
            "{smiles} -> {out}"
        );
    }
}

/// Of two acids that could balance one cation, the stronger holds the charge:
/// carnitine is the carboxylate, not the alkoxide, and a sulfonate is chosen
/// over a carboxylate and a phosphate over a carboxylate.
#[test]
fn the_strongest_acid_balances_the_cation() {
    for (expected, writings) in [
        (
            "C[N+](C)(C)CC(O)CC(=O)[O-]",
            &[
                "C[N+](C)(C)CC([O-])CC(=O)O",
                "C[N+](C)(C)CC(O)CC(=O)O",
                "C[N+](C)(C)CC(O)CC(=O)[O-]",
            ][..],
        ),
        (
            "C[N+](C)(C)CC(C(=O)O)S(=O)(=O)[O-]",
            &[
                "C[N+](C)(C)CC(C(=O)[O-])S(=O)(=O)[O-]",
                "C[N+](C)(C)CC(C(=O)[O-])S(=O)(=O)O",
                "C[N+](C)(C)CC(C(=O)O)S(=O)(=O)O",
            ],
        ),
        (
            "C[N+](C)(C)CC(C(=O)O)OP(=O)(O)[O-]",
            &[
                "C[N+](C)(C)CC(C(=O)[O-])OP(=O)(O)O",
                "C[N+](C)(C)CC(C(=O)O)OP(=O)(O)O",
            ],
        ),
        // A carboxylic acid before a phenol, whichever the record charged.
        (
            "C[N+](C)(C)C(Cc1ccc(O)cc1)C(=O)[O-]",
            &[
                "C[N+](C)(C)C(Cc1ccc([O-])cc1)C(=O)O",
                "C[N+](C)(C)C(Cc1ccc(O)cc1)C(=O)O",
            ],
        ),
    ] {
        for writing in writings {
            assert_eq!(parent(writing), canonical(expected), "{writing}");
        }
    }
}

/// The records the review found wrong in the first version of this step, each
/// now a fixed point that reads back.
#[test]
fn review_regressions_read_back_and_stay_put() {
    for smiles in [
        // A phosphorus ylide keeps its oxide separated and its E/Z.
        "C/C=[P+](/C)[O-]",
        // A phosphinine oxide stays readable.
        "C1=CC=CC=[P+]1[O-]",
        "C1=CC=C2C=C3C(=CC2=C1)C=CC=[P+]3[O-]",
        // Carbocations keep their protons.
        "c1cc[cH+]ccc1",
        "[B-](F)(F)(F)F.C1=CC=C2C=CC=CC(=C1)[CH+]2",
        // PubChem's cationic phosphinic and sulfonic acids.
        "CC(C[P+](=O)O)C(=O)O",
        "C1[C@H]2[C@@H](C[S+]1(=O)O)N(C(=S)N2)C3=CC=CC=C3",
        "CCCC[P+](CCC)(CCCC)O",
        // An open-chain oxonium whose tautomer writes its stereo marks the
        // other way round from how they read back (PubChem CID 156642185).
        "CC1=CC(=CC(=C1)OC)/C=C/C=[O+]/C=C\\C=C(\\C(=O)C)/N",
        // Thiophosphates whose O and S anions were written in either order.
        "CCCCCCCCCCCCC1=C(C=C(C=C1)O[P+]([O-])([S-])[S-])CCCCCCCCCCCC.[Zn+2]",
    ] {
        let once = parent(smiles);
        assert_eq!(parent(&once), once, "{smiles}");
        assert_eq!(canonical(&once), once, "{smiles} -> {once}");
    }
    let ylide = parent("C/C=[P+](/C)[O-]");
    assert!(ylide.contains('/') && ylide.contains("[O-]"), "{ylide}");
}

/// What decides is the molecule, not how it was written down: every record
/// here gives one parent under shuffled atoms and shuffled components.
#[test]
fn the_parent_does_not_depend_on_writing_order() {
    for (first, second) in [
        // Two equal carboxylates, either one of which balances the cation.
        (
            "C[N+](C)(C)C(CC(=O)O)CC(=O)[O-]",
            "C[N+](C)(C)C(CC(=O)[O-])CC(=O)O",
        ),
        (
            "OC(=O)CC(CC([O-])=O)[N+](C)(C)C",
            "C[N+](C)(C)C(CC(=O)O)CC(=O)O",
        ),
        // Components in either order, and repeated.
        ("CCN.Cl", "Cl.CCN"),
        ("CCN.CCN.OS(=O)(=O)O", "CCN"),
    ] {
        assert_eq!(parent(first), parent(second), "{first} and {second}");
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261110);
    for smiles in [
        "OC(=O)C[N+](C)(C)CC(O)=O",
        "OC(=O)CC[N+](C)(C)CC[N+](C)(C)CCC(=O)O.[Cl-].[Cl-]",
        "C[N+](C)(C)CCOP(=O)(O)OP(=O)(O)OCC1OC(n2ccc(N)nc2=O)C(O)C1O",
        "Cc1c[n+](CCCCC(N)C(=O)O)cn1CCCCC(N)C(=O)O",
        "C[P+](C)([O-])[S-]",
        "C[P+](C)([18O-])[O-]",
        "C/C=[P+](/C)[O-]",
        "C[S@+](CC)[O-]",
        "CC(=O)O[Na].OC(=O)C(F)(F)F.CCN",
        "[O-]c1cccc[n+]1C",
        "NCC[B-](F)(F)F.[K+]",
        "C[N+](C)(C)C.[O-]C(=O)c1ccccc1",
        "CCCCCCCCCCCCCCCCCC(=O)O[Mg]OC(=O)CCCCCCCCCCCCCCCCC",
    ] {
        let expected = parent(smiles);
        let components = MolecularGraph::from_smiles(smiles).unwrap();
        for _ in 0..8 {
            let mut shuffled = components.clone();
            shuffled.shuffle(&mut rng);
            for component in &mut shuffled {
                let mut order: Vec<usize> = (0..component.len()).collect();
                order.shuffle(&mut rng);
                reorder_molecule(component, &order);
            }
            let parents = parent_components(shuffled, &ParentOptions::default());
            let written: Vec<String> = parents.iter().map(|part| part.to_smiles()).collect();
            assert_eq!(written.join("."), expected, "{smiles}");
        }
    }
}

/// Applying it twice changes nothing.
#[test]
fn the_parent_is_a_fixed_point() {
    for smiles in [
        "CC(=O)O[Na]",
        "C[N+](C)(C)CC(=O)O",
        "C[S+](C)[O-]",
        "CN=N#N",
        "OCC[N+](C)(C)C.[Cl-]",
        "C[C@H]([NH3+])C(=O)[O-]",
        "CC(=O)[N+](C)([Na])CCO",
        // A fixed cation and two equally acidic sites: which one balances it
        // must not move when the answer is read back in. From GNPS records.
        "Cc1c[n+](CCCCC(N)C(=O)O)cn1CCCCC(N)C(=O)O",
        "Cc1ncc(C[n+]2csc(CCOP(=O)(O)OP(=O)(O)O)c2C)c(N)n1",
        "C[N+](C)(C)CCOP(=O)(O)OP(=O)(O)OCC1OC(n2ccc(N)nc2=O)C(O)C1O",
        "C[N+](C)(C)CC(O)CC(=O)O",
        "NCC[B-](F)(F)F",
        "C1=CC2=C(C=C1O)OC3=CC(=O)C=CC3=[N+]2[O-]",
        "CC1CCCC(C)N1[n+]1cc(=N)o[n-]1",
        "Cc1cc(C(=O)NCCCCNCCCNC(=O)[c-]2cccc2)nn1C",
        "Oc1cc(O)c2cc(O)c(-c3ccc(O)c(O)c3)[o+]c2c1",
    ] {
        let once = parent(smiles);
        assert_eq!(parent(&once), once, "{smiles}");
    }
}

/// Counterions are dropped only beside something else, repeated components
/// are written once, and two compounds neither of which is a counterion are
/// both kept rather than one being guessed at.
#[test]
fn salts_are_stripped_without_guessing() {
    for (smiles, expected) in [
        // A counterion larger than the drug is still the counterion.
        (
            "OCCOCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1.OC(=O)c1cc2ccccc2c(Cc2c(O)c(cc3ccccc23)C(O)=O)c1O",
            "OCCOCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1",
        ),
        (
            "CCCCCCCCCCCCCCCCCC(=O)O.CN1CCC[C@H]1c1cccnc1",
            "CN1CCC[C@H]1c1cccnc1",
        ),
        // A solvate is the compound.
        ("CC(=O)Nc1ccc(O)cc1.CCO.O", "CC(=O)Nc1ccc(O)cc1"),
        // Nothing but counterions: the ones with carbon.
        ("CC(=O)O", "CC(=O)O"),
        ("CC(=O)[O-].[Na+]", "CC(=O)O"),
        (
            "OC(=O)CC(O)(CC(=O)[O-])C(=O)[O-].[Na+].[Na+]",
            "OC(=O)CC(O)(CC(=O)O)C(=O)O",
        ),
        (
            "CCCCCCCCCCCCCCCCCC(=O)O[Mg]OC(=O)CCCCCCCCCCCCCCCCC",
            "CCCCCCCCCCCCCCCCCC(=O)O",
        ),
        (
            "C[N+](C)(C)C.[O-]C(=O)c1ccccc1",
            "C[N+](C)(C)C.OC(=O)c1ccccc1",
        ),
        // Two compounds: both stay.
        ("CCCO.c1ccccc1N", "CCCO.c1ccccc1N"),
        (
            "OCC(O)CO.CCCCCCCCCCCCCCCC(=O)O",
            "OCC(O)CO.CCCCCCCCCCCCCCCC(=O)O",
        ),
        // A racemate drawn as both enantiomers is both.
        (
            "C[C@H](N)C(=O)O.C[C@@H](N)C(=O)O",
            "C[C@H](N)C(=O)O.C[C@@H](N)C(=O)O",
        ),
        // Repeats are written once.
        ("CCN.CCN", "CCN"),
        // No carbon at all: the record as it was written.
        ("[Na+].[Cl-]", "[Na+].[Cl-]"),
        ("N.N.Cl[Pt]Cl", "N.N.Cl[Pt]Cl"),
    ] {
        assert_eq!(parent(smiles), canonical(expected), "{smiles}");
    }
}

/// The counterion table: every entry reads, no two are the same compound, an
/// entry on its own is kept, and beside a drug it is dropped.
#[test]
fn the_counterion_table_is_consistent() {
    let mut seen = std::collections::HashSet::new();
    let drug = "CN1CCC[C@H]1c1cccnc1";
    for &entry in COUNTERIONS {
        let alone = parent(entry);
        assert!(!alone.is_empty(), "{entry}");
        let mut plain = canonical(&alone);
        plain.retain(|c| !matches!(c, '@' | '/' | '\\'));
        assert!(seen.insert(plain), "{entry} is listed twice");
        assert_eq!(parent(&format!("{entry}.{drug}")), parent(drug), "{entry}");
    }
}

/// Transition metals are part of the compound: bonded, they stay bonded, and
/// written as a separate ion (heme as RefMet writes it) the complex is kept
/// whole rather than losing its metal as if it were a counterion.
#[test]
fn transition_metals_stay() {
    let out = parent("C1=CC=[N+]2C1=C[Fe-2]2");
    assert!(out.contains("Fe"), "{out}");
    let heme = "C=Cc1c(C)c2/C=C3/C(=C(CCC(=O)O)C(=N3)/C=c3/c(CCC(=O)O)c(C)/c(=C/C4=N/C(=C\\c1[n-]2)/C(=C4C=C)C)[n-]3)C.[Fe+2]";
    let out = parent(heme);
    assert!(out.contains("Fe") && out.contains("[n-]"), "{out}");
    assert_eq!(parent(&out), out);
    assert_eq!(
        parent("[Zn+2].OCC(O)C([O-])=O.OCC(O)C([O-])=O")
            .matches('.')
            .count(),
        2
    );
}

/// Each step can be switched off on its own.
#[test]
fn steps_can_be_switched_off() {
    let only = |options: ParentOptions, smiles: &str| parent_smiles(smiles, &options).unwrap();
    // Off, the input comes back as its components' canonical tautomers.
    assert_eq!(
        only(options(false, false, false), "CC[NH3+].[Cl-]"),
        canonical("CC[NH3+].[Cl-]")
    );
    assert!(only(options(true, false, true), "CC[NH3+].[Cl-]").contains('+'));
    assert!(only(options(true, true, false), "CCN.Cl").contains('.'));
    assert_eq!(
        only(options(true, true, false), "CCN.CCN")
            .matches('.')
            .count(),
        1
    );
    assert!(only(options(false, true, true), "CC(=O)O[Na]").contains("Na"));
}

/// The use it was written for: the parent, then the canonical tautomer, gives
/// one structure for a compound however the record wrote it.
#[test]
fn parent_then_tautomer_unifies_library_writings() {
    let harmonize = |smiles: &str| {
        normalized_smiles(&parent(smiles), &TautomerRules::new()).expect("the parent parses")
    };
    for (first, second) in [
        ("O=c1cccc[nH]1.Cl", "[O-]c1ccccn1.[Na+]"),
        ("CC(=O)N.O", "CC(O)=N"),
        ("C[N+](C)(C)CC(=O)O", "C[N+](C)(C)CC([O-])=O"),
    ] {
        assert_eq!(harmonize(first), harmonize(second), "{first} and {second}");
    }
}

/// The records the full PubChem sweep found: each is its own parent's parent,
/// and the parent of its canonical tautomer too.
#[test]
fn full_sweep_regressions_are_fixed_points() {
    for smiles in [
        // A diazo carbanion that already has its double bond is not
        // rewritten into a cumulene (CID 156008921).
        "CC(C)C1=C(C(=CC=C1)C(C)C)N2C=C[N+](=C2[C-]=[N+]=[N-])C3=C(C=CC=C3C(C)C)C(C)C.Cl[Au]",
        // A sodium selenide breaks into one anion whichever tautomer it was
        // written as (CIDs 58773492, 161671774).
        "C1=CC=C(C=C1)[Se](=[Se][Na])C2=CC=CC=C2",
        "C1=CC=C(C=C1)[Se](=[SeH][Na])C2=CC=CC=C2",
        // Amidine chains whose geometry cannot be written keep none, and the
        // tautomer is asked again without it (CIDs 154668272, 156487585).
        "CC/C=C/N=C/C=C(\\C)/C1=CC(=CC=C1)C(=C)N=C(NC(=O)CN)SC",
        "CCCC/C(=C\\N=C(C)NC(=N/C=C\\C)/C=C/CC)/CCC1=C(C=CC(=C1)N)C",
    ] {
        let once = parent(smiles);
        assert_eq!(parent(&once), once, "{smiles}");
        let tautomer = normalized_smiles(smiles, &TautomerRules::new()).unwrap();
        assert_eq!(parent(&tautomer), once, "{smiles}");
    }
}
