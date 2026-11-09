//! Parent compounds.
//!
//! Each case here is a way a spectral library or registry record really writes
//! a compound -- a salt, a sodium drawn as a bond, a zwitterion, the other
//! spelling of a sulfoxide -- and the property is that every writing of one
//! compound reduces to the same parent, while compounds that differ do not.

use crate::molecule::parent::{ParentOptions, parent_smiles};
use crate::molecule::tautomer::{TautomerRules, normalized_smiles};
use crate::prelude::*;

fn parent(smiles: &str) -> String {
    parent_smiles(smiles, &ParentOptions::default()).expect("the SMILES parses")
}

fn net_charge(smiles: &str) -> i32 {
    MolecularGraph::from_smiles(smiles)
        .expect("the SMILES parses")
        .iter()
        .map(|component| component.formal_charge())
        .sum()
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
        &["C[N+](C)(C)CC(O)CC(=O)[O-]", "OC(C[N+](C)(C)C)CC(O)=O"],
        &[
            "C[N+](C)(C)CCOP(=O)([O-])OCC(O)CO",
            "C[N+](C)(C)CCOP(=O)(O)OCC(O)CO",
        ],
        // A cation with nothing to balance it stays a cation, salt or not.
        &[
            "OCC[N+](C)(C)C",
            "OCC[N+](C)(C)C.[Cl-]",
            "[Cl-].OCC[N+](C)(C)C",
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
        // Isotopes.
        ("[13CH3]C(=O)O", "CC(=O)O"),
        // A quaternary ammonium is not the tertiary amine it was made from.
        ("C[N+](C)(C)CC(=O)[O-]", "CN(C)CC(=O)O"),
        // Choline is not its dehydration product or its acid.
        ("OCC[N+](C)(C)C", "C=C[N+](C)(C)C"),
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
        ("O=[N+]([O-])[O-]", 0),
        // A protonated tertiary amine has a proton to give back.
        ("CC[NH+](C)C", 0),
        // A quaternary ammonium has none and nothing to pair with.
        ("OCC[N+](C)(C)C", 1),
        ("C[n+]1ccccc1", 1),
        // A phenol is not acid enough to be the balancing anion.
        ("C[N+](C)(C)Cc1ccc(O)cc1", 1),
        // Two fixed cations take two acids' protons.
        ("C[N+](C)(C)CCC(=O)O.C[N+](C)(C)CCC(=O)O", 0),
        ("OC(=O)CC[N+](C)(C)CC[N+](C)(C)CCC(=O)O", 0),
    ] {
        assert_eq!(net_charge(&parent(smiles)), expected, "{smiles}");
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
    ] {
        assert_eq!(parent(smiles), parent(&parent(smiles)), "{smiles}");
        let out = parent(smiles);
        assert_eq!(net_charge(&out), 0, "{smiles} -> {out}");
        assert!(
            !out.contains("C(=O)[O-]") && !out.contains("[O-]C(=O)"),
            "{smiles} -> {out}"
        );
    }
}

/// Of two anions that could balance one cation, the more acidic keeps the
/// charge: carnitine is the carboxylate, not the alkoxide, and a sulfonate
/// is chosen over a carboxylate.
#[test]
fn the_most_acidic_anion_balances_the_cation() {
    let as_written = |smiles: &str| {
        parent_smiles(
            smiles,
            &ParentOptions {
                neutralize: false,
                ..ParentOptions::default()
            },
        )
        .unwrap()
    };
    // Carnitine, from the alkoxide, the cation and the zwitterion.
    let carboxylate = as_written("C[N+](C)(C)CC(O)CC(=O)[O-]");
    for writing in [
        "C[N+](C)(C)CC([O-])CC(=O)O",
        "C[N+](C)(C)CC(O)CC(=O)O",
        "C[N+](C)(C)CC(O)CC(=O)[O-]",
    ] {
        assert_eq!(parent(writing), carboxylate, "{writing}");
    }
    // A sulfonate over a carboxylate, from either.
    let sulfonate = as_written("C[N+](C)(C)CC(C(=O)O)S(=O)(=O)[O-]");
    for writing in [
        "C[N+](C)(C)CC(C(=O)[O-])S(=O)(=O)[O-]",
        "C[N+](C)(C)CC(C(=O)[O-])S(=O)(=O)O",
        "C[N+](C)(C)CC(C(=O)O)S(=O)(=O)O",
    ] {
        assert_eq!(parent(writing), sulfonate, "{writing}");
    }
}

/// What decides is the molecule, not how it was written down.
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
        // Components in either order, and duplicated.
        ("CCN.Cl", "Cl.CCN"),
        ("CCN.CCN.OS(=O)(=O)O", "CCN"),
    ] {
        assert_eq!(parent(first), parent(second), "{first} and {second}");
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
    ] {
        let once = parent(smiles);
        assert_eq!(parent(&once), once, "{smiles}");
    }
}

/// A record that is nothing but counterions keeps its largest component
/// rather than becoming empty, so the counterions themselves stay findable.
#[test]
fn a_counterion_on_its_own_is_kept() {
    for (smiles, expected) in [
        ("CC(=O)O", "CC(=O)O"),
        ("OC(=O)C(F)(F)F", "OC(=O)C(F)(F)F"),
        ("OCC(O)CO", "OCC(O)CO"),
    ] {
        assert_eq!(
            parent(smiles),
            parent_smiles(
                expected,
                &ParentOptions {
                    disconnect_metals: false,
                    normalize_groups: false,
                    neutralize: false,
                    strip_salts: false,
                }
            )
            .unwrap(),
            "{smiles}"
        );
    }
    assert!(!parent("[Na+].[Cl-]").is_empty());
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
}

/// Each step can be switched off on its own.
#[test]
fn steps_can_be_switched_off() {
    let only = |options: ParentOptions, smiles: &str| parent_smiles(smiles, &options).unwrap();
    let none = ParentOptions {
        disconnect_metals: false,
        normalize_groups: false,
        neutralize: false,
        strip_salts: false,
    };
    // Off, the input comes back as its canonical SMILES.
    assert_eq!(only(none, "CC[NH3+].[Cl-]").matches('.').count(), 1);
    assert!(
        only(
            ParentOptions {
                neutralize: false,
                ..Default::default()
            },
            "CC[NH3+].[Cl-]"
        )
        .contains('+')
    );
    assert!(
        only(
            ParentOptions {
                strip_salts: false,
                ..Default::default()
            },
            "CCN.Cl"
        )
        .contains('.')
    );
    assert!(
        only(
            ParentOptions {
                normalize_groups: false,
                ..Default::default()
            },
            "C[S+](C)[O-]"
        )
        .contains('+')
    );
    assert!(
        only(
            ParentOptions {
                disconnect_metals: false,
                ..Default::default()
            },
            "CC(=O)O[Na]"
        )
        .contains("Na")
    );
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
