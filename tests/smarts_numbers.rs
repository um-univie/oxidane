use oxidane::io::{AtomExpression, parse_cxsmarts};
use oxidane::prelude::*;

#[test]
fn numeric_properties_reject_values_that_would_wrap() {
    for query in [
        "[#262]",
        "[D256]",
        "[H256]",
        "[h256]",
        "[R256]",
        "[r256]",
        "[v256]",
        "[X256]",
        "[^256]",
        "[65549C]",
        "[C+256]",
        "[C-256]",
        "[C+128]",
        "[C-129]",
        "[r{3-256}]",
        "[r{256-}]",
        "[r{8-3}]",
        "[#99999999999999999999999999999999999999999999]",
    ] {
        assert!(parse_cxsmarts(query).is_err(), "accepted {query}");
    }
}

#[test]
fn representable_numeric_boundaries_are_preserved() {
    for (query, expected) in [
        ("[#0]", AtomExpression::AtomicNumber(0)),
        ("[#255]", AtomExpression::AtomicNumber(255)),
        ("[D255]", AtomExpression::Degree(255)),
        ("[H255]", AtomExpression::TotalHydrogens(255)),
        ("[h255]", AtomExpression::ImplicitHydrogens(255)),
        ("[R255]", AtomExpression::RingMembership(255)),
        ("[r255]", AtomExpression::RingSize(255)),
        ("[v255]", AtomExpression::Valence(255)),
        ("[X255]", AtomExpression::Connectivity(255)),
        ("[65535]", AtomExpression::Isotope(65535)),
        ("[+127]", AtomExpression::Charge(127)),
        ("[-128]", AtomExpression::Charge(-128)),
        ("[+0]", AtomExpression::Charge(0)),
        ("[-0]", AtomExpression::Charge(0)),
        (
            "[r{3-255}]",
            AtomExpression::RingSizeRange(Some(3), Some(255)),
        ),
    ] {
        assert_eq!(
            parse_cxsmarts(query).unwrap().atoms,
            vec![expected],
            "{query}"
        );
    }
}

#[test]
fn signed_charge_boundaries_and_repeated_signs_match_exactly() {
    for (query, charge) in [
        ("[C+127]", 127),
        ("[C-128]", -128),
        ("[C++]", 2),
        ("[C---]", -3),
    ] {
        let pattern = parse_cxsmarts(query).unwrap();
        let mut molecule = MolecularGraph::from_smiles_single("C").unwrap();
        assert!(
            !pattern.matches(&molecule),
            "{query} matched neutral carbon"
        );
        molecule.charges_mut()[0] = charge;
        assert!(
            pattern.matches(&molecule),
            "{query} did not match charge {charge}"
        );
    }
    assert!(parse_cxsmarts(&format!("[C{}]", "+".repeat(128))).is_err());
    assert!(parse_cxsmarts(&format!("[C{}]", "-".repeat(129))).is_err());
}

#[test]
fn all_extended_stereo_classes_validate_their_number() {
    for (class, maximum) in [("TH", 2), ("AL", 2), ("SP", 3), ("TB", 20), ("OH", 30)] {
        for number in [0, maximum + 1, 257] {
            let query = format!("[C@{class}{number}]");
            assert!(parse_cxsmarts(&query).is_err(), "accepted {query}");
        }
        for number in 1..=maximum {
            assert!(parse_cxsmarts(&format!("[C@{class}{number}]")).is_ok());
        }
    }
}
