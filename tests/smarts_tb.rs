use oxidane::io::{MoleculeMatchContext, parse_cxsmarts};
use oxidane::prelude::*;

#[test]
fn trigonal_bipyramidal_queries_distinguish_all_twenty_configurations() {
    for query_class in 1..=20 {
        let query = format!("S[As@TB{query_class}](F)(Cl)(Br)N");
        let pattern = parse_cxsmarts(&query).unwrap();
        for target_class in 1..=20 {
            let target = format!("S[As@TB{target_class}](F)(Cl)(Br)N");
            let molecule = MolecularGraph::from_smiles_single(&target).unwrap();
            let expected = query_class == target_class;
            assert_eq!(
                pattern.matches(&molecule),
                expected,
                "{query} against {target}"
            );
            assert_eq!(!pattern.find_matches(&molecule).is_empty(), expected);
            let context = MoleculeMatchContext::new(&molecule);
            assert_eq!(pattern.matches_with(&molecule, &context), expected);
        }
    }
}

#[test]
fn trigonal_bipyramidal_queries_follow_ligand_order_and_require_stereo() {
    // Swapping two equatorial ligands exchanges TB1 and TB2.
    let pattern = parse_cxsmarts("S[As@TB1](F)(Cl)(Br)N").unwrap();
    for (target, expected) in [
        ("S[As@TB2](Cl)(F)(Br)N", true),
        ("S[As@TB1](Cl)(F)(Br)N", false),
        ("S[As](F)(Cl)(Br)N", false),
    ] {
        let molecule = MolecularGraph::from_smiles_single(target).unwrap();
        assert_eq!(pattern.matches(&molecule), expected, "{target}");
    }
}

#[test]
fn trigonal_bipyramidal_class_numbers_do_not_wrap_or_escape_their_range() {
    for query in [
        "S[As@TB0](F)(Cl)(Br)N",
        "S[As@TB21](F)(Cl)(Br)N",
        "S[As@TB257](F)(Cl)(Br)N",
        "S[As@TB999999999999999999999999999999999999](F)(Cl)(Br)N",
    ] {
        assert!(parse_cxsmarts(query).is_err(), "{query}");
    }
}
