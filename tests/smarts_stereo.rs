use oxidane::io::{MoleculeMatchContext, parse_cxsmarts};
use oxidane::prelude::*;

fn check(query: &str, target: &str, expected: bool) {
    let pattern = parse_cxsmarts(query).unwrap_or_else(|error| panic!("{query}: {error:?}"));
    let molecule = MolecularGraph::from_smiles_single(target).unwrap();
    assert_eq!(
        pattern.matches(&molecule),
        expected,
        "{query} against {target}"
    );
    assert_eq!(
        !pattern.find_matches(&molecule).is_empty(),
        expected,
        "all: {query} against {target}"
    );
    let context = MoleculeMatchContext::new(&molecule);
    assert_eq!(
        pattern.matches_with(&molecule, &context),
        expected,
        "cached: {query} against {target}"
    );
}

#[test]
fn tetrahedral_queries_distinguish_configuration_from_spelling() {
    for (query, target, expected) in [
        ("N[C@@H](C)C(=O)O", "N[C@@H](C)C(=O)O", true),
        ("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O", false),
        ("N[C@@H](C)C(=O)O", "NC(C)C(=O)O", false),
        ("NC(C)C(=O)O", "N[C@H](C)C(=O)O", true),
        ("N[C@@H](C)C(=O)O", "C[C@H](N)C(=O)O", true),
        ("[C@H](F)(Cl)Br", "F[C@@H](Cl)Br", true),
        ("[C@H](F)(Cl)Br", "F[C@H](Cl)Br", false),
        ("F[C@]([#1])(Cl)Br", "F[C@H](Cl)Br", true),
        ("N[C;!@@H](C)C(=O)O", "N[C@@H](C)C(=O)O", false),
        ("N[C;!@@H](C)C(=O)O", "N[C@H](C)C(=O)O", true),
    ] {
        check(query, target, expected);
    }
}

#[test]
fn ring_closures_keep_their_position_in_the_carrier_order() {
    check("F[C@]1(Cl)CCCO1", "F[C@]1(Cl)CCCO1", true);
    check("F[C@]1(Cl)CCCO1", "F[C@@]1(Cl)CCCO1", false);
    check("F[C@]1(Cl)CCCO1", "Cl[C@@]1(F)CCCO1", true);
    check("C[P@]1CCCO1", "C[P@]1CCCO1", true);
    check("C[P@]1CCCO1", "C[P@@]1CCCO1", false);
}

#[test]
fn directional_queries_require_single_bonds_and_relative_geometry() {
    for (query, target, expected) in [
        ("C/C", "C=C", false),
        ("C/C", "CC", true),
        ("F/C=C/F", "F/C=C/F", true),
        ("F/C=C/F", "F\\C=C\\F", true),
        ("F/C=C/F", "F/C=C\\F", false),
        ("F/C=C/F", "FC=CF", false),
        ("FC=CF", "F/C=C\\F", true),
        ("F/C=C\\F", "F/C=C\\F", true),
        ("F/?C=C/?F", "FC=CF", true),
        ("F/?C=C/?F", "F/C=C\\F", false),
        ("F/C(Cl)=C(/Br)I", "Cl/C(F)=C(Br)/I", true),
        ("F/C(Cl)=C(/Br)I", "Cl/C(F)=C(/Br)I", false),
        ("F/C=C/C=C/F", "F/C=C/C=C/F", true),
        ("F/C=C/C=C/F", "F/C=C/C=C\\F", false),
        ("F/C=C1/CCNCCC1", "F/C=C1CCNCCC/1", true),
        ("F/C=C1/CCNCCC1", "F/C=C1CCNCCC\\1", false),
    ] {
        check(query, target, expected);
    }
}

#[test]
fn a_wrong_first_mapping_does_not_hide_a_later_stereo_match() {
    check("N[C@@H](C)C(=O)", "N[C@H](C)C(=O)N[C@@H](C)C(=O)O", true);
}

#[test]
fn recursive_queries_apply_stereo_to_the_inner_mapping() {
    check("[$(N[C@@H](C)C(=O)O)]", "N[C@@H](C)C(=O)O", true);
    check("[$(N[C@@H](C)C(=O)O)]", "N[C@H](C)C(=O)O", false);
}

#[test]
fn extended_classes_are_constraints_too() {
    for (query, opposite) in [
        ("FC=[C@]=C(Cl)Br", "FC=[C@@]=C(Cl)Br"),
        ("F[Pt@SP1](Cl)(Br)I", "F[Pt@SP2](Cl)(Br)I"),
        ("F[Co@OH1](Cl)(Br)(I)(N)O", "F[Co@OH2](Cl)(Br)(I)(N)O"),
    ] {
        check(query, query, true);
        check(query, opposite, false);
    }
}
