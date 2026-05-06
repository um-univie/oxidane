use oxidane::io::{MoleculeMatchContext, parse_cxsmarts};
use oxidane::prelude::*;

fn check_count(query: &str, target: &str, expected: usize) {
    let pattern = parse_cxsmarts(query).unwrap();
    let molecule = MolecularGraph::from_smiles_single(target).unwrap();
    assert_eq!(
        pattern.find_matches(&molecule).len(),
        expected,
        "{query} in {target}"
    );
    assert_eq!(
        pattern.matches(&molecule),
        expected != 0,
        "{query} in {target}"
    );
    let context = MoleculeMatchContext::new(&molecule);
    assert_eq!(pattern.matches_with(&molecule, &context), expected != 0);
}

#[test]
fn hydrogen_atoms_are_distinct_from_attached_hydrogen_counts() {
    for (query, target, expected) in [
        ("[H]", "C", 4),
        ("[H]", "O", 2),
        ("[H]", "CO", 4),
        ("[H:7]", "C", 4),
        ("[H1]", "CO", 1),
        ("[O;H]", "CO", 1),
        ("[CH3]", "CO", 1),
        ("[H,C]", "CO", 2),
        ("[H;D1]", "CO", 0),
        ("[H;D2]", "CO", 1),
        ("[!H]", "CO", 5),
        ("[H0]", "CO", 4),
        ("[$([H])]", "CO", 4),
        ("[H]O[H]", "O", 2),
    ] {
        check_count(query, target, expected);
    }
    assert_eq!(parse_cxsmarts("[H:7]").unwrap().atom_classes[&0], 7);
}

#[test]
fn hydrogen_atoms_keep_isotope_charge_and_mapping_constraints() {
    for (query, target, expected) in [
        ("[H]", "[2H]O[2H]", 2),
        ("[2H]", "[2H]O[2H]", 2),
        ("[2H:7]", "[2H]O[2H]", 2),
        ("[3H]", "[2H]O[2H]", 0),
        ("[2H]", "O", 0),
        ("[H+]", "[H+]", 1),
        ("[H+]", "O", 0),
        ("[H-]", "[H-]", 1),
        ("[H-]", "[H+]", 0),
        ("[2H+:7]", "[2H+]", 1),
        ("[2H+]", "[3H+]", 0),
        ("[H+0]", "O", 2),
        ("[H++]", "[H+2]", 1),
        ("[H--]", "[H-2]", 1),
    ] {
        check_count(query, target, expected);
    }
}

#[test]
fn explicit_hydrogen_stereo_queries_match_the_correct_configuration() {
    check_count("F[C@]([H])(Cl)Br", "F[C@H](Cl)Br", 1);
    check_count("F[C@]([H])(Cl)Br", "F[C@@H](Cl)Br", 0);
}
