use oxidane::io::{MOLParseError, SDFParseError, parse_sdf_file};
use std::io::Cursor;

const PENTENE: &str = include_str!("pentene.sdf");

#[test]
fn malformed_atom_and_bond_lines_are_errors() {
    for broken in [
        PENTENE.replacen("   -0.0436", "BADCOORD!", 1),
        PENTENE.replacen("  1  2  1", "  1 99  1", 1),
        PENTENE.replacen("  1  2  1", "  1  2  X", 1),
        PENTENE.replacen(" 15 14", " 14 14", 1),
        PENTENE.replacen(" 15 14", " 15 13", 1),
        PENTENE.replacen(" 15 14", " 16 14", 1),
        PENTENE.replacen(" 15 14", "BAD 14", 1),
        PENTENE.replacen("V2000", "V3000", 1),
    ] {
        assert!(parse_sdf_file(Cursor::new(broken)).is_err());
    }
}

#[test]
fn every_truncation_inside_a_mol_block_is_an_error() {
    let lines: Vec<_> = PENTENE.lines().collect();
    let end = lines.iter().position(|line| *line == "M  END").unwrap();
    let complete = format!("{}\n$$$$\n", lines[..=end].join("\n"));
    for length in 1..=end {
        let partial = format!("{}\n", lines[..length].join("\n"));
        for input in [partial.clone(), format!("{complete}{partial}")] {
            assert!(
                matches!(
                    parse_sdf_file(Cursor::new(input)),
                    Err(SDFParseError::MOLParseError(
                        MOLParseError::TruncatedRecord(_)
                    ))
                ),
                "length {length}"
            );
        }
    }
    assert_eq!(parse_sdf_file(Cursor::new(complete)).unwrap().len(), 1);
    assert!(parse_sdf_file(Cursor::new("")).unwrap().is_empty());
}

#[test]
fn complete_fixture_keeps_its_declared_structure() {
    let entries = parse_sdf_file(Cursor::new(PENTENE)).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].mol_file.atoms.len(), 15);
    assert_eq!(entries[0].mol_file.bonds.len(), 14);
}
