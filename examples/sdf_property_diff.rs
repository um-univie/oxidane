//! Compares the charges, radicals and isotopes we read out of a MOL block
//! against the ones RDKit reads out of the same block.
//!
//! Generate the cases first; the oracle has three modes, described in its own
//! docstring:
//!
//! ```text
//! for mode in modern oldstyle sweep; do
//!   python3 notes/sdf_property_oracle.py $mode notes/cases_$mode.json
//!   cargo run --release --example sdf_property_diff -- notes/cases_$mode.json
//! done
//! ```
//!
//! `modern` uses RDKit's own output, so it checks that we read what a real
//! writer produces rather than only what the specification allows; `oldstyle`
//! moves the same information into the atom block columns; `sweep` is
//! exhaustive over elements and codes.

use oxidane::io::{SdfParseOptions, parse_sdf_file};
use oxidane::molecule::base::MoleculeCore;
use oxidane::prelude::Molecule3D;
use std::io::Cursor;

/// Reads one JSON string field, undoing the escaping the oracle applies.
fn string_field(entry: &str, name: &str) -> String {
    let key = format!("\"{name}\": \"");
    let start = entry.find(&key).expect("missing string field") + key.len();
    let mut out = String::new();
    let mut chars = entry[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => break,
            },
            other => out.push(other),
        }
    }
    out
}

fn int_list(entry: &str, name: &str) -> Vec<i64> {
    let key = format!("\"{name}\": [");
    let start = entry.find(&key).expect("missing list field") + key.len();
    let rest = &entry[start..];
    let end = rest.find(']').expect("unterminated list");
    rest[..end]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().expect("not an integer"))
        .collect()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: sdf_property_diff <cases.json>");
    let raw = std::fs::read_to_string(&path).expect("cases file not found");

    // The oracle writes one flat array of objects and no nested objects, so
    // splitting on the object boundaries is enough and saves a json dependency
    // in an example.
    let entries: Vec<&str> = raw
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split("}, {")
        .collect();

    let (mut total, mut agree) = (0, 0);
    for entry in entries {
        let smiles = string_field(entry, "smiles");
        let molblock = string_field(entry, "molblock");
        let expected_charges = int_list(entry, "charges");
        let expected_radicals = int_list(entry, "radicals");
        let expected_isotopes = int_list(entry, "isotopes");

        let sdf = format!("{molblock}\n$$$$\n");
        // Perception off: what is being compared is what the property block
        // says, atom for atom, and hydrogens worked out from the valence model
        // would append entries RDKit's heavy-atom vectors do not have.
        let options = SdfParseOptions {
            hydrogens: false,
            stereochemistry: false,
            aromaticity: None,
        };
        let parsed = parse_sdf_file(Cursor::new(sdf.as_str()))
            .ok()
            .and_then(|entries| Molecule3D::from_sdf_entries_with(entries, options).ok())
            .and_then(|mut molecules| {
                (!molecules.is_empty()).then(|| molecules.remove(0))
            });

        total += 1;
        let Some(molecule) = parsed else {
            println!("{smiles}: parse failed");
            continue;
        };

        let charges: Vec<i64> = molecule.charges().iter().map(|&c| c as i64).collect();
        let radicals: Vec<i64> = molecule
            .radical_states()
            .iter()
            .map(|&r| r as i64)
            .collect();
        let isotopes: Vec<i64> = (0..molecule.atomic_numbers().len())
            .map(|index| molecule.get_isotope(index).unwrap_or(0) as i64)
            .collect();

        let mut differences = Vec::new();
        if charges != expected_charges {
            differences.push(format!("charges {charges:?} vs {expected_charges:?}"));
        }
        if radicals != expected_radicals {
            differences.push(format!("radicals {radicals:?} vs {expected_radicals:?}"));
        }
        if isotopes != expected_isotopes {
            differences.push(format!("isotopes {isotopes:?} vs {expected_isotopes:?}"));
        }

        if differences.is_empty() {
            agree += 1;
        } else {
            println!("{smiles}: {}", differences.join("; "));
        }
    }

    println!("{agree}/{total} agree with RDKit");
}
