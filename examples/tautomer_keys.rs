//! Reads SMILES on stdin, one per line, and writes `<smiles>\t<tautomer key>`.
//!
//! The Rust half of `tests/tautomer_oracle.py`, which checks that every
//! tautomer RDKit generates for a molecule comes back with one key.
//!
//! ```text
//! printf 'O=c1cccc[nH]1\nOc1ccccn1\n' | cargo run --release --example tautomer_keys
//! ```

use oxidane::molecule::tautomer::{TautomerRules, tautomer_key_from_smiles};
use std::io::{BufWriter, Read, Write};

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let mut rules = if arguments
        .iter()
        .any(|argument| argument == "--carbon-endpoints")
    {
        TautomerRules::with_carbon_endpoints()
    } else {
        TautomerRules::new()
    };
    if arguments
        .iter()
        .any(|argument| argument == "--ionizable-endpoints")
    {
        rules.ionizable_endpoints = true;
    }

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("stdin is readable");
    let mut output = BufWriter::new(std::io::stdout().lock());
    for smiles in input.lines().map(str::trim).filter(|line| !line.is_empty()) {
        match tautomer_key_from_smiles(smiles, &rules) {
            Ok(key) => writeln!(output, "{smiles}\t{key}").expect("stdout is writable"),
            Err(error) => eprintln!("{smiles}: {error:?}"),
        }
    }
}
