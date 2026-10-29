//! Reads SMILES on stdin, writes `<smiles>\t<canonical tautomer SMILES>`.
use oxidane::molecule::tautomer::{TautomerRules, normalized_smiles};
use std::io::{BufWriter, Read, Write};
fn main() {
    let rules = TautomerRules::new();
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("stdin is readable");
    let mut output = BufWriter::new(std::io::stdout().lock());
    for smiles in input.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if let Ok(normal) = normalized_smiles(smiles, &rules) {
            writeln!(output, "{smiles}\t{normal}").expect("stdout is writable");
        }
    }
}
