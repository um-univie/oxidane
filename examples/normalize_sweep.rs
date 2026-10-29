//! Checks tautomer normalization over every SMILES corpus in the repository.
//!
//! Six properties, none of which a hand-written test list can cover, because
//! the interesting failures are all molecules nobody thought to write down:
//!
//! * the normalized molecule has the formula it started with,
//! * its SMILES parses back,
//! * normalizing it again changes nothing,
//! * it shares a tautomer key with what it came from -- the two halves of the
//!   module agreeing that they are looking at the same compound,
//! * relabelling the input's atoms does not change the answer,
//! * and how often no answer exists at all.
//!
//! ```text
//! cargo run --release --example normalize_sweep        # five relabellings each
//! cargo run --release --example normalize_sweep -- 20
//! ```

use oxidane::graph_algorithms::mol_canonization::reorder_molecule;
use oxidane::molecular_formula::MolecularFormula;
use oxidane::molecule::tautomer::{TautomerRules, normalize, tautomer_key};
use oxidane::prelude::*;
use rand::prelude::*;

/// Every SMILES the repository has, deduplicated.
fn corpus() -> Vec<String> {
    let mut smiles = Vec::new();
    for path in ["tests/smiles.txt", "tests/smiles_tree_permutations.txt"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            smiles.extend(text.split_whitespace().map(str::to_string));
        }
    }
    // The aromatization corpus is `written, expected  # comment` per line.
    if let Ok(text) = std::fs::read_to_string("tests/aromatization_tests.smi") {
        for line in text.lines() {
            for field in line.split('#').next().unwrap_or("").split(',') {
                let field = field.trim();
                if !field.is_empty() {
                    smiles.push(field.to_string());
                }
            }
        }
    }
    smiles.sort();
    smiles.dedup();
    smiles
}

fn main() {
    let trials: usize = std::env::args()
        .nth(1)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(5);
    let rules = TautomerRules::new();
    let mut rng = StdRng::seed_from_u64(0xBEEF);

    let mut fragments = 0usize;
    let (mut fallback, mut formula, mut reparse) = (0usize, 0usize, 0usize);
    let (mut keyed, mut idempotence, mut permutation) = (0usize, 0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    let note = |examples: &mut Vec<String>, text: String| {
        if examples.len() < 10 {
            examples.push(text);
        }
    };

    for smiles in &corpus() {
        let Ok(molecules) = MolecularGraph::from_smiles(smiles) else {
            continue;
        };
        for molecule in &molecules {
            fragments += 1;
            let Some(normal) = normalize(molecule, &rules) else {
                fallback += 1;
                continue;
            };
            let written = normal.to_smiles();

            if MolecularFormula::from_molecule(&normal) != MolecularFormula::from_molecule(molecule)
            {
                formula += 1;
                note(&mut examples, format!("formula  {smiles} -> {written}"));
            }

            match MolecularGraph::from_smiles_single(&written) {
                Ok(reparsed) => {
                    if tautomer_key(&reparsed, &rules) != tautomer_key(molecule, &rules) {
                        keyed += 1;
                        note(&mut examples, format!("key      {smiles} -> {written}"));
                    }
                    let again = normalize(&reparsed, &rules).map(|molecule| molecule.to_smiles());
                    if again.as_deref() != Some(written.as_str()) {
                        idempotence += 1;
                        note(&mut examples, format!("idempot  {written} -> {again:?}"));
                    }
                }
                Err(_) => {
                    reparse += 1;
                    note(&mut examples, format!("reparse  {smiles} -> {written}"));
                }
            }

            for _ in 0..trials {
                let mut labeling: Vec<usize> = (0..molecule.len()).collect();
                labeling.shuffle(&mut rng);
                let mut permuted = molecule.clone();
                reorder_molecule(&mut permuted, &labeling);
                let relabelled = normalize(&permuted, &rules).map(|m| m.to_smiles());
                if relabelled.as_deref() != Some(written.as_str()) {
                    permutation += 1;
                    note(&mut examples, format!("permute  {smiles}: {relabelled:?}"));
                    break;
                }
            }
        }
    }

    let percent = 100.0 * fallback as f64 / fragments.max(1) as f64;
    println!("{fragments} fragments, {trials} relabellings each");
    println!("fell back (no assignment): {fallback}  ({percent:.2}%)");
    println!("formula changed          : {formula}");
    println!("output did not reparse   : {reparse}");
    println!("key changed              : {keyed}");
    println!("not idempotent           : {idempotence}");
    println!("relabelling changed it   : {permutation}");
    for example in &examples {
        println!("  {example}");
    }
}
