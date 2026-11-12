//! Draws the pictures in the README with oxidane's own depiction code.
//!
//! ```sh
//! cargo run --example readme_figures   # writes assets/readme/*.svg
//! ```
//!
//! The reaction and the peptides are not drawn from SMILES typed in by hand:
//! they are what the README's rules give when applied, so the pictures show
//! the library's output.

use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use oxidane::io::SmilesParseOptions;
use oxidane::molecule::draw::{Draw, DrawOptions};
use oxidane::molecule::reaction::ReactionSmarts;
use oxidane::prelude::*;
use std::collections::BTreeSet;

const INK: &str = "#243247";
const ESTER: &str = "[C:1](=[O:2])[OX2H1:3].[OX2H1:4][CX4:5]>>[C:1](=[O:2])[O:4][C:5].[O:3]";
const AMIDE: &str = "[C:1](=[O:2])[OX2H1:3].[NX3H2:4][CX4:5]>>[C:1](=[O:2])[N:4][C:5].[O:3]";

/// A molecule as drawn: Kekulé bonds rather than aromatic rings.
fn molecule(smiles: &str) -> MolecularGraph {
    let options = SmilesParseOptions {
        aromaticity: None,
        ..SmilesParseOptions::default()
    };
    MolecularGraph::from_smiles_single_with(smiles, options).expect("the SMILES parses")
}

/// One drawn molecule, placed at `x`, `y` in a figure.
fn panel(molecule: &MolecularGraph, x: f64, y: f64, width: u32, height: u32) -> String {
    let options = DrawOptions {
        width,
        height,
        cleanup: true,
        ..DrawOptions::default()
    };
    let svg = molecule.to_svg_with_options(&options);
    svg.replacen("<svg ", &format!("<svg x=\"{x}\" y=\"{y}\" "), 1)
}

fn text(x: f64, y: f64, size: f64, label: &str) -> String {
    format!(
        "<text x=\"{x}\" y=\"{y}\" font-family=\"sans-serif\" font-size=\"{size}\" \
         fill=\"{INK}\" text-anchor=\"middle\">{label}</text>"
    )
}

fn arrow(x: f64, y: f64, length: f64) -> String {
    let end = x + length;
    format!(
        "<path d=\"M {x} {y} H {end} M {} {} L {end} {y} L {} {}\" fill=\"none\" \
         stroke=\"{INK}\" stroke-width=\"2\"/>",
        end - 9.0,
        y - 7.0,
        end - 9.0,
        y + 7.0
    )
}

/// A figure on a white card, so it reads on a dark page too.
fn figure(width: f64, height: f64, title: &str, body: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" \
         viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{title}\">\
         <title>{title}</title>\
         <rect width=\"{width}\" height=\"{height}\" rx=\"12\" fill=\"#ffffff\"/>{body}</svg>\n"
    )
}

fn canonical(smiles: &str) -> String {
    canonicalize_and_sort_smiles(smiles, false).expect("the SMILES parses")
}

fn molecules() -> String {
    let entries = [
        ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("(S)-ibuprofen", "CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O"),
        ("serotonin", "NCCC1=CNC2=C1C=C(O)C=C2"),
    ];
    let mut body = String::new();
    for (index, (name, smiles)) in entries.iter().enumerate() {
        let x = 10.0 + index as f64 * 230.0;
        body += &panel(&molecule(smiles), x, 10.0, 225, 190);
        body += &text(x + 112.0, 215.0, 15.0, name);
    }
    figure(940.0, 235.0, "Molecules drawn by oxidane", &body)
}

/// Acetic acid and ethanol under the ester rule, products as the rule gives
/// them.
fn reaction() -> String {
    let rule = ReactionSmarts::parse(ESTER).unwrap();
    let acid = MolecularGraph::from_smiles_single("CC(=O)O").unwrap();
    let alcohol = MolecularGraph::from_smiles_single("CCO").unwrap();
    let products = rule.apply_to(&[&acid, &alcohol]);
    let written = canonical(&products[0].to_smiles());
    let mut parts: Vec<&str> = written.split('.').collect();
    // The ester first, then the water.
    parts.sort_by_key(|part| std::cmp::Reverse(part.len()));
    assert_eq!(parts, [canonical("CCOC(C)=O").as_str(), "O"]);

    let mut body = String::new();
    body += &panel(&molecule("CC(=O)O"), 10.0, 20.0, 170, 150);
    body += &text(195.0, 100.0, 24.0, "+");
    body += &panel(&molecule("CCO"), 210.0, 20.0, 150, 150);
    body += &arrow(380.0, 95.0, 70.0);
    body += &panel(&molecule(parts[0]), 465.0, 20.0, 210, 150);
    body += &text(690.0, 100.0, 24.0, "+");
    body += &panel(&molecule(parts[1]), 700.0, 20.0, 110, 150);
    body += &text(95.0, 185.0, 14.0, "acetic acid");
    body += &text(285.0, 185.0, 14.0, "ethanol");
    body += &text(570.0, 185.0, 14.0, "ethyl acetate");
    body += &text(755.0, 185.0, 14.0, "water");
    figure(820.0, 205.0, "Ester formation applied by oxidane", &body)
}

/// Glycine and its peptides as two rounds of the README's rules make them.
fn peptides() -> String {
    let rules = [ESTER, AMIDE].map(|rule| ReactionSmarts::parse(rule).unwrap());
    let mut known: BTreeSet<String> = ["CC(=O)O", "CCO", "NCC(=O)O"]
        .into_iter()
        .map(canonical)
        .collect();
    for _ in 0..2 {
        let current: Vec<MolecularGraph> = known
            .iter()
            .map(|smiles| MolecularGraph::from_smiles_single(smiles).unwrap())
            .collect();
        for rule in &rules {
            for first in &current {
                for second in &current {
                    for product in rule.apply_to(&[first, second]) {
                        let written = canonical(&product.to_smiles());
                        known.extend(written.split('.').map(String::from));
                    }
                }
            }
        }
    }
    let chain = [
        "NCC(=O)O",
        "NCC(=O)NCC(=O)O",
        "NCC(=O)NCC(=O)NCC(=O)NCC(=O)O",
    ];
    for smiles in chain {
        assert!(known.contains(&canonical(smiles)), "{smiles} was not made");
    }

    let mut body = String::new();
    body += &panel(&molecule(chain[0]), 10.0, 10.0, 150, 140);
    body += &arrow(170.0, 80.0, 60.0);
    body += &text(200.0, 65.0, 12.0, "round 1");
    body += &panel(&molecule(chain[1]), 240.0, 10.0, 230, 140);
    body += &arrow(480.0, 80.0, 60.0);
    body += &text(510.0, 65.0, 12.0, "round 2");
    body += &panel(&molecule(chain[2]), 550.0, 10.0, 390, 140);
    body += &text(85.0, 165.0, 14.0, "glycine");
    body += &text(355.0, 165.0, 14.0, "diglycine");
    body += &text(745.0, 165.0, 14.0, "tetraglycine");
    figure(
        950.0,
        185.0,
        "Peptides grown from glycine by oxidane",
        &body,
    )
}

fn main() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/readme");
    std::fs::create_dir_all(&directory).expect("assets/readme can be created");
    for (name, svg) in [
        ("molecules.svg", molecules()),
        ("reaction.svg", reaction()),
        ("peptides.svg", peptides()),
    ] {
        std::fs::write(directory.join(name), svg).expect("the figure can be written");
        println!("wrote assets/readme/{name}");
    }
}
