//! Where the parse time goes.
use molecules::io::{SMILESParser, SmilesParseOptions};
use molecules::prelude::*;
use std::time::Instant;

fn corpus() -> Vec<String> {
    let mut c: Vec<String> = std::fs::read_to_string("tests/smiles.txt").unwrap_or_default()
        .lines().filter_map(|l| l.split_whitespace().next().map(str::to_string)).collect();
    c.extend(std::fs::read_to_string("notes/steroids.smi").unwrap_or_default()
        .lines().map(str::to_string).filter(|l| !l.is_empty()));
    c.retain(|s| MolecularGraph::from_smiles_single(s).is_ok());
    c
}

fn time(name: &str, corpus: &[String], rounds: u32, f: impl Fn(&str)) {
    let start = Instant::now();
    for _ in 0..rounds { for s in corpus { f(s); } }
    let e = start.elapsed();
    println!("  {name:34} {:>7.1?} per molecule", e / (rounds * corpus.len() as u32));
}

fn main() {
    let c = corpus();
    println!("{} molecules", c.len());
    let rounds = 200;
    time("raw (tokenise + graph only)", &c, rounds, |s| { std::hint::black_box(MolecularGraph::from_smiles_raw(s).unwrap()); });
    time("no aromaticity perception", &c, rounds, |s| {
        std::hint::black_box(MolecularGraph::from_smiles_single_with(s, SmilesParseOptions { aromaticity: None }).unwrap()); });
    time("full default", &c, rounds, |s| { std::hint::black_box(MolecularGraph::from_smiles_single(s).unwrap()); });
    time("parse_smiles (all fragments)", &c, rounds, |s| { std::hint::black_box(SMILESParser::parse_smiles(s).unwrap()); });
}
