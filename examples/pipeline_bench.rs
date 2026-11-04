//! Compare complete SMILES-to-result operations, and time the sweep's stages.
//! pipeline_bench <CID-SMILES> <parse|canon|canon-safe|key|normalize|sweep> [limit] [repeats]
//! Input is loaded before timing; output is consumed, not written to disk.
use oxidane::graph_algorithms::mol_canonization::Canonizer;
use oxidane::molecular_formula::MolecularFormula;
use oxidane::molecule::tautomer::{TautomerRules, tautomer_key_with, try_normalize_with};
use oxidane::prelude::*;
use std::hint::black_box;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

#[derive(Default)]
struct Timings {
    parse: f64,
    operation: f64,
    write: f64,
    formula: f64,
    reparse: f64,
    key_input: f64,
    key_output: f64,
    normalize_again: f64,
    components: usize,
    bytes: usize,
}
fn elapsed(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let path = &args[1];
    let mode = args[2].as_str();
    assert!(matches!(
        mode,
        "parse" | "canon" | "canon-safe" | "key" | "normalize" | "sweep"
    ));
    let limit = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(50_000);
    let repeats = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(3);
    let reader = BufReader::new(std::fs::File::open(path).unwrap());
    let corpus: Vec<(String, String)> = reader
        .lines()
        .take(limit)
        .map(|line| {
            let line = line.unwrap();
            line.split_once('\t')
                .map(|(cid, s)| (cid.to_owned(), s.to_owned()))
                .unwrap_or_else(|| (String::new(), line))
        })
        .collect();
    // Optional output capture is for exact baseline comparisons, not timing.
    let mut dump = args
        .get(6)
        .map(|path| BufWriter::new(std::fs::File::create(path).unwrap()));
    let rules = TautomerRules::new();
    let mut canonizer = Canonizer::new();
    for repeat in 0..repeats {
        let mut t = Timings::default();
        let started = Instant::now();
        for (cid, smiles) in &corpus {
            let record_started = Instant::now();
            let start = Instant::now();
            let molecules = MolecularGraph::from_smiles(smiles).expect("sample must parse");
            t.parse += elapsed(start);
            let mut outputs = Vec::new();
            for molecule in molecules {
                t.components += 1;
                let start = Instant::now();
                match mode {
                    "parse" => {
                        black_box(&molecule);
                    }
                    "key" => {
                        let key = tautomer_key_with(&mut canonizer, &molecule, &rules);
                        t.operation += elapsed(start);
                        outputs.push(key);
                    }
                    "canon" | "canon-safe" | "normalize" => {
                        let answer = match mode {
                            "canon" => canonizer.canonize_perceived(molecule),
                            "canon-safe" => canonizer.canonize(&molecule),
                            _ => try_normalize_with(&mut canonizer, &molecule, &rules).unwrap(),
                        };
                        t.operation += elapsed(start);
                        let start = Instant::now();
                        outputs.push(answer.to_smiles());
                        t.write += elapsed(start);
                    }
                    "sweep" => {
                        let normal = try_normalize_with(&mut canonizer, &molecule, &rules).unwrap();
                        t.operation += elapsed(start);
                        let start = Instant::now();
                        let written = normal.to_smiles();
                        t.write += elapsed(start);
                        let start = Instant::now();
                        assert_eq!(
                            MolecularFormula::from_molecule(&normal),
                            MolecularFormula::from_molecule(&molecule)
                        );
                        t.formula += elapsed(start);
                        let start = Instant::now();
                        let reparsed = MolecularGraph::from_smiles_single(&written).unwrap();
                        t.reparse += elapsed(start);
                        let start = Instant::now();
                        let output_key = tautomer_key_with(&mut canonizer, &reparsed, &rules);
                        t.key_output += elapsed(start);
                        let start = Instant::now();
                        let input_key = tautomer_key_with(&mut canonizer, &molecule, &rules);
                        t.key_input += elapsed(start);
                        assert_eq!(input_key, output_key);
                        let start = Instant::now();
                        let again = try_normalize_with(&mut canonizer, &reparsed, &rules)
                            .unwrap()
                            .to_smiles();
                        t.normalize_again += elapsed(start);
                        assert_eq!(written, again);
                        outputs.push(written);
                    }
                    _ => unreachable!(),
                }
            }
            outputs.sort();
            let output = outputs.join(".");
            if let Some(writer) = dump.as_mut() {
                writeln!(writer, "{output}").unwrap();
            }
            t.bytes += black_box(output).len();
            let record_seconds = elapsed(record_started);
            if record_seconds > 0.02 {
                eprintln!("slow\t{mode}\t{cid}\t{record_seconds:.6}\t{smiles}");
            }
        }
        let wall = elapsed(started);
        println!(
            "{{\"mode\":\"{mode}\",\"repeat\":{repeat},\"records\":{},\"components\":{},\"seconds\":{wall:.6},\"records_per_second\":{:.2},\"parse\":{:.6},\"operation\":{:.6},\"write\":{:.6},\"formula\":{:.6},\"reparse\":{:.6},\"key_input\":{:.6},\"key_output\":{:.6},\"normalize_again\":{:.6},\"output_bytes\":{}}}",
            corpus.len(),
            t.components,
            corpus.len() as f64 / wall,
            t.parse,
            t.operation,
            t.write,
            t.formula,
            t.reparse,
            t.key_input,
            t.key_output,
            t.normalize_again,
            t.bytes
        );
    }
}
