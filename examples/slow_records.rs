//! Canonizes `<cid>\t<smiles>` records and reports the ones that take too long.
//!
//! ```text
//! cargo run --release --example slow_records -- <corpus> [--over-ms N] [--limit N]
//! ```
//! Perception follows `OXIDANE_AROMATICITY`, so the two models can be compared
//! on the records that cost the most.
use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let path = &arguments[1];
    let over = arguments
        .windows(2)
        .find(|pair| pair[0] == "--over-ms")
        .and_then(|pair| pair[1].parse::<u128>().ok())
        .unwrap_or(100);
    let limit = arguments
        .windows(2)
        .find(|pair| pair[0] == "--limit")
        .and_then(|pair| pair[1].parse::<usize>().ok())
        .unwrap_or(usize::MAX);

    let reader = BufReader::new(std::fs::File::open(path).expect("the corpus opens"));
    let mut worst: Vec<(u128, String)> = Vec::new();
    let started = Instant::now();
    let mut seen = 0usize;
    for line in reader.lines().take(limit) {
        let Ok(line) = line else { break };
        let mut fields = line.split('\t');
        let (Some(cid), Some(smiles)) = (fields.next(), fields.next()) else {
            continue;
        };
        seen += 1;
        let record = Instant::now();
        let _ = std::panic::catch_unwind(|| canonicalize_and_sort_smiles(smiles, false));
        let milliseconds = record.elapsed().as_millis();
        if milliseconds >= over {
            println!("{milliseconds}\t{cid}\t{}", &smiles[..smiles.len().min(60)]);
            let _ = std::io::stdout().flush();
            worst.push((milliseconds, cid.to_string()));
        }
    }
    worst.sort_unstable();
    worst.reverse();
    eprintln!(
        "{seen} records in {:.1}s, {} over {over} ms, worst {:?}",
        started.elapsed().as_secs_f64(),
        worst.len(),
        worst.first()
    );
}
