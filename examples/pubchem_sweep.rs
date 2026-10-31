//! Runs ordinary canonicalization and tautomer normalization over a corpus, checking the properties each
//! promises. Written for PubChem's `CID-SMILES`: 124 million lines of
//! `<cid>\t<smiles>`, streamed rather than read in.
//!
//! Every check here is one the crate can make on its own, so it needs no
//! reference implementation and can therefore run over everything rather than
//! over a sample:
//!
//! * the SMILES parses at all (counted, not a failure -- PubChem holds things
//!   this crate does not claim to read),
//! * `normalize` returns an answer. It always should: the input molecule is
//!   itself a feasible point of the constraint system, so an empty feasible set
//!   means the model disagrees with the molecule,
//! * the answer has the formula it started with,
//! * the answer parses back,
//! * normalizing it again changes nothing,
//! * it shares a tautomer key with what it came from,
//! * ordinary canonical SMILES preserves formula, reparses and is idempotent,
//! * one CID-seeded atom permutation preserves ordinary canonical SMILES,
//!   normalized tautomer SMILES and tautomer keys for every component,
//! * the canonical SMILES, read back, has the key its input had,
//! * and nothing panics.
//!
//! ```text
//! cargo run --release --example pubchem_sweep -- <path> [--limit N] [--threads N] [--max-atoms N] [--tautomers on|off] [--normalize on|off] [--ionizable on|off]
//! ```
//!
//! The corpus is mapped rather than read: the records are found by looking for
//! newlines in the mapped bytes, the file is cut into one span per worker at
//! record boundaries, and each worker walks its own span. There is no shared
//! reader and no barrier between batches, which matters for a reason beyond
//! throughput -- under the old chunked reader a single slow record left
//! twenty-three workers idle and the run silent, and finding which record it
//! was took an hour of bisection. Each worker now publishes the record it is
//! on, and a reporter prints anything that has been in flight too long.
//!
//! The default atom limit is zero (unlimited); positive limits count heavy
//! atoms per component and log every skipped record.
//! `--tautomers off` runs the canonicalization checks alone, leaving
//! normalization and tautomer keys out, so the two halves can be told apart.
//! `--normalize off` keeps the keys and leaves out only normalization, which
//! is three quarters of the sweep's time and the part fewest callers need.
//! Failures are written to stderr with their CID so they can be looked up.

use oxidane::graph_algorithms::mol_canonization::{
    canonize_mol, canonize_mol_perceived, reorder_molecule,
};
use oxidane::molecular_formula::MolecularFormula;
use oxidane::molecule::tautomer::{
    NormalizationError, TautomerRules, normalize, tautomer_key, try_normalize,
};
use oxidane::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How long a record may be in flight before the reporter names it.
const TOO_LONG: Duration = Duration::from_secs(30);

/// How often the reporter looks.
const REPORT_EVERY: Duration = Duration::from_secs(10);

/// What each worker is doing, so that a stall can say so itself.
struct InFlight {
    /// The record each worker started last, and when, in milliseconds since
    /// the sweep began. Two numbers rather than a lock: the reporter reads
    /// them while the workers write them, and a torn pair costs a misleading
    /// line rather than anything worse.
    record: Vec<AtomicU64>,
    started: Vec<AtomicU64>,
    reported: Vec<AtomicU64>,
}

impl InFlight {
    fn new(workers: usize) -> Self {
        Self {
            record: (0..workers).map(|_| AtomicU64::new(u64::MAX)).collect(),
            started: (0..workers).map(|_| AtomicU64::new(0)).collect(),
            reported: (0..workers).map(|_| AtomicU64::new(u64::MAX)).collect(),
        }
    }

    fn taking(&self, worker: usize, record: u64, elapsed: Duration) {
        self.record[worker].store(record, Ordering::Relaxed);
        self.started[worker].store(elapsed.as_millis() as u64, Ordering::Relaxed);
    }
}

/// What one sweep checks, and under what limits.
struct Options {
    max_atoms: usize,
    tautomers: bool,
    normalize: bool,
}

#[derive(Default)]
struct Counts {
    seen: AtomicU64,
    unparsed: AtomicU64,
    normalized: AtomicU64,
    no_answer: AtomicU64,
    formula: AtomicU64,
    reparse: AtomicU64,
    idempotence: AtomicU64,
    key: AtomicU64,
    oversize: AtomicU64,
    panics: AtomicU64,
    canonical_formula: AtomicU64,
    canonical_reparse: AtomicU64,
    canonical_idempotence: AtomicU64,
    canonical_permutation: AtomicU64,
    tautomer_permutation: AtomicU64,
    key_permutation: AtomicU64,
    /// The canonical SMILES, read back, keyed differently from its input.
    key_canonical: AtomicU64,
    nonconverged: AtomicU64,
    /// Records whose valences no element allows, which are not molecules to
    /// normalize and are counted apart from the failures that would be bugs.
    impossible: AtomicU64,
}

impl Counts {
    fn merge(&self, other: &Self) {
        for (target, source) in [
            (&self.seen, &other.seen),
            (&self.unparsed, &other.unparsed),
            (&self.normalized, &other.normalized),
            (&self.no_answer, &other.no_answer),
            (&self.formula, &other.formula),
            (&self.reparse, &other.reparse),
            (&self.idempotence, &other.idempotence),
            (&self.key, &other.key),
            (&self.canonical_formula, &other.canonical_formula),
            (&self.canonical_reparse, &other.canonical_reparse),
            (&self.canonical_idempotence, &other.canonical_idempotence),
            (&self.canonical_permutation, &other.canonical_permutation),
            (&self.tautomer_permutation, &other.tautomer_permutation),
            (&self.key_permutation, &other.key_permutation),
            (&self.key_canonical, &other.key_canonical),
            (&self.nonconverged, &other.nonconverged),
            (&self.impossible, &other.impossible),
            (&self.oversize, &other.oversize),
            (&self.panics, &other.panics),
        ] {
            target.fetch_add(source.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    /// Empties the counters, for a worker that has just folded its tally into
    /// the total. Keeping the running total visible is what lets the reporter
    /// print progress without the workers contending on one atomic per record.
    fn reset(&self) {
        for counter in [
            &self.seen,
            &self.unparsed,
            &self.normalized,
            &self.no_answer,
            &self.formula,
            &self.reparse,
            &self.idempotence,
            &self.key,
            &self.canonical_formula,
            &self.canonical_reparse,
            &self.canonical_idempotence,
            &self.canonical_permutation,
            &self.tautomer_permutation,
            &self.key_permutation,
            &self.key_canonical,
            &self.nonconverged,
            &self.impossible,
            &self.oversize,
            &self.panics,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    fn report(&self, elapsed: f64) {
        let seen = self.seen.load(Ordering::Relaxed);
        let rate = seen as f64 / elapsed.max(1e-9);
        println!("{seen} lines in {elapsed:.0}s ({rate:.0}/s)");
        println!(
            "  unparsed by this crate : {}",
            self.unparsed.load(Ordering::Relaxed)
        );
        println!(
            "  normalized             : {}",
            self.normalized.load(Ordering::Relaxed)
        );
        println!(
            "  no answer (should be 0): {}",
            self.no_answer.load(Ordering::Relaxed)
        );
        println!(
            "  formula changed        : {}",
            self.formula.load(Ordering::Relaxed)
        );
        println!(
            "  output did not reparse : {}",
            self.reparse.load(Ordering::Relaxed)
        );
        println!(
            "  not idempotent         : {}",
            self.idempotence.load(Ordering::Relaxed)
        );
        println!(
            "  key changed            : {}",
            self.key.load(Ordering::Relaxed)
        );
        println!(
            "  canonical_formula : {}",
            self.canonical_formula.load(Ordering::Relaxed)
        );
        println!(
            "  canonical_reparse : {}",
            self.canonical_reparse.load(Ordering::Relaxed)
        );
        println!(
            "  canonical_idempotence : {}",
            self.canonical_idempotence.load(Ordering::Relaxed)
        );
        println!(
            "  canonical_permutation : {}",
            self.canonical_permutation.load(Ordering::Relaxed)
        );
        println!(
            "  tautomer_permutation : {}",
            self.tautomer_permutation.load(Ordering::Relaxed)
        );
        println!(
            "  key_permutation : {}",
            self.key_permutation.load(Ordering::Relaxed)
        );
        println!(
            "  key_canonical : {}",
            self.key_canonical.load(Ordering::Relaxed)
        );
        println!(
            "  did not converge       : {}",
            self.nonconverged.load(Ordering::Relaxed)
        );
        println!(
            "  impossible valence     : {}",
            self.impossible.load(Ordering::Relaxed)
        );
        println!(
            "  panics                 : {}",
            self.panics.load(Ordering::Relaxed)
        );
        println!(
            "  skipped as oversize    : {}",
            self.oversize.load(Ordering::Relaxed)
        );
    }
}

/// Checks one molecule, bumping whichever counters it trips.
fn check(cid: &str, smiles: &str, rules: &TautomerRules, counts: &Counts, options: &Options) {
    counts.seen.fetch_add(1, Ordering::Relaxed);
    // A component at a time, as the rest of the module does.
    let Ok(molecules) = MolecularGraph::from_smiles(smiles) else {
        counts.unparsed.fetch_add(1, Ordering::Relaxed);
        eprintln!("unparsed\t{cid}\t{smiles}");
        return;
    };
    for molecule in &molecules {
        // Canonization uses heavy atoms as graph vertices. SMILES length is
        // neither an atom count nor a useful bound for disconnected records.
        let heavy = molecule.atomic_numbers().iter().filter(|&&z| z > 1).count();
        if options.max_atoms > 0 && heavy > options.max_atoms {
            counts.oversize.fetch_add(1, Ordering::Relaxed);
            eprintln!("oversize\t{cid}\t{smiles}\t{heavy}");
            continue;
        }
        // One deterministic random relabelling per component, reproducible across shards/threads.
        let mut labeling: Vec<usize> = (0..molecule.len()).collect();
        let mut rng = StdRng::seed_from_u64(cid.parse::<u64>().unwrap_or(0) ^ 0xBEEF);
        labeling.shuffle(&mut rng);
        let mut permuted = molecule.clone();
        reorder_molecule(&mut permuted, &labeling);
        // `from_smiles` has perceived aromaticity already, and perceiving it
        // again is a fifth of what canonizing costs. The permuted copy still
        // goes through `canonize_mol`, so perception in a different atom order
        // is checked against this answer rather than assumed to agree.
        let canonical = canonize_mol_perceived(molecule);
        let canonical_smiles = canonical.to_smiles();
        if MolecularFormula::from_molecule(&canonical) != MolecularFormula::from_molecule(molecule)
        {
            counts.canonical_formula.fetch_add(1, Ordering::Relaxed);
            eprintln!("canonical_formula\t{cid}\t{smiles}\t{canonical_smiles}");
        }
        let reread = MolecularGraph::from_smiles_single(&canonical_smiles);
        match &reread {
            Ok(again) => {
                if MolecularFormula::from_molecule(again)
                    != MolecularFormula::from_molecule(molecule)
                {
                    counts.canonical_formula.fetch_add(1, Ordering::Relaxed);
                    eprintln!("canonical_roundtrip_formula\t{cid}\t{smiles}\t{canonical_smiles}");
                }
                let rewritten = canonize_mol_perceived(again).to_smiles();
                if rewritten != canonical_smiles {
                    counts.canonical_idempotence.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "canonical_idempotence\t{cid}\t{smiles}\t{canonical_smiles}\t{rewritten}"
                    );
                }
            }
            Err(_) => {
                counts.canonical_reparse.fetch_add(1, Ordering::Relaxed);
                eprintln!("canonical_reparse\t{cid}\t{smiles}\t{canonical_smiles}");
            }
        }
        let permuted_canonical = canonize_mol(&permuted).to_smiles();
        if permuted_canonical != canonical_smiles {
            counts.canonical_permutation.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "canonical_permutation\t{cid}\t{smiles}\t{canonical_smiles}\t{permuted_canonical}"
            );
        }
        if !options.tautomers {
            continue;
        }
        let key = tautomer_key(molecule, rules);
        if tautomer_key(&permuted, rules) != key {
            counts.key_permutation.fetch_add(1, Ordering::Relaxed);
            eprintln!("key_permutation\t{cid}\t{smiles}");
        }
        if let Ok(again) = &reread
            && tautomer_key(again, rules) != key
        {
            counts.key_canonical.fetch_add(1, Ordering::Relaxed);
            eprintln!("key_canonical\t{cid}\t{smiles}\t{canonical_smiles}");
        }
        if !options.normalize {
            continue;
        }
        let normal = match try_normalize(molecule, rules) {
            Ok(normal) => normal,
            Err(error) => {
                let (counter, label) = match error {
                    NormalizationError::NoAssignment => (&counts.no_answer, "no answer"),
                    NormalizationError::DidNotConverge => (&counts.nonconverged, "nonconvergence"),
                    NormalizationError::ImpossibleValence(_) => {
                        (&counts.impossible, "impossible valence")
                    }
                };
                counter.fetch_add(1, Ordering::Relaxed);
                eprintln!("{label}\t{cid}\t{smiles}");
                continue;
            }
        };
        counts.normalized.fetch_add(1, Ordering::Relaxed);
        let written = normal.to_smiles();
        let permuted_normal = normalize(&permuted, rules).map(|m| m.to_smiles());
        if permuted_normal.as_deref() != Some(written.as_str()) {
            counts.tautomer_permutation.fetch_add(1, Ordering::Relaxed);
            eprintln!("tautomer_permutation\t{cid}\t{smiles}\t{written}\t{permuted_normal:?}");
        }

        if MolecularFormula::from_molecule(&normal) != MolecularFormula::from_molecule(molecule) {
            counts.formula.fetch_add(1, Ordering::Relaxed);
            eprintln!("formula\t{cid}\t{smiles}\t{written}");
            continue;
        }
        let Ok(reparsed) = MolecularGraph::from_smiles_single(&written) else {
            counts.reparse.fetch_add(1, Ordering::Relaxed);
            eprintln!("reparse\t{cid}\t{smiles}\t{written}");
            continue;
        };
        if MolecularFormula::from_molecule(&reparsed) != MolecularFormula::from_molecule(molecule) {
            counts.formula.fetch_add(1, Ordering::Relaxed);
            eprintln!("tautomer_roundtrip_formula\t{cid}\t{smiles}\t{written}");
        }
        if tautomer_key(&reparsed, rules) != key {
            counts.key.fetch_add(1, Ordering::Relaxed);
            eprintln!("key\t{cid}\t{smiles}\t{written}");
        }
        if normalize(&reparsed, rules)
            .map(|again| again.to_smiles())
            .as_deref()
            != Some(&written)
        {
            counts.idempotence.fetch_add(1, Ordering::Relaxed);
            eprintln!("idempotence\t{cid}\t{smiles}\t{written}");
        }
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments
        .next()
        .expect("usage: pubchem_sweep <path> [--limit N]");
    let mut limit = usize::MAX;
    let mut threads = 0usize;
    let mut max_atoms = 0usize;
    let mut tautomers = true;
    let mut normalize = true;
    let mut ionizable = false;
    while let Some(option) = arguments.next() {
        let value = arguments.next().expect("each option requires a value");
        match option.as_str() {
            "--limit" => limit = value.parse().expect("--limit takes a number"),
            "--threads" => threads = value.parse().expect("--threads takes a number"),
            "--max-atoms" => max_atoms = value.parse().expect("--max-atoms takes a number"),
            "--tautomers" => {
                tautomers = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => panic!("--tautomers takes on or off"),
                }
            }
            "--normalize" => {
                normalize = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => panic!("--normalize takes on or off"),
                }
            }
            "--ionizable" => {
                ionizable = match value.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => panic!("--ionizable takes on or off"),
                }
            }
            _ => panic!("unknown option: {option}"),
        }
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("the thread pool is built once");
    }

    let options = Options {
        max_atoms,
        tautomers,
        normalize,
    };
    let mut rules = TautomerRules::new();
    rules.ionizable_endpoints = ionizable;
    let counts = Counts::default();
    let started = Instant::now();

    let file = File::open(&path).expect("the corpus opens");
    // Safety: the corpus is read-only for the length of the sweep. A file
    // truncated underneath a mapping would fault, which is a worse failure than
    // a short read -- and is why nothing else in the crate maps a file.
    let corpus = unsafe { memmap2::Mmap::map(&file).expect("the corpus maps") };
    let corpus: &[u8] = &corpus;

    // Every record the sweep will look at, as one span of bytes.
    let end_of_work = if limit == usize::MAX {
        corpus.len()
    } else {
        let mut seen = 0;
        let mut at = 0;
        loop {
            match memchr::memchr(b'\n', &corpus[at..]) {
                Some(offset) => {
                    at += offset + 1;
                    seen += 1;
                    if seen >= limit || at >= corpus.len() {
                        break at.min(corpus.len());
                    }
                }
                None => break corpus.len(),
            }
        }
    };
    let work = &corpus[..end_of_work];

    // Spans cut at record boundaries, many per worker. No reader thread, no
    // barrier: a slow record holds up its own span and nothing else. Four per
    // worker left the last forty minutes of a full PubChem run on a handful of
    // threads, because the later records are larger and their spans finish
    // last; small spans let idle workers take the rest of the work.
    let workers = if threads > 0 {
        threads
    } else {
        rayon::current_num_threads()
    };
    let spans = cut_into_spans(work, workers * 64);
    let in_flight = InFlight::new(spans.len());
    let finished = AtomicBool::new(false);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut next = REPORT_EVERY;
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                let elapsed = started.elapsed();
                if elapsed < next {
                    continue;
                }
                next = elapsed + REPORT_EVERY;
                let seen = counts.seen.load(Ordering::Relaxed);
                eprintln!(
                    "... {seen} records, {:.0}/s",
                    seen as f64 / elapsed.as_secs_f64()
                );
                let now = elapsed.as_millis() as u64;
                for worker in 0..in_flight.record.len() {
                    let record = in_flight.record[worker].load(Ordering::Relaxed);
                    if record == u64::MAX {
                        continue;
                    }
                    let since =
                        now.saturating_sub(in_flight.started[worker].load(Ordering::Relaxed));
                    if since < TOO_LONG.as_millis() as u64 {
                        continue;
                    }
                    // Say it once per record, not once per look.
                    if in_flight.reported[worker].swap(record, Ordering::Relaxed) == record {
                        continue;
                    }
                    eprintln!("still on record {record} after {} s", since / 1000);
                }
                let _ = std::io::stderr().flush();
            }
        });

        spans.par_iter().enumerate().for_each(|(worker, span)| {
            let local = Counts::default();
            let mut at = 0usize;
            let mut record = span.first_record;
            while at < span.bytes.len() {
                let line = match memchr::memchr(b'\n', &span.bytes[at..]) {
                    Some(offset) => &span.bytes[at..at + offset],
                    None => &span.bytes[at..],
                };
                at += line.len() + 1;
                record += 1;
                if line.is_empty() {
                    continue;
                }
                in_flight.taking(worker, record, started.elapsed());
                let Ok(line) = std::str::from_utf8(line) else {
                    eprintln!("record {record} is not text");
                    continue;
                };
                let mut fields = line.split('\t');
                let (Some(cid), Some(smiles)) = (fields.next(), fields.next()) else {
                    panic!("malformed corpus record {record}");
                };
                check_catching_panics(cid, smiles, &rules, &local, &options);
                if record % 4096 == 0 {
                    counts.merge(&local);
                    local.reset();
                }
            }
            in_flight.record[worker].store(u64::MAX, Ordering::Relaxed);
            counts.merge(&local);
        });

        finished.store(true, Ordering::Relaxed);
    });

    counts.report(started.elapsed().as_secs_f64());
    if [
        &counts.no_answer,
        &counts.formula,
        &counts.reparse,
        &counts.idempotence,
        &counts.canonical_formula,
        &counts.canonical_reparse,
        &counts.canonical_idempotence,
        &counts.canonical_permutation,
        &counts.tautomer_permutation,
        &counts.key_permutation,
        &counts.key_canonical,
        &counts.key,
        &counts.panics,
        &counts.oversize,
        &counts.nonconverged,
    ]
    .iter()
    .any(|count| count.load(Ordering::Relaxed) != 0)
    {
        std::process::exit(1);
    }
}

/// One worker's share of the corpus: a span of whole records, and how many
/// records come before it so that a stall can name the line.
struct Span<'a> {
    bytes: &'a [u8],
    first_record: u64,
}

/// Cuts the corpus into roughly `pieces` spans, each ending on a record
/// boundary.
///
/// The record numbers are what the reporter prints, so they are counted rather
/// than guessed: one pass over the newlines of each cut point, which costs a
/// scan of the file and is what makes a stalled record findable by `sed -n`.
fn cut_into_spans(corpus: &[u8], pieces: usize) -> Vec<Span<'_>> {
    if corpus.is_empty() {
        return Vec::new();
    }
    let pieces = pieces.max(1);
    let stride = corpus.len().div_ceil(pieces);

    let mut cuts = vec![0usize];
    let mut at = stride.min(corpus.len());
    while at < corpus.len() {
        match memchr::memchr(b'\n', &corpus[at..]) {
            Some(offset) => {
                let cut = at + offset + 1;
                if cut >= corpus.len() {
                    break;
                }
                cuts.push(cut);
                at = (cut + stride).min(corpus.len());
            }
            None => break,
        }
    }
    cuts.push(corpus.len());

    // How many records precede each cut, counted in parallel: it is one pass
    // over the bytes and it is what lets the reporter name a line number.
    let counts: Vec<u64> = cuts
        .windows(2)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|window| memchr::memchr_iter(b'\n', &corpus[window[0]..window[1]]).count() as u64)
        .collect();

    let mut before = 0u64;
    let mut spans = Vec::with_capacity(counts.len());
    for (window, records) in cuts.windows(2).zip(&counts) {
        spans.push(Span {
            bytes: &corpus[window[0]..window[1]],
            first_record: before,
        });
        before += records;
    }
    spans
}

// Isolate a bad record while preserving its identity. The default panic hook
// still prints the cause; a nonzero exit status makes failures machine-visible.
fn check_catching_panics(
    cid: &str,
    smiles: &str,
    rules: &TautomerRules,
    counts: &Counts,
    options: &Options,
) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        check(cid, smiles, rules, counts, options);
    }))
    .is_err()
    {
        counts.panics.fetch_add(1, Ordering::Relaxed);
        eprintln!("panic\t{cid}\t{smiles}");
    }
}
