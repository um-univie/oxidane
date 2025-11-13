//! Reads one SMILES per line from stdin, prints a digest of its circular
//! fingerprint (or `ERROR`/`PANIC`) per line. Used to check that a fingerprint
//! is a property of the compound and not of how it was written.
use oxidane::molecule::crippen::crippen_types;
use oxidane::molecule::featurize::{ATOM_FEATURE_WIDTH, graph_features};
use oxidane::molecule::fingerprint::{
    DEFAULT_MAX_DISTANCE, StableHasher, atom_pair_fingerprint, circular_fingerprint,
    map4_shingles, maccs_keys, topological_torsion_fingerprint,
};
use oxidane::molecule::descriptors::MolecularDescriptors;
use oxidane::prelude::*;
use std::io::{self, BufRead, Write};

fn main() {
    // `circular:<radius>`, `pairs`, `torsions`, `features`, `map4`, `crippen`
    // or `maccs`.
    let kind = std::env::args().nth(1).unwrap_or_else(|| "circular:2".into());
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            let _ = writeln!(out);
            continue;
        }
        let digest = std::panic::catch_unwind(|| {
            #[allow(clippy::needless_return)]
            // Every fragment, not just the first: which fragment comes first
            // is a property of the string rather than of what is in the vial.
            let fragments = MolecularGraph::from_smiles(smiles)?;
            let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
            if kind == "features" {
                // Rows come out in the molecule's atom order, which is a
                // property of the string; the multiset of them is not.
                let features = graph_features(&molecule);
                let mut rows: Vec<Vec<u32>> = (0..features.atom_count)
                    .map(|row| {
                        features
                            .atom_row(row)
                            .iter()
                            .map(|value| value.to_bits())
                            .collect()
                    })
                    .collect();
                rows.sort_unstable();
                let mut hasher = StableHasher::new(1);
                hasher.write_u32(ATOM_FEATURE_WIDTH as u32);
                for row in rows {
                    for value in row {
                        hasher.write_u32(value);
                    }
                }
                return Ok(hasher.finish_u64());
            }
            if kind == "maccs" {
                let keys = maccs_keys(&molecule);
                let mut hasher = StableHasher::new(3);
                for bit in keys.set_bits() {
                    hasher.write_u32(bit as u32);
                }
                return Ok(hasher.finish_u64());
            }
            if kind == "crippen" {
                // The classes come out in atom order; sorting them makes the
                // digest a property of the compound. The two sums go in as
                // well, since they are what anyone actually reads.
                let mut labels: Vec<&str> = crippen_types(&molecule)
                    .into_iter()
                    .map(|kind| kind.label())
                    .collect();
                labels.sort_unstable();
                let mut hasher = StableHasher::new(2);
                for label in labels {
                    for byte in label.bytes() {
                        hasher.write_u8(byte);
                    }
                    hasher.write_u8(0);
                }
                let (log_p, refractivity) = molecule.crippen_descriptors();
                hasher.write_u64((log_p * 1e6).round() as i64 as u64);
                hasher.write_u64((refractivity * 1e6).round() as i64 as u64);
                return Ok(hasher.finish_u64());
            }
            let fingerprint = match kind.split_once(':') {
                Some(("circular", radius)) => {
                    circular_fingerprint(&molecule, radius.parse().unwrap_or(2)).fingerprint
                }
                _ if kind == "pairs" => atom_pair_fingerprint(&molecule, DEFAULT_MAX_DISTANCE),
                _ if kind == "torsions" => topological_torsion_fingerprint(&molecule),
                _ if kind == "map4" => map4_shingles(&molecule, 2, DEFAULT_MAX_DISTANCE),
                _ => circular_fingerprint(&molecule, 2).fingerprint,
            };
            let mut hasher = StableHasher::new(0);
            for &(id, count) in fingerprint.entries() {
                hasher.write_u32(id);
                hasher.write_u32(count);
            }
            Ok::<u64, oxidane::io::SMILESParseError>(hasher.finish_u64())
        });
        match digest {
            Ok(Ok(value)) => {
                let _ = writeln!(out, "{value:016x}");
            }
            Ok(Err(error)) => {
                let _ = writeln!(out, "ERROR: {error:?}");
            }
            Err(_) => {
                let _ = writeln!(out, "PANIC");
            }
        }
    }
}
