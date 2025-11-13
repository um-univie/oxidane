//! Times fingerprint generation alone: every molecule is parsed into memory
//! first, so the loop that is measured does nothing but featurize.
use oxidane::molecule::descriptors::MolecularDescriptors;
use oxidane::molecule::featurize::graph_features;
use oxidane::molecule::fingerprint::{
    DEFAULT_MAX_DISTANCE, MinHash, atom_pair_fingerprint, circular_fingerprint,
    map4_fingerprint, maccs_keys, standard_invariants, topological_torsion_fingerprint,
};
use oxidane::molecule::shape::{shape_descriptors, usr, usrcat};
use oxidane::prelude::*;
use std::io::{self, BufRead};
use std::time::Instant;

fn main() {
    let radius: usize = std::env::args()
        .nth(1)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(2);

    let stdin = io::stdin();
    let molecules: Vec<MolecularGraph> = stdin
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let fragments = MolecularGraph::from_smiles(line.trim()).ok()?;
            Some(MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>()))
        })
        .collect();
    let atoms: usize = molecules.iter().map(|molecule| molecule.len()).sum();
    eprintln!(
        "{} molecules, {atoms} atoms including hydrogens",
        molecules.len()
    );

    for _ in 0..2 {
        let start = Instant::now();
        let mut checksum = 0u64;
        for molecule in &molecules {
            let invariants = standard_invariants(molecule);
            checksum ^= invariants.len() as u64;
        }
        let elapsed = start.elapsed();
        eprintln!(
            "invariants only : {:>8.3?}  {:>7.2} us/mol  (checksum {checksum})",
            elapsed,
            elapsed.as_secs_f64() * 1e6 / molecules.len() as f64
        );

        let start = Instant::now();
        let mut checksum = 0u64;
        for molecule in &molecules {
            checksum ^= circular_fingerprint(molecule, radius).fingerprint.len() as u64;
        }
        let elapsed = start.elapsed();
        eprintln!(
            "circular r={radius}     : {:>8.3?}  {:>7.2} us/mol  (checksum {checksum})",
            elapsed,
            elapsed.as_secs_f64() * 1e6 / molecules.len() as f64
        );

        for (label, run) in [
            (
                "atom pairs      ",
                &(|molecule: &MolecularGraph| {
                    atom_pair_fingerprint(molecule, DEFAULT_MAX_DISTANCE).len() as u64
                }) as &dyn Fn(&MolecularGraph) -> u64,
            ),
            ("torsions        ", &|molecule: &MolecularGraph| {
                topological_torsion_fingerprint(molecule).len() as u64
            }),
            ("graph features  ", &|molecule: &MolecularGraph| {
                graph_features(molecule).atom_count as u64
            }),
            ("map4 r=2 1024  ", &|molecule: &MolecularGraph| {
                map4_fingerprint(molecule, 2, DEFAULT_MAX_DISTANCE, 1024).values()[0] as u64
            }),
            ("maccs           ", &|molecule: &MolecularGraph| {
                maccs_keys(molecule).count_ones() as u64
            }),
            ("usr             ", &|molecule: &MolecularGraph| {
                // Positions are the caller's; a plain two-dimensional layout is
                // enough to time the arithmetic.
                let positions: Vec<oxidane::vector::Vector3D> = (0..molecule.len())
                    .map(|atom| oxidane::vector::Vector3D {
                        x: atom as f64 * 0.7,
                        y: (atom % 5) as f64,
                        z: (atom % 3) as f64,
                    })
                    .collect();
                usr(&positions)[0] as u64
            }),
            ("usrcat          ", &|molecule: &MolecularGraph| {
                let positions: Vec<oxidane::vector::Vector3D> = (0..molecule.len())
                    .map(|atom| oxidane::vector::Vector3D {
                        x: atom as f64 * 0.7,
                        y: (atom % 5) as f64,
                        z: (atom % 3) as f64,
                    })
                    .collect();
                usrcat(molecule, &positions)[0] as u64
            }),
            ("shape 3d        ", &|molecule: &MolecularGraph| {
                let positions: Vec<oxidane::vector::Vector3D> = (0..molecule.len())
                    .map(|atom| oxidane::vector::Vector3D {
                        x: atom as f64 * 0.7,
                        y: (atom % 5) as f64,
                        z: (atom % 3) as f64,
                    })
                    .collect();
                shape_descriptors(molecule, &positions).first_moment as u64
            }),
            ("crippen         ", &|molecule: &MolecularGraph| {
                molecule.crippen_descriptors().0 as u64
            }),
            ("minhash 256     ", &|molecule: &MolecularGraph| {
                let fingerprint = circular_fingerprint(molecule, 2).fingerprint;
                MinHash::of(&fingerprint, 256).values()[0] as u64
            }),
            ("descriptors     ", &|molecule: &MolecularGraph| {
                (molecule.topological_polar_surface_area() as u64)
                    ^ molecule.rotatable_bond_count() as u64
                    ^ molecule.ring_count() as u64
            }),
        ] {
            let start = Instant::now();
            let mut checksum = 0u64;
            for molecule in &molecules {
                checksum ^= run(molecule);
            }
            let elapsed = start.elapsed();
            eprintln!(
                "{label}: {:>8.3?}  {:>7.2} us/mol  (checksum {checksum})",
                elapsed,
                elapsed.as_secs_f64() * 1e6 / molecules.len() as f64
            );
        }
    }
}
