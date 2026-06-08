use oxidane::molecule::embed::{Embed3D, EmbedConfig};
use oxidane::prelude::*;

fn check_relaxed_bonds(smiles: &str, reference: &[(usize, usize, f64)]) {
    let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
    for seed in [0, 42, EmbedConfig::default().seed] {
        let config = EmbedConfig {
            seed,
            ..Default::default()
        };
        let embedded = molecule.embed_3d_with_config(&config).unwrap();
        for &(first, second, expected) in reference {
            let length = (embedded.positions[first] - embedded.positions[second]).length();
            assert!(
                (length - expected).abs() < 0.003,
                "{smiles}, seed {seed}, bond {first}-{second}: {length}, reference {expected}"
            );
        }
    }
}

#[test]
fn relaxed_spiro_bonds_can_shorten_beyond_the_initial_distance_window() {
    // MMFF-relaxed reference in bounds_reference_geometry.tsv. The initial
    // 0-2 window started at 1.5075 A when this regression was added; the
    // final bond must be allowed to reach 1.4572 A. Do not pin that initial
    // window: a future bounds improvement should leave this test valid.
    check_relaxed_bonds(
        "C1CC12CC2",
        &[
            (0, 1, 1.5350),
            (0, 2, 1.4572),
            (1, 2, 1.4572),
            (2, 3, 1.4572),
            (2, 4, 1.4572),
            (3, 4, 1.5350),
        ],
    );
}

#[test]
fn relaxed_fused_small_rings_reach_the_reference_bond_lengths() {
    // Lowest of eight RDKit 2026.09.1 ETKDGv3 conformers, seed 20261019,
    // minimized to convergence with MMFF94. See docs/SMALL_RING_BOUNDS.md.
    check_relaxed_bonds(
        "C1CC2CC12",
        &[
            (0, 1, 1.5438),
            (1, 2, 1.4805),
            (2, 3, 1.4965),
            (3, 4, 1.4965),
            (0, 4, 1.4805),
            (2, 4, 1.4937),
        ],
    );
}
