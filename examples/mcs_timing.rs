//! Our side of the MCS timing table in `notes/rdkit-divergences.md`.
//! Pair with `python3 notes/mcs_timing.py`.
use molecules::graph_algorithms::mcsplit;
use molecules::prelude::*;
use std::time::Instant;

const STEROIDS: [&str; 7] = [
    "CCC(C)CC(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCCCC(C)C)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
    "CCC(C)CC(=O)O[C@@H]1C[C@H]2[C@@H](CC[C@@H]3C[C@H](OC(=O)CCCCCCCCCCC(C)C)CC[C@@]32C)[C@@H]2CC[C@H]([C@H](C)CCC(=O)O)[C@@]12C",
    "CCCC(C)C(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCCCC(C)C)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
    "CCCC(C)C(=O)O[C@@H]1C[C@H]2[C@@H](CC[C@@H]3C[C@H](OC(=O)CCCCCCCCCCC(C)C)CC[C@@]32C)[C@@H]2CC[C@H]([C@H](C)CCC(=O)O)[C@@]12C",
    "CCCCCC(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCCCC(C)C)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
    "CCCCCC(=O)O[C@@H]1C[C@H]2[C@@H](CC[C@@H]3C[C@H](OC(=O)CCCCCCCCCCC(C)C)CC[C@@]32C)[C@@H]2CC[C@H]([C@H](C)CCC(=O)O)[C@@]12C",
    "CCCCCCCCCC(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCC)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1"
];

fn main() {
    let mols: Vec<MolecularGraph> =
        STEROIDS.iter().map(|s| MolecularGraph::from_smiles_single(s).unwrap()).collect();
    println!("heavy atoms: {:?}", mols.iter()
        .map(|m| m.atomic_numbers().iter().filter(|&&z| z != 1).count()).collect::<Vec<_>>());
    for (i, j) in [(0usize, 1usize), (0, 5), (0, 2)] {
        let start = Instant::now();
        let r = mcsplit::find_maximum_common_edge_subgraph(&mols[i], &mols[j]).unwrap();
        println!(
            "  pair {i}x{j}: {} atoms / {} bonds  {:.1?}",
            r.atom_mapping.len(),
            r.bond_mapping.len(),
            start.elapsed()
        );
    }
}
