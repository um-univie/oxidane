//! Timing for CIP assignment and for the MCES that consumes it.
use molecules::graph_algorithms::mcsplit;
use molecules::prelude::*;
use std::time::Instant;

const STEROID: &str =
    "CC(=O)O[C@H]1CC[C@]2(C)[C@@H](CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H]3CC[C@@H]2OC(C)=O)C1";
const MORPHINE: &str = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5";
const CHOLESTEROL: &str =
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C";

fn main() {
    for (name, smiles) in
        [("cholesterol", CHOLESTEROL), ("morphine", MORPHINE), ("steroid diester", STEROID)]
    {
        let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
        let heavy = mol.atomic_numbers().iter().filter(|&&z| z != 1).count();
        let start = Instant::now();
        let rounds = 20;
        for _ in 0..rounds {
            std::hint::black_box(mol.assign_cip_configs());
        }
        println!("{name:16} {heavy:3} heavy atoms   CIP {:>9.1?}", start.elapsed() / rounds);
    }

    let a = MolecularGraph::from_smiles_single(STEROID).unwrap();
    let b = MolecularGraph::from_smiles_single(CHOLESTEROL).unwrap();
    let start = Instant::now();
    let result = mcsplit::find_maximum_common_edge_subgraph(&a, &b).unwrap();
    println!(
        "\nMCES steroid x cholesterol: {} atoms / {} bonds in {:.1?}",
        result.atom_mapping.len(),
        result.bond_mapping.len(),
        start.elapsed()
    );
}
