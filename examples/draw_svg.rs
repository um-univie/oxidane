//! Render a SMILES string: cargo run --release --example draw_svg -- 'CC(=O)Oc1ccccc1C(=O)O' aspirin.svg [--cleanup]
use oxidane::prelude::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cleanup = std::env::args().any(|arg| arg == "--cleanup");
    let args: Vec<_> = std::env::args().filter(|arg| arg != "--cleanup").collect();
    let smiles = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("CC(=O)Oc1ccccc1C(=O)O");
    let path = args.get(2).map(String::as_str).unwrap_or("molecule.svg");
    let fragments = MolecularGraph::from_smiles(smiles).map_err(|e| format!("{e:?}"))?;
    let molecule = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
    let options = DrawOptions {
        cleanup,
        ..Default::default()
    };
    std::fs::write(path, molecule.to_svg_with_options(&options))?;
    eprintln!("Wrote {path}");
    Ok(())
}
