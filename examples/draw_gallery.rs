//! Generate a visual review gallery and measure rendering (parsing excluded).
//! cargo run --release --example draw_gallery -- /tmp/molecules.html
use oxidane::prelude::*;
use std::{fmt::Write, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        ("Naphthalene", "c1ccc2ccccc2c1"),
        ("Spiro rings", "C1CCC2(CC1)CCCC2"),
        ("Benzonitrile", "N#Cc1ccccc1"),
        ("Acetylcholine", "CC(=O)OCC[N+](C)(C)C"),
        ("E alkene", "C/C=C/C"),
        ("Z alkene", "C/C=C\\C"),
        ("Lactic acid · @", "C[C@H](O)C(=O)O"),
        ("Lactic acid · @@", "C[C@@H](O)C(=O)O"),
        ("Two stereocenters", "C[C@H](O)[C@@H](C)O"),
        ("Four heavy substituents", "F[C@](Cl)(Br)I"),
        ("Isotope label", "[13CH3]C(=O)[O-]"),
        ("Disconnected salt", "[Na+].[O-]C(=O)c1ccccc1"),
    ];
    let molecules: Vec<_> = cases
        .iter()
        .map(|(_, s)| {
            let fragments = MolecularGraph::from_smiles(s).unwrap();
            MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>())
        })
        .collect();
    let mut html = String::from(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Oxidane · Molecular drawings</title><style>body{margin:0;background:#f1f5f9;color:#243247;font:15px system-ui}main{max-width:1250px;margin:60px auto;padding:0 24px}h1{font-size:36px;letter-spacing:-1px;margin-bottom:8px}p{color:#64748b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-top:32px}article{background:white;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden}h2{font-size:16px;margin:20px 24px 0}svg{display:block;width:100%;height:auto}</style><main><h1>Molecular drawings</h1><p>Oxidane · Deterministic 2D structures, rendered directly to SVG.</p><div class=\"grid\">",
    );
    for ((name, _), m) in cases.iter().zip(&molecules) {
        write!(html, "<article><h2>{name}</h2>{}</article>", m.to_svg())?;
    }
    html.push_str("</div><p>Stereochemistry shown with solid and hashed wedges and alkene geometry.</p></main></html>");
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "molecules.html".into());
    std::fs::write(&path, html)?;
    let start = Instant::now();
    for _ in 0..1000 {
        for m in &molecules {
            std::hint::black_box(m.to_svg());
        }
    }
    eprintln!(
        "Wrote {path}; {} renders: {:?} ({:.1} µs/molecule)",
        molecules.len() * 1000,
        start.elapsed(),
        start.elapsed().as_micros() as f64 / (molecules.len() * 1000) as f64
    );
    for size in [100, 1000] {
        let chain = MolecularGraph::from_smiles_single(&"C".repeat(size)).unwrap();
        let start = Instant::now();
        for _ in 0..100 {
            std::hint::black_box(chain.to_svg());
        }
        eprintln!(
            "{size}-carbon chain: {:.1} µs/render",
            start.elapsed().as_micros() as f64 / 100.0
        );
    }
    Ok(())
}
