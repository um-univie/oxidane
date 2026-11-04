//! Compare optional 2D cleanup: cargo run --release --example draw_cleanup -- /tmp/cleanup.html
use oxidane::prelude::*;
use std::{fmt::Write, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cocaine SMILES: https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=2286
    let cases = [
        (
            "Cocaine",
            "COC(=O)[C@H]1[C@H](C[C@H]2N([C@@H]1CC2)C)OC(=O)c1ccccc1",
        ),
        ("Bridged bicyclic core", "C1CC2CCC1C2"),
        (
            "Crowded aryl branches",
            "CC(C)(C)c1cc(C(C)(C)C)cc(C(C)(C)C)c1",
        ),
        ("Spiro system", "C1CCC2(CC1)CCCC2"),
        ("Fused aromatic rings", "c1ccc2ccccc2c1"),
        ("Cyclic alkene", "C1CC/C=C/CCC1"),
    ];
    let mut html = String::from(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Oxidane · Optional 2D cleanup</title><style>body{background:#f1f5f9;color:#243247;font:15px system-ui;margin:0}main{max-width:1100px;margin:48px auto;padding:24px}h1{font-size:34px;letter-spacing:-1px}h2{font-size:18px;margin-top:36px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden}h3{font-size:14px;margin:20px 24px 0;color:#64748b}svg{display:block;width:100%;height:auto}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style><main><h1>Optional 2D cleanup</h1><p>Fast ring layout alongside a refinement that opens crowded regions while preserving stereochemistry.</p>",
    );
    for (name, smiles) in cases {
        let m = MolecularGraph::from_smiles_single(smiles).unwrap();
        write!(html, "<h2>{name}</h2><div class=\"pair\">")?;
        for cleanup in [false, true] {
            let options = DrawOptions {
                width: 500,
                height: 330,
                cleanup,
                ..Default::default()
            };
            write!(
                html,
                "<article><h3>{}</h3>{}</article>",
                if cleanup {
                    "Cleanup enabled"
                } else {
                    "Default layout"
                },
                m.to_svg_with_options(&options)
            )?;
            let start = Instant::now();
            for _ in 0..100 {
                std::hint::black_box(m.to_svg_with_options(&options));
            }
            eprintln!(
                "{name}, cleanup={cleanup}: {:.1} µs/render",
                start.elapsed().as_micros() as f64 / 100.0
            );
        }
        html.push_str("</div>");
    }
    html.push_str("</main></html>");
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cleanup.html".into());
    std::fs::write(&path, html)?;
    eprintln!("Wrote {path}");
    Ok(())
}
