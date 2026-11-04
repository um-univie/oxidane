//! Like `embed_xyz`, but with the conformer-search budget exposed:
//! `embed_budget <SMILES> [attempts] [seed] [nosearch|norelax]`. Used to
//! compare this crate's embedder against RDKit's ETKDG at an equal number of
//! trials, and -- with `norelax` -- to see the distance-geometry structure
//! before any force field has touched it, which is where the basin the
//! relaxation then falls into is actually decided.
use chemistry_consts::ElementProperties;
use oxidane::molecule::embed::EmbedConfig;
use oxidane::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let smiles = args.next().unwrap_or_else(|| {
        eprintln!("usage: embed_budget <SMILES> [attempts] [seed]");
        std::process::exit(2);
    });
    let attempts = args.next().and_then(|a| a.parse().ok()).unwrap_or(5);
    let seed = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(0xE7D6_5EED);
    let mode = args.next();
    let mode = mode.as_deref();
    let search = mode != Some("nosearch") && mode != Some("norelax");
    let relax = mode != Some("norelax");

    let mol = match MolecularGraph::from_smiles_single(&smiles) {
        Ok(mol) => mol,
        Err(e) => {
            eprintln!("parse error: {e:?}");
            std::process::exit(1);
        }
    };

    // `nosearch` is `EmbedConfig::fast`'s configuration at the requested
    // budget: no torsion search, and each trial refined against every
    // restraint at once, since nothing downstream will fix a torsion.
    let config = EmbedConfig {
        seed,
        max_attempts: attempts,
        torsion_search: search,
        combined_refinement: !search,
        relax,
        ..Default::default()
    };
    let embedded = match mol.embed_3d_with_config(&config) {
        Ok(embedded) => embedded,
        Err(e) => {
            eprintln!("embed error: {e}");
            std::process::exit(1);
        }
    };

    println!("{}", embedded.len());
    println!("{smiles}");
    for i in 0..embedded.len() {
        let symbol = embedded.atomic_numbers()[i].atomic_symbol().unwrap_or("X");
        let p = embedded.get_atom_position(i).unwrap();
        println!("{symbol} {:.6} {:.6} {:.6}", p.x, p.y, p.z);
    }
}
