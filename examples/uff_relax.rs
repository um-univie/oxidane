//! Scratch driver for the xtb validation: embed without relaxing, print the
//! start, relax with this crate's UFF (or MMFF94), print the result.
use chemistry_consts::ElementProperties;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::embed::{Embed3D, EmbedConfig};
use oxidane::molecule::forcefield::{Field, ForceField, mmff};
use oxidane::prelude::*;

fn main() {
    let smiles = std::env::args()
        .nth(1)
        .expect("usage: uff_relax <SMILES> [uff|mmff|best]");
    let scheme = std::env::args().nth(2).unwrap_or_else(|| "uff".into());
    let config = EmbedConfig {
        relax: false,
        ..Default::default()
    };
    let mut molecule = MolecularGraph::from_smiles_single(&smiles)
        .expect("parse")
        .embed_3d_with_config(&config)
        .expect("embed");
    println!("START");
    print(&molecule);
    let field = match scheme.as_str() {
        "uff" => ForceField::new(&molecule).map(Field::Uff),
        "mmff" => mmff::ForceField::new(&molecule).map(Field::Mmff),
        "best" => Field::best_for(&molecule),
        _ => {
            eprintln!("unknown force field: {scheme}; expected uff, mmff, or best");
            std::process::exit(2);
        }
    }
    .unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });
    field.minimize(&mut molecule.positions, 5000);
    println!("END");
    print(&molecule);
}

fn print(molecule: &oxidane::molecule::molecule3d::Molecule3D) {
    for atom in 0..molecule.len() {
        let position = molecule.get_atom_position(atom).unwrap();
        let symbol = molecule.atomic_numbers()[atom]
            .atomic_symbol()
            .unwrap_or("X");
        println!(
            "{symbol} {:.9} {:.9} {:.9}",
            position.x, position.y, position.z
        );
    }
}
