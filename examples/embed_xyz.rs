//! Embeds a SMILES string into 3D and prints it as plain XYZ (element +
//! coordinates, no bond info) to stdout. Used by `tests/embed_oracle.py` to
//! cross-check `Embed3D` against RDKit's ETKDG: RDKit supplies the bond
//! connectivity/order (parsed from the same SMILES) and this program
//! supplies only the coordinates, since the crate has no SDF/molfile writer.
use chemistry_consts::ElementProperties;
use oxidane::prelude::*;

fn main() {
    let smiles = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: embed_xyz <SMILES>");
        std::process::exit(2);
    });

    let mol = match MolecularGraph::from_smiles_single(&smiles) {
        Ok(mol) => mol,
        Err(e) => {
            eprintln!("parse error: {e:?}");
            std::process::exit(1);
        }
    };

    let embedded = match mol.embed_3d() {
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
