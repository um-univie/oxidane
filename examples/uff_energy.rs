//! Prints this crate's UFF energy for a SMILES embedded by `embed_3d`,
//! alongside the XYZ, so `tests/uff_oracle.py` can ask RDKit what it thinks
//! of the same coordinates.
use chemistry_consts::ElementProperties;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::embed::Embed3D;
use oxidane::molecule::forcefield::ForceField;
use oxidane::prelude::*;

fn main() {
    let smiles = std::env::args().nth(1).expect("usage: uff_energy <SMILES>");
    let molecule = MolecularGraph::from_smiles_single(&smiles).expect("parse");
    let embedded = molecule.embed_3d().expect("embed");
    let field = ForceField::new(&embedded).expect("type");
    println!("{}", embedded.len());
    println!("ENERGY {:.6}", field.energy(&embedded.positions));
    for atom in 0..embedded.len() {
        let p = embedded.get_atom_position(atom).unwrap();
        let symbol = embedded.atomic_numbers()[atom]
            .atomic_symbol()
            .unwrap_or("X");
        println!("{symbol} {:.6} {:.6} {:.6}", p.x, p.y, p.z);
    }
}
