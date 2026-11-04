//! Prints this crate's MMFF94 energy, term by term, for a SMILES embedded by
//! `embed_3d`, alongside the XYZ -- so `tests/mmff_energy_oracle.py` can ask
//! RDKit what it thinks of the same coordinates.
//!
//! Term by term rather than as a total, because a total that agrees can be
//! two errors cancelling, and because MMFF is one of the few force fields
//! whose reference implementation will switch its terms off one at a time.
use chemistry_consts::ElementProperties;
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::embed::{Embed3D, EmbedConfig};
use oxidane::molecule::forcefield::mmff;
use oxidane::prelude::*;

fn main() {
    let smiles = std::env::args()
        .nth(1)
        .expect("usage: mmff_energy <SMILES>");
    let Ok(molecule) = MolecularGraph::from_smiles_single(&smiles) else {
        println!("SKIPPED this crate does not parse the SMILES");
        return;
    };
    // Unrelaxed: what is wanted is a geometry neither force field has had a
    // say in, so that the comparison is of the energies and not of two
    // minimisers landing in the same place.
    let config = EmbedConfig {
        relax: false,
        ..Default::default()
    };
    let Ok(embedded) = molecule.embed_3d_with_config(&config) else {
        println!("SKIPPED this crate does not embed the molecule");
        return;
    };
    let field = match mmff::ForceField::new(&embedded) {
        Ok(field) => field,
        Err(error) => {
            println!("SKIPPED {error}");
            return;
        }
    };
    let energies = field.energies(&embedded.positions);

    println!("BOND {:.6}", energies.bond);
    println!("ANGLE {:.6}", energies.angle);
    println!("STRETCH_BEND {:.6}", energies.stretch_bend);
    println!("OOP {:.6}", energies.out_of_plane);
    println!("TORSION {:.6}", energies.torsion);
    println!("VDW {:.6}", energies.van_der_waals);
    println!("ELE {:.6}", energies.electrostatic);
    println!("TOTAL {:.6}", energies.total());
    if std::env::args().any(|argument| argument == "--torsions") {
        for (i, j, k, l, kind, v1, v2, v3) in field.torsion_terms() {
            println!("T {i} {j} {k} {l} {kind} {v1:.4} {v2:.4} {v3:.4}");
        }
    }
    println!("{}", embedded.len());
    for atom in 0..embedded.len() {
        let position = embedded.get_atom_position(atom).unwrap();
        let symbol = embedded.atomic_numbers()[atom]
            .atomic_symbol()
            .unwrap_or("X");
        println!(
            "{symbol} {:.6} {:.6} {:.6}",
            position.x, position.y, position.z
        );
    }
}
