//! Reads an SDF and prints one SMILES per molecule, for comparison against
//! whatever produced the file.
//!
//! Paired with `notes/sdf_read_oracle.py`, which writes MOL blocks from a list
//! of SMILES with RDKit, runs this, and feeds our output back through RDKit's
//! canonicaliser. What that measures is whether a molecule survives the trip
//! out to a MOL block and back — the hydrogens the block leaves implicit, the
//! aromaticity it cannot write, and the stereochemistry it puts in the wedge
//! column.
//!
//! ```text
//! python3 notes/sdf_read_oracle.py tests/smiles.txt
//! ```

use molecules::io::{SdfParseOptions, ToSMILES};
use molecules::prelude::Molecule3D;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments
        .next()
        .expect("usage: sdf_read_diff <file.sdf> [--no-hydrogens] [--no-aromaticity]");
    let flags: Vec<String> = arguments.collect();

    let options = SdfParseOptions {
        hydrogens: !flags.iter().any(|f| f == "--no-hydrogens"),
        stereochemistry: !flags.iter().any(|f| f == "--no-stereochemistry"),
        aromaticity: if flags.iter().any(|f| f == "--no-aromaticity") {
            None
        } else {
            SdfParseOptions::default().aromaticity
        },
    };

    match Molecule3D::from_sdf_with(&path, options) {
        Ok(molecules) => {
            for molecule in &molecules {
                println!("{}", molecule.to_smiles());
            }
        }
        // One line per molecule either way, so a failure does not shift every
        // later comparison onto the wrong input.
        Err(error) => println!("ERROR {error:?}"),
    }
}
