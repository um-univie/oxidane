#[cfg(test)]
mod tests {
    use crate::io::parse_sdf_file;
    use crate::molecule::chirality::{ChiralClass, ChiralClassifier};
    use crate::prelude::*;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_is_chiral() {
        let mols = Molecule3D::from_xyz("tests/ethane.xyz").unwrap();
        let mut mol = mols.into_iter().next().unwrap();
        mol.identify_chiral_classes();
        assert!(!mol.is_chiral());
    }

    /// `determine_tetrahedral_chirality` previously returned `TH(1)`/`TH(2)`
    /// inverted relative to this crate's documented `ChiralClass::TH`
    /// convention (and RDKit) - found while building the `embed` module's
    /// distance-geometry embedder, which needed to read 3D geometry back
    /// into a chirality tag independently. A round-trip through
    /// `Molecule3D::to_smiles` (which calls `identify_chiral_classes` when
    /// a `Molecule3D` has no `chiral_classes` yet - e.g. one just read from
    /// an XYZ/SDF file) must reproduce the original tag.
    #[test]
    fn tetrahedral_chirality_round_trips_through_3d_and_back_to_smiles() {
        for smiles in ["[C@](F)(Cl)(Br)I", "[C@@](F)(Cl)(Br)I", "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            let mut embedded = mol.embed_3d().unwrap();
            embedded.chiral_classes = None; // simulate "freshly read from XYZ/SDF, no stereo tag yet"
            embedded.identify_chiral_classes();
            assert_eq!(
                embedded.to_smiles(),
                smiles,
                "SMILES -> embed_3d -> identify_chiral_classes -> to_smiles should reproduce the input tag"
            );
        }
    }

    #[test]
    fn test_square_planar_chirality() {
        let mols = Molecule3D::from_xyz("tests/cisplatin_sp2.xyz").unwrap();
        let mut cisplatin = mols.into_iter().next().unwrap();
        cisplatin.identify_chiral_classes();
        assert_eq!(cisplatin.chiral_classes().unwrap()[0], ChiralClass::SP(3));

        let mols = Molecule3D::from_xyz("tests/transplatin_sp1.xyz").unwrap();
        let mut transplatin = mols.into_iter().next().unwrap();
        transplatin.identify_chiral_classes();
        assert_eq!(transplatin.chiral_classes().unwrap()[0], ChiralClass::SP(1));

        let mols = Molecule3D::from_xyz("tests/transplatin_sp2.xyz").unwrap();
        let mut transplatin = mols.into_iter().next().unwrap();
        transplatin.identify_chiral_classes();
        assert_eq!(transplatin.chiral_classes().unwrap()[0], ChiralClass::SP(2));

        let mols = Molecule3D::from_xyz("tests/transplatin_sp3.xyz").unwrap();
        let mut transplatin = mols.into_iter().next().unwrap();
        transplatin.identify_chiral_classes();
        assert_eq!(transplatin.chiral_classes().unwrap()[0], ChiralClass::SP(3));
    }

    #[test]
    fn test_trigonal_bipyramidal_chirality() {
        let file = File::open("tests/ordered_molecules.sdf").unwrap();
        let reader = BufReader::new(file);
        let sdf_entries = parse_sdf_file(reader).unwrap();
        let mut chiral_classes: Vec<ChiralClass> = Vec::new();
        for sdf_entry in sdf_entries.iter() {
            let chiral_permutation = sdf_entry.data_fields.get("ChiralPermutation").unwrap();
            let chiral_permutation = chiral_permutation.parse::<u8>().unwrap();
            let chiral_type = sdf_entry.data_fields.get("ChiralType").unwrap();
            let chiral_class = match (chiral_type.as_str(), chiral_permutation) {
                ("TH", 1..=2) => ChiralClass::TH(chiral_permutation),
                ("SP", 1..=3) => ChiralClass::SP(chiral_permutation),
                ("TB", 1..=20) => ChiralClass::TB(chiral_permutation),
                ("OH", 1..=30) => ChiralClass::OH(chiral_permutation),
                ("AL", 1..=2) => ChiralClass::AL(chiral_permutation),
                _ => ChiralClass::None,
            };
            chiral_classes.push(chiral_class);
        }
        let mut pf5: Vec<Molecule3D> = Molecule3D::from_sdf_entries(sdf_entries).unwrap();

        for (mol, chiral_class) in pf5.iter_mut().zip(chiral_classes) {
            mol.identify_chiral_classes();
            assert_eq!(mol.chiral_classes().unwrap()[0], chiral_class);
        }
    }

    /// Cis and trans have to be told apart, and by the same convention the
    /// SMILES side uses.
    ///
    /// This used to assert the plain `Counterclockwise` and `Clockwise`
    /// classes, which are the TETRAHEDRAL ones, so a double bond came back
    /// carrying stereocentres. Worse, it asserted the same tag on atom 0 for
    /// both the cis and the trans file, which is a check that cannot fail for
    /// the reason it was written: under the convention, the lower-index end of
    /// a double bond is always `CisTransCounterclockwise` and all the
    /// information is in the other end.
    #[test]
    fn test_cis_trans_chirality() {
        let cis = Molecule3D::from_sdf("tests/cis_2_butene.sdf").unwrap().remove(0);
        let trans = Molecule3D::from_sdf("tests/trans_2_butene.sdf").unwrap().remove(0);

        let cis_classes = cis.chiral_classes().unwrap();
        let trans_classes = trans.chiral_classes().unwrap();

        // Same tag at both ends is cis, opposite tags are trans.
        assert_eq!(cis_classes[0], ChiralClass::CisTransCounterclockwise);
        assert_eq!(cis_classes[1], ChiralClass::CisTransCounterclockwise);
        assert_eq!(trans_classes[0], ChiralClass::CisTransCounterclockwise);
        assert_eq!(trans_classes[1], ChiralClass::CisTransClockwise);

        // And the two are actually distinguishable, which the old assertions
        // never established.
        assert_ne!(cis_classes[1], trans_classes[1]);

        // Cross-checked against RDKit's `AssignStereochemistryFrom3D`, which
        // canonicalises these two files to `C/C=C\\C` and `C/C=C/C`; both of
        // ours canonicalise to the same pair.
        assert_eq!(cis.to_smiles(), "C(=C/C)/C");
        assert_eq!(trans.to_smiles(), "C(=C\\C)/C");
    }
}
