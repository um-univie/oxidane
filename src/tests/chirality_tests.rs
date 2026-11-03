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

    /// Guards against `determine_tetrahedral_chirality` returning `TH(1)`/`TH(2)`
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

        // And the two are distinguishable.
        assert_ne!(cis_classes[1], trans_classes[1]);

        // Cross-checked against RDKit's `AssignStereochemistryFrom3D`, which
        // canonicalises these two files to `C/C=C\\C` and `C/C=C/C`; both of
        // ours canonicalise to the same pair.
        assert_eq!(cis.to_smiles(), "C(=C/C)/C");
        assert_eq!(trans.to_smiles(), "C(=C\\C)/C");
    }

    /// Every ordering of `items`, in lexicographic order of positions.
    fn permutations(items: usize) -> Vec<Vec<usize>> {
        if items == 0 {
            return vec![Vec::new()];
        }
        let mut all = Vec::new();
        for rest in permutations(items - 1) {
            for slot in 0..=rest.len() {
                let mut next = rest.clone();
                next.insert(slot, items - 1);
                all.push(next);
            }
        }
        all
    }

    /// `update_chirality_after_relabeling_in_place` returns the class a
    /// centre must carry once its neighbours are listed in their new label
    /// order. Writing the centre that way with that class is therefore the
    /// same molecule as before, which the canonical SMILES can check: for
    /// every TB and SP class under every relabelling of the neighbours, and
    /// every OH class under a spread of them.
    #[test]
    fn a_relabelled_tb_sp_or_oh_centre_keeps_its_geometry() {
        use crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
        use crate::molecule::chirality::update_chirality_after_relabeling_in_place;

        let canon = |smiles: &str| canonicalize_and_sort_smiles(smiles, false).unwrap();
        let write = |centre: &str, tag: &str, class: u8, listed: &[&str]| {
            let middle: String = listed[1..listed.len() - 1]
                .iter()
                .map(|ligand| format!("({ligand})"))
                .collect();
            format!("{}[{centre}@{tag}{class}]{middle}{}", listed[0], listed[listed.len() - 1])
        };
        let cases: [(&str, &str, u8, &[&str], usize); 3] = [
            ("As", "TB", 20, &["S", "F", "Cl", "Br", "N"], 1),
            ("Pt", "SP", 3, &["F", "Cl", "Br", "I"], 1),
            ("Co", "OH", 30, &["S", "F", "Cl", "Br", "N", "I"], 37),
        ];
        for (centre, tag, classes, ligands, stride) in cases {
            // Ligand atoms are 0 and 2.., the centre 1, as the SMILES is written.
            let atoms: Vec<usize> = std::iter::once(0).chain(2..=ligands.len()).collect();
            for class in 1..=classes {
                let written = write(centre, tag, class, ligands);
                let expected = canon(&written);
                for order in permutations(ligands.len()).into_iter().step_by(stride) {
                    let mut labeling: Vec<usize> = (0..=ligands.len()).collect();
                    for (position, &rank) in order.iter().enumerate() {
                        labeling[atoms[position]] = atoms[rank];
                    }
                    let mut molecule = MolecularGraph::from_smiles_single(&written).unwrap();
                    update_chirality_after_relabeling_in_place(&mut molecule, &labeling);
                    let relabelled = match molecule.chiral_classes().unwrap()[1] {
                        ChiralClass::TB(n) | ChiralClass::SP(n) | ChiralClass::OH(n) => n,
                        other => panic!("{written}: became {other:?}"),
                    };
                    let mut listing: Vec<usize> = (0..ligands.len()).collect();
                    listing.sort_by_key(|&position| order[position]);
                    let listed: Vec<&str> = listing.iter().map(|&position| ligands[position]).collect();
                    let rewritten = write(centre, tag, relabelled, &listed);
                    assert_eq!(
                        canon(&rewritten),
                        expected,
                        "{written} relabelled by {order:?} came back as {rewritten}"
                    );
                }
            }
        }
    }

    /// Parity cases that are easy to get wrong, checked against the
    /// OpenSMILES definitions directly. TB1 is the a-e axis with b, c, d
    /// anticlockwise seen from a: exchanging two equatorial neighbours or the
    /// two axial ones reverses that, a three-cycle of the equatorials does
    /// not. SP1 is the U path a-b-c-d round the square: exchanging a and c is
    /// a reflection of the square and leaves it a U.
    #[test]
    fn relabelling_follows_the_opensmiles_tb_and_sp_definitions() {
        use crate::molecule::chirality::update_chirality_after_relabeling_in_place;

        let relabel = |smiles: &str, labeling: &[usize]| {
            let mut molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
            update_chirality_after_relabeling_in_place(&mut molecule, labeling);
            molecule.chiral_classes().unwrap()[1]
        };
        let tb1 = "S[As@TB1](F)(Cl)(Br)N";
        assert_eq!(relabel(tb1, &[0, 1, 2, 4, 3, 5]), ChiralClass::TB(2));
        assert_eq!(relabel(tb1, &[5, 1, 2, 3, 4, 0]), ChiralClass::TB(2));
        assert_eq!(relabel(tb1, &[0, 1, 3, 4, 2, 5]), ChiralClass::TB(1));
        assert_eq!(relabel("F[Pt@SP1](Cl)(Br)I", &[3, 1, 2, 0, 4]), ChiralClass::SP(1));
    }
}
