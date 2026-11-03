//! Reading molecules from SDF.

use crate::io::{SdfParseOptions, ToSMILES, parse_sdf_file};
use crate::molecule::base::MoleculeCore;
use crate::prelude::Molecule3D;
use std::io::Cursor;
use std::path::Path;

/// Every per-atom vector has to have one entry per atom.
///
/// `from_sdf_entries` built the molecule with `..Default::default()`, and
/// `radical_states` is a plain `Vec` rather than an `Option`, so it came out
/// empty while the accessors index it directly — asking an SDF-derived
/// molecule anything about radicals, `to_smiles` included, panicked. The other
/// per-atom fields are `Option`, where `None` correctly means "nothing
/// recorded".
#[test]
fn per_atom_vectors_are_sized() {
    for file in ["tests/pentene.sdf", "tests/cis_2_butene.sdf", "tests/ordered_molecules.sdf"] {
        let molecules = Molecule3D::from_sdf(Path::new(file))
            .unwrap_or_else(|e| panic!("{file}: {e:?}"));
        assert!(!molecules.is_empty(), "{file}: no molecules");
        for (index, molecule) in molecules.iter().enumerate() {
            let atoms = molecule.atomic_numbers().len();
            assert_eq!(molecule.radical_states().len(), atoms, "{file} molecule {index}");
            assert_eq!(molecule.charges().len(), atoms, "{file} molecule {index}");
            assert_eq!(molecule.atom_bonds().len(), atoms, "{file} molecule {index}");
        }
    }
}

/// The paths that index those vectors have to survive an SDF-derived molecule.
#[test]
fn sdf_molecules_can_be_written_and_measured() {
    let molecules = Molecule3D::from_sdf(Path::new("tests/pentene.sdf")).unwrap();
    let molecule = &molecules[0];
    assert_eq!(molecule.to_smiles(), "C(C=CC)C");
    assert_eq!(molecule.degrees().len(), molecule.atomic_numbers().len());
}

/// The `M  CHG` / `M  RAD` / `M  ISO` property block.
///
/// None of the SDFs in `tests/` carries a property line other than `M  END`,
/// which is why the whole block went uninterpreted for so long without anything
/// noticing. These build the files in memory instead.
mod property_block {
    use super::*;

    /// One V2000 atom line, laid out on the columns the format fixes.
    ///
    /// Writing these by hand is how the off-by-one creeps in: the symbol field
    /// is three wide, the mass difference two, and the charge three, and the
    /// parser rejects any line shorter than 69 characters.
    fn atom_line(symbol: &str, mass_difference: i8, charge_code: u8) -> String {
        let line = format!(
            "{:>10.4}{:>10.4}{:>10.4} {:<3}{:>2}{:>3}{}",
            0.0,
            0.0,
            0.0,
            symbol,
            mass_difference,
            charge_code,
            "  0".repeat(10)
        );
        assert_eq!(line.len(), 69, "atom line off the column layout: {line:?}");
        line
    }

    /// An SDF holding one molecule with the given atoms, bonds and properties.
    fn sdf(atoms: &[String], bonds: &[&str], properties: &[&str]) -> String {
        let mut out = format!(
            "test\n  molecules\n\n{:>3}{:>3}  0     0  0  0  0  0  0999 V2000\n",
            atoms.len(),
            bonds.len()
        );
        for line in atoms {
            out.push_str(line);
            out.push('\n');
        }
        for line in bonds {
            out.push_str(line);
            out.push('\n');
        }
        for line in properties {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("M  END\n$$$$\n");
        out
    }

    /// Reads the one molecule, with perception switched off.
    ///
    /// These tests are about what the property block says, so the per-atom
    /// vectors are compared whole; hydrogens worked out from the valence model
    /// would append entries none of them is asking about, and aromatisation
    /// would rewrite bond orders they do not set.
    fn parse(content: &str) -> Molecule3D {
        let entries = parse_sdf_file(Cursor::new(content)).expect("parse failed");
        let options = SdfParseOptions {
            hydrogens: false,
            stereochemistry: false,
            aromaticity: None,
        };
        let mut molecules =
            Molecule3D::from_sdf_entries_with(entries, options).expect("conversion failed");
        assert_eq!(molecules.len(), 1);
        molecules.remove(0)
    }

    /// The atom block's charge column holds a code, not the charge.
    ///
    /// 1..=3 mean +3..=+1 and 5..=7 mean -1..=-3, so reading the column as the
    /// charge itself made every charged atom in every SDF wrong: a +1 nitrogen
    /// came back as +3 and a -1 oxygen as +5, which is not even a charge a
    /// second-row element can carry.
    #[test]
    fn the_atom_block_charge_column_is_a_code() {
        let molecule = parse(&sdf(
            &[atom_line("N", 0, 3), atom_line("O", 0, 5)],
            &["  1  2  1  0  0  0  0"],
            &[],
        ));
        assert_eq!(molecule.charges(), &[1, -1]);
    }

    /// Code 4 is a doublet radical and carries no charge at all.
    #[test]
    fn charge_code_four_is_a_radical() {
        let molecule = parse(&sdf(&[atom_line("C", 0, 4)], &[], &[]));
        assert_eq!(molecule.charges(), &[0]);
        assert_eq!(molecule.radical_states(), &[1]);
    }

    /// `M  CHG` supersedes the atom block rather than adding to it.
    ///
    /// Files commonly write both, the atom block entries being there for
    /// readers too old to know the property lines, and the two are allowed to
    /// disagree — the property line is the one that counts. This one gives
    /// atom 1 as +1 in the atom block and -1 in the property block.
    #[test]
    fn m_chg_supersedes_the_atom_block() {
        let molecule = parse(&sdf(
            &[atom_line("N", 0, 3), atom_line("O", 0, 5)],
            &["  1  2  1  0  0  0  0"],
            &["M  CHG  1   1  -1"],
        ));
        // Atom 2 is not listed, so its atom block charge is superseded by
        // nothing at all: the block replaces the whole assignment.
        assert_eq!(molecule.charges(), &[-1, 0]);
    }

    /// The first property line is the one that ends the bond block.
    ///
    /// The state machine enters its Properties state on the line it fails to
    /// read as a bond, and must record that line too: `M  CHG` is usually the
    /// first property line of a real file.
    #[test]
    fn the_first_property_line_is_kept() {
        let molecule = parse(&sdf(
            &[atom_line("N", 0, 0), atom_line("O", 0, 0)],
            &["  1  2  1  0  0  0  0"],
            &["M  CHG  2   1   1   2  -1"],
        ));
        assert_eq!(molecule.charges(), &[1, -1]);
    }

    /// A property line right after the atom block, with no bonds in between.
    #[test]
    fn the_first_property_line_is_kept_without_bonds() {
        let molecule = parse(&sdf(&[atom_line("O", 0, 0)], &[], &["M  CHG  1   1  -1"]));
        assert_eq!(molecule.charges(), &[-1]);
    }

    /// The format caps a property line at eight pairs, so larger molecules
    /// arrive split across several lines of the same kind, which accumulate.
    #[test]
    fn repeated_property_lines_accumulate() {
        let atoms: Vec<_> = (0..10).map(|_| atom_line("O", 0, 0)).collect();
        let molecule = parse(&sdf(
            &atoms,
            &[],
            &[
                "M  CHG  8   1  -1   2  -1   3  -1   4  -1   5  -1   6  -1   7  -1   8  -1",
                "M  CHG  2   9  -1  10  -1",
            ],
        ));
        assert_eq!(molecule.charges(), &[-1; 10]);
    }

    /// `M  RAD` values are codes too: singlet, doublet and triplet, which we
    /// store as the count of electrons occupying bonding slots.
    #[test]
    fn m_rad_is_read() {
        let atoms: Vec<_> = (0..3).map(|_| atom_line("C", 0, 0)).collect();
        let molecule = parse(&sdf(&atoms, &[], &["M  RAD  3   1   1   2   2   3   3"]));
        assert_eq!(molecule.radical_states(), &[2, 1, 2]);
    }

    /// `M  ISO` gives the mass number outright.
    #[test]
    fn m_iso_is_read() {
        let molecule = parse(&sdf(
            &[atom_line("C", 0, 0), atom_line("C", 0, 0)],
            &["  1  2  1  0  0  0  0"],
            &["M  ISO  1   1  13"],
        ));
        assert_eq!(molecule.isotopes(), Some(&vec![13, 0]));
    }

    /// The atom block's own isotope column is a difference from the most
    /// abundant isotope, so it needs the element resolved before it means
    /// anything.
    #[test]
    fn the_atom_block_mass_difference_is_read() {
        let molecule = parse(&sdf(
            &[atom_line("C", 1, 0), atom_line("H", 1, 0)],
            &["  1  2  1  0  0  0  0"],
            &[],
        ));
        assert_eq!(molecule.isotopes(), Some(&vec![13, 2]));
    }

    /// `M  ISO` supersedes that column as `M  CHG` supersedes the charges.
    #[test]
    fn m_iso_supersedes_the_atom_block() {
        let molecule = parse(&sdf(
            &[atom_line("C", 1, 0), atom_line("C", 1, 0)],
            &["  1  2  1  0  0  0  0"],
            &["M  ISO  1   1  14"],
        ));
        assert_eq!(molecule.isotopes(), Some(&vec![14, 0]));
    }

    /// A molecule that says nothing about isotopes stores nothing, rather than
    /// claiming every atom is at mass number zero.
    #[test]
    fn no_isotope_information_stays_none() {
        let molecule = parse(&sdf(
            &[atom_line("C", 0, 0), atom_line("O", 0, 0)],
            &["  1  2  2  0  0  0  0"],
            &[],
        ));
        assert_eq!(molecule.isotopes(), None);
    }

    /// The mass difference is counted from the most abundant isotope, which is
    /// not the lightest one.
    ///
    /// `chemistry_consts::MONOISOTOPIC_MASSES` holds the lightest isotope
    /// despite its name, and the two differ for 47 elements, so decoding
    /// against it silently shifted whole swathes of the periodic table:
    /// lithium would start from 6 rather than 7 and uranium from 233 rather
    /// than 238. Both entries below are wrong by exactly that gap if the
    /// table is confused again.
    #[test]
    fn the_mass_difference_counts_from_the_most_abundant_isotope() {
        let molecule = parse(&sdf(
            &[atom_line("Li", -1, 0), atom_line("U", -3, 0)],
            &[],
            &[],
        ));
        assert_eq!(molecule.isotopes(), Some(&vec![6, 235]));
    }

    /// A mass difference that would put the atom below mass number 1 is not a
    /// mass number, so nothing is recorded for it.
    ///
    /// RDKit stores it as an unsigned underflow instead — hydrogen with a
    /// difference of -3 comes back claiming mass number 65534.
    #[test]
    fn an_impossible_mass_difference_records_nothing() {
        let molecule = parse(&sdf(&[atom_line("H", -3, 0)], &[], &[]));
        assert_eq!(molecule.isotopes(), None);
    }

    /// Lines that are not interpreted still come through untouched.
    #[test]
    fn unknown_property_lines_are_kept_verbatim() {
        let content = sdf(
            &[atom_line("C", 0, 0)],
            &[],
            &["M  STY  1   1 SUP", "M  CHG  1   1   1"],
        );
        let entries = parse_sdf_file(Cursor::new(content.as_str())).unwrap();
        let mol_file = &entries[0].mol_file;
        assert_eq!(mol_file.properties, vec!["M  STY  1   1 SUP".to_string()]);
        assert_eq!(mol_file.charges, Some(vec![(0, 1)]));
    }
}

/// What a MOL block leaves implicit and we have to work out.
///
/// The block states which atoms are bonded to which and nothing else: the
/// hydrogens, the aromaticity and the stereochemistry are all derived, exactly
/// as they are for a SMILES. None of it was being derived, so a molecule read
/// from an SDF was a bare heavy-atom skeleton — alanine came back as C3NO2,
/// benzene had no aromatic bond, and on a corpus of 1501 stereoisomers not one
/// kept its configuration.
mod perception {
    use super::*;
    use crate::prelude::ChiralClass;

    /// A MOL block written from a 2D depiction, as RDKit and PubChem write it.
    fn block(atoms: &[&str], bonds: &[&str]) -> String {
        let mut out = format!(
            "test\n  molecules\n\n{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n",
            atoms.len(),
            bonds.len()
        );
        for line in atoms.iter().chain(bonds.iter()) {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("M  END\n$$$$\n");
        out
    }

    fn read(content: &str) -> Molecule3D {
        let entries = parse_sdf_file(Cursor::new(content)).expect("parse failed");
        Molecule3D::from_sdf_entries(entries)
            .expect("conversion failed")
            .remove(0)
    }

    fn atom(x: f64, y: f64, symbol: &str) -> String {
        // 10 + 10 + 10 for the coordinates, a space, 3 for the symbol, 2 for
        // the mass difference and 11 fields of 3. The parser rejects anything
        // shorter than the full 69.
        let line = format!(
            "{x:>10.4}{y:>10.4}{:>10.4} {symbol:<3}{:>2}{}",
            0.0,
            0,
            "  0".repeat(11)
        );
        debug_assert_eq!(line.len(), 69);
        line
    }

    /// Hydrogens the block does not list still have to be there.
    #[test]
    fn implicit_hydrogens_are_worked_out() {
        // Ethanol drawn as three heavy atoms.
        let molecule = read(&block(
            &[&atom(0.0, 0.0, "C"), &atom(1.0, 0.0, "C"), &atom(2.0, 0.0, "O")],
            &["  1  2  1  0", "  2  3  1  0"],
        ));
        assert_eq!(molecule.molecular_formula().to_string(), "C2H6O1");
        assert_eq!(molecule.atomic_numbers().len(), 9);
        // Every per-atom vector grows with them, `positions` included: it is
        // indexed directly, and leaving it short made the next question about
        // any distance panic.
        assert_eq!(molecule.positions.len(), 9);
    }

    /// A ring can only be written Kekulé, so aromaticity has to be perceived or
    /// benzene from an SDF stops matching benzene from a SMILES.
    #[test]
    fn aromaticity_is_perceived() {
        let ring: Vec<String> = (0..6)
            .map(|index| {
                let angle = std::f64::consts::TAU * index as f64 / 6.0;
                atom(angle.cos(), angle.sin(), "C")
            })
            .collect();
        let bonds = [
            "  1  2  2  0", "  2  3  1  0", "  3  4  2  0",
            "  4  5  1  0", "  5  6  2  0", "  6  1  1  0",
        ];
        let molecule = read(&block(
            &ring.iter().map(String::as_str).collect::<Vec<_>>(),
            &bonds,
        ));
        assert_eq!(molecule.to_smiles(), "c1ccccc1");
    }

    /// The wedge column is the only record of tetrahedral configuration a flat
    /// drawing has, and reversing it has to reverse the answer.
    #[test]
    fn a_wedge_gives_a_stereocentre() {
        let atoms = [
            atom(0.0, 0.0, "C"),
            atom(1.0, 0.0, "F"),
            atom(-0.5, 0.9, "Cl"),
            atom(-0.5, -0.9, "Br"),
        ];
        let atoms: Vec<&str> = atoms.iter().map(String::as_str).collect();

        let wedge = read(&block(&atoms, &["  1  2  1  1", "  1  3  1  0", "  1  4  1  0"]));
        let hash = read(&block(&atoms, &["  1  2  1  6", "  1  3  1  0", "  1  4  1  0"]));

        let (wedge_class, hash_class) = (
            wedge.chiral_classes().unwrap()[0],
            hash.chiral_classes().unwrap()[0],
        );
        assert_ne!(wedge_class, ChiralClass::None, "wedge gave no configuration");
        assert_ne!(hash_class, ChiralClass::None, "hash gave no configuration");
        assert_ne!(wedge_class, hash_class, "wedge and hash agreed");
        assert_ne!(wedge.to_smiles(), hash.to_smiles());
    }

    /// Without a wedge there is no configuration to find, however the atoms
    /// happen to be laid out.
    #[test]
    fn a_flat_drawing_without_a_wedge_has_no_stereocentre() {
        let atoms = [
            atom(0.0, 0.0, "C"),
            atom(1.0, 0.0, "F"),
            atom(-0.5, 0.9, "Cl"),
            atom(-0.5, -0.9, "Br"),
        ];
        let atoms: Vec<&str> = atoms.iter().map(String::as_str).collect();
        let molecule = read(&block(&atoms, &["  1  2  1  0", "  1  3  1  0", "  1  4  1  0"]));
        assert_eq!(molecule.chiral_classes().unwrap()[0], ChiralClass::None);
    }

    /// A flat four-coordinate atom is not a square planar centre.
    ///
    /// Every atom in a 2D depiction is coplanar with its neighbours by
    /// construction, so handing one to the full geometric perception made each
    /// quaternary carbon come back square planar: a drawn neopentane was being
    /// written out as `[C@SP3]`.
    #[test]
    fn a_flat_quaternary_carbon_is_not_square_planar() {
        let atoms = [
            atom(0.0, 0.0, "C"),
            atom(1.0, 0.0, "C"),
            atom(-1.0, 0.0, "C"),
            atom(0.0, 1.0, "C"),
            atom(0.0, -1.0, "C"),
        ];
        let atoms: Vec<&str> = atoms.iter().map(String::as_str).collect();
        let molecule = read(&block(
            &atoms,
            &["  1  2  1  0", "  1  3  1  0", "  1  4  1  0", "  1  5  1  0"],
        ));
        assert_eq!(molecule.chiral_classes().unwrap()[0], ChiralClass::None);
        assert_eq!(molecule.to_smiles(), "C(C)(C)(C)C");
    }

    /// Double bond geometry is in the coordinates — unless the file says it is
    /// not.
    ///
    /// Stereo code 3 means "either", and it binds however the atoms are placed.
    /// `tests/pentene.sdf` marks its double bond 3 while carrying 3D
    /// coordinates whose torsion across it is -179.7 degrees; reading those
    /// reports a configuration the file went out of its way to withhold, which
    /// is what RDKit calls STEREOANY.
    #[test]
    fn an_either_double_bond_keeps_no_geometry() {
        let atoms = [
            atom(0.0, 0.0, "C"),
            atom(1.0, 0.0, "C"),
            atom(-0.5, 0.9, "C"),
            atom(1.5, -0.9, "C"),
        ];
        let atoms: Vec<&str> = atoms.iter().map(String::as_str).collect();

        let stated = read(&block(&atoms, &["  1  2  2  0", "  1  3  1  0", "  2  4  1  0"]));
        let either = read(&block(&atoms, &["  1  2  2  3", "  1  3  1  0", "  2  4  1  0"]));

        assert_ne!(stated.chiral_classes().unwrap()[0], ChiralClass::None);
        assert_eq!(either.chiral_classes().unwrap()[0], ChiralClass::None);
        assert!(stated.to_smiles().contains('/'));
        assert!(!either.to_smiles().contains('/'));
    }

    /// The bond block's columns are fixed width and run together.
    ///
    /// Past ninety-nine atoms there is no space between the two indices — a
    /// bond between atoms 100 and 101 is written `100101  1  0` — so splitting
    /// the line on whitespace read one number where there are two. The
    /// connection table came apart entirely: a 116-atom peptide read back as a
    /// heap of loose methanes.
    #[test]
    fn bond_indices_past_ninety_nine_are_read() {
        let atoms: Vec<String> = (0..101)
            .map(|index| atom(index as f64, 0.0, "C"))
            .collect();
        let mut bonds: Vec<String> = (1..101)
            .map(|index| format!("{index:>3}{:>3}  1  0", index + 1))
            .collect();
        bonds.push("100101  1  0".to_string()); // already the natural spelling
        let molecule = read(&block(
            &atoms.iter().map(String::as_str).collect::<Vec<_>>(),
            &bonds.iter().map(String::as_str).collect::<Vec<_>>(),
        ));
        // One chain, so every heavy atom is reachable from the first.
        assert_eq!(molecule.atomic_numbers().iter().filter(|&&n| n == 6).count(), 101);
        assert_eq!(molecule.to_smiles().matches('.').count(), 0);
    }

    /// A stated valence overrides the model, which is how a file says "this one
    /// takes no hydrogen".
    #[test]
    fn a_stated_valence_is_honoured() {
        let mut potassium = atom(0.0, 0.0, "K");
        // Column 48..51 is the valence; 15 is the format's zero.
        potassium.replace_range(48..51, " 15");
        let molecule = read(&block(&[&potassium], &[]));
        assert_eq!(molecule.molecular_formula().to_string(), "K1");
    }
}

/// An azo `N=N` drawn flat states its geometry like any other double bond,
/// but each nitrogen has only one substituent, and the reader asked for two at
/// every end it looked from -- so neither end qualified and the geometry was
/// dropped. Both drawings are RDKit's own depictions of the SMILES.
#[test]
fn a_flat_azo_bond_keeps_its_geometry() {
    use crate::graph_algorithms::mol_canonization::canonize_mol;
    use crate::prelude::MolecularGraph;
    let block = |coordinates: [(f64, f64); 4]| {
        let symbols = ["C", "N", "N", "C"];
        let atoms: String = coordinates
            .iter()
            .zip(symbols)
            .map(|(&(x, y), symbol)| {
                format!("{x:10.4}{y:10.4}{:10.4} {symbol:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n", 0.0)
            })
            .collect();
        format!(
            "\n     RDKit          2D\n\n  4  3  0  0  0  0  0  0  0  0999 V2000\n{atoms}  1  2  1  0\n  2  3  2  0\n  3  4  1  0\nM  END\n$$$$\n"
        )
    };
    for (smiles, coordinates) in [
        ("C/N=N/C", [(-1.9796, -0.1365), (-0.5994, 0.4508), (0.5994, -0.4508), (1.9796, 0.1365)]),
        ("C/N=N\\C", [(-1.5, -0.6495), (-0.75, 0.6495), (0.75, 0.6495), (1.5, -0.6495)]),
    ] {
        let entries = parse_sdf_file(Cursor::new(block(coordinates))).unwrap();
        let read = Molecule3D::from_sdf_entries(entries).unwrap().remove(0);
        let expected = canonize_mol(&MolecularGraph::from_smiles_single(smiles).unwrap()).to_smiles();
        assert_eq!(canonize_mol(&read).to_smiles(), expected, "{smiles}");
    }
}
