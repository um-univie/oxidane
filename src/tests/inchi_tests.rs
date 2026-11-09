//! Reading InChI.
//!
//! An InChI does not state bond orders, charge positions or unpaired
//! electrons, so the reader has to choose them, and these tests pin the
//! choices: each string is the standard InChI the reference implementation
//! writes for the structure beside it, and reading it back must give that
//! structure. The sweep that checks every reconstruction gives back its own
//! InChI needs the reference implementation, and lives outside the crate.

use crate::graph_algorithms::mol_canonization::canonize_mol;
use crate::io::{InChIError, Parity, parse_inchi};
use crate::molecule::tautomer::{TautomerRules, tautomer_key_from_smiles};
use crate::prelude::*;

/// The components as one canonical string, in a fixed order.
fn canonical(molecules: &[MolecularGraph]) -> String {
    let mut components: Vec<String> = molecules
        .iter()
        .map(|molecule| canonize_mol(molecule).to_smiles())
        .collect();
    components.sort();
    components.join(".")
}

fn read(inchi: &str) -> String {
    canonical(&MolecularGraph::from_inchi(inchi).unwrap_or_else(|error| panic!("{inchi}: {error}")))
}

fn smiles(smiles: &str) -> String {
    canonical(&MolecularGraph::from_smiles(smiles).unwrap())
}

fn assert_reads(cases: &[(&str, &str)]) {
    for (inchi, expected) in cases {
        assert_eq!(read(inchi), smiles(expected), "{inchi}");
    }
}

#[test]
fn the_layers_are_read_as_written() {
    let parsed = parse_inchi(
        "InChI=1S/C3H7NO2.2Na/c1-2(4)3(5)6;;/h2H,4H2,1H3,(H,5,6);;/q;2*+1/p-2/t2-;;/m0../s1",
    )
    .unwrap();
    assert!(parsed.standard);
    assert_eq!(parsed.components.len(), 3);
    assert_eq!(parsed.protons, -2);
    assert_eq!(parsed.stereo_kind, Some(1));
    let alanine = &parsed.components[0];
    // Carbon first, then the rest alphabetically: C, C, C, N, O, O.
    assert_eq!(alanine.elements, vec![6, 6, 6, 7, 8, 8]);
    assert_eq!(alanine.bonds, vec![(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)]);
    assert_eq!(alanine.hydrogens, vec![3, 1, 0, 2, 0, 0]);
    assert_eq!(alanine.mobile_groups.len(), 1);
    assert_eq!(alanine.mobile_groups[0].atoms, vec![4, 5]);
    assert_eq!(alanine.centres[0].parity, Parity::Minus);
    assert_eq!(alanine.inverted, Some(false));
    assert_eq!(parsed.components[1].charge, 1);
    assert_eq!(parsed.components[2].charge, 1);
    assert_eq!(parsed.components[2].inverted, None);
}

/// A branch holds every atom up to its `)`, and `,` starts another branch
/// from the same atom. Citric acid's central carbon carries three.
#[test]
fn branches_and_ring_closures_connect_the_right_atoms() {
    let citric = parse_inchi(
        "InChI=1S/C6H8O7/c7-3(8)1-6(13,5(11)12)2-4(9)10/h13H,1-2H2,(H,7,8)(H,9,10)(H,11,12)",
    )
    .unwrap();
    let central = 5;
    let mut neighbours: Vec<usize> = citric.components[0]
        .bonds
        .iter()
        .filter_map(|&(a, b)| (a == central).then_some(b).or((b == central).then_some(a)))
        .collect();
    neighbours.sort_unstable();
    assert_eq!(neighbours, vec![0, 1, 4, 12]);
    assert_eq!(
        read("InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"),
        smiles("c1ccccc1")
    );
}

#[test]
fn bond_orders_follow_from_valences() {
    assert_reads(&[
        ("InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)", "CC(=O)O"),
        ("InChI=1S/C2H3N/c1-2-3/h1H3", "CC#N"),
        ("InChI=1S/C2H2O/c1-2-3/h1H2", "C=C=O"),
        (
            "InChI=1S/C5H5NO/c7-5-3-1-2-4-6-5/h1-4H,(H,6,7)",
            "O=c1cccc[nH]1",
        ),
        (
            "InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)",
            "CC(=O)Nc1ccc(O)cc1",
        ),
        // A mobile hydrogen goes on nitrogen first: a sulfamide, not an
        // `S(=N)O` imidic acid.
        ("InChI=1S/H4N2O2S/c1-5(2,3)4/h(H4,1,2,3,4)", "NS(N)(=O)=O"),
        ("InChI=1S/CH5N3/c2-1(3)4/h(H5,2,3,4)", "NC(N)=N"),
    ]);
}

/// The layers say nothing about where a charge sits, so the reader writes
/// the charge-separated forms chemists do -- and never one InChI would have
/// normalized away.
#[test]
fn charge_separations_are_written_the_conventional_way() {
    assert_reads(&[
        ("InChI=1S/CH3NO2/c1-2(3)4/h1H3", "C[N+](=O)[O-]"),
        ("InChI=1S/C5H5NO/c7-6-4-2-1-3-5-6/h1-5H", "[O-][n+]1ccccc1"),
        ("InChI=1S/O3/c1-3-2", "O=[O+][O-]"),
        ("InChI=1S/N3/c1-3-2/q-1", "[N-]=[N+]=[N-]"),
        ("InChI=1S/CH3N3/c1-3-4-2/h1H3", "CN=[N+]=[N-]"),
        ("InChI=1S/CO/c1-2", "[C-]#[O+]"),
        ("InChI=1S/C2H3N/c1-3-2/h1H3", "C[N+]#[C-]"),
        ("InChI=1S/NO3/c2-1(3)4/q-1", "[O-][N+](=O)[O-]"),
        (
            "InChI=1S/C5H11NO2/c1-6(2,3)4-5(7)8/h4H2,1-3H3",
            "C[N+](C)(C)CC(=O)[O-]",
        ),
        // No separation a higher valence can avoid.
        ("InChI=1S/C2H6OS/c1-4(2)3/h1-2H3", "CS(C)=O"),
        ("InChI=1S/C2H6O2S/c1-5(2,3)4/h1-2H3", "CS(C)(=O)=O"),
        ("InChI=1S/H3O4P/c1-5(2,3)4/h(H3,1,2,3,4)", "OP(=O)(O)O"),
    ]);
}

#[test]
fn net_charges_sit_where_inchi_left_them() {
    assert_reads(&[
        ("InChI=1S/C3H9S/c1-4(2)3/h1-3H3/q+1", "C[S+](C)C"),
        ("InChI=1S/C5H5/c1-2-4-5-3-1/h1-5H/q-1", "c1cc[cH-]c1"),
        // A carbanion and not an enolate: InChI would have put a proton on the
        // enolate's oxygen and written `/p-1`.
        (
            "InChI=1S/C5H7O2/c1-4(6)3-5(2)7/h3H,1-2H3/q-1",
            "CC(=O)[CH-]C(C)=O",
        ),
        ("InChI=1S/F6P/c1-7(2,3,4,5)6/q-1", "F[P-](F)(F)(F)(F)F"),
        ("InChI=1S/BF4/c2-1(3,4)5/q-1", "F[B-](F)(F)F"),
        // Six bonds are more than silicon's table allows at any charge: the
        // atom is left as written and holds the charge.
        ("InChI=1S/F6Si/c1-7(2,3,4,5)6/q-2", "F[Si-2](F)(F)(F)(F)F"),
        ("InChI=1S/2ClH.Fe/h2*1H;/q;;+2/p-2", "[Fe+2].[Cl-].[Cl-]"),
        ("InChI=1S/H/q-1", "[H-]"),
    ]);
}

/// `/p` moves protons on or off without saying where; the reader puts them
/// back where InChI's normalization would take them off again.
#[test]
fn protons_go_back_where_inchi_took_them_from() {
    assert_reads(&[
        ("InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)/p-1", "CC(=O)[O-]"),
        (
            "InChI=1S/C2H4O2.Na/c1-2(3)4;/h1H3,(H,3,4);/q;+1/p-1",
            "CC(=O)[O-].[Na+]",
        ),
        ("InChI=1S/C5H5N/c1-2-4-6-5-3-1/h1-5H/p+1", "c1cc[nH+]cc1"),
        ("InChI=1S/CH5N3/c2-1(3)4/h(H5,2,3,4)/p+1", "NC(N)=[NH2+]"),
        (
            "InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H/p-1",
            "[O-]c1ccccc1",
        ),
        ("InChI=1S/H2O/h1H2/p-1", "[OH-]"),
        ("InChI=1S/H2O/h1H2/p+1", "[OH3+]"),
        // A betaine's carboxylate, written `(H-,...)` beside its quaternary
        // nitrogen, takes the proton back before anything basic does.
        (
            "InChI=1S/C6H13NO2/c1-7(2,3)5-4-6(8)9/h4-5H2,1-3H3/p+1",
            "C[N+](C)(C)CCC(=O)O",
        ),
        // A proton no atom can take is a component of its own.
        ("InChI=1S/p+1", "[H+]"),
    ]);
}

#[test]
fn unpaired_electrons_are_what_no_bond_can_take() {
    assert_reads(&[
        ("InChI=1S/CH3/h1H3", "[CH3]"),
        ("InChI=1S/NO/c1-2", "[N]=O"),
        ("InChI=1S/O2/c1-2", "O=O"),
        ("InChI=1S/C4H9O/c1-4(2,3)5/h1-3H3", "CC(C)(C)[O]"),
        // A boron missing its hydrogens is drawn with its electrons, not as a
        // borenium beside an anion -- and the chlorine stays monovalent.
        ("InChI=1S/BCl/c1-2", "[B]Cl"),
        ("InChI=1S/H2/h1H", "[H][H]"),
    ]);
}

#[test]
fn stereo_parities_become_the_same_configurations() {
    assert_reads(&[
        (
            "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1",
            "C[C@H](N)C(=O)O",
        ),
        (
            "InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m1/s1",
            "C[C@@H](N)C(=O)O",
        ),
        ("InChI=1S/C4H8/c1-3-4-2/h3-4H,1-2H3/b4-3+", "C/C=C/C"),
        ("InChI=1S/C4H8/c1-3-4-2/h3-4H,1-2H3/b4-3-", "C/C=C\\C"),
        (
            "InChI=1S/C3H8OS/c1-3-5(2)4/h3H2,1-2H3/t5-/m0/s1",
            "C[S@](=O)CC",
        ),
        (
            "InChI=1S/C8H16/c1-7-5-3-4-6-8(7)2/h7-8H,3-6H2,1-2H3/t7-,8-/m0/s1",
            "C[C@H]1CCCC[C@@H]1C",
        ),
        ("InChI=1S/CBrClFI/c2-1(3,4)5/t1-/m1/s1", "F[C@](Cl)(Br)I"),
        ("InChI=1S/CH2ClNO/c2-1-3-4/h1,4H/b3-1+", "Cl/C=N/O"),
    ]);
}

#[test]
fn isotopes_are_counted_from_inchis_own_masses() {
    assert_reads(&[
        ("InChI=1S/CH4/h1H4/i1+1", "[13CH4]"),
        // Bromine's reference is 80, its rounded average mass.
        ("InChI=1S/CH3Br/c1-2/h1H3/i2-1", "C[79Br]"),
        ("InChI=1S/CH4O/c1-2/h2H,1H3/i1D3", "[2H]C([2H])([2H])O"),
        ("InChI=1S/H2O/h1H2/i/hD2", "[2H]O[2H]"),
        // The deuterium of the isotopic layer's `/h` is the record's, and here
        // it is on the water, not on the ester beside it.
        (
            "InChI=1S/C3H6O2.H2O/c1-3(4)5-2;/h1-2H3;1H2/i/hD",
            "COC(C)=O.[2H]O",
        ),
        // A stereocentre that only the label makes one.
        (
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3/i2D/t2-/m0/s1",
            "[2H][C@@H](C)O",
        ),
    ]);
}

/// The isotopic layer may say a run of components has the main layer's stereo,
/// `2m`, rather than write it again. A PubChem record of four deuterated
/// isomers.
const DEUTERATED_ISOMERS: &str = concat!(
    "InChI=1S/4C19H30O/c2*1-14-8-10-15(11-9-14)12-16(13-18(2,3)4)17(20)19(5,6)7;",
    "2*1-14-9-8-10-15(11-14)12-16(13-18(2,3)4)17(20)19(5,6)7/h4*8-11,16H,12-13H2,1-7H3",
    "/t4*16-/m1111/s1/i10D,11D;12D2;8D,9D,10D,11D,12D,16D;8D,9D,10D,11D,12D",
    "/t2m;12-,16-;12-,16+/m1110",
);

#[test]
fn isotopic_stereo_may_copy_the_main_layers() {
    let parsed = parse_inchi(DEUTERATED_ISOMERS).unwrap();
    for component in &parsed.components[..2] {
        assert_eq!(
            component.isotopic_centres.as_deref(),
            Some(component.centres.as_slice())
        );
    }
    for component in &parsed.components[2..] {
        assert_eq!(component.isotopic_centres.as_ref().map(Vec::len), Some(2));
    }
    assert_eq!(parsed.components[3].isotopic_inverted, Some(false));
    assert_eq!(
        MolecularGraph::from_inchi(DEUTERATED_ISOMERS).map(|molecules| molecules.len()),
        Ok(4)
    );
}

/// Reading an InChI and keying the structure gives the key of the structure
/// it was written from, wherever InChI and the key draw the same line.
#[test]
fn a_read_structure_keys_like_its_source() {
    let rules = TautomerRules::new();
    for (inchi, source) in [
        (
            "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        ),
        (
            "InChI=1S/C13H18O2/c1-9(2)8-11-4-6-12(7-5-11)10(3)13(14)15/h4-7,9-10H,8H2,1-3H3,(H,14,15)",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        ),
        (
            "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)",
            "Nc1nc2[nH]cnc2c(=O)[nH]1",
        ),
    ] {
        let read: Vec<String> = MolecularGraph::from_inchi(inchi)
            .unwrap()
            .iter()
            .map(|molecule| molecule.to_smiles())
            .collect();
        assert_eq!(
            tautomer_key_from_smiles(&read.join("."), &rules).unwrap(),
            tautomer_key_from_smiles(source, &rules).unwrap(),
            "{inchi}"
        );
    }
}

#[test]
fn malformed_strings_are_refused() {
    assert_eq!(parse_inchi("1S/CH4/h1H4"), Err(InChIError::MissingPrefix));
    assert!(matches!(
        parse_inchi("InChI=1S/CH4/h1H3"),
        Err(InChIError::Inconsistent(_))
    ));
    assert!(matches!(
        parse_inchi("InChI=1S/C2H6/c1-3/h1-2H3"),
        Err(InChIError::Syntax { layer: 'c', .. })
    ));
    assert!(matches!(
        parse_inchi("InChI=1S/C2H6/c1-2(/h1-2H3"),
        Err(InChIError::Syntax { layer: 'c', .. })
    ));
    assert!(matches!(
        parse_inchi("InChI=1S/Xx/h1H"),
        Err(InChIError::UnknownElement(_))
    ));
    assert!(matches!(
        parse_inchi("InChI=1B/C2H4/c1-2/h1-2H2/z101-1-2"),
        Err(InChIError::Unsupported(_))
    ));
}

/// A non-standard InChI is read as far as its main layer: the fixed-H and
/// reconnected-metal layers describe other structures.
#[test]
fn layers_describing_other_structures_are_skipped() {
    assert_eq!(
        read("InChI=1/CH4N2O/c2-1(3)4/h(H4,2,3,4)/f/h2-3H2"),
        smiles("NC(N)=O")
    );
}
