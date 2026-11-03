use crate::graph_algorithms::mcsplit::{
    find_common_edge_subgraph_multi, find_maximum_common_edge_subgraph,
    find_maximum_common_induced_subgraph,
};
use crate::molecule::base::MoleculeCore;
use crate::molecule::cip::CipConfig;
use crate::prelude::*;
use std::collections::HashSet;
use std::time::{Duration, Instant};

fn parse(smiles: &str) -> MolecularGraph {
    MolecularGraph::from_smiles(smiles).unwrap().remove(0)
}

/// Same internal-consistency check used for `mces` — a plain size assertion
/// wouldn't have caught the atom-orientation bug that check found there, so
/// every mcsplit test uses it too.
fn assert_valid_mces(mol_a: &MolecularGraph, mol_b: &MolecularGraph, mces: &MaximumCommonEdgeSubgraph) {
    let mut seen_a = HashSet::new();
    let mut seen_b = HashSet::new();
    for m in &mces.atom_mapping {
        assert!(seen_a.insert(m.atom_a), "atom_a={} mapped more than once", m.atom_a);
        assert!(seen_b.insert(m.atom_b), "atom_b={} mapped more than once", m.atom_b);
        assert_eq!(mol_a.atomic_numbers[m.atom_a], mol_b.atomic_numbers[m.atom_b]);
        assert_eq!(m.atomic_number, mol_a.atomic_numbers[m.atom_a]);
    }

    let a_to_b: std::collections::HashMap<usize, usize> =
        mces.atom_mapping.iter().map(|m| (m.atom_a, m.atom_b)).collect();
    let has_bond = |mol: &MolecularGraph, x: usize, y: usize| {
        mol.get_atom_bonds(x).is_some_and(|bonds| bonds.iter().any(|b| b.target() == y))
    };

    for bm in &mces.bond_mapping {
        let (a1, a2) = bm.bond_a;
        let (b1, b2) = bm.bond_b;
        assert!(has_bond(mol_a, a1, a2), "bond_a {:?} is not an actual bond in mol_a", bm.bond_a);
        assert!(has_bond(mol_b, b1, b2), "bond_b {:?} is not an actual bond in mol_b", bm.bond_b);

        let mapped = (a_to_b.get(&a1), a_to_b.get(&a2));
        let consistent = mapped == (Some(&b1), Some(&b2)) || mapped == (Some(&b2), Some(&b1));
        assert!(
            consistent,
            "bond {:?}<->{:?} disagrees with the atom mapping ({:?})",
            bm.bond_a, bm.bond_b, mapped
        );
    }
}

#[test]
fn identical_molecules_match_fully() {
    let benzene_a = parse("c1ccccc1");
    let benzene_b = parse("c1ccccc1");

    let mces = find_maximum_common_edge_subgraph(&benzene_a, &benzene_b).unwrap();
    assert_valid_mces(&benzene_a, &benzene_b, &mces);
    assert_eq!(mces.num_atoms(), 6);
    assert_eq!(mces.num_bonds(), 6);
}

#[test]
fn shared_linear_scaffold() {
    let ethanol = parse("CCO");
    let propanol = parse("CCCO");

    let mces = find_maximum_common_edge_subgraph(&ethanol, &propanol).unwrap();
    assert_valid_mces(&ethanol, &propanol, &mces);
    assert_eq!(mces.num_atoms(), 3);
    assert_eq!(mces.num_bonds(), 2);
}

#[test]
fn unrelated_molecules_share_no_bond() {
    let methane = parse("C");
    let water = parse("O");
    assert!(find_maximum_common_edge_subgraph(&methane, &water).is_none());
}

#[test]
fn bond_order_must_match() {
    let ethane = parse("CC");
    let ethene = parse("C=C");
    assert!(find_maximum_common_edge_subgraph(&ethane, &ethene).is_none());
}

#[test]
fn ring_vs_chain_edge_variant_matches_open_path() {
    // Same expectation as `mces`: the ring-closing bond has no counterpart
    // in the acyclic chain, so the edge-maximizing search finds a 6-atom /
    // 5-bond path, simply not counting that one bond as matched.
    let cyclohexane = parse("C1CCCCC1");
    let hexane = parse("CCCCCC");

    let mces = find_maximum_common_edge_subgraph(&cyclohexane, &hexane).unwrap();
    assert_valid_mces(&cyclohexane, &hexane, &mces);
    assert_eq!(mces.num_atoms(), 6);
    assert_eq!(mces.num_bonds(), 5);
}

#[test]
fn ring_vs_chain_induced_variant_drops_an_atom() {
    // The induced variant can't map all 6 atoms sequentially at all:
    // cyclohexane's atom0-atom5 pair is bonded (ring closure) while
    // hexane's atom0-atom5 pair isn't, which is exactly the inconsistency
    // the induced constraint forbids. It has to drop to 5 atoms / 4 bonds
    // instead — the documented induced-vs-edge-maximizing difference
    // between the module's two search variants.
    let cyclohexane = parse("C1CCCCC1");
    let hexane = parse("CCCCCC");

    let mces = find_maximum_common_induced_subgraph(&cyclohexane, &hexane).unwrap();
    assert_valid_mces(&cyclohexane, &hexane, &mces);
    assert_eq!(mces.num_atoms(), 5);
    assert_eq!(mces.num_bonds(), 4);
}

#[test]
fn branched_orientation_stays_consistent() {
    // Same case that caught a real orientation bug in `mces`'s reconstruction.
    let pentane = parse("CCCCC");
    let isopentane = parse("CCC(C)C");

    let mces = find_maximum_common_edge_subgraph(&pentane, &isopentane).unwrap();
    assert_valid_mces(&pentane, &isopentane, &mces);
    assert_eq!(mces.num_bonds(), 3);
}

#[test]
fn aromaticity_must_match() {
    let tetralin = parse("C1CCc2ccccc2C1");
    let decalin = parse("C1CCC2CCCCC2C1");

    let mces = find_maximum_common_edge_subgraph(&tetralin, &decalin).unwrap();
    assert_valid_mces(&tetralin, &decalin, &mces);
    for mapping in &mces.atom_mapping {
        assert!(!tetralin.is_atom_aromatic(mapping.atom_a));
    }
}

/// The case that motivated switching algorithms: `mces` did not finish
/// within a minute for a 14-carbon fused tricyclic ring system compared
/// against itself, even after gaining Bron-Kerbosch pivoting, a live
/// branch-and-bound bound, and bitset adjacency. McSplit's tighter
/// (per-bidomain) bound should make this fast.
#[test]
fn symmetric_fused_ring_self_match_is_fast() {
    let a = parse("C1CCC2C(C1)CCC3CCCCC23");
    let b = parse("C1CCC2C(C1)CCC3CCCCC23");

    let start = Instant::now();
    let mces = find_maximum_common_edge_subgraph(&a, &b).unwrap();
    let elapsed = start.elapsed();

    assert_valid_mces(&a, &b, &mces);
    assert_eq!(mces.num_atoms(), 14);
    assert_eq!(mces.num_bonds(), 16);
    assert!(
        elapsed < Duration::from_secs(5),
        "14-carbon tricyclic self-match took {elapsed:?}, expected well under 5s"
    );
}

/// Larger real-molecule case `mces` still cannot solve in reasonable time
/// (lovastatin vs. simvastatin, ~30 heavy atoms, differing by one methyl
/// group — RDKit's FindMCS solves this instantly). This is the actual
/// motivating case for switching algorithms.
#[test]
fn large_similar_drug_pair_is_fast() {
    let lovastatin = parse("CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12");
    let simvastatin = parse("CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12");

    let start = Instant::now();
    let mces = find_maximum_common_edge_subgraph(&lovastatin, &simvastatin).unwrap();
    let elapsed = start.elapsed();

    assert_valid_mces(&lovastatin, &simvastatin, &mces);
    assert!(
        elapsed < Duration::from_secs(10),
        "lovastatin/simvastatin match took {elapsed:?}, expected well under 10s"
    );
    // The two differ by exactly one methyl group, so almost all of the
    // smaller molecule (29 heavy atoms, 31 bonds) should match.
    assert!(mces.num_bonds() >= 28, "expected a near-total match, got {} bonds", mces.num_bonds());
}

/// Seven closely-related steroid esters (~50 heavy atoms, 53 bonds each,
/// with long alkyl ester chains). This is the workload that exposed the
/// missing bidomain/vertex selection heuristics: with arbitrary selection
/// not even the *first* pair finished in 90 s, because every C-C bond along
/// an unbranched chain carries an identical label and so landed in one large
/// bidomain that got explored exhaustively. With the standard
/// smallest-bidomain-first rule all 21 pairs finish in well under a second
/// total.
const STEROID_ESTERS: [&str; 7] = [
    "CCCC(C)C(=O)O[C@@H]1C[C@H]2[C@@H](CC[C@@H]3C[C@H](OC(=O)CCCCCCCCCCC(C)C)CC[C@@]32C)[C@@H]2CC[C@H]([C@H](C)CCC(=O)O)[C@@]12C",
    "CCCC(C)C(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCCCC(C)C)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
    "CCC(C)CC(=O)O[C@@H]1C[C@H]2[C@@H](CC[C@@H]3C[C@H](OC(=O)CCCCCCCCCCC(C)C)CC[C@@]32C)[C@@H]2CC[C@H]([C@H](C)CCC(=O)O)[C@@]12C",
    "CCC(C)CC(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCCCC(C)C)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
    "CCCCCC(=O)O[C@@H]1C[C@H]2[C@@H](CC[C@@H]3C[C@H](OC(=O)CCCCCCCCCCC(C)C)CC[C@@]32C)[C@@H]2CC[C@H]([C@H](C)CCC(=O)O)[C@@]12C",
    "CCCCCCCCCC(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCC)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
    "CCCCCC(=O)O[C@@H]1CC[C@@]2(C)[C@H](CC[C@@H]3[C@@H]2C[C@@H](OC(=O)CCCCCCCCCCC(C)C)[C@]2(C)[C@@H]([C@H](C)CCC(=O)O)CC[C@@H]32)C1",
];

#[test]
fn large_steroid_ester_set_is_fast_and_valid() {
    let mols: Vec<MolecularGraph> = STEROID_ESTERS.iter().map(|s| parse(s)).collect();
    let cips: Vec<Vec<CipConfig>> = mols.iter().map(|m| m.assign_cip_configs()).collect();

    let start = Instant::now();
    for i in 0..mols.len() {
        for j in (i + 1)..mols.len() {
            let mces = find_maximum_common_edge_subgraph(&mols[i], &mols[j])
                .unwrap_or_else(|| panic!("pair {i}x{j} unexpectedly shares no bond"));
            assert_valid_mces(&mols[i], &mols[j], &mces);
            // These differ only in ester side chains and/or ring-fusion
            // stereochemistry, so every pair still shares most of the
            // 53-bond skeleton. Pairs differing only in the side chain keep
            // all 10 stereocentres and reach 52 bonds; pairs whose steroid
            // cores differ stereochemically lose a centre and settle
            // around 47-48.
            assert!(
                mces.num_bonds() >= 47,
                "pair {i}x{j} matched only {} bonds, expected >= 47",
                mces.num_bonds()
            );

            // The point of CIP-aware labels: no matched pair may disagree
            // on absolute configuration.
            for am in &mces.atom_mapping {
                assert_eq!(
                    cips[i][am.atom_a], cips[j][am.atom_b],
                    "pair {i}x{j} matched atoms {} and {} with different CIP configs",
                    am.atom_a, am.atom_b
                );
            }
        }
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "21 steroid-ester pairs took {elapsed:?}, expected well under 30s"
    );
}

/// The n-ary reduction over the same set.
#[test]
fn n_ary_reduction_over_steroid_esters() {
    let mols: Vec<MolecularGraph> = STEROID_ESTERS.iter().map(|s| parse(s)).collect();

    let start = Instant::now();
    let common = find_common_edge_subgraph_multi(&mols).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(common.num_atoms, 48);
    assert_eq!(common.num_bonds, 49);
    assert!(elapsed < Duration::from_secs(10), "n-ary took {elapsed:?}");

    // Everything reported must be a real bond of the first molecule, and
    // every referenced atom must be accounted for.
    let reference = &mols[0];
    let has_bond = |x: usize, y: usize| {
        reference.get_atom_bonds(x).is_some_and(|bs| bs.iter().any(|b| b.target() == y))
    };
    for &(a, b) in &common.reference_bonds {
        assert!(has_bond(a, b), "reported bond ({a},{b}) is not a bond of molecule 1");
    }
    let atoms: HashSet<usize> = common.reference_atoms.iter().copied().collect();
    for &(a, b) in &common.reference_bonds {
        assert!(atoms.contains(&a) && atoms.contains(&b));
    }
}

#[test]
fn n_ary_requires_at_least_two_molecules() {
    let one = [parse("CCO")];
    assert!(find_common_edge_subgraph_multi(&one).is_none());
}

#[test]
fn n_ary_agrees_with_pairwise_for_two_molecules() {
    let ethanol = parse("CCO");
    let propanol = parse("CCCO");

    let pairwise = find_maximum_common_edge_subgraph(&ethanol, &propanol).unwrap();
    let multi = find_common_edge_subgraph_multi(&[ethanol, propanol]).unwrap();

    assert_eq!(multi.num_atoms, pairwise.num_atoms());
    assert_eq!(multi.num_bonds, pairwise.num_bonds());
}

/// Enantiomers must not match at the stereocentre. Before CIP descriptors
/// were added to the label, `AtomLabel` collapsed both handednesses to
/// `ChiralType::TH` and L- and D-alanine matched as if identical.
#[test]
fn enantiomers_do_not_match_at_the_stereocentre() {
    let l_ala = parse("N[C@@H](C)C(=O)O");
    let d_ala = parse("N[C@H](C)C(=O)O");

    let l_cip = l_ala.assign_cip_configs();
    let d_cip = d_ala.assign_cip_configs();
    assert_eq!(l_cip[1], CipConfig::S);
    assert_eq!(d_cip[1], CipConfig::R);

    let mces = find_maximum_common_edge_subgraph(&l_ala, &d_ala).unwrap();
    assert_valid_mces(&l_ala, &d_ala, &mces);

    // The rest of the molecule still matches, but the opposite-handed
    // centres must not be mapped onto each other.
    for am in &mces.atom_mapping {
        assert_eq!(
            l_cip[am.atom_a], d_cip[am.atom_b],
            "matched atoms {} and {} disagree on CIP configuration",
            am.atom_a, am.atom_b
        );
    }
    assert!(
        !mces.atom_mapping.iter().any(|am| am.atom_a == 1 && am.atom_b == 1),
        "the S centre was mapped onto the R centre"
    );
}

/// The same molecule written with two different SMILES atom orderings must
/// match completely. This is the case that rules out using the raw
/// `ChiralClass` @/@@ parity as a label: those tags differ between these
/// two spellings (`TH(2)` vs `TH(1)`) even though the molecule is identical,
/// so matching on them would report a spurious mismatch. CIP descriptors
/// agree (both S), so the match is total.
#[test]
fn same_molecule_written_differently_matches_completely() {
    let a = parse("N[C@@H](C)C(=O)O");
    let b = parse("C[C@H](N)C(=O)O");

    let a_cip = a.assign_cip_configs();
    let b_cip = b.assign_cip_configs();
    assert_eq!(a_cip[1], CipConfig::S);
    assert_eq!(b_cip[1], CipConfig::S);

    let mces = find_maximum_common_edge_subgraph(&a, &b).unwrap();
    assert_valid_mces(&a, &b, &mces);
    // Alanine has 6 heavy atoms and 5 heavy-atom bonds; all should match.
    assert_eq!(mces.num_atoms(), 6);
    assert_eq!(mces.num_bonds(), 5);
}

/// Octahedral, trigonal-bipyramidal, square-planar and allene centres get no
/// CIP descriptor — the rules do not cover them, and neither does RDKit's
/// labeller. Without a fallback label two different isomers were completely
/// interchangeable: the `@OH1`/`@OH2` pair below matched on all seven of their
/// atoms. The fallback is the chirality class read in the molecule's canonical
/// atom ordering, so it distinguishes the isomers without being sensitive to
/// how either was written (that is the other half of this test).
#[test]
fn higher_order_stereoisomers_do_not_match() {
    for (a, b) in [
        ("S[Co@OH1](F)(I)(Cl)(C)Br", "S[Co@OH2](F)(I)(Cl)(C)Br"),
        ("S[As@TB1](Br)(F)(Cl)N", "S[As@TB2](Br)(F)(Cl)N"),
        ("F[Pt@SP1](Cl)(Br)I", "F[Pt@SP2](Cl)(Br)I"),
        ("NC(Br)=[C@]=C(O)C", "NC(Br)=[C@@]=C(O)C"),
    ] {
        let (mol_a, mol_b) = (parse(a), parse(b));
        let heavy = mol_a.atomic_numbers().iter().filter(|&&z| z != 1).count();
        let matched = match find_maximum_common_edge_subgraph(&mol_a, &mol_b) {
            Some(mces) => {
                assert_valid_mces(&mol_a, &mol_b, &mces);
                mces.num_atoms()
            }
            None => 0,
        };
        assert!(
            matched < heavy,
            "{a} vs {b}: different isomers matched on all {matched} of {heavy} atoms"
        );
    }
}

/// The same higher-order stereocentre written different ways must still match
/// completely — the fallback label has to be a property of the molecule, not of
/// the SMILES. These spellings are the ones
/// `chiral_canonization_tests::invariance_octahedral` and
/// `invariance_trigonal_bipyramidal` assert are equivalent.
#[test]
fn equivalent_spellings_of_a_higher_order_centre_match_completely() {
    for (a, b, atoms, bonds) in [
        (
            "[Co@OH18]([SH])([Br])([I])([CH3])([F])[Cl]",
            "[I][Co@OH28]([Cl])([Br])([F])([CH3])[SH]",
            7,
            6,
        ),
        ("S[As@TB1](Br)(F)(Cl)N", "N[As@TB4](F)(Cl)(S)Br", 6, 5),
        // Writing an allene from the other end reverses the axial sense, so
        // the equivalent spelling of `[C@]` here is `[C@@]`; both canonize to
        // `BrC(N)=[C@]=C(O)C`.
        ("NC(Br)=[C@]=C(O)C", "OC(C)=[C@@]=C(Br)N", 7, 6),
    ] {
        let (mol_a, mol_b) = (parse(a), parse(b));
        let mces = find_maximum_common_edge_subgraph(&mol_a, &mol_b)
            .unwrap_or_else(|| panic!("{a} vs {b}: no match at all"));
        assert_valid_mces(&mol_a, &mol_b, &mces);
        assert_eq!((mces.num_atoms(), mces.num_bonds()), (atoms, bonds), "{a} vs {b}");
    }
}
