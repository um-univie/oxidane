use crate::molecule::base::MoleculeCore;
use crate::prelude::*;
use std::collections::HashSet;
use std::time::{Duration, Instant};

fn parse(smiles: &str) -> MolecularGraph {
    MolecularGraph::from_smiles(smiles).unwrap().remove(0)
}

/// Checks that a result is a genuine, self-consistent isomorphism between
/// the two molecules' matched atoms/bonds: the atom mapping is injective on
/// both sides, every bond mapping's endpoints agree with it, and every
/// mapped bond is an actual bond in its molecule.
fn assert_valid_mces(
    mol_a: &MolecularGraph,
    mol_b: &MolecularGraph,
    mces: &MaximumCommonEdgeSubgraph,
) {
    let mut seen_a = HashSet::new();
    let mut seen_b = HashSet::new();
    for m in &mces.atom_mapping {
        assert!(
            seen_a.insert(m.atom_a),
            "atom_a={} mapped more than once",
            m.atom_a
        );
        assert!(
            seen_b.insert(m.atom_b),
            "atom_b={} mapped more than once",
            m.atom_b
        );
        assert_eq!(
            mol_a.atomic_numbers[m.atom_a],
            mol_b.atomic_numbers[m.atom_b]
        );
        assert_eq!(m.atomic_number, mol_a.atomic_numbers[m.atom_a]);
    }

    let a_to_b: std::collections::HashMap<usize, usize> = mces
        .atom_mapping
        .iter()
        .map(|m| (m.atom_a, m.atom_b))
        .collect();
    let has_bond = |mol: &MolecularGraph, x: usize, y: usize| {
        mol.get_atom_bonds(x)
            .is_some_and(|bonds| bonds.iter().any(|b| b.target() == y))
    };

    for bm in &mces.bond_mapping {
        let (a1, a2) = bm.bond_a;
        let (b1, b2) = bm.bond_b;
        assert!(
            has_bond(mol_a, a1, a2),
            "bond_a {:?} is not an actual bond in mol_a",
            bm.bond_a
        );
        assert!(
            has_bond(mol_b, b1, b2),
            "bond_b {:?} is not an actual bond in mol_b",
            bm.bond_b
        );

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
    // Ethanol (C-C-O) vs propanol (C-C-C-O): largest common substructure is
    // the 3-atom / 2-bond C-C-O scaffold, hydrogens excluded.
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
    // Ethane (C-C single bond) vs ethene (C=C double bond): no bond in
    // common, even though both are two-carbon molecules.
    let ethane = parse("CC");
    let ethene = parse("C=C");

    assert!(find_maximum_common_edge_subgraph(&ethane, &ethene).is_none());
}

#[test]
fn find_all_returns_every_tied_maximum() {
    // Propane (C-C-C, 2 bonds) can map to butane (C-C-C-C, 3 bonds) at bonds {0, 1}
    // or {1, 2}. There are at least two distinct maximum common edge subgraphs
    // tied for the maximum size (2 bonds).
    let propane = parse("CCC");
    let butane = parse("CCCC");

    let all = find_all_maximum_common_edge_subgraphs(&propane, &butane);
    assert!(
        all.len() > 1,
        "expected multiple tied maximum subgraphs (propane C-C-C maps to multiple 2-bond subgraphs of butane C-C-C-C), but got {}",
        all.len()
    );
    for result in &all {
        assert_valid_mces(&propane, &butane, result);
        assert_eq!(result.num_bonds(), 2);
    }
}

#[test]
fn common_edge_subgraph_size_matches_pairwise_for_two_molecules() {
    let ethanol = parse("CCO");
    let propanol = parse("CCCO");

    let pairwise = find_maximum_common_edge_subgraph(&ethanol, &propanol).unwrap();
    let size = common_edge_subgraph_size(&[ethanol, propanol]).unwrap();

    assert_eq!(size.num_atoms, pairwise.num_atoms());
    assert_eq!(size.num_bonds, pairwise.num_bonds());
}

#[test]
fn common_edge_subgraph_size_across_three_molecules() {
    // C-C-O is common to all three: ethanol, propanol and butanol.
    let ethanol = parse("CCO");
    let propanol = parse("CCCO");
    let butanol = parse("CCCCO");

    let size = common_edge_subgraph_size(&[ethanol, propanol, butanol]).unwrap();
    assert_eq!(size.num_atoms, 3);
    assert_eq!(size.num_bonds, 2);
}

#[test]
fn common_edge_subgraph_size_requires_at_least_two_molecules() {
    let ethanol = parse("CCO");
    assert!(common_edge_subgraph_size(&[ethanol]).is_none());
}

#[test]
fn ring_vs_chain_matches_open_path_only() {
    // Cyclohexane's ring can only match hexane along an open path, since the
    // ring-closing bond has no counterpart in the acyclic chain: 6 atoms but
    // only 5 bonds (a path), not 6 (a cycle).
    let cyclohexane = parse("C1CCCCC1");
    let hexane = parse("CCCCCC");

    let mces = find_maximum_common_edge_subgraph(&cyclohexane, &hexane).unwrap();
    assert_valid_mces(&cyclohexane, &hexane, &mces);
    assert_eq!(mces.num_atoms(), 6);
    assert_eq!(mces.num_bonds(), 5);
}

/// Regression test for a real performance bug: comparing a fused bicyclic
/// ring system against itself makes every carbon compatible with every other
/// carbon (same coarse element-only label), producing a large, dense,
/// highly-symmetric modular product. Before the Bron-Kerbosch search gained
/// pivoting, a live-updated branch-and-bound bound, and O(1)-ish bitset
/// adjacency lookups, this specific case took ~560ms for just 10 heavy atoms
/// (and did not finish within a minute for 14). Budget generously (native
/// perf varies across CI/dev machines) but tightly enough to catch a
/// regression back to that behavior.
#[test]
fn symmetric_fused_ring_self_match_is_fast() {
    let decalin_a = parse("C1CCC2CCCCC2C1");
    let decalin_b = parse("C1CCC2CCCCC2C1");

    let start = Instant::now();
    let mces = find_maximum_common_edge_subgraph(&decalin_a, &decalin_b).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(mces.num_atoms(), 10);
    assert_eq!(mces.num_bonds(), 11);
    assert!(
        elapsed < Duration::from_secs(5),
        "decalin self-match took {elapsed:?}, expected well under 5s (regression?)"
    );
}

/// Regression test for a real correctness bug in the atom-orientation
/// derivation: a bond-pair's shared atom can sit at *different* relative
/// positions across the two molecules (e.g. the second atom of one bond but
/// the first atom of the next), and an early implementation guessed
/// orientation independently per bond-pair, sometimes committing to a choice
/// that a later, structurally-forced bond-pair then silently contradicted —
/// producing a self-inconsistent result (the same target atom claimed by two
/// different source atoms) that nonetheless passed a plain size check.
/// Pentane's branch point (isopentane's atom 2, bonded to atoms 1, 3 and 4)
/// is exactly this case. `assert_valid_mces` is the check that would have
/// caught it; a bare atom/bond count would not have.
#[test]
fn branched_orientation_stays_consistent() {
    let pentane = parse("CCCCC");
    let isopentane = parse("CCC(C)C");

    let mces = find_maximum_common_edge_subgraph(&pentane, &isopentane).unwrap();
    assert_eq!(mces.num_bonds(), 3);
    assert_valid_mces(&pentane, &isopentane, &mces);
}

/// A genuinely disconnected common substructure: adipic acid's two
/// carboxylic-acid-plus-CH2 ends (not bonded to each other in the matched
/// result — the connecting CH2-CH2-CH2 in the middle isn't part of the
/// match) both match succinic acid's two ends independently, for a bigger
/// combined result than matching just one connected end would give.
#[test]
fn disconnected_fragments_combine() {
    let adipic_acid = parse("OC(=O)CCCCC(=O)O");
    let succinic_acid = parse("OC(=O)CCC(=O)O");

    let mces = find_maximum_common_edge_subgraph(&adipic_acid, &succinic_acid).unwrap();
    assert_valid_mces(&adipic_acid, &succinic_acid, &mces);
    // Both ends: 2x (O-C(=O)-C), 8 atoms / 6 bonds total — bigger than the
    // 6-atom/5-bond single connected end a connected-only search would find.
    assert_eq!(mces.num_atoms(), 8);
    assert_eq!(mces.num_bonds(), 6);
}

/// Atom comparison is stricter than element-only: an aromatic and a
/// non-aromatic carbon never match, even when a looser (element-only) tool
/// would allow it. Tetralin's saturated ring shares only 4 atoms / 3 bonds
/// with decalin (avoiding tetralin's two aromatic ring-fusion carbons),
/// *not* the full 10/11-atom skeletal match a plain element comparison
/// would report.
#[test]
fn aromaticity_must_match() {
    let tetralin = parse("C1CCc2ccccc2C1");
    let decalin = parse("C1CCC2CCCCC2C1");

    let mces = find_maximum_common_edge_subgraph(&tetralin, &decalin).unwrap();
    assert_valid_mces(&tetralin, &decalin, &mces);
    assert_eq!(mces.num_atoms(), 4);
    assert_eq!(mces.num_bonds(), 3);
    for mapping in &mces.atom_mapping {
        assert!(!tetralin.is_atom_aromatic(mapping.atom_a));
    }
}
