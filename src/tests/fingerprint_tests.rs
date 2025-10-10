//! Fingerprints, and the property that matters most about them: two writings
//! of one molecule must fingerprint identically, or every similarity built on
//! them is a similarity between SMILES strings rather than between compounds.

#[cfg(test)]
mod fingerprint_tests {
    use crate::molecule::fingerprint::{
        BitFingerprint, Fingerprint, circular_fingerprint, circular_fingerprint_with_invariants,
        feature_invariants,
    };
    use crate::molecule::topology::{DistanceMatrix, topological_distances};
    use crate::prelude::*;

    fn parse(smiles: &str) -> MolecularGraph {
        MolecularGraph::from_smiles_single(smiles).expect("should parse")
    }

    #[test]
    fn identifiers_are_counted_and_kept_in_order() {
        let fingerprint = Fingerprint::from_ids([7u32, 3, 7, 9, 7]);
        assert_eq!(fingerprint.entries(), &[(3, 1), (7, 3), (9, 1)]);
        assert_eq!(fingerprint.count(7), 3);
        assert_eq!(fingerprint.count(4), 0);
        assert_eq!(fingerprint.len(), 3);
        assert_eq!(fingerprint.total_count(), 5);
    }

    #[test]
    fn similarity_of_known_sets() {
        let left = Fingerprint::from_ids([1u32, 2, 3, 4]);
        let right = Fingerprint::from_ids([3u32, 4, 5]);
        // Two shared out of five between them.
        assert!((left.tanimoto(&right) - 2.0 / 5.0).abs() < 1e-12);
        assert!((left.dice(&right) - 4.0 / 7.0).abs() < 1e-12);
        assert!((left.cosine(&right) - 2.0 / (4.0f64 * 3.0).sqrt()).abs() < 1e-12);
        assert!((left.tversky(&right, 1.0, 1.0) - left.tanimoto(&right)).abs() < 1e-12);
        assert!((left.tversky(&right, 0.5, 0.5) - left.dice(&right)).abs() < 1e-12);
        assert_eq!(left.tanimoto(&left), 1.0);
        assert_eq!(
            left.tanimoto(&Fingerprint::from_ids([100u32, 200])),
            0.0
        );
        // Nothing said either way is agreement, not disagreement.
        assert_eq!(Fingerprint::new().tanimoto(&Fingerprint::new()), 1.0);
    }

    #[test]
    fn counts_separate_what_presence_cannot() {
        let one_ring = Fingerprint::from_ids([1u32]);
        let four_rings = Fingerprint::from_ids([1u32, 1, 1, 1]);
        assert_eq!(one_ring.tanimoto(&four_rings), 1.0);
        assert!((one_ring.count_tanimoto(&four_rings) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn folding_keeps_what_it_can() {
        let fingerprint = Fingerprint::from_ids([1u32, 65, 129]);
        let folded = fingerprint.fold(64);
        assert_eq!(folded.len(), 64);
        // All three identifiers are congruent to 1 modulo 64, so a 64-bit
        // vector cannot tell them apart -- which is the cost of folding.
        assert_eq!(folded.count_ones(), 1);
        assert!(folded.get(1));

        let wider = fingerprint.fold(1024);
        assert_eq!(wider.count_ones(), 3);
        assert_eq!(wider.set_bits().collect::<Vec<_>>(), vec![1, 65, 129]);
    }

    #[test]
    fn bit_vectors_compare_the_same_way() {
        let mut left = BitFingerprint::new(128);
        let mut right = BitFingerprint::new(128);
        for bit in [1, 2, 3, 100] {
            left.set(bit);
        }
        for bit in [3, 100, 127] {
            right.set(bit);
        }
        assert_eq!(left.intersection_with(&right), 2);
        assert_eq!(left.union_with(&right), 5);
        assert!((left.tanimoto(&right) - 0.4).abs() < 1e-12);
        assert_eq!(left.tanimoto(&left), 1.0);
        assert_eq!(
            BitFingerprint::new(64).tanimoto(&BitFingerprint::new(64)),
            1.0
        );
    }

    #[test]
    fn distances_along_a_chain_and_around_a_ring() {
        let hexane = parse("CCCCCC");
        let distances = topological_distances(&hexane);
        // The chain is written in order, so the carbons are atoms 0 to 5.
        assert_eq!(distances.get(0, 0), 0);
        assert_eq!(distances.get(0, 5), 5);
        assert_eq!(distances.get(2, 4), 2);

        let benzene = parse("c1ccccc1");
        let distances = topological_distances(&benzene);
        // Opposite corners of a six-ring are three bonds apart whichever way
        // round you go, and no pair is further.
        assert_eq!(distances.get(0, 3), 3);
        for first in 0..6 {
            for second in 0..6 {
                assert!(distances.get(first, second) <= 3);
            }
        }

        // A hydrogen is a leaf, so it never shortens a path between carbons.
        let ethane = parse("CC");
        let distances = topological_distances(&ethane);
        assert_eq!(distances.get(0, 1), 1);

    }

    #[test]
    fn the_wiener_index_of_hexane() {
        let hexane = parse("CCCCCC");
        let distances = topological_distances(&hexane);
        let heavy: Vec<bool> = (0..hexane.len())
            .map(|atom| hexane.get_atomic_number(atom) != 1)
            .collect();
        // The unbranched six-carbon chain is the textbook value.
        assert_eq!(distances.distance_sum(&heavy), 35);
    }

    #[test]
    fn a_fingerprint_does_not_depend_on_how_the_molecule_was_written() {
        for writings in [
            ["CCO", "OCC", "C(O)C"].as_slice(),
            ["c1ccccc1", "C1=CC=CC=C1", "c1cc(cc(c1))"].as_slice(),
            [
                "CC(=O)Oc1ccccc1C(=O)O",
                "OC(=O)C1=CC=CC=C1OC(C)=O",
                "O=C(O)c1ccccc1OC(C)=O",
            ]
            .as_slice(),
            [
                "O=c1cc[nH]c(=O)[nH]1",
                "O=C1NC(=O)NC=C1",
                "C1=CNC(=O)NC1=O",
            ]
            .as_slice(),
            // Two atoms can grow environments covering the same atoms and be
            // given different identifiers, and only one may be kept. Choosing
            // by atom index made these fingerprint several different ways,
            // because the indices are an accident of the writing.
            [
                "CN(C(C[NH2+]CC(c1ccccc1)O)c1occc1)C",
                "C(C[NH2+]CC(c1ccccc1)O)(N(C)C)c1ccco1",
                "c1cccc(C(C[NH2+]CC(N(C)C)c2ccco2)O)c1",
                "c1cccc(c1)C(C[NH2+]CC(N(C)C)c1ccco1)O",
            ]
            .as_slice(),
            [
                "C(NC(=O)Cc1csc(NC(=O)Cc2cccs2)n1)(CC)c1c(OCC(=O)NC)cccc1",
                "C(C(NC(CC)c1c(OCC(NC)=O)cccc1)=O)c1csc(NC(Cc2sccc2)=O)n1",
                "c1ccc(CC(=O)Nc2nc(cs2)CC(=O)NC(CC)c2ccccc2OCC(=O)NC)s1",
            ]
            .as_slice(),
        ] {
            let reference = circular_fingerprint(&parse(writings[0]), 2).fingerprint;
            for &writing in &writings[1..] {
                assert_eq!(
                    circular_fingerprint(&parse(writing), 2).fingerprint,
                    reference,
                    "{writing} fingerprinted differently from {}",
                    writings[0]
                );
            }
        }
    }

    #[test]
    fn benzene_says_one_thing_six_times() {
        let benzene = parse("c1ccccc1");
        let result = circular_fingerprint(&benzene, 1);
        // Every carbon is the same carbon, so radius zero and radius one each
        // give one identifier, six times over.
        assert_eq!(result.fingerprint.len(), 2);
        assert_eq!(result.fingerprint.total_count(), 12);
        for &(_, count) in result.fingerprint.entries() {
            assert_eq!(count, 6);
        }
    }

    #[test]
    fn an_environment_that_has_already_been_named_is_not_named_again() {
        let benzene = parse("c1ccccc1");
        // By radius three every atom's environment is the whole ring, so five
        // of the six are the same substructure reached from a different atom.
        let result = circular_fingerprint(&benzene, 3);
        let whole_ring: Vec<usize> = result
            .environments
            .iter()
            .filter(|environment| environment.atoms.len() == 6)
            .map(|environment| environment.center)
            .collect();
        assert_eq!(whole_ring.len(), 1);
    }

    #[test]
    fn environments_point_back_at_their_atoms() {
        let ethanol = parse("CCO");
        let result = circular_fingerprint(&ethanol, 1);
        // Three heavy atoms, so three radius-zero environments, each its own
        // atom and no hydrogens.
        let zero: Vec<&_> = result
            .environments
            .iter()
            .filter(|environment| environment.radius == 0)
            .collect();
        assert_eq!(zero.len(), 3);
        for environment in zero {
            assert_eq!(environment.atoms, vec![environment.center]);
            assert!(ethanol.get_atomic_number(environment.center) != 1);
        }
        // The carbon in the middle reaches both its neighbours at radius one.
        let middle = result
            .environments
            .iter()
            .find(|environment| environment.radius == 1 && environment.center == 1)
            .expect("the middle atom has an environment");
        assert_eq!(middle.atoms, vec![0, 1, 2]);
    }

    #[test]
    fn radius_zero_is_the_atom_alone() {
        let result = circular_fingerprint(&parse("CCO"), 0);
        // Before any neighbour is taken into account the three heavy atoms are
        // still distinct: the oxygen by element, and the two carbons by how
        // many heavy neighbours and hydrogens each has.
        assert_eq!(result.fingerprint.len(), 3);
        assert_eq!(result.fingerprint.total_count(), 3);
    }

    #[test]
    fn different_molecules_are_not_the_same_fingerprint() {
        let ethanol = circular_fingerprint(&parse("CCO"), 2).fingerprint;
        let propanol = circular_fingerprint(&parse("CCCO"), 2).fingerprint;
        let benzene = circular_fingerprint(&parse("c1ccccc1"), 2).fingerprint;
        assert_ne!(ethanol, propanol);
        assert!(ethanol.tanimoto(&propanol) > 0.0);
        assert!(ethanol.tanimoto(&propanol) < 1.0);
        // Nothing in common at all.
        assert_eq!(ethanol.tanimoto(&benzene), 0.0);
    }

    #[test]
    fn features_generalise_where_elements_do_not() {
        let chloro = parse("Clc1ccccc1");
        let bromo = parse("Brc1ccccc1");
        // The two halogens are different elements and the same feature.
        assert_ne!(
            circular_fingerprint(&chloro, 2).fingerprint,
            circular_fingerprint(&bromo, 2).fingerprint
        );
        let by_feature = |molecule: &MolecularGraph| {
            let invariants = feature_invariants(molecule);
            circular_fingerprint_with_invariants(molecule, 2, &invariants).fingerprint
        };
        assert_eq!(by_feature(&chloro), by_feature(&bromo));
    }

    #[test]
    fn a_fragment_is_contained_in_what_contains_it() {
        let phenol = circular_fingerprint(&parse("Oc1ccccc1"), 2).fingerprint;
        let paracetamol = circular_fingerprint(&parse("CC(=O)Nc1ccc(O)cc1"), 2).fingerprint;
        // Asymmetric Tversky asks how much of the first the second holds, and
        // a phenol ring is most of what phenol has to say.
        let contained = phenol.tversky(&paracetamol, 1.0, 0.0);
        assert!(
            contained > 0.5,
            "phenol should be largely contained in paracetamol, got {contained}"
        );
        assert!(contained > phenol.tanimoto(&paracetamol));
    }

    #[test]
    fn atoms_in_different_fragments_never_reach_each_other() {
        // `from_smiles` splits on the `.`, so a molecule holding two fragments
        // has to be built rather than parsed.
        let ethanol = parse("CCO");
        let chloride = parse("[Cl-]");
        let salt = MolecularGraph::combine(&[&ethanol, &chloride]);
        let distances = topological_distances(&salt);
        let far = salt.len() - 1;
        assert_eq!(distances.get(0, far), DistanceMatrix::UNREACHABLE);
        assert_eq!(distances.get(far, far), 0);
    }
}

mod pair_and_torsion_tests {
    use crate::molecule::fingerprint::{
        DEFAULT_MAX_DISTANCE, Fingerprint, atom_pair_fingerprint, atom_pair_types,
        topological_torsion_fingerprint,
    };
    use crate::prelude::*;

    fn parse(smiles: &str) -> MolecularGraph {
        MolecularGraph::from_smiles_single(smiles).expect("should parse")
    }

    fn pairs(smiles: &str) -> Fingerprint {
        atom_pair_fingerprint(&parse(smiles), DEFAULT_MAX_DISTANCE)
    }

    fn torsions(smiles: &str) -> Fingerprint {
        topological_torsion_fingerprint(&parse(smiles))
    }

    #[test]
    fn every_pair_of_heavy_atoms_is_counted_once() {
        // Six carbons make fifteen pairs, and hydrogens make none.
        assert_eq!(pairs("CCCCCC").total_count(), 15);
        assert_eq!(pairs("c1ccccc1").total_count(), 15);
        // A ring is closer to itself than a chain is: benzene has only three
        // distinct distances to a chain's five, and every carbon is alike, so
        // three identifiers rather than the chain's many.
        assert_eq!(pairs("c1ccccc1").len(), 3);
        assert!(pairs("CCCCCC").len() > 3);
    }

    #[test]
    fn a_pair_knows_what_the_atoms_are_and_how_far_apart() {
        // Propane: two end-to-middle pairs at one bond, one end-to-end pair at
        // two. The ends are the same type as each other and not as the middle.
        let propane = pairs("CCC");
        assert_eq!(propane.total_count(), 3);
        assert_eq!(propane.len(), 2);
        assert_eq!(
            propane.entries().iter().map(|&(_, n)| n).max(),
            Some(2)
        );

        // The same skeleton with a different element somewhere is a different
        // set of pairs.
        assert_ne!(pairs("CCC"), pairs("CCO"));
        // And so is the same atoms at a different distance.
        assert_ne!(pairs("CCCC"), pairs("CC(C)C"));
    }

    #[test]
    fn the_type_is_element_pi_and_heavy_neighbours() {
        let ethene = parse("C=C");
        let ethane = parse("CC");
        let types_ethene = atom_pair_types(&ethene);
        let types_ethane = atom_pair_types(&ethane);
        // Same element and the same one heavy neighbour, told apart by the π
        // electron the double bond puts on each end.
        assert_ne!(types_ethene[0], types_ethane[0]);
        // Hydrogens have no type at all.
        for (atom, atomic_number) in ethane.atomic_numbers().iter().enumerate() {
            if *atomic_number == 1 {
                assert_eq!(types_ethane[atom], 0);
            }
        }
        // An aromatic carbon holds one π electron, like an alkene carbon.
        let benzene = parse("c1ccccc1");
        let butadiene = parse("C=CC=C");
        let benzene_types = atom_pair_types(&benzene);
        let diene_types = atom_pair_types(&butadiene);
        // Benzene's carbons have two heavy neighbours; butadiene's inner ones
        // have two as well, and both carry one π electron.
        assert_eq!(benzene_types[0], diene_types[1]);
    }

    #[test]
    fn distant_pairs_can_be_left_out() {
        let decane = parse("CCCCCCCCCC");
        let all = atom_pair_fingerprint(&decane, DEFAULT_MAX_DISTANCE);
        let near = atom_pair_fingerprint(&decane, 3);
        assert_eq!(all.total_count(), 45);
        assert!(near.total_count() < all.total_count());
        // Every identifier the short cut-off keeps is one the long one had.
        for id in near.ids() {
            assert!(all.count(id) > 0);
        }
    }

    #[test]
    fn atoms_in_different_fragments_make_no_pair() {
        let ethanol = parse("CCO");
        let chloride = parse("[Cl-]");
        let salt = MolecularGraph::combine(&[&ethanol, &chloride]);
        // Three heavy atoms in one fragment and one in the other: three pairs,
        // not six.
        assert_eq!(
            atom_pair_fingerprint(&salt, DEFAULT_MAX_DISTANCE).total_count(),
            3
        );
    }

    #[test]
    fn a_torsion_needs_four_atoms_in_a_row() {
        assert!(torsions("CC").is_empty());
        assert!(torsions("CCC").is_empty());
        assert_eq!(torsions("CCCC").total_count(), 1);
        // Isobutane has a branch point and no four in a row.
        assert!(torsions("CC(C)C").is_empty());
        // Neopentane likewise.
        assert!(torsions("CC(C)(C)C").is_empty());
        // Pentane has two overlapping four-atom paths, and they are the same
        // path read from each end of a symmetric molecule -- but not the same
        // path, so two.
        assert_eq!(torsions("CCCCC").total_count(), 2);
    }

    #[test]
    fn a_ring_has_a_torsion_at_every_bond() {
        // Six ring bonds, each the middle of exactly one four-atom path.
        assert_eq!(torsions("c1ccccc1").total_count(), 6);
        assert_eq!(torsions("C1CCCCC1").total_count(), 6);
        // Every one of them is the same, so one identifier.
        assert_eq!(torsions("c1ccccc1").len(), 1);
        // Cyclopropane has three atoms and so no path of four distinct ones.
        assert!(torsions("C1CC1").is_empty());
    }

    #[test]
    fn a_torsion_reads_the_same_from_either_end() {
        // Butanol and its mirror writing are one torsion, not two.
        assert_eq!(torsions("CCCO"), torsions("OCCC"));
        assert_eq!(torsions("CCCO").total_count(), 1);
        // And a torsion with different ends is not the same as one without.
        assert_ne!(torsions("CCCO"), torsions("CCCC"));
    }

    #[test]
    fn neither_depends_on_how_the_molecule_was_written() {
        for writings in [
            ["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)C1=CC=CC=C1OC(C)=O"],
            ["O=c1cc[nH]c(=O)[nH]1", "C1=CNC(=O)NC1=O"],
            ["CN(C(C[NH2+]CC(c1ccccc1)O)c1occc1)C", "C(C[NH2+]CC(c1ccccc1)O)(N(C)C)c1ccco1"],
        ] {
            assert_eq!(pairs(writings[0]), pairs(writings[1]), "{}", writings[1]);
            assert_eq!(
                torsions(writings[0]),
                torsions(writings[1]),
                "{}",
                writings[1]
            );
        }
    }

    #[test]
    fn similarity_behaves_as_it_should() {
        let benzene = pairs("c1ccccc1");
        let toluene = pairs("Cc1ccccc1");
        let hexane = pairs("CCCCCC");
        assert!(benzene.tanimoto(&toluene) > benzene.tanimoto(&hexane));
        assert_eq!(benzene.tanimoto(&benzene), 1.0);
    }
}

mod featurize_tests {
    use crate::molecule::featurize::{
        ATOM_FEATURE_WIDTH, BOND_FEATURE_WIDTH, Hybridization, graph_features, hybridization,
    };
    use crate::prelude::*;

    fn parse(smiles: &str) -> MolecularGraph {
        MolecularGraph::from_smiles_single(smiles).expect("should parse")
    }

    #[test]
    fn hydrogens_are_counted_rather_than_given_rows() {
        let ethanol = parse("CCO");
        let features = graph_features(&ethanol);
        assert_eq!(features.atom_count, 3);
        assert_eq!(features.atoms, vec![0, 1, 2]);
        assert_eq!(features.atom_features.len(), 3 * ATOM_FEATURE_WIDTH);
        // Two bonds, each given both ways round.
        assert_eq!(features.edge_count, 4);
        assert_eq!(features.bond_features.len(), 4 * BOND_FEATURE_WIDTH);
        assert_eq!(features.edge_index.len(), 8);
        for edge in 0..features.edge_count {
            let (from, to) = features.edge(edge);
            assert!(from < 3 && to < 3 && from != to);
        }
    }

    #[test]
    fn every_row_says_exactly_one_thing_per_field() {
        let molecule = parse("CC(=O)Oc1ccccc1C(=O)O");
        let features = graph_features(&molecule);
        for row in 0..features.atom_count {
            let values = features.atom_row(row);
            // Thirteen element columns, six degree, six charge, five hydrogen,
            // six hybridization, four chirality: each block has one entry set.
            let mut cursor = 0;
            for width in [13usize, 6, 6, 5, 6] {
                let block = &values[cursor..cursor + width];
                assert_eq!(
                    block.iter().filter(|&&value| value == 1.0).count(),
                    1,
                    "row {row} block at {cursor}"
                );
                cursor += width;
            }
            // Aromatic and in-ring are flags.
            assert!(values[cursor] == 0.0 || values[cursor] == 1.0);
            assert!(values[cursor + 1] == 0.0 || values[cursor + 1] == 1.0);
            cursor += 2;
            assert_eq!(
                values[cursor..cursor + 4]
                    .iter()
                    .filter(|&&value| value == 1.0)
                    .count(),
                1
            );
        }
    }

    /// Hybridisation is how many orbitals the atom needs, not how many double
    /// bonds it has. Every expectation here is RDKit's for the same atom.
    #[test]
    fn hybridization_counts_orbitals_and_not_double_bonds() {
        // Four sigma bonds is four orbitals however many of them are double,
        // which is what reading the double bonds gets wrong about every
        // sulfone, sulfoxide and phosphate.
        assert_eq!(hybridization(&parse("CS(=O)(=O)C"), 1), Hybridization::Sp3);
        assert_eq!(hybridization(&parse("CP(=O)(O)O"), 1), Hybridization::Sp3);
        // Lone pairs take corners too: water is bent, not linear.
        assert_eq!(hybridization(&parse("O"), 0), Hybridization::Sp3);
        assert_eq!(hybridization(&parse("CN"), 1), Hybridization::Sp3);
        // A lone pair delocalised into a neighbouring pi system flattens the
        // atom, which is what an amide and an aryl ether are.
        assert_eq!(hybridization(&parse("CC(=O)NC"), 3), Hybridization::Sp2);
        assert_eq!(hybridization(&parse("COC=C"), 1), Hybridization::Sp2);
        assert_eq!(hybridization(&parse("CC(=O)O"), 3), Hybridization::Sp2);
        // Below the second row it does not, unless a charge makes it reach: a
        // thioether beside a vinyl group and a sulfonamide's nitrogen stay
        // tetrahedral, a thiolate on a ring does not, and an aromatic sulfur is
        // flat because its bonds are aromatic rather than single.
        assert_eq!(hybridization(&parse("CSC=C"), 1), Hybridization::Sp3);
        assert_eq!(hybridization(&parse("CN(C)S(C)(=O)=O"), 1), Hybridization::Sp3);
        assert_eq!(hybridization(&parse("c1ccccc1[S-]"), 6), Hybridization::Sp2);
        assert_eq!(hybridization(&parse("c1ccsc1"), 3), Hybridization::Sp2);
        // A hydrogen has no shape of its own and a bare ion has nothing to
        // hybridise.
        assert_eq!(hybridization(&parse("CC"), 2), Hybridization::Unspecified);
        assert_eq!(hybridization(&parse("[Na+]"), 0), Hybridization::S);
    }

    #[test]
    fn hybridization_reads_the_bonds() {
        assert_eq!(hybridization(&parse("CC"), 0), Hybridization::Sp3);
        assert_eq!(hybridization(&parse("C=C"), 0), Hybridization::Sp2);
        assert_eq!(hybridization(&parse("C#C"), 0), Hybridization::Sp);
        assert_eq!(hybridization(&parse("C=C=C"), 1), Hybridization::Sp);
        assert_eq!(hybridization(&parse("c1ccccc1"), 0), Hybridization::Sp2);
        // Written in Kekulé form it is the same atom, because perception has
        // already made the two writings agree.
        assert_eq!(hybridization(&parse("C1=CC=CC=C1"), 0), Hybridization::Sp2);
        // Hypervalent by neighbour count.
        assert_eq!(
            hybridization(&parse("FS(F)(F)(F)(F)F"), 1),
            Hybridization::Sp3d2
        );
        assert_eq!(hybridization(&parse("FP(F)(F)(F)F"), 1), Hybridization::Sp3d);
    }

    #[test]
    fn a_bond_row_records_order_conjugation_and_ring() {
        let amide = parse("CC(=O)NC");
        let features = graph_features(&amide);
        // The C-N of an amide is single, conjugated, and not in a ring.
        let (carbon, nitrogen) = (1usize, 3usize);
        let edge = (0..features.edge_count)
            .find(|&edge| {
                let (from, to) = features.edge(edge);
                features.atoms[from] == carbon && features.atoms[to] == nitrogen
            })
            .expect("the amide bond has an edge");
        let row = features.bond_row(edge);
        assert_eq!(row[0], 1.0, "single");
        assert_eq!(row[4], 1.0, "conjugated");
        assert_eq!(row[5], 0.0, "not in a ring");

        // An ethane C-C is single and conjugated with nothing.
        let ethane = graph_features(&parse("CC"));
        assert_eq!(ethane.bond_row(0)[4], 0.0);

        // A benzene bond is aromatic, conjugated and in a ring.
        let benzene = graph_features(&parse("c1ccccc1"));
        let row = benzene.bond_row(0);
        assert_eq!(row[3], 1.0, "aromatic");
        assert_eq!(row[4], 1.0, "conjugated");
        assert_eq!(row[5], 1.0, "in a ring");
    }

    #[test]
    fn the_matrix_does_not_depend_on_how_the_molecule_was_written() {
        // Rows are in the molecule's atom order, which differs between
        // writings, so what has to match is the multiset of rows.
        let sorted_rows = |smiles: &str| {
            let molecule = parse(smiles);
            let features = graph_features(&molecule);
            let mut rows: Vec<Vec<u32>> = (0..features.atom_count)
                .map(|row| {
                    features
                        .atom_row(row)
                        .iter()
                        .map(|value| value.to_bits())
                        .collect()
                })
                .collect();
            rows.sort_unstable();
            rows
        };
        for writings in [
            ["CC(=O)Oc1ccccc1C(=O)O", "OC(=O)C1=CC=CC=C1OC(C)=O"],
            ["O=c1cc[nH]c(=O)[nH]1", "C1=CNC(=O)NC1=O"],
            ["CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"],
        ] {
            assert_eq!(sorted_rows(writings[0]), sorted_rows(writings[1]), "{}", writings[1]);
        }
    }
}

mod minhash_tests {
    use crate::molecule::fingerprint::{Fingerprint, MinHash, circular_fingerprint};
    use crate::prelude::*;

    fn sketch(ids: impl IntoIterator<Item = u32>, permutations: usize) -> MinHash {
        MinHash::of(&Fingerprint::from_ids(ids), permutations)
    }

    #[test]
    fn a_sketch_of_a_set_agrees_with_itself() {
        let set = sketch(0..500, 256);
        assert_eq!(set.len(), 256);
        assert_eq!(set.jaccard(&set), 1.0);
        assert_eq!(set.distance(&set), 0.0);
    }

    #[test]
    fn disjoint_sets_hardly_agree_and_nested_ones_mostly_do() {
        let left = sketch(0..1000, 512);
        let right = sketch(1000..2000, 512);
        assert!(left.jaccard(&right) < 0.05, "{}", left.jaccard(&right));

        // Half of one inside the other: five hundred shared out of a thousand
        // between them.
        let whole = sketch(0..1000, 512);
        let part = sketch(0..500, 512);
        let estimate = whole.jaccard(&part);
        assert!(
            (estimate - 0.5).abs() < 0.06,
            "expected about a half, got {estimate}"
        );
    }

    #[test]
    fn a_longer_sketch_estimates_more_closely() {
        let exact = |a: u32, b: u32| {
            let left = Fingerprint::from_ids(0..a);
            let right = Fingerprint::from_ids((a - b)..(2 * a - b));
            left.tanimoto(&right)
        };
        let truth = exact(1000, 400);
        let short_error = (sketch(0..1000, 32).jaccard(&sketch(600..1600, 32)) - truth).abs();
        let long_error = (sketch(0..1000, 2048).jaccard(&sketch(600..1600, 2048)) - truth).abs();
        assert!(
            long_error <= short_error.max(0.02),
            "short {short_error}, long {long_error}"
        );
        assert!(long_error < 0.05);
    }

    #[test]
    fn sketches_of_different_lengths_are_not_comparable() {
        assert_eq!(sketch(0..100, 64).jaccard(&sketch(0..100, 128)), 0.0);
        // Two sketches of nothing say the same nothing.
        assert_eq!(
            MinHash::of(&Fingerprint::new(), 64).jaccard(&MinHash::of(&Fingerprint::new(), 64)),
            1.0
        );
    }

    #[test]
    fn it_tracks_the_similarity_of_real_molecules() {
        let of = |smiles: &str| {
            let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
            let fingerprint = circular_fingerprint(&molecule, 2).fingerprint;
            (MinHash::of(&fingerprint, 1024), fingerprint)
        };
        let (aspirin_sketch, aspirin) = of("CC(=O)Oc1ccccc1C(=O)O");
        let (salicylic_sketch, salicylic) = of("OC(=O)c1ccccc1O");
        let (glucose_sketch, glucose) = of("OCC1OC(O)C(O)C(O)C1O");

        // The sketch estimates the Tanimoto it was made from.
        for (sketch, fingerprint) in [
            (&salicylic_sketch, &salicylic),
            (&glucose_sketch, &glucose),
        ] {
            let estimate = aspirin_sketch.jaccard(sketch);
            let truth = aspirin.tanimoto(fingerprint);
            assert!(
                (estimate - truth).abs() < 0.06,
                "estimate {estimate} against truth {truth}"
            );
        }
        // And it keeps the ordering that matters.
        assert!(aspirin_sketch.jaccard(&salicylic_sketch) > aspirin_sketch.jaccard(&glucose_sketch));
    }

    #[test]
    fn a_sketch_does_not_depend_on_how_the_molecule_was_written() {
        let of = |smiles: &str| {
            let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
            MinHash::of(&circular_fingerprint(&molecule, 2).fingerprint, 128)
        };
        assert_eq!(
            of("CC(=O)Oc1ccccc1C(=O)O"),
            of("OC(=O)C1=CC=CC=C1OC(C)=O")
        );
    }
}

mod shingle_tests {
    use crate::molecule::fingerprint::{
        Fingerprint, MinHash, atom_pair_fingerprint, circular_fingerprint, map4_fingerprint,
        map4_shingles, mhfp_fingerprint, shingle_id,
    };
    use crate::prelude::*;

    fn parse(smiles: &str) -> MolecularGraph {
        MolecularGraph::from_smiles_single(smiles).expect("should parse")
    }

    /// A peptide of the given one-letter sequence, written the obvious way.
    fn peptide(sequence: &str) -> MolecularGraph {
        let mut smiles = String::new();
        for residue in sequence.chars() {
            smiles.push_str(match residue {
                'G' => "NCC(=O)",
                'A' => "NC(C)C(=O)",
                'S' => "NC(CO)C(=O)",
                'F' => "NC(Cc1ccccc1)C(=O)",
                _ => "NCC(=O)",
            });
        }
        smiles.push('O');
        parse(&smiles)
    }

    #[test]
    fn there_is_one_shingle_per_pair_per_layer() {
        // Four heavy atoms, so six pairs, and two layers each.
        let shingles = map4_shingles(&parse("CCCC"), 2, 30);
        assert_eq!(shingles.total_count(), 12);
        // A butane chain has three kinds of atom pair by environment, but the
        // ends are alike and the middles are alike, so far fewer distinct
        // shingles than occurrences.
        assert!(shingles.len() < 12);
    }

    #[test]
    fn a_shingle_reads_the_same_from_either_end() {
        assert_eq!(shingle_id(11, 97, 3, 1), shingle_id(97, 11, 3, 1));
        // And the distance and the layer are both part of what it says.
        assert_ne!(shingle_id(11, 97, 3, 1), shingle_id(11, 97, 4, 1));
        assert_ne!(shingle_id(11, 97, 3, 1), shingle_id(11, 97, 3, 2));
    }

    #[test]
    fn distant_pairs_and_separate_fragments_are_left_out() {
        let decane = parse("CCCCCCCCCC");
        assert!(map4_shingles(&decane, 1, 3).total_count() < map4_shingles(&decane, 1, 30).total_count());

        // Two ethanes never meet, so they contribute only their own pairs: one
        // per molecule per layer, and none of the four across.
        let fragments = MolecularGraph::from_smiles("CC.CC").expect("should parse");
        let apart = MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>());
        assert_eq!(map4_shingles(&apart, 1, 30).total_count(), 2);
    }

    #[test]
    fn layer_zero_would_be_the_atom_pair_fingerprint_and_is_not_offered() {
        assert!(map4_shingles(&parse("CCCC"), 0, 30).is_empty());
        assert!(!atom_pair_fingerprint(&parse("CCCC"), 30).is_empty());
    }

    #[test]
    fn neither_depends_on_how_the_molecule_was_written() {
        let one = parse("CC(=O)Oc1ccccc1C(=O)O");
        let other = parse("OC(=O)C1=CC=CC=C1OC(C)=O");
        assert_eq!(map4_shingles(&one, 2, 30), map4_shingles(&other, 2, 30));
        assert_eq!(
            map4_fingerprint(&one, 2, 30, 256),
            map4_fingerprint(&other, 2, 30, 256)
        );
        assert_eq!(mhfp_fingerprint(&one, 3, 256), mhfp_fingerprint(&other, 3, 256));
    }

    #[test]
    fn mhfp_is_the_circular_environments_sketched() {
        let molecule = parse("CC(=O)Oc1ccccc1C(=O)O");
        assert_eq!(
            mhfp_fingerprint(&molecule, 3, 128),
            MinHash::of(&circular_fingerprint(&molecule, 3).fingerprint, 128)
        );
    }

    #[test]
    fn a_sketch_estimates_the_similarity_of_the_shingles_it_was_made_from() {
        let aspirin = parse("CC(=O)Oc1ccccc1C(=O)O");
        let salicylic = parse("OC(=O)c1ccccc1O");
        let truth = map4_shingles(&aspirin, 2, 30).tanimoto(&map4_shingles(&salicylic, 2, 30));
        let estimate =
            map4_fingerprint(&aspirin, 2, 30, 2048).jaccard(&map4_fingerprint(&salicylic, 2, 30, 2048));
        assert!(
            (estimate - truth).abs() < 0.05,
            "estimate {estimate} against truth {truth}"
        );
    }

    #[test]
    fn it_keeps_telling_peptides_apart_where_a_folded_vector_gives_up() {
        // Two forty-residue peptides that differ in a quarter of their
        // residues. Folded small enough, every bit that could separate them has
        // collided with one that cannot, and they look nearly identical; the
        // sketch is over the identifiers themselves and does not.
        let left = peptide(&"GAFS".repeat(10));
        let right = peptide(&"GAFG".repeat(10));

        let folded = |molecule: &MolecularGraph| circular_fingerprint(molecule, 2).fingerprint.fold(256);
        let folded_similarity = folded(&left).tanimoto(&folded(&right));
        let sketched = map4_fingerprint(&left, 2, 30, 1024).jaccard(&map4_fingerprint(&right, 2, 30, 1024));
        assert!(
            sketched < folded_similarity,
            "folded {folded_similarity}, sketched {sketched}"
        );
    }

    #[test]
    fn similarity_behaves_as_it_should() {
        let of = |smiles: &str| map4_fingerprint(&parse(smiles), 2, 30, 1024);
        let aspirin = of("CC(=O)Oc1ccccc1C(=O)O");
        assert_eq!(aspirin.jaccard(&aspirin), 1.0);
        assert!(aspirin.jaccard(&of("OC(=O)c1ccccc1O")) > aspirin.jaccard(&of("CCCCCCCCCCCC")));
        // Nothing shared with nothing is still everything they agree on.
        assert_eq!(MinHash::of(&Fingerprint::new(), 64).len(), 64);
    }
}
