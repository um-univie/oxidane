//! Tautomer keys.
//!
//! Two properties, and they pull in opposite directions. Every tautomer of a
//! compound must get one key, or a registry keeps duplicates; and two compounds
//! that are not tautomers must keep two, or it merges things that differ. The
//! sweep over a corpus lives in `tests/tautomer_oracle.py`, which compares the
//! partition against RDKit's tautomer hashes; what is here is the set of cases
//! worth naming.

use crate::molecule::tautomer::{
    TautomerRules, erased_skeleton, perceive, tautomer_key, tautomer_key_from_smiles,
};
use crate::prelude::*;

fn key(smiles: &str) -> String {
    tautomer_key_from_smiles(smiles, &TautomerRules::new()).expect("the SMILES parses")
}

fn carbon_key(smiles: &str) -> String {
    tautomer_key_from_smiles(smiles, &TautomerRules::with_carbon_endpoints())
        .expect("the SMILES parses")
}

/// Every pair here is one compound written as two of its tautomers.
#[test]
fn tautomers_share_a_key() {
    for (first, second) in [
        // The 1,3-shift, in the forms it takes across the periodic table.
        ("CC(=O)N", "CC(O)=N"),
        ("CC(=N)N", "CC(N)=N"),
        ("CC(=S)N", "CC(S)=N"),
        // Three endpoints sharing one pool.
        ("NC(=O)N", "N=C(O)N"),
        ("NC(=N)N", "N=C(N)N"),
        // A negative charge is mobile in the same way a hydrogen is.
        ("CC(=O)[O-]", "CC([O-])=O"),
        // Around a ring, where the shift travels along aromatic bonds.
        ("O=c1cccc[nH]1", "Oc1ccccn1"),
        ("O=c1cc[nH]cc1", "Oc1ccncc1"),
        ("c1cnc[nH]1", "c1c[nH]cn1"),
        ("c1nc[nH]n1", "c1n[nH]cn1"),
        ("c1nnn[nH]1", "c1n[nH]nn1"),
        // The nucleobases, which is where this matters most in practice.
        ("Nc1cc[nH]c(=O)n1", "Nc1ccnc(O)n1"),
        ("O=c1cc[nH]c(=O)[nH]1", "Oc1ccnc(O)n1"),
        ("Nc1nc2[nH]cnc2c(=O)[nH]1", "Nc1nc2nc[nH]c2c(O)n1"),
        // Warfarin's coumarin, the case that motivated the module.
        ("O=c1cc(O)c2ccccc2o1", "Oc1cc(=O)oc2ccccc12"),
        // The 1,5-shift, which needs three intermediates rather than one.
        ("OC=CC=O", "O=CC=CO"),
    ] {
        assert_eq!(key(first), key(second), "{first} and {second}");
    }
}

/// None of these pairs is a tautomer, and a key that merged them would be
/// merging compounds a registry has to keep apart.
#[test]
fn distinct_compounds_keep_distinct_keys() {
    for (first, second) in [
        // Keto and enol differ only at a carbon, which the default rules do not
        // treat as an endpoint.
        ("CC(C)=O", "CC(O)=C"),
        // Different elements at the endpoints.
        ("CC(=O)N", "CC(=O)O"),
        // Different ring positions for the same shift.
        ("O=c1cccc[nH]1", "O=c1cc[nH]cc1"),
        ("c1cnc[nH]1", "c1cc[nH]n1"),
        ("Nc1ncnc2[nH]cnc12", "Nc1nc2[nH]cnc2c(=O)[nH]1"),
        ("O=c1cc[nH]c(=O)[nH]1", "Nc1cc[nH]c(=O)n1"),
        // Nothing mobile in either, and different skeletons.
        ("Oc1ccccc1", "O=C1CCCCC1"),
        ("NC(=O)c1ccccc1", "NC(=O)C1CCCCC1"),
        // The group totals are what stops the erased skeleton of an amide from
        // colliding with a structure whose bonds are genuinely single.
        ("CC(=O)N", "CC(N)O"),
        ("O=c1cccc[nH]1", "c1ccncc1"),
    ] {
        assert_ne!(key(first), key(second), "{first} and {second}");
    }
}

/// The key depends on the molecule, not on the order its atoms were written in.
#[test]
fn the_key_is_independent_of_input_order() {
    for (first, second) in [
        ("O=c1cccc[nH]1", "[nH]1ccccc1=O"),
        ("CC(=O)N", "NC(C)=O"),
        ("Nc1nc2[nH]cnc2c(=O)[nH]1", "O=c1[nH]c(N)nc2c1nc[nH]2"),
    ] {
        assert_eq!(key(first), key(second), "{first} and {second}");
    }
}

/// A molecule with nothing mobile keeps its ordinary canonical SMILES, so keys
/// for the bulk of a registry stay readable and stay comparable with the ones
/// canonization already produced.
#[test]
fn a_molecule_with_no_group_keeps_its_canonical_smiles() {
    for smiles in ["c1ccccc1", "CCO", "CC(C)Cc1ccc(cc1)C(C)C(=O)O".trim()] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let groups = perceive(&molecule, &TautomerRules::new());
        if groups.is_empty() {
            // No group field, and under the default rules no marker either.
            assert!(!key(smiles).contains('/'), "{smiles}");
        }
    }
    assert!(
        perceive(
            &MolecularGraph::from_smiles_single("c1ccccc1").unwrap(),
            &TautomerRules::new()
        )
        .is_empty()
    );
}

/// Carbon is an endpoint only when the rules say so, and then keto and enol are
/// the same compound.
#[test]
fn carbon_endpoints_merge_keto_and_enol() {
    assert_ne!(key("CC(C)=O"), key("CC(O)=C"));
    assert_eq!(carbon_key("CC(C)=O"), carbon_key("CC(O)=C"));
    assert_eq!(
        carbon_key("CC(=O)CC(C)=O"),
        carbon_key("CC(O)=CC(C)=O"),
        "acetylacetone"
    );
}

/// The totals a group conserves are what the erasure trades the bond orders and
/// hydrogen positions for, so they have to be right.
#[test]
fn a_group_carries_the_totals_it_conserves() {
    let rules = TautomerRules::new();

    let urea = MolecularGraph::from_smiles_single("NC(=O)N").unwrap();
    let groups = perceive(&urea, &rules);
    assert_eq!(groups.groups().len(), 1);
    let group = &groups.groups()[0];
    assert_eq!(group.endpoints.len(), 3, "one oxygen and two nitrogens");
    assert_eq!(group.mobile_hydrogens, 4);
    assert_eq!(group.mobile_charge, 0);

    // Acetate's mobile charge is the negative one, and it has no hydrogen.
    let acetate = MolecularGraph::from_smiles_single("CC(=O)[O-]").unwrap();
    let groups = perceive(&acetate, &rules);
    assert_eq!(groups.groups().len(), 1);
    assert_eq!(groups.groups()[0].mobile_hydrogens, 0);
    assert_eq!(groups.groups()[0].mobile_charge, -1);
}

/// The skeleton is what the erasure leaves: the group's bonds all single, its
/// mobile hydrogens gone, its charges zeroed.
#[test]
fn the_skeleton_has_nothing_left_to_tell_tautomers_apart() {
    let rules = TautomerRules::new();
    let (from_amide, _) = erased_skeleton(
        &MolecularGraph::from_smiles_single("CC(=O)N").unwrap(),
        &rules,
    );
    let (from_imidic, _) = erased_skeleton(
        &MolecularGraph::from_smiles_single("CC(O)=N").unwrap(),
        &rules,
    );

    for skeleton in [&from_amide, &from_imidic] {
        let hydrogens = skeleton
            .atomic_numbers()
            .iter()
            .filter(|&&atomic_number| atomic_number == 1)
            .count();
        assert_eq!(hydrogens, 3, "only the methyl's hydrogens survive");
        for atom in 0..skeleton.atomic_numbers().len() {
            for bond in skeleton.get_atom_bonds(atom).unwrap() {
                let target = bond.target();
                let heteroatom =
                    skeleton.atomic_numbers()[atom] > 6 || skeleton.atomic_numbers()[target] > 6;
                if heteroatom {
                    assert_eq!(bond.bond_order(), BondOrder::Single, "atom {atom}");
                }
            }
        }
    }
}

/// A salt is keyed component by component, so which side was written first does
/// not change the answer.
#[test]
fn components_are_keyed_independently_and_sorted() {
    assert_eq!(key("CC(=O)[O-].[Na+]"), key("[Na+].CC(=O)[O-]"));
    assert_ne!(key("CC(=O)[O-].[Na+]"), key("CC(=O)[O-]"));
}

/// The canonizer's method and the free function are the same answer.
#[test]
fn the_canonizer_method_agrees_with_the_free_function() {
    use crate::graph_algorithms::mol_canonization::Canonizer;

    let rules = TautomerRules::new();
    let mut canonizer = Canonizer::new();
    for smiles in ["O=c1cccc[nH]1", "Oc1ccccn1", "CC(=O)N", "c1ccccc1"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        assert_eq!(
            canonizer.tautomer_key(&molecule, &rules),
            tautomer_key(&molecule, &rules),
            "{smiles}"
        );
    }
}

/// Perception has to terminate on the shapes that make its closure grow: a
/// fully conjugated macrocycle, a fused polycyclic, and a molecule whose every
/// heteroatom is an endpoint.
#[test]
fn perception_terminates_on_dense_conjugation() {
    let rules = TautomerRules::new();
    for smiles in [
        "c1ccc2c(c1)ccc1c2ccc2c1cccc2",
        "O=C1NC(=O)NC(=O)C1",
        "Nc1nc(N)nc(N)n1",
        "O=c1[nH]c(=O)[nH]c(=O)[nH]1",
        "N=C(N)NC(=N)N",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let groups = perceive(&molecule, &rules);
        for group in groups.groups() {
            assert!(
                group.atoms.len() <= molecule.atomic_numbers().len(),
                "{smiles}"
            );
        }
        // And the answer does not depend on being asked twice.
        assert_eq!(key(smiles), key(smiles), "{smiles}");
    }
}

/// An empty molecule and a lone atom must not panic. A lone atom is a group
/// only where the key pools a whole record, since a bare ion is where a salt's
/// proton goes.
#[test]
fn degenerate_inputs_are_handled() {
    let rules = TautomerRules::new();
    let water = MolecularGraph::from_smiles_single("O").unwrap();
    assert!(perceive(&water, &TautomerRules::heteroatom_only()).is_empty());
    assert_eq!(perceive(&water, &rules).groups().len(), 1);
    assert_eq!(key("O"), tautomer_key(&water, &rules));

    let empty = MolecularGraph::default();
    assert!(perceive(&empty, &rules).is_empty());
}

// ---------------------------------------------------------------------------
// Normalization: choosing one tautomer to stand for the class.
// ---------------------------------------------------------------------------

use crate::molecule::tautomer::{normalize, normalized_smiles};

fn normalized(smiles: &str) -> String {
    normalized_smiles(smiles, &TautomerRules::new()).expect("the SMILES parses")
}

/// The point of normalizing: the tautomers of a compound do not merely share a
/// key, they come back as one structure.
#[test]
fn tautomers_normalize_to_one_structure() {
    for (first, second) in [
        ("CC(=O)N", "CC(O)=N"),
        ("NC(=O)N", "N=C(O)N"),
        ("CC(=N)N", "CC(N)=N"),
        ("CC(=S)N", "CC(S)=N"),
        ("CC(=O)[O-]", "CC([O-])=O"),
        ("O=c1cccc[nH]1", "Oc1ccccn1"),
        ("O=c1cc[nH]cc1", "Oc1ccncc1"),
        ("c1cnc[nH]1", "c1c[nH]cn1"),
        ("c1nc[nH]n1", "c1n[nH]cn1"),
        ("c1nnn[nH]1", "c1n[nH]nn1"),
        ("Nc1cc[nH]c(=O)n1", "Nc1ccnc(O)n1"),
        ("O=c1cc[nH]c(=O)[nH]1", "Oc1ccnc(O)n1"),
        ("Nc1nc2[nH]cnc2c(=O)[nH]1", "Nc1nc2nc[nH]c2c(O)n1"),
        ("O=c1cc(O)c2ccccc2o1", "Oc1cc(=O)oc2ccccc12"),
        ("OC=CC=O", "O=CC=CO"),
        ("NC(=[NH2+])N", "[NH2+]=C(N)N"),
        ("c1nn[n-]n1", "[n-]1nncn1"),
    ] {
        assert_eq!(normalized(first), normalized(second), "{first}, {second}");
    }
}

/// PubChem records already written as the form to store, which normalizing
/// used to rewrite: a nitro group beside two NH donors into `N(O)O`, a nitro
/// group beside a carboxylic acid into `[N+](O)=O` and a carboxylate, a
/// nitrile into a ketenimine, a cation onto a methoxy or ester oxygen, a phenol
/// beside a pyridone or a lactone into a quinone methide or a zwitterion, and
/// a salicylidene imine into its keto-amine.
#[test]
fn normalization_keeps_forms_that_are_already_preferred() {
    use crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
    for smiles in [
        "CC(C)(C)CN(C)S(=O)(=O)c1ccc(NN)c([N+](=O)[O-])c1",
        "CCCc1nn(C)c(N2CCOCC2C(=O)O)c1[N+](=O)[O-]",
        "NC=CC#N",
        "CCc1ccc(NC(=O)C(C#N)=Cc2ccc(NC(C)=O)cc2)cc1",
        "COc1cc([N+](=O)[O-])ccc1NC(=O)CNc1ccccc1N(C)Cc1ccccc1",
        "CC1(C)C(C=Cc2ccc(OC)c(OCc3ccccc3)c2)=[N+](CCCCCC(=O)NCCN)c2ccc(Nc3ccccc3)cc21",
        "O=C(CCS)Oc1ccc2[nH]c(-c3ccccc3)[nH+]c2c1",
        "Cn1c(-c2ccc(O)cc2)cccc1=O",
        "C[C@@H]1OC(=O)c2c(O)cccc21",
        "Oc1ccccc1C=Nc1ccccc1",
    ] {
        let canonical = canonicalize_and_sort_smiles(smiles, true).unwrap();
        assert_eq!(normalized(smiles), canonical, "{smiles}");
    }
}

/// The form chosen is the one a registry would store: the carbonyl, not the
/// enol; the amide, not the imidic acid.
#[test]
fn normalization_prefers_the_carbonyl() {
    // Written either way, an amide stays an amide.
    for smiles in ["CC(=O)N", "CC(O)=N"] {
        let normal = normalized(smiles);
        assert!(normal.contains("O=C"), "{smiles} gave {normal}");
    }
    // 2-pyridone, not 2-hydroxypyridine: the nitrogen keeps the hydrogen.
    for smiles in ["O=c1cccc[nH]1", "Oc1ccccn1"] {
        let normal = normalized(smiles);
        assert!(normal.contains("[nH]"), "{smiles} gave {normal}");
        assert!(normal.starts_with("O="), "{smiles} gave {normal}");
    }
    // Uracil as the dione, counted on the graph rather than on the string --
    // the first carbonyl is written `O=c1`, not `=O`.
    let uracil = MolecularGraph::from_smiles_single(&normalized("Oc1ccnc(O)n1")).unwrap();
    let carbonyls = (0..uracil.atomic_numbers().len())
        .filter(|&atom| uracil.atomic_numbers()[atom] == 8)
        .filter(|&atom| {
            uracil.get_atom_bonds(atom).unwrap().iter().any(|bond| {
                bond.bond_order() == BondOrder::Double
                    && uracil.atomic_numbers()[bond.target()] == 6
            })
        })
        .count();
    assert_eq!(carbonyls, 2, "{}", uracil.to_smiles());
}

/// Normalization may move hydrogens about; it may not create or destroy them.
#[test]
fn normalization_preserves_the_formula() {
    use crate::molecular_formula::MolecularFormula;

    let rules = TautomerRules::new();
    for smiles in [
        "CC(O)=N",
        "Oc1ccccn1",
        "NC(=[NH2+])N",
        "CC(=[NH2+])N",
        "C[N+](=O)[O-]",
        "CC(=O)[O-]",
        "NS(=O)(=O)c1ccccc1",
        "N#Cc1ccc(O)cc1",
        "Nc1nc2nc[nH]c2c(O)n1",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let normal = normalize(&molecule, &rules).unwrap_or_else(|| panic!("{smiles}"));
        assert_eq!(
            MolecularFormula::from_molecule(&normal),
            MolecularFormula::from_molecule(&molecule),
            "{smiles} gave {}",
            normal.to_smiles()
        );
    }
}

/// Normalizing an already normal molecule changes nothing, and what comes out
/// is a molecule a parser will take back.
#[test]
fn normalization_is_idempotent_and_its_output_parses() {
    for smiles in [
        "CC(O)=N",
        "Oc1ccccn1",
        "O=c1cc[nH]c(=O)[nH]1",
        "Nc1nc2[nH]cnc2c(=O)[nH]1",
        "NS(=O)(=O)c1ccccc1",
        "N#Cc1ccc(O)cc1",
        "CC(=O)Oc1ccccc1C(=O)O",
    ] {
        let once = normalized(smiles);
        MolecularGraph::from_smiles(&once).unwrap_or_else(|_| panic!("{once} should parse"));
        assert_eq!(normalized(&once), once, "{smiles}");
    }
}

/// Which of several interchangeable arms holds a mobile hydrogen is the
/// skeleton's choice, so `normalize` has to make it the same way every time.
///
/// This porphyrazine carries four N-O arms: three nitroxide radicals and one
/// hydroxylamine, so the arms hold one mobile hydrogen and three mobile
/// electrons between them -- and erasure takes all four out, leaving four arms
/// the skeleton cannot tell apart. `normalize` put the
/// hydrogen on a different arm from the deposited form than from its own
/// canonical SMILES, so it was not idempotent, though the key was right both
/// times.
///
/// The ambiguity test that settles this could not be asked of the skeleton the
/// way the chirality one is: a chirality tag survives erasure and a mobile
/// hydrogen is what erasure removes, so it has to be asked of the molecule the
/// placement built. PubChem CID 177484357.
#[test]
fn an_arm_that_holds_a_mobile_hydrogen_is_chosen_the_same_way_twice() {
    let rules = TautomerRules::new();
    let smiles = "CC1(C2=CC3=C4NC(=C3C=C2C(N1[O])(C)C)N=C5C6=CC7=C(C=C6C(=N5)N=C8C9=CC1=C(C=C9C(=NC2=NC(=N4)C3=CC4=C(C=C32)C(N(C4(C)C)O)(C)C)N8)C(N(C1(C)C)[O])(C)C)C(N(C7(C)C)[O])(C)C)C";
    let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
    let answer = crate::molecule::tautomer::normalize(&molecule, &rules).expect("an answer");
    let written = answer.to_smiles();
    let back = MolecularGraph::from_smiles_single(&written).expect("the answer parses back");
    assert_eq!(
        tautomer_key(&back, &rules),
        tautomer_key(&molecule, &rules),
        "the answer does not key as what it was written from"
    );
    let again = crate::molecule::tautomer::normalize(&back, &rules).expect("an answer");
    assert_eq!(
        again.to_smiles(),
        written,
        "normalizing again moved the hydrogen"
    );
}

/// A ring closed through a hypervalent centre keeps one group whichever of its
/// bonds the pi happens to sit on.
///
/// A sulfur(VI) in a ring has four sigma bonds and a valence of six, so two of
/// its ring bonds carry pi -- and *which* two is exactly the choice one writing
/// of the molecule makes and another does not. The carbon that loses the pi
/// gains a hydrogen, which takes it out of `is_conjugatable_under`, which takes
/// it out of the conjugated component, so the group came out one size from the
/// deposited form and another from its own canonical SMILES, and the two did
/// not share a key. PubChem CID 123563520 is the case this is cut down from.
#[test]
fn a_hypervalent_ring_centre_keeps_one_group() {
    let rules = TautomerRules::new();
    let thiadiazine =
        MolecularGraph::from_smiles_single("CS=12=CC(=NN(C1)C)C2").expect("the SMILES parses");
    let answer = crate::molecule::tautomer::normalize(&thiadiazine, &rules)
        .expect("the molecule is a feasible point of its own system");
    let written = answer.to_smiles();
    let back = MolecularGraph::from_smiles_single(&written).expect("the answer parses back");
    assert_eq!(
        tautomer_key(&back, &rules),
        tautomer_key(&thiadiazine, &rules),
        "{written} does not key as what it was written from"
    );
    // The group has to be the whole bicycle, both times: the sulfur, both
    // carbons its pi can reach, and the ring between them.
    let before = perceive(&thiadiazine, &rules);
    let after = perceive(&back, &rules);
    assert_eq!(before.groups().len(), 1);
    assert_eq!(
        before.groups()[0].atoms.len(),
        after.groups()[0].atoms.len(),
        "the group changed size across a write and a read"
    );
}

/// The two halves have to agree: a molecule and its normalization are the same
/// compound, so they share a key.
#[test]
fn a_molecule_and_its_normalization_share_a_key() {
    let rules = TautomerRules::new();
    for smiles in [
        "CC(O)=N",
        "Oc1ccccn1",
        "c1c[nH]cn1",
        "Nc1ccnc(O)n1",
        "Oc1cc(=O)oc2ccccc12",
        "N#Cc1ccc(O)cc1",
        "NS(=O)(=O)c1ccccc1",
        "CC(=O)[O-]",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let normal = normalize(&molecule, &rules).unwrap_or_else(|| panic!("{smiles}"));
        assert_eq!(
            tautomer_key(&normal, &rules),
            tautomer_key(&molecule, &rules),
            "{smiles} gave {}",
            normal.to_smiles()
        );
    }
}

/// Bonds a proton shift cannot reach must come through untouched. If the group
/// erased such a bond order, the solver could spend the freed valence on a
/// hydrogen out of the pool -- same formula, same totals, different compound.
#[test]
fn bonds_a_shift_cannot_reach_survive() {
    // A nitrile is not an imine: `C#N` must not come back as `=C=N`.
    //
    // Including on a ring that also carries an endpoint. `N#Cc1ccc(O)cc1` used
    // to come back as a quinone-imine: the phenol oxygen and the nitrile
    // nitrogen are one conjugated component, and "prefer the carbonyl" spent
    // the ring carbon's π bond on the oxygen, leaving the ring quinoid and the
    // nitrogen short by a hydrogen. Closing the ring -- forbidding π on the
    // bonds that leave it -- obliges the ring to satisfy itself instead.
    for smiles in [
        "N#CC(=O)N",
        "N#CSc1ccc(C)cc1",
        "N#CCC(=O)N",
        "N#Cc1ccc(O)cc1",
        "N#Cc1cc[nH]c1",
    ] {
        let normal = normalized(smiles);
        assert!(normal.contains("#N"), "{smiles} gave {normal}");
    }

    // And the ring it was stealing from keeps its aromaticity.
    for smiles in ["N#Cc1ccc(O)cc1", "Oc1ccccc1"] {
        let normal = normalized(smiles);
        assert!(
            normal.contains("c1"),
            "{smiles} gave {normal}, which is not aromatic"
        );
    }
    // A sulfonyl oxygen is not a hydroxyl: `S(=O)(=O)` must not become `S(O)(O)`.
    for smiles in ["NS(=O)(=O)c1ccccc1", "CS(=O)(=O)Nc1ccccc1"] {
        let normal = normalized(smiles);
        assert!(normal.contains("S(=O)(=O)"), "{smiles} gave {normal}");
    }
    // And a thioether sulfur stays divalent rather than being promoted so that
    // it can carry a double bond it never wanted.
    for smiles in ["N#CSc1ccc(C)cc1", "CSc1nc2ccccc2[nH]1"] {
        let normal = normalized(smiles);
        assert!(!normal.contains("[SH]"), "{smiles} gave {normal}");
    }
}

/// A charge has to land somewhere, and where it lands is part of the answer.
#[test]
fn charges_are_placed() {
    // Conserved where the group already had one.
    assert!(normalized("CC([O-])=O").contains("[O-]"));
    assert!(normalized("[n-]1nncn1").contains("[n-]"));
    // Separated where the molecule cannot be written without it, and not
    // otherwise: `MAX_CHARGE_SEPARATION` is tried from zero up.
    let nitro = normalized("C[N+]([O-])=O");
    assert!(nitro.contains("[N+]") && nitro.contains("[O-]"), "{nitro}");
    assert!(!normalized("CC(=O)N").contains('+'));
    assert!(!normalized("NS(=O)(=O)c1ccccc1").contains('+'));
}

/// Salts normalize component by component.
#[test]
fn components_normalize_independently() {
    assert_eq!(
        normalized("CC(=O)[O-].[Na+]"),
        normalized("[Na+].CC([O-])=O")
    );
}

/// The solver underneath, on its own terms.
#[test]
fn the_degree_constrained_solver_is_exact() {
    use crate::molecule::matching::{
        degree_constrained_subgraph, lexicographic_degree_constrained_subgraph,
    };

    // Benzene: six atoms in a ring, each wanting one π bond -- three of them.
    let ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
    let taken = degree_constrained_subgraph(6, &ring, &[1; 6]).expect("benzene kekulizes");
    assert_eq!(taken.iter().filter(|&&t| t).count(), 3);

    // An odd ring cannot give every atom a π bond, and a greedy walk that
    // thought it could is what wrote a five-valent carbon into pyrrole.
    let five = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
    assert!(degree_constrained_subgraph(5, &five, &[1; 5]).is_none());

    // Two π bonds on one edge is a triple bond, and parallel edges are how the
    // f-factor says so.
    let parallel = [(0, 1), (0, 1)];
    let taken = degree_constrained_subgraph(2, &parallel, &[2, 2]).expect("a triple bond");
    assert_eq!(taken, vec![true, true]);

    // The lexicographic form takes the first edge it is offered when it can.
    let taken = lexicographic_degree_constrained_subgraph(6, &ring, &[1; 6], &[(0, true)])
        .expect("benzene kekulizes");
    assert!(taken[0]);
    let taken = lexicographic_degree_constrained_subgraph(6, &ring, &[1; 6], &[(1, true)])
        .expect("benzene kekulizes");
    assert!(taken[1]);

    // And leaves one out when asked, which is what closing a ring needs: the
    // preference is a value, not merely an order.
    let taken = lexicographic_degree_constrained_subgraph(6, &ring, &[1; 6], &[(0, false)])
        .expect("benzene kekulizes the other way");
    assert!(!taken[0]);
}

// ---------------------------------------------------------------------------
// Regressions found by running PubChem.
//
// Every one of these passed the rest of this file and the whole curated corpus
// before it was fixed. They are here because a corpus of 91,000 hand-picked
// fragments held none of them and 124 million deposited compounds hold all of
// them -- and because NEXT.md, where they are written up, does not run.
// ---------------------------------------------------------------------------

/// Unpaired electrons spend valence, wherever they sit.
///
/// A `[N]` radical has one bond and two unpaired electrons. Reading its valence
/// off the element alone asked for the two further π bonds a three-valent
/// nitrogen implies, and its single bond could not carry them, so the group had
/// no solution at all. 187 of PubChem's first 204 unsolvable groups were this.
#[test]
fn radicals_spend_valence() {
    let rules = TautomerRules::new();
    for smiles in ["[CH]CCCCC(=O)[N]", "[CH]C1=C(C(=CC=C1)O[CH])O", "[N]C(=O)C"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        assert!(
            normalize(&molecule, &rules).is_some(),
            "{smiles} has no assignment, but the input is one"
        );
    }
}

/// A proton moved from one group to another is the same compound, so the two
/// structures get one key and one normalized form.
///
/// Both references this convention exists to agree with say so: InChI gives
/// each of these pairs one identifier, and so does RDKit's `HetAtomTautomer`
/// hash. Keeping the totals one set per group gave them two, and three of the
/// four classes a PubChem sample split were this.
#[test]
fn a_proton_moved_between_groups_keeps_one_key() {
    for (first, second) in [
        // A succinate monoanion, the charge on either carboxyl.
        ("OC(=O)CCC(=O)[O-]", "[O-]C(=O)CCC(=O)O"),
        // An amidine and an acid, neutral or as the zwitterion.
        ("NC(=N)NCCC(=O)O", "NC(=[NH2+])NCCC(=O)[O-]"),
        // The case from the corpus: a glycine diamide with three carboxyl
        // oxygens between two groups and a proton that can sit on either.
        (
            "O=C([O-])CN(C=C([O-])O)C=C(O)O",
            "O=C(O)CN(C=C([O-])O)C=C([O-])O",
        ),
    ] {
        assert_eq!(key(first), key(second), "{first}, {second}");
        assert_eq!(normalized(first), normalized(second), "{first}, {second}");
    }
}

/// A proton on a site with no partner of its own is mobile too, once
/// [`TautomerRules::ionizable_endpoints`] admits it -- which is what merges a
/// zwitterion with its neutral form.
#[test]
fn ionizable_endpoints_merge_a_zwitterion_with_its_neutral_form() {
    let ionizable = TautomerRules::with_ionizable_endpoints();
    let ionized =
        |smiles: &str| tautomer_key_from_smiles(smiles, &ionizable).expect("the SMILES parses");
    for (neutral, zwitterion) in [
        ("NCC(=O)O", "[NH3+]CC(=O)[O-]"),
        ("NCCS(=O)(=O)O", "[NH3+]CCS(=O)(=O)[O-]"),
        ("OC(=O)CCCC[NH2]", "[O-]C(=O)CCCC[NH3+]"),
    ] {
        assert_eq!(ionized(neutral), ionized(zwitterion), "{neutral}");
        // So does the default key, which pools the whole record.
        assert_eq!(key(neutral), key(zwitterion), "{neutral}");
        // Heteroatom tautomers alone keep them apart: an aliphatic amine has
        // no second endpoint in its own conjugated system.
        let only = TautomerRules::heteroatom_only();
        assert_ne!(
            tautomer_key_from_smiles(neutral, &only).unwrap(),
            tautomer_key_from_smiles(zwitterion, &only).unwrap(),
            "{neutral}"
        );
    }
    // The marker says which relation was asked for, so the two cannot be
    // compared against each other by accident.
    assert!(ionized("NCC(=O)O").starts_with("i/"));
}

/// What the wider relation still refuses to merge: anything the formula tells
/// apart, and anything in another component.
#[test]
fn ionizable_endpoints_keep_the_formula() {
    let ionizable = TautomerRules::with_ionizable_endpoints();
    let ionized =
        |smiles: &str| tautomer_key_from_smiles(smiles, &ionizable).expect("the SMILES parses");
    // Pyridine and the pyridinium cation differ by a hydrogen, and the pooled
    // totals say so.
    assert_ne!(ionized("c1ccncc1"), ionized("c1cc[nH+]cc1"));
    // A free base with its acid beside it is not the salt: each component is
    // keyed on its own.
    assert_ne!(ionized("CCN.Cl"), ionized("CC[NH3+].[Cl-]"));
}

/// When the ordinary preferences leave the pools with nowhere to go, the
/// hydrogens and electrons are spread before any atom is let down its ladder.
///
/// Selenocarbonyl diselenide, `C(=[SeH])=[SeH]`, is two seleniums each holding
/// a hydrogen and an electron. Deciding the ladders first let the first drop to
/// a valence of two, and the second was then the only endpoint with room: it
/// took both hydrogens and both electrons and came out as `[Se]=C=[SeH2]`,
/// which is neither the compound nor a structure a SMILES can state, so the
/// group had no answer at all. PubChem CIDs 71438000 and 139263036.
#[test]
fn the_pools_are_spread_when_nothing_else_writes() {
    let rules = TautomerRules::new();
    for smiles in ["C(=[SeH])=[SeH]", "C1=CC=C2C(=C1)[Se]=C3C=CC=C(C3=[Se]2)S"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let electrons: u32 = molecule
            .radical_states()
            .iter()
            .map(|&state| u32::from(state))
            .sum();
        let answer = crate::molecule::tautomer::try_normalize(&molecule, &rules)
            .expect("the input is itself a feasible point");
        let written = answer.to_smiles();
        let back = MolecularGraph::from_smiles_single(&written)
            .unwrap_or_else(|_| panic!("{written} should parse"));
        assert_eq!(
            back.radical_states()
                .iter()
                .map(|&state| u32::from(state))
                .sum::<u32>(),
            electrons,
            "{smiles} came back as {written} with a different number of electrons"
        );
        assert_eq!(
            tautomer_key(&back, &rules),
            tautomer_key(&molecule, &rules),
            "{smiles}"
        );
    }
}

/// A labelled hydrogen stays on the atom the input put it on.
///
/// It used to go into the pool with the rest and come back out as an ordinary
/// hydrogen: `[2H]N=C(N)N` normalized to `NC(N)=N`, a labelled compound
/// silently turned into an unlabelled one. The formula check could not see it,
/// because a formula counts elements and both are hydrogen; the corpus found it
/// through a tritiated sulfoxide whose stereocentre then answered differently
/// depending on the order its atoms arrived in. PubChem CID 123949882.
#[test]
fn a_labelled_hydrogen_does_not_move() {
    let rules = TautomerRules::new();
    for smiles in ["[2H]N=C(N)N", "[3H][S@](=O)N=CC(C)(C)C", "[2H]Oc1ccccc1"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let answer = normalize(&molecule, &rules).expect("an answer");
        let written = answer.to_smiles();
        let labels = |graph: &MolecularGraph| -> usize {
            (0..graph.atomic_numbers().len())
                .filter(|&atom| {
                    graph.atomic_numbers()[atom] == 1 && graph.get_isotope(atom).is_some()
                })
                .count()
        };
        let back = MolecularGraph::from_smiles_single(&written)
            .unwrap_or_else(|_| panic!("{written} should parse"));
        assert_eq!(
            labels(&back),
            labels(&molecule),
            "{smiles} became {written}"
        );
    }

    // Two labellings of one tautomer class are two compounds, because the label
    // marks a position and the shift is not allowed to move it.
    assert_ne!(key("[2H]N=C(N)N"), key("N=C(N)[2H]"));
    // The unlabelled hydrogens beside it move as they always did.
    assert_eq!(key("N=C(N)N"), key("NC(=N)N"));
}

/// A structure whose valences do not exist is refused by name, rather than
/// reported as a search that failed.
///
/// Everything `normalize` promises rests on the input being a feasible point of
/// its own constraint system, and that needs the input to be a molecule.
/// PubChem deposits `[B]=[SH-](=P)=PF`, whose sulfur spends seven; RDKit
/// declines to read it at all. It is reported as invalid input rather than as
/// "no feasible assignment", which keeps the invariant the corpus sweep leans
/// on -- that an answerless molecule is a bug.
#[test]
fn a_structure_with_no_possible_valence_is_refused() {
    use crate::molecule::tautomer::{NormalizationError, impossible_valence, try_normalize};

    let rules = TautomerRules::new();
    for smiles in ["[B]=[SH-](=P)=PF", "[2H]P=[SH-](=[B])=[V]"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        assert!(impossible_valence(&molecule, &rules).is_some(), "{smiles}");
        assert!(
            matches!(
                try_normalize(&molecule, &rules),
                Err(NormalizationError::ImpossibleValence(_))
            ),
            "{smiles} should be refused, not searched for"
        );
    }

    // What the check must not refuse: a valence the table does not list on an
    // atom no group types. A perchlorate's chlorine spends seven, a dialkyl
    // H-phosphonate's phosphorus four, and both are ordinary chemistry.
    for smiles in ["O=Cl(=O)(=O)[O-]", "COP(=O)OC", "[Mn]", "CC(=O)N"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        assert_eq!(impossible_valence(&molecule, &rules), None, "{smiles}");
        assert!(try_normalize(&molecule, &rules).is_ok(), "{smiles}");
    }
}

/// An atom on the lowest valence its element has can carry every electron
/// between its bonds and that valence, because nothing below it explains the
/// same bonds.
///
/// A mesityl tellurium cation is one bond and two unpaired electrons on a
/// tellurium whose charged valences start at three. Capping every endpoint at
/// one electron -- which is right for an atom on a higher rung, where the rung
/// below would be read instead -- left this group with nowhere to put the
/// second, and it had no assignment at all. PubChem CID 11147098.
#[test]
fn an_atom_on_its_lowest_valence_keeps_every_electron() {
    let rules = TautomerRules::with_ionizable_endpoints();
    let smiles = "CC1=CC(=C(C(=C1)C)[Te+])C";
    let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
    let electrons: u32 = molecule
        .radical_states()
        .iter()
        .map(|&state| u32::from(state))
        .sum();
    assert_eq!(electrons, 2);

    let answer = crate::molecule::tautomer::try_normalize(&molecule, &rules)
        .expect("the input is itself a feasible point");
    let written = answer.to_smiles();
    let back = MolecularGraph::from_smiles_single(&written)
        .unwrap_or_else(|_| panic!("{written} should parse"));
    assert_eq!(
        back.radical_states()
            .iter()
            .map(|&state| u32::from(state))
            .sum::<u32>(),
        electrons,
        "{written} came back with a different number of electrons"
    );
    assert_eq!(tautomer_key(&back, &rules), tautomer_key(&molecule, &rules));
}

/// A lone ion is nobody's group: there is nothing for its proton to move
/// between, and the valence model has no answer for an atom that spends
/// nothing.
#[test]
fn a_bare_ion_is_its_own_compound() {
    let ionizable = TautomerRules::with_ionizable_endpoints();
    for smiles in ["[NH4+]", "[O-2]", "[Se+4]"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        assert!(
            perceive(&molecule, &ionizable).is_empty(),
            "{smiles} should have no group"
        );
        // And it still normalizes, rather than failing for want of an answer.
        assert!(normalize(&molecule, &ionizable).is_some(), "{smiles}");
    }
}

/// The default key pools the whole record, so a proton crosses from one
/// component to another: benzoic acid beside ammonia is ammonium benzoate,
/// and an amine beside hydrogen chloride its hydrochloride. Pooling each
/// component on its own keeps them apart.
#[test]
fn a_proton_crosses_between_components_only_when_the_record_is_pooled() {
    for (first, second) in [
        ("OC(=O)c1ccccc1.N", "[O-]C(=O)c1ccccc1.[NH4+]"),
        ("CN.Cl", "C[NH3+].[Cl-]"),
    ] {
        assert_eq!(key(first), key(second), "{first}");
        let apart = TautomerRules::with_ionizable_endpoints();
        assert_ne!(
            tautomer_key_from_smiles(first, &apart).unwrap(),
            tautomer_key_from_smiles(second, &apart).unwrap(),
            "{first}"
        );
    }
    // Only moved, never added: the record keeps its formula and its charge.
    for (first, second) in [
        ("CC(=O)O", "CC(=O)[O-]"),
        ("CN", "C[NH3+]"),
        ("C.[Ti]", "[CH3-].[Ti]"),
        ("CC(=O)O.[Na+]", "CC(=O)[O-].[Na+]"),
    ] {
        assert_ne!(key(first), key(second), "{first}");
    }
}

/// The totals are written once, after the groups.
#[test]
fn the_key_writes_the_component_totals_once() {
    // Two carboxyl groups, one proton and one charge between them.
    let monoanion = key("OC(=O)CCC(=O)[O-]");
    let fields: Vec<&str> = monoanion.split('/').collect();
    // The SMILES, a field per group, and the totals.
    assert_eq!(fields.len(), 4, "{monoanion}");
    assert_eq!(fields[1], "0,0");
    assert_eq!(fields[2], "0,0");
    assert_eq!(fields[3], "1:-1");
}

/// An unpaired electron moves between endpoints exactly as a hydrogen does, so
/// the structures it distinguishes are one compound and get one key.
///
/// Before `t4` a component holding one was refused a group outright, and each
/// of these wrote its plain canonical SMILES as its key -- three keys for one
/// compound, and none of them saying that a shift relates them.
#[test]
fn radical_tautomers_share_a_key() {
    for forms in [
        // A pyrimidinedione with one of its oxygens reduced to an oxyl
        // radical: the hydrogen and the electron move around the ring
        // together.
        vec!["[O]c1nccc(O)n1", "[O]c1nccc(=O)[nH]1", "[O]c1nc(=O)cc[nH]1"],
        // Cyanamide's aminyl radical, written with the electron on either
        // nitrogen.
        vec!["[NH]C#N", "N=C=[N]"],
    ] {
        let first = key(forms[0]);
        for form in &forms[1..] {
            assert_eq!(key(form), first, "{} and {}", forms[0], form);
        }
        // And one structure, not merely one key.
        let answer = normalized(forms[0]);
        for form in &forms[1..] {
            assert_eq!(normalized(form), answer, "{} and {}", forms[0], form);
        }
    }
}

/// The electron is counted against the group and taken out of the skeleton,
/// which is what makes two members of a class present the same graph.
#[test]
fn a_mobile_electron_is_counted_rather_than_placed() {
    let rules = TautomerRules::new();
    let molecule = MolecularGraph::from_smiles_single("[NH]C#N").expect("the SMILES parses");
    let groups = perceive(&molecule, &rules);
    assert_eq!(groups.groups().len(), 1);
    assert_eq!(groups.groups()[0].mobile_radicals, 1);
    assert_eq!(groups.groups()[0].mobile_hydrogens, 1);

    let (skeleton, groups) = erased_skeleton(&molecule, &rules);
    assert_eq!(groups.groups()[0].mobile_radicals, 1);
    for &endpoint in &groups.groups()[0].endpoints {
        assert_eq!(
            skeleton
                .radical_states()
                .get(endpoint)
                .copied()
                .unwrap_or(0),
            0,
            "the skeleton still says which endpoint holds the electron"
        );
    }
}

/// Normalizing puts the electrons back, all of them, and lands on a fixed
/// point.
#[test]
fn normalizing_a_radical_keeps_its_electrons_and_repeats() {
    let rules = TautomerRules::new();
    for smiles in [
        "[O]c1nccc(O)n1",
        "[NH]C#N",
        "CC(=O)[N]",
        // A selenophosphorus cage, four unpaired electrons over five seleniums.
        "[P+]12=[Se][Se]=[P+]([Se]1)[Se]=[Se]2",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let electrons: u32 = molecule
            .radical_states()
            .iter()
            .map(|&state| u32::from(state))
            .sum();
        let answer = normalize(&molecule, &rules).expect("an answer");
        let written = answer.to_smiles();
        let back = MolecularGraph::from_smiles_single(&written)
            .unwrap_or_else(|_| panic!("{written} should parse"));
        // Written and read back, the answer has to be the compound it was
        // solved from: SMILES states an unpaired electron only as the gap
        // between the bonds and the valence they reach into, so an electron
        // piled where no such gap can be written is one the reader loses.
        assert_eq!(
            back.radical_states()
                .iter()
                .map(|&state| u32::from(state))
                .sum::<u32>(),
            electrons,
            "{smiles} came back with a different number of electrons as {written}"
        );
        assert_eq!(
            tautomer_key(&back, &rules),
            tautomer_key(&molecule, &rules),
            "{smiles} normalized to something of another class"
        );
        assert_eq!(
            normalized(&written),
            written,
            "{smiles} is not a fixed point"
        );
    }
}

/// An electron the group does not count as mobile still has to be visible to
/// the skeleton, or the erasure invents a symmetry the molecule has not got.
///
/// This cage carries three imidazolylidene rings and one imidazolium. Erasure
/// takes the charge off the imidazolium and the orders off every ring bond, and
/// what is left of a carbene carbon -- one carbon, two ring neighbours, one
/// hydrogen -- is exactly what is left of the imidazolium's C2. With the
/// electron invisible the four rings became one orbit, the `+1` was paired with
/// a carbene ring, and that group had no solution at all. PubChem CID 60018328.
#[test]
fn a_carbene_ring_does_not_answer_to_a_charged_one() {
    let rules = TautomerRules::new();
    let smiles = "C1N2C=CN([CH]2)CN3C=CN([CH]3)C[N+]4=CN(CN5C=CN1[CH]5)C=C4";
    let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
    let answer = crate::molecule::tautomer::try_normalize(&molecule, &rules)
        .expect("the input is itself a feasible point");
    let written = answer.to_smiles();
    let back = MolecularGraph::from_smiles_single(&written)
        .unwrap_or_else(|_| panic!("{written} should parse"));
    assert_eq!(tautomer_key(&back, &rules), tautomer_key(&molecule, &rules));
}

/// A valence the input itself uses is legal whether or not the table lists it.
///
/// PubChem writes dialkyl H-phosphonates without the P-H, so their phosphorus
/// spends four; the table offers three, five and seven, and the check threw the
/// correct answer away.
#[test]
fn a_valence_the_input_uses_is_accepted() {
    let rules = TautomerRules::new();
    for smiles in [
        "COP(=O)OC",
        "CC(C)OP(=O)OC(C)C",
        "C1=CC=C(C=C1)OP(=O)OC2=CC=CC=C2",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        assert!(
            normalize(&molecule, &rules).is_some(),
            "{smiles} has no assignment"
        );
    }
}

/// A symmetric skeleton cannot say which of its groups holds what, so the
/// answer must not depend on which atom happened to be written first.
///
/// The mono-anion of a symmetric diacid is the case: erasure leaves the two
/// carboxyls indistinguishable, and the totals -- one with a hydrogen, one with
/// the charge -- are exactly the data erasure removed from the graph. The key
/// takes the smallest field ordering over the skeleton's automorphisms.
#[test]
fn a_symmetric_skeleton_keys_the_same_either_way() {
    for (first, second) in [
        ("OC(=O)CCC(=O)[O-]", "[O-]C(=O)CCC(=O)O"),
        ("OC(=O)CC(=O)[O-]", "[O-]C(=O)CC(=O)O"),
        ("OC(=O)CCCCC(=O)[O-]", "[O-]C(=O)CCCCC(=O)O"),
        (
            "OC(=O)[C@H](O)[C@@H](O)C(=O)[O-]",
            "[O-]C(=O)[C@H](O)[C@@H](O)C(=O)O",
        ),
        (
            "OC(=O)CN(CC(=O)[O-])CCN(CC(=O)[O-])CC(=O)[O-]",
            "[O-]C(=O)CN(CC(=O)O)CCN(CC(=O)[O-])CC(=O)[O-]",
        ),
    ] {
        assert_eq!(key(first), key(second), "{first} and {second}");
    }
}

/// The geometry of a double bond the shift can move is erased with its order.
///
/// A shift moves the bond and nothing holds the substituents on their side of
/// it while it moves, so cis and trans are not conserved. Erasing the order but
/// keeping the tag meant the key read a geometry off the input that `normalize`
/// then destroyed, and the two disagreed -- on about one molecule in twenty
/// through the stereochemically busy parts of PubChem, and on none at all in
/// the first two hundred thousand, which is why no curated corpus caught it.
#[test]
fn geometry_the_shift_can_move_is_erased_with_the_bond() {
    let rules = TautomerRules::new();
    for smiles in [
        "CC1=NN=C2N1NC(=O)/C(=C/C3=CC=C(C=C3)Cl)/S2",
        "CC1=NN=C2N1NC(=O)/C(=C\\C3=CC=C(C=C3)Cl)/S2",
        "CN/C=C(/C(=O)C(F)(F)F)\\Cl",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let normal = normalize(&molecule, &rules).unwrap_or_else(|| panic!("{smiles}"));
        assert_eq!(
            tautomer_key(&normal, &rules),
            tautomer_key(&molecule, &rules),
            "{smiles} normalized to {}",
            normal.to_smiles()
        );
    }
    // An enaminone's C=C is one the shift reaches: the amine's hydrogen goes to
    // the ketone oxygen and the double bond goes with it, so the two geometries
    // are one compound and the erasure is what says so.
    assert_eq!(
        key("CN/C=C(/C(=O)C(F)(F)F)\\Cl"),
        key("CN/C=C(\\C(=O)C(F)(F)F)/Cl"),
    );
    assert_ne!(key("F/C=C/F"), key("F/C=C\\F"));
}

/// A geometry is only as good as the neighbourhood its class is read against.
///
/// The class on each end of a double bond says which side its substituents sit,
/// and which substituent it is talking about is read off the *first single
/// bond* at that atom. Erasing a neighbouring double bond puts a new bond at
/// the head of that list, so a kept class goes on being believed while meaning
/// the opposite -- and which neighbour comes first is a property of the order
/// the input was written in, not of the compound. PubChem's phosphorus ylides
/// are where it shows: the `P=N` is erased out from under the mark on the
/// `P=C`, and one molecule written two ways came back with one `/` and one
/// `\\`.
#[test]
fn a_geometry_whose_neighbours_are_erased_is_not_kept() {
    for (first, second) in [
        (
            "CC(C)(C)/C=P(=NC(C)(C)C)/N(C(C)(C)C)C(C)(C)C",
            "CC(C)(C)/C=P(=NC(C)(C)C)\\N(C(C)(C)C)C(C)(C)C",
        ),
        (
            "C/C(=P(=S)/N([Si](C)(C)C)[Si](C)(C)C)/C(C)(C)C",
            "C/C(=P(=S)\\N([Si](C)(C)C)[Si](C)(C)C)/C(C)(C)C",
        ),
    ] {
        assert_eq!(key(first), key(second), "{first} and {second}");
    }
}

/// A geometry no shift can reach is kept, and the bond's order with it.
///
/// Erasure is about the orders a shift makes meaningless, and a bond it cannot
/// move has none. Erasing those as well merged compounds that are not
/// tautomers of each other at all: fumaric with maleic acid, the two methyl
/// crotonates -- an ester, with no mobile hydrogen anywhere in it -- and E with
/// Z cinnamamide. Over a PubChem sample it cost 94% of the molecules that state
/// a double-bond geometry theirs, a third of them with nothing mobile in the
/// group. RDKit's `HetAtomTautomer` erases these too and its v2 does not; these
/// are the answers v2 gives.
#[test]
fn a_geometry_no_shift_can_reach_is_kept() {
    for (first, second) in [
        // Nothing is mobile at all: a tertiary amide has no N-H and no O-H.
        ("CN(C)C(=O)/C=C/c1ccccc1", "CN(C)C(=O)/C=C\\c1ccccc1"),
        // A mobile hydrogen that has nowhere to go but the other oxygen.
        ("OC(=O)/C=C/C(=O)O", "OC(=O)/C=C\\C(=O)O"),
        ("C/C=C/C(=O)O", "C/C=C\\C(=O)O"),
        ("C/C=C/C(=O)OC", "C/C=C\\C(=O)OC"),
    ] {
        assert_ne!(key(first), key(second), "{first} and {second}");
    }

    // What is still erased, and why. A geometry whose bond has a ring atom at
    // either end goes the way it always did: the group may aromatize that ring,
    // and a mark on an aromatic atom is one the writer cannot place, so
    // `normalize` handed back a structure without the tag its input had and the
    // next pass answered differently. That cost 32 molecules of a PubChem
    // sample their idempotence and 10 more their keys, against the benzylidenes
    // and arylidenes it would have kept. The bond's own ring membership is not
    // enough -- an exocyclic bond is not a ring bond, and it was the ring atom
    // at its far end that the aromatization reached.
    for (first, second) in [(
        "CC1=NN=C2N1NC(=O)/C(=C/C3=CC=C(C=C3)Cl)/S2",
        "CC1=NN=C2N1NC(=O)/C(=C\\C3=CC=C(C=C3)Cl)/S2",
    )] {
        assert_eq!(key(first), key(second), "{first} and {second}");
    }

    // And `normalize` writes the geometry it kept, rather than handing back a
    // structure that has quietly lost it.
    let rules = TautomerRules::new();
    for smiles in ["CN(C)C(=O)/C=C/c1ccccc1", "CN(C)C(=O)/C=C\\c1ccccc1"] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let normal = normalize(&molecule, &rules).expect("it normalizes");
        assert!(
            normal.to_smiles().contains('/'),
            "{smiles} normalized to {}",
            normal.to_smiles()
        );
    }
}

/// A key made under other than the default rules says which, so keys from
/// different relations cannot be compared by accident; the default rules write
/// no marker at all.
#[test]
fn keys_say_which_rules_made_them() {
    let both = TautomerRules {
        carbon_endpoints: true,
        ..TautomerRules::with_ionizable_endpoints()
    };
    assert_eq!(key("c1ccccc1"), "c1ccccc1");
    for (rules, marker) in [
        (TautomerRules::with_carbon_endpoints(), "c/"),
        (TautomerRules::with_ionizable_endpoints(), "i/"),
        (TautomerRules::with_charge_reduction(), "p/"),
        (TautomerRules::heteroatom_only(), "h/"),
        (both, "ci/"),
    ] {
        let marked = tautomer_key_from_smiles("CC(=O)N", &rules).expect("the SMILES parses");
        assert!(marked.starts_with(marker), "{marked}");
    }

    // Whatever the rules, the property the key exists for still holds.
    for rules in [
        TautomerRules::new(),
        TautomerRules::with_carbon_endpoints(),
        TautomerRules::with_ionizable_endpoints(),
        TautomerRules::with_charge_reduction(),
        TautomerRules::heteroatom_only(),
    ] {
        assert_eq!(
            tautomer_key_from_smiles("O=c1cccc[nH]1", &rules).unwrap(),
            tautomer_key_from_smiles("Oc1ccccn1", &rules).unwrap(),
        );
    }
}

/// A double bond's geometry survives canonization even when the mark has to be
/// written on a ring closure.
///
/// In a ring exactly one bond is the closure and the rest are tree edges, so
/// the double bond beside it has a carbon whose only substituent lies across
/// that closure. The writer would not put a mark on a ring-closure digit, found
/// no reference it could use, and passed the bond over unmarked -- the geometry
/// went out silently, and one more mark was lost on each further pass.
///
/// This is not a tautomer property; it lives here because a corpus sweep of the
/// tautomer machinery is what found it. `normalize` leaves these molecules
/// alone -- they have no tautomeric group at all -- so every failure was the
/// canonical writer's.
#[test]
fn ring_closure_double_bonds_keep_their_geometry() {
    use crate::graph_algorithms::mol_canonization::canonize_mol;

    let cis_trans = |molecule: &MolecularGraph| {
        (0..molecule.atomic_numbers().len())
            .filter(|&atom| {
                matches!(
                    molecule.get_chiral_class(atom),
                    ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
                )
            })
            .count()
    };

    // Eight-membered rings only: they are slack enough for cis and trans to be
    // different compounds, so the geometry is real and has to survive. A double
    // bond in a smaller ring is held cis by the ring itself, carries no
    // geometry to keep, and is written without marks -- which is what RDKit
    // does with cycloheptene too.
    for smiles in [
        "C1/C=C\\CC/C=C\\C1",
        "C1/C=C\\CC/C=C/C1",
        "C1C/C=C\\C=C/CC1",
    ] {
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("the SMILES parses");
        let canonical = canonize_mol(&molecule);
        let written = canonical.to_smiles();
        let reparsed = MolecularGraph::from_smiles_single(&written).expect("the output parses");

        assert_eq!(
            cis_trans(&reparsed),
            cis_trans(&molecule),
            "{smiles} wrote {written}, losing geometry"
        );
        // Idempotence is *not* asserted, and that is a known gap rather than
        // an oversight. The marks now survive, but the writer and the parser
        // read a mark on a ring-closure digit with opposite conventions, so
        // the string alternates between two spellings of the same geometry --
        // `C1CCCC/C=C\\1` and `C1CCCC\\C=C\\1` -- with a period of two.
        // Geometry is the part worth pinning until the conventions are
        // reconciled.
    }

    // The two geometries must stay distinct, not merely both be marked. The
    // first attempt at this fix kept every mark and inverted them: cis,cis
    // cycloocta-1,5-diene came back as cis,trans, and only a comparison with a
    // reference implementation showed it.
    let cis_cis = canonize_mol(&MolecularGraph::from_smiles_single("C1/C=C\\CC/C=C\\C1").unwrap())
        .to_smiles();
    let cis_trans_form =
        canonize_mol(&MolecularGraph::from_smiles_single("C1/C=C\\CC/C=C/C1").unwrap()).to_smiles();
    assert_ne!(cis_cis, cis_trans_form);

    // And a double bond the ring holds cis is written without marks, rather
    // than with marks that then fight the genuinely stereogenic bond beside it.
    let cycloheptene =
        canonize_mol(&MolecularGraph::from_smiles_single("C1CC/C=C\\CC1").unwrap()).to_smiles();
    assert!(
        !cycloheptene.contains('/') && !cycloheptene.contains('\\'),
        "{cycloheptene} marks a bond the ring already decides"
    );
}

#[test]
fn fused_phosphorus_closure_is_independent_of_numbering() {
    use crate::graph_algorithms::mol_canonization::reorder_molecule;
    use crate::molecule::tautomer::normalize;
    use rand::{SeedableRng, seq::SliceRandom};
    let record = include_str!("../../tests/pubchem_normalization_regressions.tsv")
        .lines()
        .find(|line| line.starts_with("170056601\t"))
        .unwrap();
    let molecule = MolecularGraph::from_smiles_single(record.split_once('\t').unwrap().1).unwrap();
    let rules = TautomerRules::new();
    let expected_key = tautomer_key(&molecule, &rules);
    let expected = normalize(&molecule, &rules).unwrap().to_smiles();
    let reparsed = MolecularGraph::from_smiles_single(&expected).unwrap();
    assert_eq!(tautomer_key(&reparsed, &rules), expected_key);
    assert_eq!(normalize(&reparsed, &rules).unwrap().to_smiles(), expected);
    let mut rng = rand::rngs::StdRng::seed_from_u64(170056601);
    for _ in 0..12 {
        let mut copy = molecule.clone();
        let mut order: Vec<_> = (0..copy.len()).collect();
        order.shuffle(&mut rng);
        reorder_molecule(&mut copy, &order);
        assert_eq!(tautomer_key(&copy, &rules), expected_key);
        assert_eq!(normalize(&copy, &rules).unwrap().to_smiles(), expected);
    }
}

/// A carbon endpoint's stereocentre is made by a mobile hydrogen, so it does
/// not survive the shift: both enantiomers of a ketone with an alpha
/// stereocentre and their shared enol are one tautomer class. The key gave
/// them three strings; RDKit's `HetAtomTautomerv2` gives them one.
/// Heteroatom rules leave that carbon alone, and with it the stereocentre.
#[test]
fn a_carbon_endpoint_loses_its_stereocentre_to_the_shift() {
    let forms = ["CC[C@H](C)C(=O)C", "CC[C@@H](C)C(=O)C", "CCC(C)=C(O)C"];
    for form in &forms[1..] {
        assert_eq!(carbon_key(forms[0]), carbon_key(form), "{form}");
    }
    let rules = TautomerRules::with_carbon_endpoints();
    assert_eq!(
        normalized_smiles(forms[0], &rules).unwrap(),
        normalized_smiles(forms[1], &rules).unwrap()
    );
    assert_ne!(key(forms[0]), key(forms[1]));
}

/// With carbon endpoints a ring carbon's hydrogens are mobile, and the solver,
/// placing π before hydrogens, used to give a ring bond both its π units or a
/// ring carbon π on both sides: phenol came back as `O=C1CCCC#C1` and
/// pyridinium as `[N+]#1C=CCCC#1`. Pyridinium still comes back with a
/// cumulated ring nitrogen, `[N+]=1=CCC=CC=1`, so it is held to the first rule
/// only.
#[test]
fn carbon_endpoints_write_no_ring_triple_bonds_or_ring_allenes() {
    let rules = TautomerRules::with_carbon_endpoints();
    let ring_bond_orders = |smiles: &str| {
        let normalized = normalized_smiles(smiles, &rules).unwrap();
        let molecule = MolecularGraph::from_smiles(&normalized).unwrap().remove(0);
        let (_, ring_bonds) = molecule.find_cyclic_bonds_and_atoms();
        let order = |first: usize, second: usize| {
            molecule
                .get_atom_bonds(first)
                .unwrap()
                .iter()
                .find(|bond| bond.target() == second)
                .unwrap()
                .bond_order()
        };
        let triple = ring_bonds
            .iter()
            .any(|&(first, second)| order(first, second) == BondOrder::Triple);
        let cumulated = (0..molecule.atomic_numbers().len()).any(|atom| {
            ring_bonds
                .iter()
                .filter(|&&(first, second)| {
                    (first == atom || second == atom) && order(first, second) == BondOrder::Double
                })
                .count()
                >= 2
        });
        (normalized, triple, cumulated)
    };
    for smiles in ["Oc1ccccc1", "C[n+]1ccccc1", "c1ccncc1", "c1cc[nH+]cc1"] {
        let (normalized, triple, cumulated) = ring_bond_orders(smiles);
        assert!(!triple, "{smiles} -> {normalized}");
        assert!(
            smiles == "c1cc[nH+]cc1" || !cumulated,
            "{smiles} -> {normalized}"
        );
    }
    assert_eq!(normalized_smiles("Oc1ccccc1", &rules).unwrap(), "Oc1ccccc1");
}

/// An N-oxide keeps its own negative charge, and an acid beside it its
/// proton: nicotinic acid N-oxide came back as `O[n+]1cccc(c1)C(=O)[O-]`. A
/// hydroxamic acid, with the same σ shape, still gives its proton up to a
/// carboxylate before an N-oxide would.
#[test]
fn an_n_oxide_keeps_its_charge_beside_an_acid() {
    assert_eq!(
        normalized("OC(=O)c1ccc[n+]([O-])c1"),
        normalized("[O-]C(=O)c1ccc[n+](O)c1")
    );
    assert_eq!(
        normalized("OC(=O)c1ccc[n+]([O-])c1"),
        "OC(=O)c1ccc[n+]([O-])c1"
    );
    assert_eq!(normalized("[O-]C(=O)CCC(=O)N(O)C"), "ON(C)C(=O)CCC(=O)[O-]");
}

/// An acylamino group on an aromatic ring is written as the amide, not as the
/// acyl imine on a ring that has taken the hydrogen; an imine on a saturated
/// ring, as in clonidine, is left as the major form it is.
#[test]
fn an_aminoheteroarene_is_written_as_the_amine() {
    assert_eq!(
        normalized("CC(=O)Nc1cc[nH]n1"),
        normalized("CC(=O)N=c1cc[nH][nH]1")
    );
    assert!(!normalized("CC(=O)Nc1cc[nH]n1").contains("N=c"));
    assert!(!normalized("CC(=O)Nc1nccs1").contains("N=c"));
    assert!(normalized("Clc1cccc(Cl)c1NC1=NCCN1").contains("N=C"));
}

/// `S=O` and `[S+][O-]` are one sulfoxide, and the bond between them is
/// frozen, so the charge-separated writing is rewritten before erasure.
#[test]
fn a_charge_separated_oxide_is_one_compound_with_its_double_bonded_form() {
    for (double, separated) in [
        ("CS(C)=O", "C[S+](C)[O-]"),
        ("CS(C)(=O)=O", "C[S+2](C)([O-])[O-]"),
        ("CP(C)(C)=O", "C[P+](C)(C)[O-]"),
        ("C[S@](=O)c1ccccc1", "C[S@+]([O-])c1ccccc1"),
    ] {
        assert_eq!(key(double), key(separated), "{double}");
        assert_eq!(normalized(double), normalized(separated), "{double}");
    }
    // An N-oxide has no double-bonded form to rewrite into.
    assert_eq!(normalized("[O-][n+]1ccccc1"), "[O-][n+]1ccccc1");
}

fn reduced_key(smiles: &str) -> String {
    tautomer_key_from_smiles(smiles, &TautomerRules::with_charge_reduction())
        .expect("the SMILES parses")
}

/// Every protonation state of a compound shares the charge-reduced key: a
/// proton gained or lost raises a group's hydrogens and its charge together,
/// and the key keeps only their difference.
#[test]
fn protonation_states_share_the_charge_reduced_key() {
    for forms in [
        &["CC(=O)O", "CC(=O)[O-]"][..],
        &["CCN", "CC[NH3+]"],
        &["CCN.Cl", "CC[NH3+].[Cl-]"],
        &[
            "NCC(=O)O",
            "[NH3+]CC(=O)[O-]",
            "[NH3+]CC(=O)O",
            "NCC(=O)[O-]",
        ],
        &["c1ccncc1", "c1cc[nH+]cc1"],
        &["OP(=O)(O)O", "OP(=O)([O-])[O-]", "[O-]P(=O)([O-])[O-]"],
        &["N", "[NH4+]"],
        &["O", "[OH-]"],
        &["Cl", "[Cl-]"],
        &["C[N+](C)(C)CC(=O)O", "C[N+](C)(C)CC(=O)[O-]"],
        &["Oc1ccccn1", "O=c1cccc[nH]1", "[O-]c1cccc[nH+]1"],
    ] {
        for form in &forms[1..] {
            assert_eq!(reduced_key(forms[0]), reduced_key(form), "{form}");
        }
    }
}

/// What no proton can change keeps the keys apart: a quaternary nitrogen's
/// charge, a metal's, and a different number of heavy atoms or bonds.
#[test]
fn the_charge_reduced_key_keeps_what_protons_cannot_change() {
    for (first, second) in [
        ("C[N+](C)(C)C", "CN(C)C"),
        ("CC(=O)[O-].[Na+]", "CC(=O)O"),
        ("CC(=O)O", "CC(=O)OC"),
        ("CC(=O)C", "CC(O)=C"),
        ("Clc1ccccc1", "c1ccccc1.Cl"),
    ] {
        assert_ne!(reduced_key(first), reduced_key(second), "{first} {second}");
    }
}
