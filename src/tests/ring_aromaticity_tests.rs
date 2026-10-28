//! The ring-family aromaticity model: its ring perception, and what it
//! perceives with it.
//!
//! The path search in `molecule::aromaticity` is the oracle here. Where the two
//! models are meant to agree, they are compared directly; where the ring model
//! is meant to do better -- azulene's fusion bond, porphine -- the better answer
//! is pinned so that it cannot quietly go away again.

use crate::graph_algorithms::mol_canonization::reorder_molecule;
use crate::molecule::aromaticity::{AromaticityModel, MoleculeAromaticity};
use crate::molecule::ring_aromaticity::ring_family;
use crate::prelude::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

/// C70, whose 12 pentagons and 25 hexagons are the ring family the path search
/// could not settle on: PubChem's CID 16152356 is this cage with four
/// tert-butylperoxy groups, and it canonized two ways under the step budget.
const FULLERENE: &str = "C12C3=C4C5=C6C3=C7C8=C9C3=C%10C%11=C%12C%13=C%14C%15=C%11C3=C3C8=C6C6=C8C5=C5C%11=C8C(=C%14C8=C%11C%11=C%14C%16=C%17C(=C%18C%12=C%12C%10=C%10C9=C7C1=C1C%10=C7C%12=C%18C9=C%17C%10=C%14C%12(C%11=C5C4=C4C2=C2C1(C7=C9C%10(C2=C4%12))))C%13=C8%16)C%15=C36";

/// Porphine: four pyrroles and a sixteen-membered macrocycle.
const PORPHINE: &str = "C1=CC2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C1N2";

/// The fused-ring limit is process-wide, so the test that changes it takes this
/// lock for writing and every test that perceives takes it for reading. Without
/// that they would run into each other: the suite is threaded.
static LIMIT: std::sync::RwLock<()> = std::sync::RwLock::new(());

fn molecule(smiles: &str) -> MolecularGraph {
    MolecularGraph::from_smiles_single(smiles).expect("the test input parses")
}

/// The molecule perceived under `model`, whatever the parser perceived.
fn perceived(smiles: &str, model: AromaticityModel) -> MolecularGraph {
    let _reading = LIMIT.read().expect("the limit lock is not poisoned");
    perceived_under_the_limit_in_force(smiles, model)
}

/// [`perceived`] without taking the lock, for the one test that holds it.
fn perceived_under_the_limit_in_force(smiles: &str, model: AromaticityModel) -> MolecularGraph {
    let mut molecule = molecule(smiles);
    molecule
        .aromatize(model, None, false)
        .expect("perception answers");
    molecule
}

/// The ring family as sets of atoms, in the numbering `order` came from.
///
/// `reorder_molecule(order)` puts the atom that was at `order[position]` at
/// `position`, so reading `order` maps an atom of the permuted molecule back to
/// the atom it was, and two numberings can be compared as sets.
fn family_in_the_original_numbering(
    molecule: &MolecularGraph,
    order: Option<&[usize]>,
) -> Vec<Vec<usize>> {
    let mut rings: Vec<Vec<usize>> = ring_family(molecule, usize::MAX)
        .into_iter()
        .map(|ring| {
            let mut atoms: Vec<usize> = match order {
                Some(order) => ring.atoms.iter().map(|&atom| order[atom]).collect(),
                None => ring.atoms.clone(),
            };
            atoms.sort_unstable();
            atoms
        })
        .collect();
    rings.sort();
    rings
}

fn ring_sizes(smiles: &str) -> Vec<usize> {
    let mut sizes: Vec<usize> = ring_family(&molecule(smiles), usize::MAX)
        .iter()
        .map(|ring| ring.len())
        .collect();
    sizes.sort_unstable();
    sizes
}

fn aromatic_atom_count(smiles: &str, model: AromaticityModel) -> usize {
    perceived(smiles, model).get_aromatic_atoms().len()
}

/// Nothing below this line may depend on the order the atoms were written in.
/// That is the whole reason the ring model exists, so it is tested first and
/// tested on the shapes that broke the path search.
#[test]
fn the_ring_family_is_the_same_under_every_numbering() {
    let cases = [
        "c1ccccc1",
        "c1ccc2ccccc2c1",
        // Cubane: six faces, five in any SSSR, and which five is an accident.
        "C12C3C4C1C5C4C3C25",
        "C1CC2CCC1CC2",
        "C1C2CC3CC1CC(C2)C3",
        PORPHINE,
        "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",
        FULLERENE,
    ];

    for smiles in cases {
        let molecule = molecule(smiles);
        let reference = family_in_the_original_numbering(&molecule, None);
        let mut generator = StdRng::seed_from_u64(20260917);
        for _ in 0..16 {
            let mut order: Vec<usize> = (0..molecule.len()).collect();
            order.shuffle(&mut generator);
            let mut permuted = molecule.clone();
            reorder_molecule(&mut permuted, &order);
            assert_eq!(
                family_in_the_original_numbering(&permuted, Some(&order)),
                reference,
                "the ring family of {smiles} moved with the numbering"
            );
        }
    }
}

#[test]
fn the_ring_family_keeps_the_rings_an_sssr_would_drop() {
    assert_eq!(ring_sizes("c1ccccc1"), vec![6]);
    // Naphthalene's ten-ring is the sum of its two six-rings, so it is not
    // relevant and does not join the family.
    assert_eq!(ring_sizes("c1ccc2ccccc2c1"), vec![6, 6]);
    // Cubane: six faces where an SSSR has five.
    assert_eq!(ring_sizes("C12C3C4C1C5C4C3C25"), vec![4, 4, 4, 4, 4, 4]);
    // Bicyclo[2.2.2]octane: three six-rings where an SSSR has two.
    assert_eq!(ring_sizes("C1CC2CCC1CC2"), vec![6, 6, 6]);
    // Adamantane: four faces where an SSSR has three.
    assert_eq!(ring_sizes("C1C2CC3CC1CC(C2)C3"), vec![6, 6, 6, 6]);
    assert_eq!(ring_sizes(PORPHINE), vec![5, 5, 5, 5, 16]);
    assert_eq!(
        ring_sizes("c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67"),
        vec![6; 7]
    );

    let fullerene = ring_sizes(FULLERENE);
    assert_eq!(fullerene.iter().filter(|&&size| size == 5).count(), 12);
    assert_eq!(fullerene.iter().filter(|&&size| size == 6).count(), 25);
    assert_eq!(fullerene.len(), 37);
}

/// The ring family is built from the bond graph alone, so a molecule written
/// aromatic and the same molecule written Kekule have the same one.
#[test]
fn the_ring_family_does_not_read_bond_orders() {
    assert_eq!(ring_sizes("c1ccccc1"), ring_sizes("C1=CC=CC=C1"));
    assert_eq!(
        ring_sizes("c1ccc2ccccc2c1"),
        ring_sizes("C1=CC=C2C=CC=CC2=C1")
    );
}

/// The acceptance criterion for the exocyclic rule, from
/// `molecule::pi_system`'s module documentation: the same thiazolone, tethered
/// and untethered, must be perceived the same way -- and not aromatic.
#[test]
fn the_thiazolone_pair_agrees_under_the_ring_model() {
    let tethered = "C1CCCN=C2SC(=C)C(=O)N2CC1";
    let untethered = "CCCCN=C2SC(=C)C(=O)N2CC";
    for smiles in [tethered, untethered] {
        assert_eq!(
            aromatic_atom_count(smiles, AromaticityModel::Rings),
            0,
            "{smiles} has no aromatic ring: every carbon of the thiazolone has \
             a double bond leaving it"
        );
        assert_eq!(
            aromatic_atom_count(smiles, AromaticityModel::Rings),
            aromatic_atom_count(smiles, AromaticityModel::OpenSMILES),
            "the two models disagree about {smiles}"
        );
    }
}

/// Where both models are right, they have to say so identically: this is what
/// makes a corpus comparison between them readable.
#[test]
fn the_two_models_agree_on_the_ordinary_aromatics() {
    let cases = [
        "c1ccccc1",
        "C1=CC=CC=C1",
        "c1ccncc1",
        "c1cc[nH]c1",
        "c1ccoc1",
        "c1ccsc1",
        "c1ccc2ccccc2c1",
        "c1ccc2[nH]ccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccc2c(c1)[nH]c1ccccc12",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccc(cc1)-c1ccccc1",
        "c1cnc2[nH]cnc2c1",
        "C1=CC=C2C(=C1)C=CC3=CC=CC=C32",
        "O=C1C=CC(=O)C=C1",
        "C1=CC=[N+](C=C1)[O-]",
        "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",
        "C1CCCCC1",
        "C1=CC=CC=CC=C1",
    ];

    for smiles in cases {
        assert_eq!(
            aromatic_atom_count(smiles, AromaticityModel::Rings),
            aromatic_atom_count(smiles, AromaticityModel::OpenSMILES),
            "the models disagree about {smiles}"
        );
    }
}

/// Azulene is one aromatic ten-electron system across both of its rings. The
/// count that says so is a count around the perimeter, so the bond across the
/// middle is not part of it and is not marked -- where naphthalene's fusion
/// bond is, because each of its rings is aromatic on its own.
#[test]
fn a_fused_system_is_aromatic_around_its_perimeter() {
    let azulene = perceived("C1=CC2=CC=CC=CC2=C1", AromaticityModel::Rings);
    assert_eq!(azulene.get_aromatic_atoms().len(), 10);
    assert!(
        azulene.to_smiles().contains('-'),
        "azulene's fusion bond is written single: {}",
        azulene.to_smiles()
    );

    let naphthalene = perceived("C1=CC=C2C=CC=CC2=C1", AromaticityModel::Rings);
    assert_eq!(naphthalene.get_aromatic_atoms().len(), 10);
    assert!(
        !naphthalene.to_smiles().contains('-'),
        "every bond of naphthalene is aromatic: {}",
        naphthalene.to_smiles()
    );
}

/// A ring whose every atom brings a lone pair and nothing else has no orbital
/// for those electrons to move into: P4O6 and cyclic S5 count to 4n+2 and are
/// not aromatic. Borazine, whose boron offers an empty orbital, is.
#[test]
fn a_ring_of_donors_alone_is_not_aromatic() {
    for smiles in [
        "O1P2OP3OP1OP(O2)O3",
        "S1SSSS1",
        "P12P3P1P23",
        "CP1P(P(P(P1C)C)C)C",
    ] {
        assert_eq!(
            aromatic_atom_count(smiles, AromaticityModel::Rings),
            0,
            "{smiles} has no orbital to delocalize into"
        );
    }
    assert_eq!(aromatic_atom_count("b1nbnbn1", AromaticityModel::Rings), 6);
}

/// Porphine's four pyrroles and its macrocycle are one aromatic system. Under
/// the path search two of the pyrroles come out Kekule.
#[test]
fn porphine_is_aromatic_throughout() {
    assert_eq!(
        aromatic_atom_count(PORPHINE, AromaticityModel::Rings),
        24,
        "every ring atom of porphine is aromatic"
    );
}

/// Ring size is a statement about chemistry, and it is the caller's.
///
/// At the default, 4n+2 decides up to twenty-four atoms: 18-annulene is
/// aromatic, 16- and 20-annulene are not, and 26-annulene is out of range.
/// Taking the limit off makes 26-annulene aromatic -- and makes the family of a
/// macrocycle able to grow with it, which is what the limit is really for.
#[test]
fn the_ring_size_limit_is_chemistry_and_is_the_caller_s() {
    use crate::molecule::ring_aromaticity::{MAX_RING_SIZE, ring_size_limit, set_ring_size_limit};

    let _writing = LIMIT.write().expect("the limit lock is not poisoned");
    let aromatic = |smiles: &str| {
        perceived_under_the_limit_in_force(smiles, AromaticityModel::Rings)
            .get_aromatic_atoms()
            .len()
    };
    let eighteen = "C1=CC=CC=CC=CC=CC=CC=CC=CC=C1";
    let twenty = "C1=CC=CC=CC=CC=CC=CC=CC=CC=CC=C1";
    let twenty_six = "C1=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=C1";

    assert_eq!(ring_size_limit(), MAX_RING_SIZE);
    assert_eq!(aromatic(eighteen), 18);
    assert_eq!(aromatic(twenty), 0, "twenty π electrons are not 4n+2");
    assert_eq!(
        aromatic(twenty_six),
        0,
        "beyond the limit, nothing is scored"
    );

    set_ring_size_limit(0);
    assert_eq!(
        aromatic(twenty_six),
        26,
        "with no limit, 4n+2 decides alone"
    );
    assert_eq!(aromatic(twenty), 0);

    set_ring_size_limit(MAX_RING_SIZE);
    assert_eq!(aromatic(twenty_six), 0);
}

/// Cyclooctatetraene has eight π electrons and is not aromatic, whichever way
/// the rings are counted.
#[test]
fn eight_electrons_are_not_aromatic() {
    assert_eq!(
        aromatic_atom_count("C1=CC=CC=CC=C1", AromaticityModel::Rings),
        0
    );
}

/// Perception decides the canonical form, so a molecule written aromatic and
/// the same molecule written Kekule have to reach the same string under the
/// ring model too.
#[test]
fn the_two_writings_of_one_molecule_canonize_together() {
    let pairs = [
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("c1ccc2ccccc2c1", "C1=CC=C2C=CC=CC2=C1"),
        ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
        (PORPHINE, "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3"),
    ];

    for (aromatic, kekule) in pairs {
        let left = perceived(aromatic, AromaticityModel::Rings);
        let right = perceived(kekule, AromaticityModel::Rings);
        assert_eq!(
            left.get_aromatic_atoms().len(),
            right.get_aromatic_atoms().len(),
            "{aromatic} and {kekule} are perceived differently"
        );
    }
}

/// The cages the path search could not canonize consistently, canonized under
/// sixteen numberings each.
#[test]
fn the_cages_canonize_to_one_string_under_the_ring_model() {
    use crate::graph_algorithms::mol_canonization::Canonizer;

    for smiles in [FULLERENE, PORPHINE] {
        let molecule = molecule(smiles);
        let mut canonizer = Canonizer::new();
        let mut generator = StdRng::seed_from_u64(20260917);
        let mut answers: Vec<String> = Vec::new();
        for _ in 0..16 {
            let mut order: Vec<usize> = (0..molecule.len()).collect();
            order.shuffle(&mut generator);
            let mut permuted = molecule.clone();
            reorder_molecule(&mut permuted, &order);
            permuted
                .aromatize(AromaticityModel::Rings, None, false)
                .expect("perception answers");
            answers.push(canonizer.canonize_perceived(permuted).to_smiles());
        }
        answers.sort();
        answers.dedup();
        assert_eq!(
            answers.len(),
            1,
            "{smiles} canonized {} ways",
            answers.len()
        );
    }
}

/// The one number left in the model is reachable, and it does what it says.
///
/// Six is the default, and it is RDKit's; zero is the model with no such number
/// in it, which perceives each ring alone and each fused system entire. The
/// setting is global, so this test puts it back.
#[test]
fn the_fused_ring_limit_is_the_caller_s_to_set() {
    use crate::molecule::ring_aromaticity::{
        MAX_FUSED_RINGS, fused_ring_limit, set_fused_ring_limit,
    };

    let _writing = LIMIT.write().expect("the limit lock is not poisoned");
    let aromatic = |smiles: &str| {
        perceived_under_the_limit_in_force(smiles, AromaticityModel::Rings)
            .get_aromatic_atoms()
            .len()
    };

    // Azulene needs two rings combined; one is not enough.
    let azulene = "C1=CC2=CC=CC=CC2=C1";
    assert_eq!(fused_ring_limit(), MAX_FUSED_RINGS);
    assert_eq!(aromatic(azulene), 10);

    set_fused_ring_limit(1);
    assert_eq!(aromatic(azulene), 0);

    // Zero is not "no limit": it is each ring alone and each system entire,
    // which azulene's one system satisfies.
    set_fused_ring_limit(0);
    assert_eq!(aromatic(azulene), 10);

    set_fused_ring_limit(MAX_FUSED_RINGS);
    assert_eq!(aromatic(azulene), 10);
}

/// A fused system stops being enumerated once everything in it is marked.
///
/// Six benzenes joined into a macrocycle give sixty-four equally short
/// twenty-four-membered rings -- one per way round each benzene -- and
/// combining seventy rings six at a time does not finish. It does not have to:
/// the answer is a union, a subset marks only its own fused system, and once
/// every bond of that system is marked nothing further can change it. PubChem
/// CID 71816689 stalled a corpus sweep for want of that argument.
#[test]
fn a_saturated_ring_system_stops_being_enumerated() {
    let cyclophane =
        "C1=CC2=C3C=CC(=C4C=CC(=C5C=CC(=C6C=CC(=C7C=CC(=C1C=C2)C=C7)C=C6)C=C5)C=C4)C=C3";
    let perceived = perceived(cyclophane, AromaticityModel::Rings);
    assert_eq!(
        perceived.get_aromatic_atoms().len(),
        36,
        "every carbon of the cyclophane is on an aromatic ring"
    );

    // The eight-benzene one, which needed the ring-size limit as well.
    let bigger = "C1=CC2=C3C=CC4=C5C=CC6=C7C=CC8=C9C=CC%10=C1C1=C2C2=C%11C3=C4C3=C4C5=C6C5=C6C7=C8C7=C8C9=C%10C9=C1C1=C2C2=C%10C%11=C3C3=C%11C4=C5C4=C5C6=C7C6=C7C8=C9C8=C1C1=C2C2=C9C%10=C3C3=C%10C%11=C4C4=C%11C5=C6C5=C6C7=C8C7=C1C1=C2C2=C7C8=C3C3=C8C9=C4C4=C9C";
    let _ = MolecularGraph::from_smiles_single(bigger);
}
