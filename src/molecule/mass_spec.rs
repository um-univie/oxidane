//! Mass-spectrometry conventions: the parent compound a spectral library
//! records.
//!
//! A library entry names the neutral molecule M whose ion was measured -- the
//! M of `[M+H]+`, `[M-H]-` or `[M+Na]+` -- and records deposit it in whatever
//! form was in the vial: a hydrochloride, a sodium salt with the sodium drawn
//! as a bond, the cation an `[M+H]+` was drawn from, a zwitterion. None of that
//! changes the spectrum, but it makes one compound look like several.
//! [`parent_smiles`] writes M, by these steps:
//!
//! 1. **Metals.** A single bond from an alkali or alkaline-earth metal to N, O,
//!    S, Se or a halogen is broken; the metal leaves as its cation and the atom
//!    it was bonded to keeps the electron, so `CC(=O)O[Na]` becomes acetate and
//!    sodium. This changes no formula and no net charge.
//! 2. **Spellings no tautomer reaches.** `N=N#N`, an azide drawn with a
//!    five-valent nitrogen, becomes `N=[N+]=[N-]`, and `[C-][N+]#N` becomes the
//!    diazo compound `C=[N+]=[N-]`.
//! 3. **The canonical tautomer.** Every component is replaced by its canonical
//!    tautomer, as `canonical_tautomer` writes it, which keeps each component's
//!    formula and charge: `[S+][O-]` as `S=O` (a phosphorus ylide's oxide stays
//!    separated), 2-hydroxypyridine as 2-pyridone, a pyridone drawn as its
//!    betaine as the pyridone. Everything below starts from that structure, so
//!    the parent does not depend on which tautomer the record was drawn as.
//! 4. **Charges.** Every cation on N, O, S, P, As, Se or Te that holds a
//!    proton gives it up, and a cation with a hydroxyl gives up that one and
//!    becomes the charge pair it is a protonated form of (a protonated nitro
//!    group, N-oxide or phosphoryl). Every anion on N, O, S, Se or Te and every
//!    bare halide takes one. What is left are charges no proton removes: a
//!    quaternary nitrogen's, a carbocation's, a borate's, a flavylium's. Each
//!    fixed cation is then balanced by the strongest acid in the component --
//!    a sulfonic or sulfuric acid, then any acid of phosphorus or arsenic,
//!    then a carboxylic acid -- and each fixed anion by the strongest base, an
//!    amidine or guanidine before an aliphatic amine. That is the convention
//!    spectral libraries record: carnitine and the betaines are zwitterions,
//!    observed as `[M+H]+`; choline, with no acid, is a cation, observed as
//!    `[M]+`. A phenol, an alcohol or an NH is never taken for an acid, and an
//!    aromatic or amide nitrogen never for a base, so a quaternary ammonium
//!    with a phenol, a flavylium or a phenazinium dye stays a cation. Nor is
//!    a net charge ever taken further from zero than the record had it: a
//!    zwitterion whose anion is too weak to balance its cation -- a phenolate
//!    betaine, a mesoionic sydnone imine -- is its own parent, though an
//!    alkoxide, an amide ion or hydroxide, the anions of no acid that exists
//!    in water, always takes its proton. Charge pairs
//!    that are how a group is written -- a nitro group, an N-oxide, an azide,
//!    an ylide -- are not ions and are left alone.
//! 5. **Salts.** Identical components are written once. Components without
//!    carbon and the common counterions and solvents of crystallization in
//!    [`COUNTERIONS`] are dropped when anything else is left. Nothing else is
//!    guessed: a record of two compounds neither of which is a known
//!    counterion keeps both, and one that is only counterions keeps the ones
//!    with carbon, so acetic acid on its own stays acetic acid.
//!
//! A record with a transition-metal ion as its own component -- heme as a
//! porphyrin dianion and `[Fe+2]` -- is a complex written ionically, not a
//! salt, and a record with no carbon at all has no organic compound to be
//! the parent of. Both come back after steps 1 and 2 only.
//!
//! What is promised, and checked over PubChem by `examples/pubchem_sweep`
//! with `--parent on`: the parent does not depend on the order of the atoms or
//! the components, nor on which tautomer the record was written as; it is a
//! fixed point, the parent of a parent being itself; it reads back as the same
//! SMILES; and it holds no heavy atom the record did not.

use std::sync::OnceLock;

use rustc_hash::FxHashSet;

use crate::atom::Atom;
use crate::graph_algorithms::mol_canonization::{Canonizer, with_canonizer};
use crate::io::SMILESParseError;
use crate::molecule::bond::{BondOrder, BondTarget};
use crate::molecule::chirality::ChiralClass;
use crate::molecule::tautomer::{TautomerRules, perceive, try_normalize_with};
use crate::prelude::*;

/// Which of the steps [`parent_smiles`] takes. All are on by default; the
/// canonical tautomer and the spelling repairs are always made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParentOptions {
    /// Break bonds from alkali and alkaline-earth metals to N, O, S, Se and
    /// halogens.
    pub disconnect_metals: bool,
    /// Add and remove protons until the net charge is as close to zero as the
    /// fixed charges allow.
    pub neutralize: bool,
    /// Drop counterions, solvents and repeated components.
    pub strip_salts: bool,
}

impl Default for ParentOptions {
    fn default() -> Self {
        Self {
            disconnect_metals: true,
            neutralize: true,
            strip_salts: true,
        }
    }
}

/// The parent compound of a SMILES, as canonical SMILES; see the module
/// documentation for what that is.
///
/// ```
/// use oxidane::molecule::mass_spec::{parent_smiles, ParentOptions};
///
/// let parent = |smiles| parent_smiles(smiles, &ParentOptions::default()).unwrap();
///
/// // The hydrochloride and the free base are one compound.
/// assert_eq!(parent("CCN.Cl"), parent("CC[NH3+].[Cl-]"));
/// assert_eq!(parent("CC[NH3+].[Cl-]"), parent("CCN"));
/// // A sodium drawn as a bond, a sodium drawn as an ion, and the free acid.
/// assert_eq!(parent("CC(=O)O[Na]"), parent("CC(=O)O"));
/// assert_eq!(parent("CC(=O)[O-].[Na+]"), parent("CC(=O)O"));
/// // A zwitterion with nothing fixed holding it apart is the neutral form...
/// assert_eq!(parent("[NH3+]CC([O-])=O"), parent("NCC(=O)O"));
/// // ...but a quaternary ammonium keeps the carboxylate that balances it.
/// assert_eq!(parent("C[N+](C)(C)CC(=O)O"), parent("C[N+](C)(C)CC(=O)[O-]"));
/// assert_eq!(parent("C[N+](C)(C)CC(=O)O").matches('-').count(), 1);
/// // One spelling of a sulfoxide.
/// assert_eq!(parent("C[S+](C)[O-]"), parent("CS(C)=O"));
/// // Two compounds, neither a known counterion: both are kept.
/// assert_eq!(parent("CCO.c1ccccc1N").matches('.').count(), 0);
/// assert_eq!(parent("CCCO.c1ccccc1N").matches('.').count(), 1);
/// ```
pub fn parent_smiles(smiles: &str, options: &ParentOptions) -> Result<String, SMILESParseError> {
    let components = MolecularGraph::from_smiles(smiles)?;
    Ok(with_canonizer(|canonizer| {
        parents_with(canonizer, components, options)
            .iter()
            .map(|component| component.to_smiles())
            .collect::<Vec<_>>()
            .join(".")
    }))
}

/// [`parent_smiles`] on components already read: the parent's components, each
/// in canonical atom order, sorted by their canonical SMILES.
pub fn parent_components(
    components: Vec<MolecularGraph>,
    options: &ParentOptions,
) -> Vec<MolecularGraph> {
    with_canonizer(|canonizer| parents_with(canonizer, components, options))
}

fn parents_with(
    canonizer: &mut Canonizer,
    components: Vec<MolecularGraph>,
    options: &ParentOptions,
) -> Vec<MolecularGraph> {
    let mut components: Vec<MolecularGraph> = if options.disconnect_metals {
        components
            .into_iter()
            .flat_map(|component| {
                // The canonical tautomer first where a metal bond is to be
                // broken, so that the tautomer the record was written as does
                // not decide what is left once it is: a sodium selenide's
                // `[Se]=[Se][Na]` and its canonical tautomer `[Se][Se][Na]`
                // broke into different anions.
                let mut component = if has_ionic_metal_bond(&component) {
                    canonical_tautomer(canonizer, &component)
                } else {
                    component
                };
                if disconnect_metals(&mut component) {
                    component
                        .split_components()
                        .unwrap_or_else(|_| vec![component])
                } else {
                    vec![component]
                }
            })
            .collect()
    } else {
        components
    };
    for component in &mut components {
        repair_spellings(component);
    }
    // A transition-metal ion written as its own component is the centre of a
    // complex written ionically -- heme as a porphyrin dianion and `[Fe+2]` --
    // not a counterion, and the ligand's charges are where the metal binds.
    // A record without carbon has no organic compound to stand for it. Either
    // is written back as it came.
    let complex = components.iter().any(is_metal_ion);
    let organic = components.iter().any(has_carbon);
    let tautomerized = !complex && organic;
    let parents: Vec<MolecularGraph> = if !tautomerized {
        components
    } else {
        let parents: Vec<MolecularGraph> = components
            .iter()
            .map(|component| {
                if options.neutralize {
                    neutralized(canonizer, component)
                } else {
                    canonical_tautomer(canonizer, component)
                }
            })
            .collect();
        if options.strip_salts {
            strip_salts(canonizer, parents)
        } else {
            parents
        }
    };
    // The parent is what its SMILES says, and says nothing its structure does
    // not. The canonical tautomer drops the geometry of a double bond at an
    // endpoint, where the proton may move it, and keeps those beside it; with
    // one on each side, the marks they need land on both ends of the bond that
    // has none, and SMILES reads a geometry into it -- PubChem CID 156642185,
    // an open-chain oxonium, read back with its C=O+ Z where it had been E.
    // Such a component keeps no geometry at all rather than one it does not
    // have. Each component is written, read back and canonized, so the string
    // a caller gets is the canonical SMILES of what it denotes.
    let mut parents: Vec<MolecularGraph> = parents
        .iter()
        .map(|component| {
            let again = MolecularGraph::from_smiles_single(&component.to_smiles())
                .map(|again| canonizer.canonize(&again))
                .unwrap_or_else(|_| canonizer.canonize(component));
            if geometries(&again) <= geometries(component) {
                return again;
            }
            let mut plain = component.clone();
            if let Some(classes) = plain.chiral_classes_mut() {
                for class in classes.iter_mut() {
                    if matches!(
                        class,
                        ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
                    ) {
                        *class = ChiralClass::None;
                    }
                }
            }
            // Which tautomer is canonical can depend on the geometries a
            // structure carries, so one that has given them up is asked again.
            let plain = if tautomerized {
                canonical_tautomer(canonizer, &plain)
            } else {
                plain
            };
            let again = MolecularGraph::from_smiles_single(&plain.to_smiles())
                .map(|again| canonizer.canonize(&again))
                .unwrap_or_else(|_| canonizer.canonize(&plain));
            again
        })
        .collect();
    parents.sort_by_cached_key(|component| component.to_smiles());
    parents
}

/// The atoms carrying a double bond's geometry.
fn geometries(molecule: &MolecularGraph) -> usize {
    (0..molecule.atomic_numbers().len())
        .filter(|&atom| {
            matches!(
                molecule.get_chiral_class(atom),
                ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
            )
        })
        .count()
}

/// A single metal atom other than an alkali or alkaline-earth metal.
fn is_metal_ion(component: &MolecularGraph) -> bool {
    let heavy: Vec<u8> = component
        .atomic_numbers()
        .iter()
        .copied()
        .filter(|&z| z != 1)
        .collect();
    heavy.len() == 1 && is_complexing_metal(heavy[0])
}

/// Metals that form complexes rather than salts: the transition metals, the
/// lanthanides and actinides, and the post-transition metals.
fn is_complexing_metal(atomic_number: u8) -> bool {
    matches!(atomic_number, 13 | 21..=31 | 39..=50 | 57..=83 | 89..=103)
}

fn has_carbon(component: &MolecularGraph) -> bool {
    component.atomic_numbers().contains(&6)
}

// ---------------------------------------------------------------- metals

/// Alkali and alkaline-earth metals: drawn bonded to a heteroatom they are a
/// counterion written the wrong way, never a part of the compound.
fn is_ionic_metal(atomic_number: u8) -> bool {
    matches!(atomic_number, 3 | 11 | 19 | 37 | 55 | 4 | 12 | 20 | 38 | 56)
}

/// The atoms a metal is ever bonded to as a salt: N, O, S, Se and halogens.
fn takes_metal_electron(atomic_number: u8) -> bool {
    matches!(atomic_number, 7 | 8 | 16 | 34 | 9 | 17 | 35 | 53)
}

/// Whether [`disconnect_metals`] has a bond to break.
fn has_ionic_metal_bond(molecule: &MolecularGraph) -> bool {
    (0..molecule.atomic_numbers().len()).any(|metal| {
        is_ionic_metal(molecule.atomic_numbers()[metal])
            && molecule
                .get_atom_bonds(metal)
                .unwrap_or_default()
                .iter()
                .any(|bond| {
                    bond.bond_order() == BondOrder::Single
                        && takes_metal_electron(molecule.atomic_numbers()[bond.target()])
                })
    })
}

/// Breaks every single bond between an ionic metal and an atom that takes its
/// electron. Returns whether anything was broken.
fn disconnect_metals(molecule: &mut MolecularGraph) -> bool {
    let mut broken = Vec::new();
    for metal in 0..molecule.atomic_numbers().len() {
        if !is_ionic_metal(molecule.atomic_numbers()[metal]) {
            continue;
        }
        for bond in molecule.get_atom_bonds(metal).unwrap_or_default() {
            let partner = bond.target();
            if bond.bond_order() == BondOrder::Single
                && takes_metal_electron(molecule.atomic_numbers()[partner])
            {
                broken.push((metal, partner));
            }
        }
    }
    for &(metal, partner) in &broken {
        remove_bond(molecule, metal, partner);
        *molecule.get_atom_charge_mut(metal) += 1;
        *molecule.get_atom_charge_mut(partner) -= 1;
        clear_tetrahedral(molecule, partner);
    }
    !broken.is_empty()
}

// ---------------------------------------------------------------- spellings

/// Rewrites the drawings of a group that no valence model accepts, or that the
/// canonical tautomer cannot reach because the charge sits on carbon. Applied
/// until nothing changes.
fn repair_spellings(molecule: &mut MolecularGraph) {
    while pentavalent_azide(molecule) || diazo_carbanion(molecule) {}
}

/// `N=N#N` puts five bonds on a neutral nitrogen; the azide is `N=[N+]=[N-]`.
fn pentavalent_azide(molecule: &mut MolecularGraph) -> bool {
    for middle in 0..molecule.atomic_numbers().len() {
        if molecule.atomic_numbers()[middle] != 7 || molecule.get_atom_charge(middle) != 0 {
            continue;
        }
        let bonds = molecule.get_atom_bonds(middle).unwrap_or_default();
        let double_to_n = bonds.iter().any(|bond| {
            bond.bond_order() == BondOrder::Double && molecule.atomic_numbers()[bond.target()] == 7
        });
        let triple_end = bonds
            .iter()
            .find(|bond| {
                bond.bond_order() == BondOrder::Triple
                    && molecule.atomic_numbers()[bond.target()] == 7
                    && heavy_degree(molecule, bond.target()) == 1
            })
            .map(|bond| bond.target());
        if let (true, Some(end)) = (double_to_n, triple_end) {
            set_bond_order(molecule, middle, end, BondOrder::Double);
            *molecule.get_atom_charge_mut(middle) += 1;
            *molecule.get_atom_charge_mut(end) -= 1;
            return true;
        }
    }
    false
}

/// `[C-][N+]#N` is the other resonance form of a diazo compound; the one
/// written is `C=[N+]=[N-]`. An aromatic carbanion, or one with a double bond
/// already, is left alone: a second double bond on it is a cumulene, not a
/// spelling of the first.
fn diazo_carbanion(molecule: &mut MolecularGraph) -> bool {
    for carbon in 0..molecule.atomic_numbers().len() {
        if molecule.atomic_numbers()[carbon] != 6
            || molecule.get_atom_charge(carbon) != -1
            || molecule.is_atom_aromatic(carbon)
            || molecule
                .get_atom_bonds(carbon)
                .unwrap_or_default()
                .iter()
                .any(|bond| bond.bond_order() != BondOrder::Single)
        {
            continue;
        }
        for bond in molecule.get_atom_bonds(carbon).unwrap_or_default() {
            let middle = bond.target();
            if bond.bond_order() != BondOrder::Single
                || molecule.atomic_numbers()[middle] != 7
                || molecule.get_atom_charge(middle) != 1
            {
                continue;
            }
            let end = molecule
                .get_atom_bonds(middle)
                .unwrap_or_default()
                .iter()
                .find(|other| {
                    other.bond_order() == BondOrder::Triple
                        && molecule.atomic_numbers()[other.target()] == 7
                        && heavy_degree(molecule, other.target()) == 1
                })
                .map(|other| other.target());
            if let Some(end) = end {
                set_bond_order(molecule, carbon, middle, BondOrder::Double);
                set_bond_order(molecule, middle, end, BondOrder::Double);
                *molecule.get_atom_charge_mut(carbon) += 1;
                *molecule.get_atom_charge_mut(end) -= 1;
                clear_tetrahedral(molecule, carbon);
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------- charges

/// The canonical tautomer of a component, as [`TautomerRules::new`] writes it,
/// or its canonical form where it has none.
///
/// These are the rules `canonical_tautomer` is written under, not the
/// ionizable ones, which pool every heteroatom's protons into one and place
/// the charges they leave by tautomer conventions rather than by acidity:
/// they write carnitine with its charge on the alcohol, a nitro acid as a
/// protonated nitro group beside a carboxylate, and AZT with a proton moved
/// from its thymine onto its azide. Which protomer a component is drawn as
/// is the charge step's to decide.
fn canonical_tautomer(canonizer: &mut Canonizer, molecule: &MolecularGraph) -> MolecularGraph {
    try_normalize_with(canonizer, molecule, &TautomerRules::new())
        .unwrap_or_else(|_| canonizer.canonize(molecule))
}

/// How many times [`neutralized_once`] is applied in search of a structure
/// it leaves alone. Two is usual: one to neutralize, one to confirm.
const ROUNDS: usize = 8;

/// A component's neutralized form; see the module documentation.
///
/// The parent of a parent has to be itself, and one pass need not get there:
/// the balanced form is a different protonation state from the one the pass
/// started on, and its canonical tautomer can write a charge where a site rule
/// reads it differently -- a protonated thiazole as `[s+]` beside an `[nH]`
/// rather than as `[nH+]`. So the pass is repeated until it changes nothing,
/// and the structure it settles on is the answer. Should it cycle instead, the
/// member of the cycle with the least canonical SMILES stands for it, which
/// the cycle reaches from any of its members.
fn neutralized(canonizer: &mut Canonizer, molecule: &MolecularGraph) -> MolecularGraph {
    let (mut current, mut moved) = neutralized_once(canonizer, molecule);
    let mut seen: Vec<(String, MolecularGraph)> = Vec::new();
    while moved && seen.len() < ROUNDS {
        let written = current.to_smiles();
        if let Some(start) = seen.iter().position(|(earlier, _)| *earlier == written) {
            return seen
                .drain(start..)
                .min_by(|left, right| left.0.cmp(&right.0))
                .map(|(_, molecule)| molecule)
                .expect("a cycle has a member");
        }
        seen.push((written, current.clone()));
        (current, moved) = neutralized_once(canonizer, &current);
    }
    current
}

/// One pass of neutralization, and whether it moved a proton.
///
/// The removable charges go first, round by round, each round followed by the
/// canonical tautomer. The fixed charges left are then balanced on that form,
/// which every protonation state of the component reaches, so the site that
/// takes or gives the balancing proton is chosen by the molecule, not by where
/// the record put its charges. The balanced form is kept as it is rather than
/// handed back to the tautomer, which places a charge by its own conventions
/// -- it would give carnitine's to the alcohol -- unless the tautomer writes it
/// with fewer separated charges: `C[P+](=O)[O-]`, made from a phosphinic acid
/// balancing its own cationic phosphorus, is `CP(=O)=O`.
///
/// A pass that would leave the net charge further from zero than it found it
/// is not made: a neutral zwitterion whose anion is too weak an acid to
/// balance its cation -- a phenolate betaine, or resazurin, whose canonical
/// tautomer writes the N-oxide's charge on the far oxygen -- is its own
/// parent.
fn neutralized_once(
    canonizer: &mut Canonizer,
    molecule: &MolecularGraph,
) -> (MolecularGraph, bool) {
    let mut start = canonical_tautomer(canonizer, molecule);
    if charged_atoms(&start) == 0 {
        return (start, false);
    }
    let mut moved = false;
    // An alkoxide, an amide ion or hydroxide is the anion of no acid that
    // exists in water, so it takes its proton before the net charge is
    // weighed: choline drawn as `C[N+](C)(C)CC[O-]` is choline.
    if protonate_non_acid_anions(&mut start) {
        moved = true;
        start = canonical_tautomer(canonizer, &start);
    }
    let mut form = start.clone();
    // Every round removes a charge, so there are at most as many as charges.
    for _ in 0..=charged_atoms(&form) {
        if remove_removable_charges(&mut form) {
            form = canonical_tautomer(canonizer, &form);
        } else if let Some(released) = released_through_the_pool(canonizer, &form) {
            form = released;
        } else {
            break;
        }
        moved = true;
    }
    let mut balanced = form.clone();
    let result = if balance_fixed_charges(&mut balanced) {
        moved = true;
        let balanced = canonizer.canonize(&balanced);
        let rewritten = canonical_tautomer(canonizer, &balanced);
        if charge_score(&rewritten) < charge_score(&balanced) {
            rewritten
        } else {
            balanced
        }
    } else {
        form
    };
    if result.formal_charge().abs() > start.formal_charge().abs() {
        return (start, moved);
    }
    (result, moved)
}

/// Puts a proton on every anion no acid in water leaves behind; see
/// [`is_non_acid_anion`]. Returns whether there was one.
fn protonate_non_acid_anions(molecule: &mut MolecularGraph) -> bool {
    let paired = charge_pairs(molecule);
    let anions: Vec<usize> = (0..molecule.atomic_numbers().len())
        .filter(|&atom| !paired.contains(&atom) && is_non_acid_anion(molecule, atom))
        .collect();
    if anions.is_empty() {
        return false;
    }
    move_protons(molecule, &[], &anions);
    true
}

/// Whether the anion at `atom` is an oxygen or nitrogen with only saturated
/// carbons around it, or nothing at all: an alkoxide, an amide ion, hydroxide.
/// Their acids are weaker than water, so no zwitterion holds one.
fn is_non_acid_anion(molecule: &MolecularGraph, atom: usize) -> bool {
    let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
    molecule.get_atom_charge(atom) == -1
        && matches!(molecule.atomic_numbers()[atom], 7 | 8)
        && !molecule.is_atom_aromatic(atom)
        && bonds
            .iter()
            .all(|bond| bond.bond_order() == BondOrder::Single)
        && heavy_neighbours(molecule, atom).all(|carbon| {
            molecule.atomic_numbers()[carbon] == 6
                && !molecule.is_atom_aromatic(carbon)
                && molecule
                    .get_atom_bonds(carbon)
                    .unwrap_or_default()
                    .iter()
                    .all(|bond| bond.bond_order() == BondOrder::Single)
        })
}

/// A proton taken from, or given to, the component's pool of mobile
/// hydrogens, where the canonical tautomer can then write the component with
/// fewer separated charges and no fewer aromatic bonds.
///
/// The site rules read a charge where the canonical tautomer writes it, and
/// the tautomer writes it by conventions of its own -- a protonated
/// benzothiadiazole as `[s+]` beside an `[nH]`, since a chalcogen takes a
/// positive charge before a nitrogen does -- so a cation that is the
/// protonated form of a neutral base need not have its proton on the charged
/// atom. Asked of the pool, the question is the compound's: does one proton
/// fewer leave a neutral molecule? The pool is the component's, every
/// endpoint's hydrogens one total, so which endpoint gives the proton up does
/// not matter, and the first in atom order stands for all of them.
///
/// The aromatic bonds are what tell a protonated heteroaromatic from a dye.
/// A protonated imidazole, thiazole or benzothiadiazole gives back an
/// aromatic base. A flavylium, a safranin or a 9-hydroxyellipticinium could
/// lose a proton too, but only to become a quinoid anhydrobase, and none of
/// them is an acid at the pH a library measures them at: they are cations,
/// and the ring they would give up says so.
fn released_through_the_pool(
    canonizer: &mut Canonizer,
    form: &MolecularGraph,
) -> Option<MolecularGraph> {
    let paired = charge_pairs(form);
    let unpaired = |sign: i8| {
        (0..form.atomic_numbers().len())
            .any(|atom| !paired.contains(&atom) && form.get_atom_charge(atom).signum() == sign)
    };
    let groups = perceive(form, &TautomerRules::new());
    let mut endpoints: Vec<usize> = groups
        .groups()
        .iter()
        .flat_map(|group| group.endpoints.iter().copied())
        .collect();
    endpoints.sort_unstable();
    let mut tries: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    if unpaired(1)
        && let Some(&atom) = endpoints
            .iter()
            .find(|&&atom| !hydrogens_on(form, atom).is_empty())
    {
        tries.push((vec![atom], Vec::new()));
    }
    if unpaired(-1)
        && let Some(&atom) = endpoints.first()
    {
        tries.push((Vec::new(), vec![atom]));
    }
    let (unpaired_before, _) = charge_score(form);
    let aromatic_before = aromatic_bonds(form);
    tries.into_iter().find_map(|(lose, gain)| {
        let mut moved = form.clone();
        move_protons(&mut moved, &lose, &gain);
        let rewritten = try_normalize_with(canonizer, &moved, &TautomerRules::new()).ok()?;
        (charge_score(&rewritten).0 < unpaired_before
            && aromatic_bonds(&rewritten) >= aromatic_before)
            .then_some(rewritten)
    })
}

fn aromatic_bonds(molecule: &MolecularGraph) -> usize {
    molecule
        .atom_bonds()
        .iter()
        .flatten()
        .filter(|bond| bond.bond_order() == BondOrder::Aromatic)
        .count()
}

fn charged_atoms(molecule: &MolecularGraph) -> usize {
    (0..molecule.atomic_numbers().len())
        .filter(|&atom| molecule.get_atom_charge(atom) != 0)
        .count()
}

/// How charged a structure is, for choosing between two writings of it: the
/// charged atoms that are not part of a charge pair, then all of them.
fn charge_score(molecule: &MolecularGraph) -> (usize, usize) {
    let paired = charge_pairs(molecule);
    let unpaired = (0..molecule.atomic_numbers().len())
        .filter(|&atom| molecule.get_atom_charge(atom) != 0 && !paired.contains(&atom))
        .count();
    (unpaired, charged_atoms(molecule))
}

/// Elements whose cations give up a proton they hold: onium ions, never a
/// carbocation, whose proton is not an acid's.
fn gives_proton_as_cation(atomic_number: u8) -> bool {
    matches!(atomic_number, 7 | 8 | 15 | 16 | 33 | 34 | 52)
}

/// Whether a singly charged anion takes a proton: one on N, O, S, Se or Te, or
/// a bare halide. A carbanion's or a borate's charge stays.
fn takes_proton_as_anion(molecule: &MolecularGraph, atom: usize) -> bool {
    let element = molecule.atomic_numbers()[atom];
    matches!(element, 7 | 8 | 16 | 34 | 52)
        || (matches!(element, 9 | 17 | 35 | 53) && heavy_degree(molecule, atom) == 0)
}

/// The charges that are how a group is written rather than ions: each
/// positive atom is paired with as many negatively charged neighbours as its
/// charge, in atom order. A nitro group or an N-oxide is one such pair, and
/// nitrate, whose nitrogen has two `[O-]` neighbours and one positive charge,
/// is a pair and one free anion.
fn charge_pairs(molecule: &MolecularGraph) -> FxHashSet<usize> {
    let mut paired = FxHashSet::default();
    for atom in 0..molecule.atomic_numbers().len() {
        let charge = molecule.get_atom_charge(atom);
        if charge <= 0 {
            continue;
        }
        let mut partners: Vec<usize> = molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .map(|bond| bond.target())
            .filter(|&other| molecule.get_atom_charge(other) < 0 && !paired.contains(&other))
            .collect();
        if partners.is_empty() {
            continue;
        }
        partners.sort_unstable();
        partners.truncate(charge as usize);
        paired.insert(atom);
        paired.extend(partners);
    }
    paired
}

/// Takes the proton off every cation that holds one, and off the hydroxyl of
/// a cation that does not -- a protonated nitro group, N-oxide or phosphoryl,
/// which become the charge pair they are a protonated form of -- and puts one
/// on every anion that takes one, leaving charge pairs alone. Returns whether
/// anything changed.
fn remove_removable_charges(molecule: &mut MolecularGraph) -> bool {
    let paired = charge_pairs(molecule);
    let mut lose: Vec<usize> = Vec::new();
    let mut gain: Vec<usize> = Vec::new();
    for atom in 0..molecule.atomic_numbers().len() {
        let charge = molecule.get_atom_charge(atom);
        if charge == 0 || paired.contains(&atom) {
            continue;
        }
        if charge > 0 {
            let mut removable = 0;
            if gives_proton_as_cation(molecule.atomic_numbers()[atom]) {
                removable = (charge as usize).min(hydrogens_on(molecule, atom).len());
                lose.extend(std::iter::repeat_n(atom, removable));
            }
            let hydroxyls: Vec<usize> = heavy_neighbours(molecule, atom)
                .filter(|&oxygen| {
                    molecule.atomic_numbers()[oxygen] == 8
                        && molecule.get_atom_charge(oxygen) == 0
                        && heavy_degree(molecule, oxygen) == 1
                        && !hydrogens_on(molecule, oxygen).is_empty()
                        && !lose.contains(&oxygen)
                })
                .collect();
            lose.extend(hydroxyls.into_iter().take(charge as usize - removable));
        } else if charge == -1 && takes_proton_as_anion(molecule, atom) {
            gain.push(atom);
        }
    }
    if lose.is_empty() && gain.is_empty() {
        return false;
    }
    move_protons(molecule, &lose, &gain);
    true
}

/// Balances the charges no proton removes: a fixed cation by the strongest
/// acid there is, a fixed anion by the strongest base. Ties go to the lower
/// atom index, which on a canonical molecule is a property of the molecule.
/// Returns whether anything changed.
fn balance_fixed_charges(molecule: &mut MolecularGraph) -> bool {
    let paired = charge_pairs(molecule);
    let net: i32 = (0..molecule.atomic_numbers().len())
        .filter(|atom| !paired.contains(atom))
        .map(|atom| molecule.get_atom_charge(atom) as i32)
        .sum();
    if net == 0 {
        return false;
    }
    let rank: fn(&MolecularGraph, usize) -> Option<u8> =
        if net > 0 { acid_rank } else { base_rank };
    let mut sites: Vec<(u8, usize)> = (0..molecule.atomic_numbers().len())
        .filter_map(|atom| rank(molecule, atom).map(|rank| (rank, atom)))
        .collect();
    if sites.is_empty() {
        return false;
    }
    sites.sort_unstable();
    let chosen: Vec<usize> = sites
        .iter()
        .take(net.unsigned_abs() as usize)
        .map(|&(_, atom)| atom)
        .collect();
    if net > 0 {
        move_protons(molecule, &chosen, &[]);
    } else {
        move_protons(molecule, &[], &chosen);
    }
    true
}

/// How strong an acid the OH at `atom` is, lower first: a sulfonic, sulfuric
/// or sulfinic acid (and the selenium ones), then an acid of phosphorus or
/// arsenic, then a carboxylic or thiocarboxylic acid. `None` for anything
/// weaker -- a phenol, an alcohol, an enol, an imidic acid -- which is never
/// deprotonated to balance a charge.
fn acid_rank(molecule: &MolecularGraph, atom: usize) -> Option<u8> {
    if molecule.atomic_numbers()[atom] != 8
        || molecule.get_atom_charge(atom) != 0
        || hydrogens_on(molecule, atom).is_empty()
    {
        return None;
    }
    let mut heavy = heavy_neighbours(molecule, atom);
    let (Some(center), None) = (heavy.next(), heavy.next()) else {
        return None;
    };
    let double_to = |elements: &[u8]| {
        molecule
            .get_atom_bonds(center)
            .unwrap_or_default()
            .iter()
            .any(|bond| {
                bond.bond_order() == BondOrder::Double
                    && elements.contains(&molecule.atomic_numbers()[bond.target()])
            })
    };
    match molecule.atomic_numbers()[center] {
        16 | 34 if double_to(&[8]) => Some(0),
        15 | 33 => Some(1),
        6 if !molecule.is_atom_aromatic(center) && double_to(&[8, 16]) => Some(2),
        _ => None,
    }
}

/// How strong a base the nitrogen at `atom` is, lower first: the imine
/// nitrogen of an amidine or guanidine, then an aliphatic amine. `None` for
/// anything weaker -- an aromatic nitrogen, an aniline, an amide, a
/// hydroxylamine -- which is never protonated to balance a charge.
fn base_rank(molecule: &MolecularGraph, atom: usize) -> Option<u8> {
    if molecule.atomic_numbers()[atom] != 7
        || molecule.get_atom_charge(atom) != 0
        || molecule.is_atom_aromatic(atom)
    {
        return None;
    }
    let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
    let plain_carbon = |carbon: usize| {
        molecule.atomic_numbers()[carbon] == 6 && !molecule.is_atom_aromatic(carbon)
    };
    let heavy: Vec<_> = bonds
        .iter()
        .filter(|bond| molecule.atomic_numbers()[bond.target()] != 1)
        .collect();
    if let [imine] = heavy
        .iter()
        .filter(|bond| bond.bond_order() == BondOrder::Double)
        .collect::<Vec<_>>()[..]
    {
        let carbon = imine.target();
        let amidine = plain_carbon(carbon)
            && molecule
                .get_atom_bonds(carbon)
                .unwrap_or_default()
                .iter()
                .any(|bond| {
                    bond.target() != atom
                        && bond.bond_order() == BondOrder::Single
                        && molecule.atomic_numbers()[bond.target()] == 7
                        && molecule.get_atom_charge(bond.target()) == 0
                        && !molecule.is_atom_aromatic(bond.target())
                });
        return amidine.then_some(0);
    }
    let aliphatic = heavy.iter().all(|bond| {
        let carbon = bond.target();
        bond.bond_order() == BondOrder::Single
            && plain_carbon(carbon)
            && molecule
                .get_atom_bonds(carbon)
                .unwrap_or_default()
                .iter()
                .all(|other| other.bond_order() == BondOrder::Single)
    });
    aliphatic.then_some(1)
}

/// Takes one proton off each atom in `lose` (an atom listed twice loses two)
/// and puts one on each atom in `gain`, moving their charges with them.
fn move_protons(molecule: &mut MolecularGraph, lose: &[usize], gain: &[usize]) {
    for &atom in gain {
        *molecule.get_atom_charge_mut(atom) += 1;
        clear_tetrahedral(molecule, atom);
        let mut hydrogen = Atom::new(1);
        hydrogen.add_bond(BondTarget::single(atom));
        molecule.add_atom(hydrogen);
    }
    // Removing atoms renumbers everything after them, so the hydrogens are
    // collected first and taken off from the highest index down.
    let mut doomed: Vec<usize> = Vec::new();
    for &atom in lose {
        *molecule.get_atom_charge_mut(atom) -= 1;
        clear_tetrahedral(molecule, atom);
        if let Some(&hydrogen) = hydrogens_on(molecule, atom)
            .iter()
            .find(|hydrogen| !doomed.contains(hydrogen))
        {
            doomed.push(hydrogen);
        }
    }
    doomed.sort_unstable_by(|left, right| right.cmp(left));
    for hydrogen in doomed {
        molecule.remove_atom(hydrogen);
    }
}

// ---------------------------------------------------------------- salts

/// Counterions and solvents of crystallization, as the neutral acid, base or
/// solvent. Matched without stereo, so maleate and fumarate, and every
/// tartrate, are one entry. Water and the inorganic acids and bases have no
/// carbon and need no entry. Only compounds that are deposited as the other
/// half of a salt or as a solvate are listed: an entry is dropped whenever
/// the record holds something else, so a metabolite that is also a salt
/// former -- lysine, choline, ethanolamine -- is not.
pub const COUNTERIONS: &[&str] = &[
    // Acids.
    "OC=O",                                              // formic acid
    "CC(=O)O",                                           // acetic acid
    "OC(=O)C(F)(F)F",                                    // trifluoroacetic acid
    "OC(O)=O",                                           // carbonic acid
    "OC(=O)C(=O)O",                                      // oxalic acid
    "OC(=O)C=CC(=O)O",                                   // maleic and fumaric acid
    "OC(=O)CCC(=O)O",                                    // succinic acid
    "OC(=O)CCCCC(=O)O",                                  // adipic acid
    "OC(=O)CC(O)C(=O)O",                                 // malic acid
    "OC(=O)C(O)C(O)C(=O)O",                              // tartaric acid
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",                        // citric acid
    "CC(O)C(=O)O",                                       // lactic acid
    "OCC(O)C(O)C(O)C(O)C(=O)O",                          // gluconic acid
    "CS(=O)(=O)O",                                       // methanesulfonic acid
    "COS(=O)(=O)O",                                      // methyl sulfate
    "CCS(=O)(=O)O",                                      // ethanesulfonic acid
    "OS(=O)(=O)CCS(=O)(=O)O",                            // edisylic acid
    "OCCS(=O)(=O)O",                                     // isethionic acid
    "OS(=O)(=O)C(F)(F)F",                                // triflic acid
    "OS(=O)(=O)c1ccccc1",                                // benzenesulfonic acid
    "Cc1ccc(cc1)S(=O)(=O)O",                             // p-toluenesulfonic acid
    "OS(=O)(=O)c1ccc2ccccc2c1",                          // napsylic acid
    "OS(=O)(=O)c1cccc2c(cccc12)S(=O)(=O)O",              // napadisylic acid
    "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2",                  // camphorsulfonic acid
    "CCCCC(CC)COC(=O)CC(C(=O)OCC(CC)CCCC)S(=O)(=O)O",    // docusate
    "OC(=O)c1ccccc1",                                    // benzoic acid
    "OC(=O)c1ccc2ccccc2c1O",                             // xinafoic acid
    "Cn1c(=O)c2[nH]c(Cl)nc2n(C)c1=O",                    // 8-chlorotheophylline
    "OC(=O)c1cc2ccccc2c(Cc2c(O)c(cc3ccccc23)C(O)=O)c1O", // pamoic acid
    "CCCCCCCCCCCCCCCCCC(=O)O",                           // stearic acid
    // Bases.
    "CCN(CC)CC",                                  // triethylamine
    "CCNCC",                                      // diethylamine
    "CC(C)(C)N",                                  // tert-butylamine (erbumine)
    "NCCN",                                       // ethylenediamine
    "C1CNCCN1",                                   // piperazine
    "NC(CO)(CO)CO",                               // tromethamine
    "CNCC(O)C(O)C(O)C(O)CO",                      // meglumine
    "C(CNCc1ccccc1)NCc1ccccc1",                   // benzathine
    "C[N+](C)(C)C",                               // tetramethylammonium
    "CC[N+](CC)(CC)CC",                           // tetraethylammonium
    "CCCC[N+](CCCC)(CCCC)CCCC",                   // tetrabutylammonium
    "[B-](c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1", // tetraphenylborate
    // Solvents.
    "CO",        // methanol
    "CCO",       // ethanol
    "CC(C)O",    // 2-propanol
    "CC(C)=O",   // acetone
    "CC#N",      // acetonitrile
    "CS(C)=O",   // dimethyl sulfoxide
    "CN(C)C=O",  // dimethylformamide
    "ClCCl",     // dichloromethane
    "ClC(Cl)Cl", // chloroform
    "CCOC(C)=O", // ethyl acetate
    "CCOCC",     // diethyl ether
    "C1CCOC1",   // tetrahydrofuran
    "C1COCCO1",  // 1,4-dioxane
];

/// The counterion table as the canonical SMILES its entries have as parents,
/// without stereo, built once.
fn counterion_keys() -> &'static FxHashSet<String> {
    static KEYS: OnceLock<FxHashSet<String>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut canonizer = Canonizer::new();
        COUNTERIONS
            .iter()
            .map(|smiles| {
                let molecule = MolecularGraph::from_smiles_single(smiles)
                    .expect("every counterion in the table parses");
                let parent = neutralized(&mut canonizer, &molecule);
                lookup_key(&mut canonizer, &parent)
            })
            .collect()
    })
}

/// A parent component as it is looked up in the counterion table: without
/// stereo, so that maleate and fumarate, and every tartrate, match.
fn lookup_key(canonizer: &mut Canonizer, molecule: &MolecularGraph) -> String {
    let mut plain = molecule.clone();
    if let Some(classes) = plain.chiral_classes_mut() {
        classes
            .iter_mut()
            .for_each(|class| *class = ChiralClass::None);
    }
    canonizer.canonize(&plain).to_smiles()
}

/// Writes repeated components once and drops counterions; see the module
/// documentation for the rules.
fn strip_salts(canonizer: &mut Canonizer, components: Vec<MolecularGraph>) -> Vec<MolecularGraph> {
    let mut seen = FxHashSet::default();
    let components: Vec<MolecularGraph> = components
        .into_iter()
        .filter(|component| seen.insert(component.to_smiles()))
        .collect();
    if components.len() <= 1 {
        return components;
    }
    let table = counterion_keys();
    let (counterions, kept): (Vec<MolecularGraph>, Vec<MolecularGraph>) =
        components.into_iter().partition(|component| {
            !has_carbon(component) || table.contains(&lookup_key(canonizer, component))
        });
    if !kept.is_empty() {
        return kept;
    }
    // Nothing but counterions: the ones with carbon are what the record is of.
    let organic: Vec<MolecularGraph> = counterions
        .iter()
        .filter(|c| has_carbon(c))
        .cloned()
        .collect();
    if organic.is_empty() {
        counterions
    } else {
        organic
    }
}

// ---------------------------------------------------------------- helpers

fn hydrogens_on(molecule: &MolecularGraph, atom: usize) -> Vec<usize> {
    molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .map(|bond| bond.target())
        .filter(|&neighbour| {
            molecule.atomic_numbers()[neighbour] == 1
                && molecule
                    .get_atom_bonds(neighbour)
                    .is_some_and(|bonds| bonds.len() == 1)
        })
        .collect()
}

fn heavy_neighbours(molecule: &MolecularGraph, atom: usize) -> impl Iterator<Item = usize> + '_ {
    molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .map(|bond| bond.target())
        .filter(|&neighbour| molecule.atomic_numbers()[neighbour] != 1)
}

fn heavy_degree(molecule: &MolecularGraph, atom: usize) -> usize {
    heavy_neighbours(molecule, atom).count()
}

fn set_bond_order(molecule: &mut MolecularGraph, first: usize, second: usize, order: BondOrder) {
    for (from, to) in [(first, second), (second, first)] {
        if let Some(bonds) = molecule.atom_bonds_mut().get_mut(from) {
            for bond in bonds.iter_mut() {
                if bond.target == to {
                    bond.bond_order = order;
                }
            }
        }
    }
}

fn remove_bond(molecule: &mut MolecularGraph, first: usize, second: usize) {
    for (from, to) in [(first, second), (second, first)] {
        if let Some(bonds) = molecule.atom_bonds_mut().get_mut(from) {
            bonds.retain(|bond| bond.target != to);
        }
    }
}

/// A tetrahedral tag describes the neighbours an atom had; once a proton or a
/// bond is added or taken away it describes a centre that is not there.
fn clear_tetrahedral(molecule: &mut MolecularGraph, atom: usize) {
    if let Some(classes) = molecule.chiral_classes_mut()
        && let Some(class) = classes.get_mut(atom)
        && !matches!(
            class,
            ChiralClass::None
                | ChiralClass::CisTransClockwise
                | ChiralClass::CisTransCounterclockwise
        )
    {
        *class = ChiralClass::None;
    }
}
