//! The parent compound: what is left of a record once counterions, metals
//! drawn as bonds, alternative spellings of a group, and protons that only an
//! ionized form carries are taken away.
//!
//! Registries and spectral libraries hold the same compound written many ways:
//! as the hydrochloride, the sodium salt with the sodium bonded to the oxygen,
//! a sulfoxide as `S=O` or as `[S+][O-]`, an amino acid as the zwitterion.
//! [`parent_smiles`] reduces each of them to one structure:
//!
//! 1. **Metals.** A bond from an alkali or alkaline-earth metal to N, O, S, Se
//!    or a halogen is broken; the metal leaves as its cation and the atom it
//!    was bonded to keeps the electron, so `CC(=O)O[Na]` becomes acetate and
//!    sodium. Transition metals stay where they are: the cobalt of cobalamin is
//!    part of the compound.
//! 2. **Groups.** One spelling per group: neutral where the element has the
//!    valence for it (`S=O`, `P=O`, `Se=O`), charge-separated where it does not
//!    (`N=[N+]=[N-]` for azides, `C=[N+]=[N-]` for diazo compounds; nitro groups
//!    and N-oxides already arrive that way from the reader).
//! 3. **Charges.** Protons come off cations that carry one and go onto anions,
//!    except where a cation that cannot lose a proton -- a quaternary ammonium,
//!    a sulfonium -- needs an anion to balance it. Those anions are chosen by
//!    acidity, so carnitine stays the carboxylate zwitterion rather than
//!    becoming a cation. Charge pairs that are the normal way to write a group
//!    (nitro, N-oxide, azide) are left as they are.
//! 4. **Salts.** Components without carbon and common counterions (acetate,
//!    trifluoroacetate, mesylate, citrate, ...) are dropped and the largest
//!    remaining component is kept. A record that is nothing but counterions
//!    keeps its largest component, so acetic acid on its own stays acetic acid.
//!
//! Every step can be switched off with [`ParentOptions`]. Nothing here changes
//! what any other function in the crate returns; the tautomer functions take
//! the parent's SMILES like any other input.

use std::sync::OnceLock;

use rustc_hash::FxHashSet;

use crate::atom::Atom;
use crate::graph_algorithms::mol_canonization::{Canonizer, with_canonizer};
use crate::io::SMILESParseError;
use crate::molecule::bond::{BondOrder, BondTarget};
use crate::molecule::chirality::ChiralClass;
use crate::molecule::descriptors::MolecularDescriptors;
use crate::prelude::*;

/// Which of the steps [`parent_smiles`] takes. All are on by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParentOptions {
    /// Break bonds to alkali and alkaline-earth metals.
    pub disconnect_metals: bool,
    /// Rewrite groups to one spelling (`[S+][O-]` to `S=O`, `N=N#N` to
    /// `N=[N+]=[N-]`, ...).
    pub normalize_groups: bool,
    /// Add and remove protons until the net charge is as close to zero as the
    /// fixed charges allow.
    pub neutralize: bool,
    /// Drop counterions and keep the largest component.
    pub strip_salts: bool,
}

impl Default for ParentOptions {
    fn default() -> Self {
        Self {
            disconnect_metals: true,
            normalize_groups: true,
            neutralize: true,
            strip_salts: true,
        }
    }
}

/// The parent compound of a SMILES, as canonical SMILES.
///
/// ```
/// use oxidane::molecule::parent::{parent_smiles, ParentOptions};
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
/// ```
pub fn parent_smiles(smiles: &str, options: &ParentOptions) -> Result<String, SMILESParseError> {
    let components = MolecularGraph::from_smiles(smiles)?;
    Ok(with_canonizer(|canonizer| {
        let parents = parent_components(canonizer, components, options);
        let mut written: Vec<String> = parents
            .iter()
            .map(|component| canonizer.canonize(component).to_smiles())
            .collect();
        written.sort();
        written.join(".")
    }))
}

/// [`parent_smiles`] on components already read.
pub fn parent_components(
    canonizer: &mut Canonizer,
    components: Vec<MolecularGraph>,
    options: &ParentOptions,
) -> Vec<MolecularGraph> {
    let mut components = if options.disconnect_metals {
        components
            .into_iter()
            .flat_map(|mut component| {
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
    if options.normalize_groups {
        for component in &mut components {
            normalize_groups(component);
        }
    }
    // A transition-metal ion written as its own component is the centre of a
    // complex written ionically -- heme as a porphyrin dianion and `[Fe+2]` --
    // not a counterion. Neither neutralizing the ligand nor dropping the metal
    // gives the compound back, so such a record is left as it was written.
    let complex = components.iter().any(is_metal_ion);
    if options.neutralize && !complex {
        components = components
            .iter()
            .map(|component| neutralized(canonizer, component))
            .collect();
    }
    if options.strip_salts && !complex {
        components = strip_salts(canonizer, components);
    }
    components
}

/// A component's neutralized form. Which of two equally acidic sites ends up
/// holding a balancing charge is decided in the canonical order of the fully
/// protonated molecule, so it is the same wherever the record put the charge
/// to begin with -- and the same when the answer is neutralized again.
fn neutralized(canonizer: &mut Canonizer, component: &MolecularGraph) -> MolecularGraph {
    let mut protonated = canonizer.canonize(component);
    neutralize(&mut protonated, false);
    let mut canonical = canonizer.canonize(&protonated);
    neutralize(&mut canonical, true);
    canonical
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

// ---------------------------------------------------------------- groups

/// Rewrites groups to one spelling. Applied until nothing changes, since a
/// sulfone written `[S+2]([O-])[O-]` takes two passes.
fn normalize_groups(molecule: &mut MolecularGraph) {
    while charge_separated_oxide(molecule)
        || pentavalent_azide(molecule)
        || diazo_carbanion(molecule)
    {}
}

/// `[S+][O-]`, `[P+][O-]`, `[Se+][O-]` (and the sulfur analogues) become the
/// double bond these elements have the valence for.
fn charge_separated_oxide(molecule: &mut MolecularGraph) -> bool {
    for center in 0..molecule.atomic_numbers().len() {
        if !matches!(molecule.atomic_numbers()[center], 15 | 16 | 33 | 34)
            || molecule.get_atom_charge(center) <= 0
        {
            continue;
        }
        let terminal = molecule
            .get_atom_bonds(center)
            .unwrap_or_default()
            .iter()
            .find(|bond| {
                let atom = bond.target();
                bond.bond_order() == BondOrder::Single
                    && matches!(molecule.atomic_numbers()[atom], 8 | 16)
                    && molecule.get_atom_charge(atom) == -1
                    && heavy_degree(molecule, atom) == 1
            })
            .map(|bond| bond.target());
        if let Some(oxygen) = terminal {
            set_bond_order(molecule, center, oxygen, BondOrder::Double);
            *molecule.get_atom_charge_mut(center) -= 1;
            *molecule.get_atom_charge_mut(oxygen) += 1;
            return true;
        }
    }
    false
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
/// written is `C=[N+]=[N-]`.
fn diazo_carbanion(molecule: &mut MolecularGraph) -> bool {
    for carbon in 0..molecule.atomic_numbers().len() {
        if molecule.atomic_numbers()[carbon] != 6 || molecule.get_atom_charge(carbon) != -1 {
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

/// How readily an anion holds its charge; lower is more acidic, and the most
/// acidic anions are the ones kept when a fixed cation needs balancing.
fn acidity_rank(molecule: &MolecularGraph, anion: usize) -> u8 {
    let element = molecule.atomic_numbers()[anion];
    let bonds = molecule.get_atom_bonds(anion).unwrap_or_default();
    let heavy: Vec<usize> = bonds
        .iter()
        .map(|bond| bond.target())
        .filter(|&atom| molecule.atomic_numbers()[atom] != 1)
        .collect();
    if element == 7 {
        // Tetrazoles, acyl sulfonamides and imides: an anion inside a
        // conjugated system or next to a sulfonyl.
        let next_to_sulfonyl = heavy
            .iter()
            .any(|&atom| molecule.atomic_numbers()[atom] == 16);
        return if next_to_sulfonyl || molecule.is_atom_aromatic(anion) {
            3
        } else {
            5
        };
    }
    let Some(&center) = heavy.first() else {
        return 5;
    };
    let center_element = molecule.atomic_numbers()[center];
    let center_has_double_to_hetero = molecule
        .get_atom_bonds(center)
        .unwrap_or_default()
        .iter()
        .any(|bond| {
            bond.bond_order() == BondOrder::Double
                && matches!(molecule.atomic_numbers()[bond.target()], 7 | 8 | 16)
        });
    match center_element {
        16 | 34 if center_has_double_to_hetero => 0, // sulfonate, sulfate, sulfinate
        15 | 33 => 1,                                // phosphate, phosphonate, arsonate
        6 if center_has_double_to_hetero => 2,       // carboxylate, thiocarboxylate
        6 if molecule.is_atom_aromatic(center) => 3, // phenolate
        _ if element == 16 => 3,                     // thiolate
        _ => 4,                                      // alkoxide and the rest
    }
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

/// Moves protons on and off until the net charge is as close to zero as the
/// charges that no proton can remove allow -- or, with `balance` off, until
/// every site that can be neutral is, fixed cations or not. Works on a
/// molecule in canonical order, which is what makes the choice among equal
/// sites canonical.
fn neutralize(molecule: &mut MolecularGraph, balance: bool) {
    let atoms = molecule.atomic_numbers().len();
    let mut fixed = 0i32;
    let mut lose_hydrogen: Vec<usize> = Vec::new();
    let mut anions: Vec<(u8, usize)> = Vec::new();
    let paired = charge_pairs(molecule);
    for atom in 0..atoms {
        let charge = molecule.get_atom_charge(atom) as i32;
        if charge == 0 || molecule.atomic_numbers()[atom] == 1 || paired.contains(&atom) {
            continue;
        }
        if charge > 0 {
            let hydrogens = hydrogens_on(molecule, atom).len() as i32;
            let removable = charge.min(hydrogens);
            lose_hydrogen.extend(std::iter::repeat_n(atom, removable as usize));
            fixed += charge - removable;
        } else if charge == -1 && matches!(molecule.atomic_numbers()[atom], 9 | 17 | 35 | 53) {
            // A free halide: the hydrohalic acid, which the salt step drops.
            anions.push((0, atom));
        } else if charge == -1 && matches!(molecule.atomic_numbers()[atom], 7 | 8 | 16 | 34) {
            anions.push((acidity_rank(molecule, atom), atom));
        } else {
            fixed += charge;
        }
    }
    // The fixed positive charge left over is balanced by that many anions.
    // Which ones is decided over the anions present and the acid groups that
    // could become one alike, the most acidic first and ties in canonical
    // order, so that the answer does not depend on which site the record put
    // the charge on: carnitine written as an alkoxide and a carboxylic acid,
    // or as `[N+]...C(=O)O` with no anion at all (which is the [M+H]+ ion, not
    // the compound), comes out as the one carboxylate zwitterion. Only real
    // acids (sulfonic, phosphoric, carboxylic) are offered; a phenol or an
    // alcohol keeps its proton, and a molecule with nothing better stays a
    // cation.
    let keep = if balance { fixed.max(0) as usize } else { 0 };
    let mut sites: Vec<(u8, usize, bool)> = anions
        .iter()
        .map(|&(rank, atom)| (rank, atom, true))
        .collect();
    if keep > 0 {
        for atom in 0..atoms {
            if molecule.atomic_numbers()[atom] == 8
                && molecule.get_atom_charge(atom) == 0
                && heavy_degree(molecule, atom) == 1
                && !hydrogens_on(molecule, atom).is_empty()
            {
                let rank = acidity_rank(molecule, atom);
                if rank <= 2 {
                    sites.push((rank, atom, false));
                }
            }
        }
    }
    sites.sort();
    let (charged, neutral): (Vec<_>, Vec<_>) = sites
        .iter()
        .enumerate()
        .partition(|(place, _)| *place < keep);
    // Anions not chosen take a proton; acids chosen give one up, the way a
    // cation does.
    let gain_hydrogen: Vec<usize> = neutral
        .iter()
        .filter(|(_, (_, _, is_anion))| *is_anion)
        .map(|(_, (_, atom, _))| *atom)
        .collect();
    lose_hydrogen.extend(
        charged
            .iter()
            .filter(|(_, (_, _, is_anion))| !*is_anion)
            .map(|(_, (_, atom, _))| *atom),
    );
    if lose_hydrogen.is_empty() && gain_hydrogen.is_empty() {
        return;
    }

    for &atom in &gain_hydrogen {
        *molecule.get_atom_charge_mut(atom) += 1;
        clear_tetrahedral(molecule, atom);
        let mut hydrogen = Atom::new(1);
        hydrogen.add_bond(BondTarget::single(atom));
        molecule.add_atom(hydrogen);
    }
    // Removing atoms renumbers everything after them, so the hydrogens are
    // collected first and taken off from the highest index down.
    let mut doomed: Vec<usize> = Vec::new();
    let mut taken: FxHashSet<usize> = FxHashSet::default();
    for &atom in &lose_hydrogen {
        *molecule.get_atom_charge_mut(atom) -= 1;
        clear_tetrahedral(molecule, atom);
        if let Some(&hydrogen) = hydrogens_on(molecule, atom)
            .iter()
            .find(|hydrogen| !taken.contains(hydrogen))
        {
            taken.insert(hydrogen);
            doomed.push(hydrogen);
        }
    }
    doomed.sort_unstable_by(|left, right| right.cmp(left));
    for hydrogen in doomed {
        molecule.remove_atom(hydrogen);
    }
}

// ---------------------------------------------------------------- salts

/// Counterions and solvents of crystallization, neutral, without stereo.
const COUNTERIONS: &[&str] = &[
    "O",                          // water
    "OC=O",                       // formic acid
    "CC(=O)O",                    // acetic acid
    "OC(=O)C(F)(F)F",             // trifluoroacetic acid
    "OC(=O)C(=O)O",               // oxalic acid
    "OC(=O)C=CC(=O)O",            // maleic / fumaric acid
    "OC(=O)CCC(=O)O",             // succinic acid
    "OC(=O)CC(O)C(=O)O",          // malic acid
    "OC(=O)C(O)C(O)C(=O)O",       // tartaric acid
    "OC(=O)CC(O)(CC(=O)O)C(=O)O", // citric acid
    "CC(O)C(=O)O",                // lactic acid
    "CS(=O)(=O)O",                // methanesulfonic acid
    "CCS(=O)(=O)O",               // ethanesulfonic acid
    "OCCS(=O)(=O)O",              // isethionic acid
    "OS(=O)(=O)c1ccccc1",         // benzenesulfonic acid
    "Cc1ccc(S(=O)(=O)O)cc1",      // p-toluenesulfonic acid
    "OC(=O)c1ccccc1",             // benzoic acid
    "CCN(CC)CC",                  // triethylamine
    "CCNCC",                      // diethylamine
    "NC(CO)(CO)CO",               // tris
    "OCC(O)C(O)C(O)C(O)CNC",      // meglumine
    "OC(=O)CCCCC(=O)O",           // adipic acid
    "C[N+](C)(C)C",               // tetramethylammonium
    "CCCC[N+](CCCC)(CCCC)CCCC",   // tetrabutylammonium
];

/// The counterion table as canonical SMILES of the neutral, stereo-free form,
/// built once.
fn counterion_keys() -> &'static FxHashSet<String> {
    static KEYS: OnceLock<FxHashSet<String>> = OnceLock::new();
    KEYS.get_or_init(|| {
        with_canonizer(|canonizer| {
            COUNTERIONS
                .iter()
                .filter_map(|smiles| MolecularGraph::from_smiles_single(smiles).ok())
                .map(|molecule| lookup_key(canonizer, &molecule))
                .collect()
        })
    })
}

/// A component as it is looked up in the counterion table: neutralized and
/// without stereo, so that maleate and fumarate, and every tartrate, match.
fn lookup_key(canonizer: &mut Canonizer, molecule: &MolecularGraph) -> String {
    let mut plain = neutralized(canonizer, molecule);
    if let Some(classes) = plain.chiral_classes_mut() {
        classes
            .iter_mut()
            .for_each(|class| *class = ChiralClass::None);
    }
    canonizer.canonize(&plain).to_smiles()
}

fn heavy_atoms(molecule: &MolecularGraph) -> usize {
    molecule
        .atomic_numbers()
        .iter()
        .filter(|&&z| z != 1)
        .count()
}

/// Drops counterions and keeps the largest of what is left; see the module
/// documentation for the rules.
fn strip_salts(canonizer: &mut Canonizer, components: Vec<MolecularGraph>) -> Vec<MolecularGraph> {
    if components.len() <= 1 {
        return components;
    }
    let table = counterion_keys();
    let is_counterion = |canonizer: &mut Canonizer, molecule: &MolecularGraph| {
        !molecule.atomic_numbers().contains(&6) || table.contains(&lookup_key(canonizer, molecule))
    };
    let flags: Vec<bool> = components
        .iter()
        .map(|component| is_counterion(canonizer, component))
        .collect();
    let candidates: Vec<&MolecularGraph> = if flags.iter().all(|&flag| flag) {
        components.iter().collect()
    } else {
        components
            .iter()
            .zip(&flags)
            .filter(|(_, flag)| !**flag)
            .map(|(component, _)| component)
            .collect()
    };
    // Largest by heavy atoms, then by mass, then by canonical SMILES, so that
    // the choice never depends on the order the components were written in.
    let best = candidates
        .into_iter()
        .map(|component| {
            let smiles = canonizer.canonize(component).to_smiles();
            (
                heavy_atoms(component),
                component.molecular_weight(),
                smiles,
                component,
            )
        })
        .max_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.total_cmp(&right.1))
                .then(right.2.cmp(&left.2))
        })
        .map(|(_, _, _, component)| component.clone());
    best.into_iter().collect()
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

fn heavy_degree(molecule: &MolecularGraph, atom: usize) -> usize {
    molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .filter(|bond| molecule.atomic_numbers()[bond.target()] != 1)
        .count()
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
