//! Wildman-Crippen atomic contributions to log P and molar refractivity.
//!
//! Both numbers are sums over atoms: every atom is put into one of sixty-eight
//! classes by what it is and what it is bonded to, and each class carries a
//! contribution fitted once, in 1999, against measured partition coefficients
//! and refractivities. The model has no conformer, no charge model and no
//! notion of intramolecular hydrogen bonding; what it has is that it is
//! reproducible, costs one pass over the atoms, and is the number every
//! published filter -- Lipinski, Ghose, the lead-like ranges -- was actually
//! drawn against.
//!
//! Hydrogens are classified in their own right rather than folded into the
//! atoms they hang off, which is why they must be present as atoms. They are,
//! after parsing.
//!
//! The classes are tried in a fixed order and the first that fits wins, so the
//! order is part of the definition rather than an implementation detail: a
//! carbon bonded to a heteroatom is `C3` because `C1` was tried first and did
//! not fit, and a carboxylate oxygen is `O12` only because `O12` is tried
//! before `O7`. That order is Wildman and Crippen's, as published and as
//! distributed with RDKit, and the contributions here are theirs to four
//! significant figures.
//!
//! Where the two toolkits can differ is in what counts as aromatic, since the
//! classes are stated in terms of aromatic and aliphatic atoms and each toolkit
//! perceives that itself. A ring this crate calls aromatic and another does not
//! will be typed differently by both, correctly, under different models.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::descriptors::MolecularDescriptors;
//!
//! let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
//! let (log_p, refractivity) = aspirin.crippen_descriptors();
//! assert!((log_p - 1.3101).abs() < 1e-3);
//! assert!((refractivity - 44.7103).abs() < 1e-3);
//! ```

use super::traits::MoleculeCore;
use crate::molecule::bond::BondOrder;

pub use chemistry_consts::crippen::CrippenType;

/// Every atom's class, indexed by atom.
pub fn crippen_types<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<CrippenType> {
    let view = View::new(molecule);
    (0..molecule.len())
        .map(|atom| view.classify(atom))
        .collect()
}

/// Every atom's `(log P, molar refractivity)` contribution, indexed by atom.
///
/// Useful for colouring a structure by where its lipophilicity comes from, and
/// for arguing with a number rather than merely quoting it.
pub fn crippen_contributions<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<(f64, f64)> {
    let view = View::new(molecule);
    (0..molecule.len())
        .map(|atom| view.classify(atom).contribution())
        .collect()
}

/// The molecule's `(log P, molar refractivity)`.
pub fn crippen_descriptors<M: MoleculeCore + ?Sized>(molecule: &M) -> (f64, f64) {
    let view = View::new(molecule);
    let mut log_p = 0.0;
    let mut refractivity = 0.0;
    for atom in 0..molecule.len() {
        let (atom_log_p, atom_refractivity) = view.classify(atom).contribution();
        log_p += atom_log_p;
        refractivity += atom_refractivity;
    }
    (log_p, refractivity)
}

/// The elements `C3` and `C4` name as heteroatoms.
const C3_HETEROATOMS: [u8; 8] = [7, 8, 15, 16, 9, 17, 35, 53];

/// What `C13`'s substituent may not be: carbon, nitrogen, oxygen, sulfur, a
/// halogen, or hydrogen.
const C13_EXCLUDED: [u8; 9] = [1, 6, 7, 8, 16, 9, 17, 35, 53];

/// What `C27`'s substituent may not be. `C13`'s list with phosphorus added.
const C27_EXCLUDED: [u8; 10] = [1, 6, 7, 8, 15, 16, 9, 17, 35, 53];

/// The elements `Me1` covers: the s-block and the heavier p-block metals and
/// metalloids.
const ME1_ELEMENTS: [u8; 22] = [
    3, 11, 19, 37, 55, 4, 12, 20, 38, 56, 5, 13, 31, 49, 81, 14, 32, 50, 82, 33, 51, 83,
];

/// The alkali metals, which are `Hal` when charged and `Me1` when not.
const ALKALI_METALS: [u8; 5] = [3, 11, 19, 37, 55];

/// One molecule, with the per-atom facts every rule asks about gathered once.
struct View<'a, M: MoleculeCore + ?Sized> {
    molecule: &'a M,
}

/// What a neighbour is, from the point of view of a rule.
#[derive(Clone, Copy)]
struct Neighbour {
    atom: usize,
    atomic_number: u8,
    aromatic: bool,
    order: BondOrder,
}

impl Neighbour {
    /// Whether the bond would match a SMARTS pattern that names no bond at all,
    /// which means single or aromatic and, in particular, not double.
    fn plain(&self) -> bool {
        matches!(
            self.order,
            BondOrder::Single
                | BondOrder::Up
                | BondOrder::Down
                | BondOrder::Coordinate
                | BondOrder::Aromatic
        )
    }

    /// Whether the bond is a single bond written as such.
    fn single(&self) -> bool {
        matches!(
            self.order,
            BondOrder::Single | BondOrder::Up | BondOrder::Down | BondOrder::Coordinate
        )
    }

    fn double(&self) -> bool {
        self.order == BondOrder::Double
    }

    fn triple(&self) -> bool {
        self.order == BondOrder::Triple
    }

    fn aromatic_bond(&self) -> bool {
        self.order == BondOrder::Aromatic
    }

    /// An aliphatic atom that is not hydrogen: SMARTS `[A;!#1]`.
    fn aliphatic_heavy(&self) -> bool {
        !self.aromatic && self.atomic_number != 1
    }

    /// Any atom that is not hydrogen: SMARTS `[!#1;A,a]`.
    fn heavy(&self) -> bool {
        self.atomic_number != 1
    }

    /// An aliphatic atom of the given element: SMARTS `C`, `N`, `O` and so on.
    fn aliphatic(&self, atomic_number: u8) -> bool {
        !self.aromatic && self.atomic_number == atomic_number
    }
}

impl<'a, M: MoleculeCore + ?Sized> View<'a, M> {
    fn new(molecule: &'a M) -> Self {
        Self { molecule }
    }

    fn neighbours(&self, atom: usize) -> impl Iterator<Item = Neighbour> + '_ {
        self.molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .map(|bond| {
                let target = bond.target();
                Neighbour {
                    atom: target,
                    atomic_number: self.molecule.get_atomic_number(target),
                    aromatic: self.molecule.is_atom_aromatic(target),
                    order: bond.bond_order(),
                }
            })
    }

    fn hydrogens(&self, atom: usize) -> usize {
        self.neighbours(atom)
            .filter(|neighbour| neighbour.atomic_number == 1)
            .count()
    }

    /// How many connections an atom has, which is SMARTS `X` -- and is only the
    /// bond count because hydrogens are atoms here.
    fn connections(&self, atom: usize) -> usize {
        self.molecule.get_atom_bonds(atom).unwrap_or_default().len()
    }

    fn classify(&self, atom: usize) -> CrippenType {
        let atomic_number = self.molecule.get_atomic_number(atom);
        let aromatic = self.molecule.is_atom_aromatic(atom);
        match (atomic_number, aromatic) {
            (1, _) => self.hydrogen(atom),
            (6, false) => self.aliphatic_carbon(atom),
            (6, true) => self.aromatic_carbon(atom),
            (7, false) => self.aliphatic_nitrogen(atom),
            (7, true) => self.aromatic_nitrogen(atom),
            (8, false) => self.aliphatic_oxygen(atom),
            (8, true) => CrippenType::O1,
            (16, true) => CrippenType::S3,
            (16, false) => self.aliphatic_sulfur(atom),
            (15, _) => CrippenType::P,
            _ => self.other_element(atom, atomic_number),
        }
    }

    // ---------------------------------------------------------------- carbon

    fn aliphatic_carbon(&self, atom: usize) -> CrippenType {
        use CrippenType::*;
        let hydrogens = self.hydrogens(atom);
        let connections = self.connections(atom);

        // C1 and C2: nothing but carbon and hydrogen around.
        let plain_carbons = self
            .neighbours(atom)
            .filter(|neighbour| neighbour.plain() && neighbour.aliphatic(6))
            .count();
        match hydrogens {
            4 => return C1,
            3 if plain_carbons >= 1 => return C1,
            2 if plain_carbons >= 2 => return C1,
            1 if plain_carbons >= 3 => return C2,
            0 if plain_carbons >= 4 => return C2,
            _ => (),
        }

        // C3 and C4: an sp3 carbon carrying a heteroatom, its remaining
        // neighbours aliphatic.
        if connections == 4 {
            let heavy: Vec<Neighbour> = self.neighbours(atom).filter(Neighbour::heavy).collect();
            let carries_heteroatom = heavy.iter().any(|neighbour| {
                !neighbour.aromatic && C3_HETEROATOMS.contains(&neighbour.atomic_number)
            });
            if carries_heteroatom && heavy.iter().all(Neighbour::aliphatic_heavy) {
                return if hydrogens >= 2 { C3 } else { C4 };
            }
        }

        // C5: a double bond to anything aliphatic that is not carbon.
        if self.neighbours(atom).any(|neighbour| {
            neighbour.double() && neighbour.aliphatic_heavy() && neighbour.atomic_number != 6
        }) {
            return C5;
        }

        // C6: an alkene carbon whose other neighbours are aliphatic.
        let doubly_bonded_carbons = self
            .neighbours(atom)
            .filter(|neighbour| neighbour.double() && neighbour.aliphatic(6))
            .count();
        if doubly_bonded_carbons >= 1 {
            let others = self
                .neighbours(atom)
                .filter(|neighbour| !neighbour.double() && neighbour.aliphatic_heavy())
                .count();
            let matches_c6 = match hydrogens {
                2 => true,
                1 => others >= 1,
                0 => others >= 2 || doubly_bonded_carbons >= 2,
                _ => false,
            };
            if matches_c6 {
                return C6;
            }
        }

        // C7: a terminal triple bond.
        if connections == 2
            && self
                .neighbours(atom)
                .any(|neighbour| neighbour.triple() && neighbour.aliphatic_heavy())
        {
            return C7;
        }

        // C8 to C12: a saturated carbon hanging off a ring.
        let on_aromatic_carbon = self.neighbours(atom).any(|neighbour| {
            neighbour.plain() && neighbour.aromatic && neighbour.atomic_number == 6
        });
        let on_aromatic = self
            .neighbours(atom)
            .any(|neighbour| neighbour.plain() && neighbour.aromatic);
        if hydrogens == 3 && on_aromatic_carbon {
            return C8;
        }
        if hydrogens == 3 && on_aromatic {
            return C9;
        }
        if connections == 4 && on_aromatic {
            return match hydrogens {
                2 => C10,
                1 => C11,
                _ => C12,
            };
        }

        // C26: an alkene carbon that touches a ring.
        if self.neighbours(atom).any(|neighbour| {
            neighbour.double() && neighbour.aromatic && neighbour.atomic_number == 6
        }) {
            return C26;
        }
        if doubly_bonded_carbons >= 1 {
            let aromatic_neighbours = self
                .neighbours(atom)
                .filter(|neighbour| !neighbour.double() && neighbour.aromatic)
                .count();
            let aromatic_carbons = self
                .neighbours(atom)
                .filter(|neighbour| {
                    !neighbour.double() && neighbour.aromatic && neighbour.atomic_number == 6
                })
                .count();
            let aliphatic_others = self
                .neighbours(atom)
                .filter(|neighbour| !neighbour.double() && neighbour.aliphatic_heavy())
                .count();
            if (aromatic_neighbours >= 1 && aliphatic_others >= 1)
                || (aromatic_carbons >= 1 && aromatic_neighbours >= 2)
                || (hydrogens == 1 && aromatic_neighbours >= 1)
            {
                return C26;
            }
        }

        // C27: an sp3 carbon on something exotic -- a silicon, a boron, a metal.
        if connections == 4
            && self.neighbours(atom).any(|neighbour| {
                neighbour.plain()
                    && neighbour.aliphatic_heavy()
                    && !C27_EXCLUDED.contains(&neighbour.atomic_number)
            })
        {
            return C27;
        }

        CS
    }

    fn aromatic_carbon(&self, atom: usize) -> CrippenType {
        use CrippenType::*;
        let hydrogens = self.hydrogens(atom);

        // C13: a substituted ring carbon carrying something exotic.
        if hydrogens == 0
            && self.neighbours(atom).any(|neighbour| {
                neighbour.single()
                    && neighbour.aliphatic_heavy()
                    && !C13_EXCLUDED.contains(&neighbour.atomic_number)
            })
        {
            return C13;
        }

        // C14 to C17: a ring carbon carrying a halogen. Each halogen has its
        // own class, since this is where most of a halogen's effect on log P
        // shows up.
        for (element, class) in [(9, C14), (17, C15), (35, C16), (53, C17)] {
            if self
                .neighbours(atom)
                .any(|neighbour| neighbour.plain() && neighbour.atomic_number == element)
            {
                return class;
            }
        }

        if hydrogens == 1 {
            return C18;
        }

        // C19 to C25: a ring carbon by what its one substituent is. Everything
        // here needs the two ring bonds first.
        let ring_bonds = self
            .neighbours(atom)
            .filter(|neighbour| neighbour.aromatic_bond() && neighbour.aromatic)
            .count();
        if ring_bonds >= 3 {
            return C19;
        }
        if ring_bonds >= 2 {
            for neighbour in self.neighbours(atom) {
                if neighbour.single() {
                    if neighbour.aromatic {
                        return C20;
                    }
                    match neighbour.atomic_number {
                        6 => return C21,
                        7 => return C22,
                        8 => return C23,
                        16 => return C24,
                        _ => (),
                    }
                }
                if neighbour.double()
                    && matches!(neighbour.atomic_number, 6 | 7 | 8)
                    && !neighbour.aromatic
                {
                    return C25;
                }
            }
        }

        CS
    }

    // -------------------------------------------------------------- hydrogen

    fn hydrogen(&self, atom: usize) -> CrippenType {
        use CrippenType::*;
        let Some(attached) = self.neighbours(atom).next() else {
            return HS;
        };

        // H1: on carbon or on another hydrogen.
        if matches!(attached.atomic_number, 1 | 6) {
            return H1;
        }

        // A hydroxyl hydrogen is told apart by what the oxygen itself carries,
        // so it is decided from there. The forms are tried in the published
        // order, and a hydroxide ion -- an oxygen with nothing on it but this
        // hydrogen -- fits none of them and falls through to `HS`.
        if attached.atomic_number == 8 {
            // Every form names its oxygen's second bond as a plain one, so a
            // hydrogen on a doubly bonded oxygen -- a protonated ketone --
            // fits none of them.
            let beyond = || {
                self.neighbours(attached.atom)
                    .filter(move |neighbour| neighbour.atom != atom && neighbour.plain())
            };
            // H2: on an oxygen bonded to a saturated or aromatic carbon ...
            if beyond().any(|neighbour| {
                neighbour.atomic_number == 6
                    && (neighbour.aromatic || self.connections(neighbour.atom) == 4)
            }) {
                return H2;
            }
            // ... or to anything that is not carbon, nitrogen, oxygen or
            // sulfur, which includes the other hydrogen of water and the
            // phosphorus of a phosphate.
            if beyond().any(|neighbour| !matches!(neighbour.atomic_number, 6 | 7 | 8 | 16)) {
                return H2;
            }
            // H3: on an oxygen bonded to a nitrogen -- a hydroxylamine, an oxime.
            if beyond().any(|neighbour| neighbour.atomic_number == 7) {
                return H3;
            }
            // H4: on an acid or an enol, and on a peroxide or a thioperoxide.
            if beyond().any(|neighbour| {
                neighbour.aliphatic(6)
                    && self.neighbours(neighbour.atom).any(|further| {
                        further.double()
                            && (matches!(further.atomic_number, 6 | 7)
                                || (further.aliphatic_heavy()
                                    && matches!(further.atomic_number, 8 | 16)))
                    })
            }) {
                return H4;
            }
            if beyond()
                .any(|neighbour| !neighbour.aromatic && matches!(neighbour.atomic_number, 8 | 16))
            {
                return H4;
            }
            return HS;
        }

        // H3: on a nitrogen, aromatic or not. The published order puts H2's
        // last form -- a hydrogen on anything that is not carbon, nitrogen or
        // oxygen -- above this, and that form does match an aromatic nitrogen's
        // hydrogen as SMARTS; RDKit nonetheless calls it H3, and agreeing with
        // the toolkit everyone's numbers were computed with is worth more here
        // than agreeing with a reading of the table.
        if attached.atomic_number == 7 {
            return H3;
        }

        // H2: on a sulfur, a phosphorus, a silicon, a boron, a metal.
        H2
    }

    // -------------------------------------------------------------- nitrogen

    fn aromatic_nitrogen(&self, atom: usize) -> CrippenType {
        match self.molecule.get_atom_charge(atom) {
            0 => CrippenType::N11,
            charge if charge > 0 => CrippenType::N12,
            _ => CrippenType::NS,
        }
    }

    fn aliphatic_nitrogen(&self, atom: usize) -> CrippenType {
        use CrippenType::*;
        let hydrogens = self.hydrogens(atom);
        let charge = self.molecule.get_atom_charge(atom);

        if charge == 0 {
            let plain_aliphatic = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.plain() && neighbour.aliphatic_heavy())
                .count();
            let plain_aromatic = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.plain() && neighbour.aromatic)
                .count();
            let plain_heavy = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.plain() && neighbour.heavy())
                .count();
            let doubly_bonded = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.double() && neighbour.heavy())
                .count();

            if hydrogens == 2 && plain_aliphatic >= 1 {
                return N1;
            }
            if hydrogens == 1 && plain_aliphatic >= 2 {
                return N2;
            }
            if hydrogens == 2 && plain_aromatic >= 1 {
                return N3;
            }
            if hydrogens == 1 && plain_aromatic >= 1 && plain_heavy >= 2 {
                return N4;
            }
            if hydrogens == 1 && doubly_bonded >= 1 {
                return N5;
            }
            if doubly_bonded >= 1 && (plain_heavy >= 1 || doubly_bonded >= 2) {
                return N6;
            }
            if plain_aliphatic >= 3 {
                return N7;
            }
            if plain_heavy >= 3 && plain_aromatic >= 1 {
                return N8;
            }
            if self
                .neighbours(atom)
                .any(|neighbour| neighbour.triple() && neighbour.aliphatic_heavy())
            {
                return N9;
            }
            return NS;
        }

        if charge > 0 {
            if (1..=3).contains(&hydrogens) {
                return N10;
            }
            // N14 before N13 for the triple bond, which N13 would otherwise
            // never see; the published order puts N13 first, but its forms all
            // name single or double bonds and so cannot match a nitrile.
            if self
                .neighbours(atom)
                .any(|neighbour| neighbour.triple() && neighbour.aliphatic_heavy())
            {
                return N14;
            }
            let plain_aliphatic = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.plain() && neighbour.aliphatic_heavy())
                .count();
            let doubly_bonded_aliphatic = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.double() && neighbour.aliphatic_heavy())
                .count();
            let plain_heavy = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.plain() && neighbour.heavy())
                .count();
            let doubly_bonded_carbon_and_nitrogen = self
                .neighbours(atom)
                .filter(|neighbour| neighbour.double() && neighbour.atomic_number == 6)
                .count()
                >= 1
                && self
                    .neighbours(atom)
                    .filter(|neighbour| neighbour.double() && neighbour.atomic_number == 7)
                    .count()
                    >= 1;
            if plain_aliphatic >= 4
                || (doubly_bonded_aliphatic >= 1 && plain_aliphatic >= 1 && plain_heavy >= 2)
                || doubly_bonded_carbon_and_nitrogen
            {
                return N13;
            }
            // N14's last form: a diazonium or an azide's middle nitrogen.
            if self.neighbours(atom).any(|neighbour| {
                neighbour.double()
                    && neighbour.atomic_number == 7
                    && self.molecule.get_atom_charge(neighbour.atom) < 0
            }) && self
                .neighbours(atom)
                .filter(|neighbour| neighbour.double() && neighbour.atomic_number == 7)
                .count()
                >= 2
            {
                return N14;
            }
            return NS;
        }

        N14
    }

    // ---------------------------------------------------------------- oxygen

    fn aliphatic_oxygen(&self, atom: usize) -> CrippenType {
        use CrippenType::*;
        let hydrogens = self.hydrogens(atom);
        let charge = self.molecule.get_atom_charge(atom);
        let connections = self.connections(atom);

        if (1..=2).contains(&hydrogens) {
            return O2;
        }

        let plain_aliphatic = self
            .neighbours(atom)
            .filter(|neighbour| neighbour.plain() && neighbour.aliphatic_heavy())
            .count();
        if plain_aliphatic >= 2 {
            return O3;
        }
        let plain_aromatic = self
            .neighbours(atom)
            .filter(|neighbour| neighbour.plain() && neighbour.aromatic)
            .count();
        let plain_heavy = self
            .neighbours(atom)
            .filter(|neighbour| neighbour.plain() && neighbour.heavy())
            .count();
        if plain_aromatic >= 1 && plain_heavy >= 2 {
            return O4;
        }

        // O5 and O6: an oxygen given away to a nitrogen or a sulfur, written
        // either as a double bond or as a charge.
        if self
            .neighbours(atom)
            .any(|neighbour| neighbour.double() && matches!(neighbour.atomic_number, 7 | 8))
        {
            return O5;
        }
        if connections == 1 && charge < 0 {
            let attached = self.neighbours(atom).next();
            if let Some(neighbour) = attached {
                if neighbour.atomic_number == 7 {
                    return O5;
                }
                if neighbour.atomic_number == 16 {
                    return O6;
                }
            }
        }
        if charge == 0
            && self.neighbours(atom).any(|neighbour| {
                neighbour.double()
                    && neighbour.atomic_number == 16
                    && self.molecule.get_atom_charge(neighbour.atom) == 0
            })
        {
            return O6;
        }

        // O12 before O7: a carboxylate oxygen would otherwise be
        // called an alkoxide.
        if charge == -1
            && self.neighbours(atom).any(|neighbour| {
                neighbour.plain()
                    && neighbour.aliphatic(6)
                    && self
                        .neighbours(neighbour.atom)
                        .any(|further| further.double() && further.aliphatic(8))
            })
        {
            return O12;
        }
        if connections == 1 && charge < 0 {
            if let Some(neighbour) = self.neighbours(atom).next() {
                if neighbour.atomic_number != 1
                    && !neighbour.aliphatic(7)
                    && !neighbour.aliphatic(16)
                {
                    return O7;
                }
            }
        }

        // O8 to O11: a carbonyl oxygen, by what the carbon carries.
        let carbonyl = self
            .neighbours(atom)
            .find(|neighbour| neighbour.double() && neighbour.atomic_number == 6);
        if let Some(carbon) = carbonyl {
            if carbon.aromatic {
                return O8;
            }
            let carbon_hydrogens = self.hydrogens(carbon.atom);
            let mut aliphatic_carbons = 0;
            let mut aromatic_carbons = 0;
            let mut aliphatic_heavy = 0;
            let mut aromatic_heavy = 0;
            let mut aliphatic_nitrogen_or_oxygen = 0;
            let mut heavy_not_carbon = 0;
            let mut doubly_bonded_oxygens = 0;
            for beyond in self.neighbours(carbon.atom) {
                if beyond.atom == atom {
                    continue;
                }
                if beyond.double() && beyond.atomic_number == 8 {
                    doubly_bonded_oxygens += 1;
                }
                if !beyond.heavy() {
                    continue;
                }
                if beyond.aromatic {
                    aromatic_heavy += 1;
                    if beyond.atomic_number == 6 {
                        aromatic_carbons += 1;
                    }
                } else {
                    aliphatic_heavy += 1;
                    if beyond.atomic_number == 6 {
                        aliphatic_carbons += 1;
                    }
                    if matches!(beyond.atomic_number, 7 | 8) {
                        aliphatic_nitrogen_or_oxygen += 1;
                    }
                }
                if beyond.atomic_number != 6 {
                    heavy_not_carbon += 1;
                }
            }

            // O9: an aldehyde, a ketone, an amide or a carbonate, all of them
            // with nothing aromatic attached.
            if (carbon_hydrogens == 1 && aliphatic_carbons >= 1)
                || (aliphatic_carbons >= 1 && aliphatic_heavy >= 2)
                || (carbon_hydrogens == 1 && aliphatic_nitrogen_or_oxygen >= 1)
                || carbon_hydrogens == 2
                || (self.connections(carbon.atom) == 2 && doubly_bonded_oxygens >= 1)
            {
                return O9;
            }
            // O10: the same, conjugated into a ring.
            if (carbon_hydrogens == 1 && aromatic_carbons >= 1)
                || ((aliphatic_carbons + aromatic_carbons) >= 1 && aromatic_heavy >= 1)
                || (aromatic_carbons >= 1 && aliphatic_heavy >= 1)
            {
                return O10;
            }
            // O11: a carbonyl with no carbon on it at all -- a urea, a
            // carbonate, a carbamate.
            if heavy_not_carbon >= 2 {
                return O11;
            }
        }

        OS
    }

    // ---------------------------------------------------------------- others

    fn aliphatic_sulfur(&self, atom: usize) -> CrippenType {
        let charge = self.molecule.get_atom_charge(atom);
        if matches!(charge, -4..=-1 | 1..=3 | 5 | 6) {
            return CrippenType::S2;
        }
        if charge == 0
            && self.neighbours(atom).any(|neighbour| {
                neighbour.double()
                    && matches!(neighbour.atomic_number, 7 | 8 | 15 | 16)
                    && !neighbour.aromatic
            })
        {
            return CrippenType::S2;
        }
        CrippenType::S1
    }

    fn other_element(&self, atom: usize, atomic_number: u8) -> CrippenType {
        use CrippenType::*;
        let charge = self.molecule.get_atom_charge(atom);
        match atomic_number {
            9 | 17 | 35 | 53 if charge == 0 => match atomic_number {
                9 => F,
                17 => Cl,
                35 => Br,
                _ => I,
            },
            9 | 17 | 35 | 53 if charge < 0 => Hal,
            53 if charge > 0 => Hal,
            element if charge > 0 && ALKALI_METALS.contains(&element) => Hal,
            element if ME1_ELEMENTS.contains(&element) => Me1,
            34 | 52 | 84 => Me1,
            21..=30 | 39..=48 | 72..=80 => Me2,
            _ => None,
        }
    }
}
