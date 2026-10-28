//! Applying reaction SMARTS to molecules.
//!
//! A reaction SMARTS is two patterns separated by `>>`. Applying it means
//! finding where the left-hand side occurs in a molecule and rewriting that
//! part into the right-hand side. What connects the two sides is the *atom
//! map*: `[C:1]` on the left and `[C:1]` on the right name the same atom, so
//! the difference between the two templates says exactly which bonds to make,
//! break or change.
//!
//! # What is supported
//!
//! Every atom on both sides must carry a map index, and the indices must be
//! unique within each side. That restriction is what makes the rewrite
//! unambiguous: with it, each atom of the match has a known fate, so there is
//! no need to guess how an unmapped template atom relates to the molecule.
//!
//! Within that:
//!
//! - **Bonds change.** A bond present on one side and absent on the other is
//!   made or broken; a bond present on both with a different order is changed.
//! - **Atoms disappear** when their map index is on the left only, taking
//!   their bonds with them.
//! - **Atoms appear** when their map index is on the right only. The product
//!   template has to name a concrete element for them, since there is nothing
//!   in the molecule to copy.
//! - **Elements and charges change** when the product template states a
//!   different one.
//!
//! Hydrogens are re-derived afterwards for every atom whose bonding changed,
//! because a template that turns `C-O` into `C=O` says nothing about the two
//! hydrogens that have to go, and a product with the wrong hydrogen count is
//! not a product.
//!
//! That is RDKit's reading, and the default. [`ReactionSmarts::parse_dpo`]
//! reads the same SMARTS as a double-pushout rule instead, where nothing
//! happens that the rule does not write — see [`Semantics::Dpo`]. Such rules
//! can be inverted, and read from or written to MØD's GML
//! ([`ReactionSmarts::from_gml`], [`ReactionSmarts::to_gml`]).
//!
//! # Example
//!
//! ```rust
//! # use oxidane::prelude::*;
//! use oxidane::molecule::reaction::ReactionSmarts;
//!
//! // Oxidise a primary alcohol to an aldehyde.
//! let reaction = ReactionSmarts::parse("[C:1]-[O:2]>>[C:1]=[O:2]").unwrap();
//! let products = reaction.apply(&MolecularGraph::from_smiles_single("CCO").unwrap());
//! assert!(products.iter().any(|p| p.to_smiles() == "CC=O"));
//! ```

use std::collections::HashMap;

use crate::atom::Atom;
use crate::io::{
    AtomExpression, BondExpression, CxSMARTSError, CxSMARTSPattern, LogicalOperator,
    SmilesParseOptions, find_smarts_matches, parse_cxsmarts,
};
use crate::molecule::aromaticity::MoleculeAromaticity;
use crate::molecule::base::{MoleculeCore, charge_valency_shift, saturating_valency};
use crate::molecule::bond::BondOrder;
use crate::molecule::chirality::{ChiralClass, index_sorted_neighbors, tetrahedral_tag_for_order};
use crate::molecule::molecular_graph::MolecularGraph;

/// Helper returning standard chemical element symbol from atomic number.
pub fn element_symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        18 => "Ar",
        19 => "K",
        20 => "Ca",
        35 => "Br",
        53 => "I",
        _ => "X",
    }
}

/// Details on why a reaction SMARTS violates mass conservation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MassConservationError {
    /// An atom on the product side has no corresponding reactant atom (mass created from nothing).
    AtomCreated {
        map_index: usize,
        element: Option<u8>,
    },
    /// An atom on the reactant side has no corresponding product atom (mass destroyed).
    AtomDestroyed {
        map_index: usize,
        element: Option<u8>,
    },
    /// An atom changes its chemical element between reactants and products (element transmutation).
    ElementTransmutation {
        map_index: usize,
        reactant_element: u8,
        product_element: u8,
    },
}

impl std::fmt::Display for MassConservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MassConservationError::AtomCreated { map_index, element } => {
                let elem_str = element.map(element_symbol).unwrap_or("unknown");
                write!(
                    f,
                    "Product atom :{} (element {}) has no corresponding reactant atom (mass created from nothing)",
                    map_index, elem_str
                )
            }
            MassConservationError::AtomDestroyed { map_index, element } => {
                let elem_str = element.map(element_symbol).unwrap_or("unknown");
                write!(
                    f,
                    "Reactant atom :{} (element {}) has no corresponding product atom (mass destroyed)",
                    map_index, elem_str
                )
            }
            MassConservationError::ElementTransmutation {
                map_index,
                reactant_element,
                product_element,
            } => {
                write!(
                    f,
                    "Atom :{} transmutes element from Z={} ({}) to Z={} ({})",
                    map_index,
                    reactant_element,
                    element_symbol(*reactant_element),
                    product_element,
                    element_symbol(*product_element)
                )
            }
        }
    }
}

impl std::error::Error for MassConservationError {}

/// Why a reaction SMARTS could not be used.
#[derive(Debug, Clone, PartialEq)]
pub enum ReactionError {
    /// The SMARTS itself did not parse.
    Smarts(CxSMARTSError),
    /// No `>>`, so this is a plain pattern rather than a reaction.
    NotAReaction,
    /// One side of the `>>` is empty.
    EmptySide(&'static str),
    /// An atom carries no map index. Every atom on both sides must have one —
    /// see the module docs.
    UnmappedAtom { side: &'static str, atom: usize },
    /// The same map index is used twice on one side, so it names no single
    /// atom.
    DuplicateMapIndex { side: &'static str, index: usize },
    /// An atom that only the product side mentions has to say which element it
    /// is; there is nothing in the molecule to copy it from.
    NewAtomWithoutElement { index: usize },
    /// The reaction violates mass conservation.
    MassNotConserved(MassConservationError),
    /// Under [`Semantics::Dpo`], a product label that tests rather than
    /// assigns. The product side says what an atom or bond *becomes*, so
    /// `[C,N:1]`, `[CH2:1]` or `-,=` there has no meaning.
    ProductQuery { index: usize, what: String },
    /// Under [`Semantics::Dpo`], an atom whose bonds the rule changes by an
    /// amount no pair of its element's normal valences differs by. With every
    /// hydrogen an explicit atom, nothing outside the rule makes up the
    /// difference, so every product would be rejected.
    ValenceNotBalanced { index: usize, change: i8 },
    /// Only a [`Semantics::Dpo`] rule has an inverse: RDKit's reading is not
    /// symmetric in its two sides.
    NotInvertible,
    /// An application condition names a map index the reactant side does
    /// not, so there is no matched atom to pin it to.
    UnknownConditionIndex { index: usize },
    /// A rule with application conditions has no inverse here: a condition
    /// on L would have to be carried across the rule onto R.
    ConditionsNotInvertible,
}

impl std::fmt::Display for ReactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReactionError::Smarts(err) => write!(f, "SMARTS parse error: {}", err),
            ReactionError::NotAReaction => write!(f, "Not a reaction (missing '>>')"),
            ReactionError::EmptySide(side) => write!(f, "Empty side: {}", side),
            ReactionError::UnmappedAtom { side, atom } => {
                write!(f, "Unmapped atom on {} side at index {}", side, atom)
            }
            ReactionError::DuplicateMapIndex { side, index } => {
                write!(f, "Duplicate map index :{} on {} side", index, side)
            }
            ReactionError::NewAtomWithoutElement { index } => {
                write!(f, "New atom with map index :{} without element", index)
            }
            ReactionError::MassNotConserved(err) => {
                write!(f, "Mass conservation violation: {}", err)
            }
            ReactionError::ProductQuery { index, what } => {
                write!(
                    f,
                    "Product label :{} tests rather than assigns: {}",
                    index, what
                )
            }
            ReactionError::ValenceNotBalanced { index, change } => write!(
                f,
                "Atom :{} changes its bonding by {} with nothing to balance it",
                index, change
            ),
            ReactionError::NotInvertible => {
                write!(f, "Only a double-pushout rule has an inverse")
            }
            ReactionError::UnknownConditionIndex { index } => {
                write!(
                    f,
                    "Condition names :{} which the reactant side does not",
                    index
                )
            }
            ReactionError::ConditionsNotInvertible => {
                write!(f, "A rule with application conditions has no inverse here")
            }
        }
    }
}

impl std::error::Error for ReactionError {}

impl From<CxSMARTSError> for ReactionError {
    fn from(error: CxSMARTSError) -> Self {
        ReactionError::Smarts(error)
    }
}

impl From<MassConservationError> for ReactionError {
    fn from(error: MassConservationError) -> Self {
        ReactionError::MassNotConserved(error)
    }
}

/// One side of a reaction, indexed by atom map.
#[derive(Clone)]
struct Side {
    pattern: CxSMARTSPattern,
    /// map index -> pattern atom
    by_map_index: HashMap<usize, usize>,
    /// pattern atom -> map index
    map_index_of: Vec<usize>,
}

impl Side {
    /// Indexes one side by map index.
    ///
    /// With `next_free` given, an unmapped atom is numbered from it rather
    /// than rejected. The caller starts it above every index either side
    /// writes and shares it between the sides, so an atom numbered here never
    /// pairs with an atom on the other side: an unmapped reactant atom is
    /// consumed and an unmapped product atom is created, as in RDKit.
    fn build(
        mut pattern: CxSMARTSPattern,
        side: &'static str,
        mut next_free: Option<&mut usize>,
    ) -> Result<Self, ReactionError> {
        if pattern.atoms.is_empty() {
            return Err(ReactionError::EmptySide(side));
        }
        let mut by_map_index = HashMap::new();
        let mut map_index_of = vec![0usize; pattern.atoms.len()];

        for atom in 0..pattern.atoms.len() {
            let index = match (pattern.atom_classes.get(&atom), next_free.as_deref_mut()) {
                (Some(&index), _) => index,
                (None, Some(next_free)) => {
                    let index = *next_free;
                    *next_free += 1;
                    pattern.atom_classes.insert(atom, index);
                    index
                }
                (None, None) => return Err(ReactionError::UnmappedAtom { side, atom }),
            };
            if by_map_index.insert(index, atom).is_some() {
                return Err(ReactionError::DuplicateMapIndex { side, index });
            }
            map_index_of[atom] = index;
        }
        Ok(Side {
            pattern,
            by_map_index,
            map_index_of,
        })
    }

    /// The bond this side writes between two map indices, if it joins them
    /// at all.
    fn bond_between(&self, first_index: usize, second_index: usize) -> Option<&BondExpression> {
        let (Some(&first), Some(&second)) = (
            self.by_map_index.get(&first_index),
            self.by_map_index.get(&second_index),
        ) else {
            return None;
        };
        self.pattern
            .bonds
            .iter()
            .find(|&&(a, b, _)| (a == first && b == second) || (a == second && b == first))
            .map(|(_, _, expression)| expression)
    }
}

/// The order a bond expression names outright, if it names one.
///
/// `-`, `=`, `#` and `:` name an order, as does a conjunction with one of them
/// in it (`=;@`). Anything that only tests — `~`, `!-`, `-,=`, the unwritten
/// single-or-aromatic bond — names none. A direction (`/`, `\`) is a single
/// bond that also says which way it points.
pub(crate) fn simple_order(expression: &BondExpression) -> Option<BondOrder> {
    match expression {
        BondExpression::Single
        | BondExpression::DirectionalUp
        | BondExpression::DirectionalDown => Some(BondOrder::Single),
        BondExpression::Double => Some(BondOrder::Double),
        BondExpression::Triple => Some(BondOrder::Triple),
        BondExpression::Aromatic => Some(BondOrder::Aromatic),
        BondExpression::Logical {
            operator: LogicalOperator::HighAnd | LogicalOperator::LowAnd,
            operands,
        } => operands.iter().find_map(simple_order),
        _ => None,
    }
}

/// The order a bond both templates write ends up with, or `None` to leave the
/// molecule's bond as it is.
///
/// This is RDKit's reading. A product bond that names an order sets it. One
/// that only tests (`~`, `-,=`) keeps what was matched. An unwritten product
/// bond keeps the matched bond too, but only when the reactant side left it
/// unwritten as well: `[c:1][c:2]>>[c:1][c:2]` leaves benzene aromatic, while
/// `[C:1]=[C:2]>>[C:1][C:2]` writes the double bond on one side and not the
/// other, and that difference is the reaction asking for a single bond.
pub(crate) fn product_order(before: &BondExpression, after: &BondExpression) -> Option<BondOrder> {
    if let Some(order) = simple_order(after) {
        return Some(order);
    }
    let unwritten = |expression: &BondExpression| *expression == BondExpression::SingleOrAromatic;
    (unwritten(after) && !unwritten(before)).then_some(BondOrder::Single)
}

/// The first value `pick` finds among the primitives a template atom asserts
/// outright, those reached through conjunctions alone.
///
/// A primitive under `!` or `,` is a test the atom may fail, or pass some
/// other way, so it states nothing a product could copy: `[!C:1]>>[!C:1]` and
/// `[C,N:1]>>[C,N:1]` keep whatever element was matched.
fn asserted<T>(
    expression: &AtomExpression,
    pick: &impl Fn(&AtomExpression) -> Option<T>,
) -> Option<T> {
    match expression {
        AtomExpression::Logical {
            operator: LogicalOperator::HighAnd | LogicalOperator::LowAnd,
            operands,
        } => operands.iter().find_map(|operand| asserted(operand, pick)),
        _ => pick(expression),
    }
}

/// The element a template atom names outright, if it names one.
pub(crate) fn element_of(expression: &AtomExpression) -> Option<u8> {
    asserted(expression, &|primitive| match primitive {
        AtomExpression::AtomicNumber(atomic_number) => Some(*atomic_number),
        AtomExpression::Element(symbol) => {
            use chemistry_consts::ElementProperties;
            symbol.as_str().atomic_number()
        }
        _ => None,
    })
}

/// The formal charge a template atom states outright, if it states one.
pub(crate) fn charge_of(expression: &AtomExpression) -> Option<i8> {
    asserted(expression, &|primitive| match primitive {
        AtomExpression::Charge(charge) => Some(*charge),
        _ => None,
    })
}

/// The isotope a template atom states outright, if it states one.
pub(crate) fn isotope_of(expression: &AtomExpression) -> Option<u16> {
    asserted(expression, &|primitive| match primitive {
        AtomExpression::Isotope(isotope) => Some(*isotope),
        _ => None,
    })
}

/// The SMARTS (and GML) symbol for a bond order.
pub(crate) fn bond_order_symbol(order: BondOrder) -> &'static str {
    match order {
        BondOrder::Double => "=",
        BondOrder::Triple => "#",
        BondOrder::Aromatic => ":",
        _ => "-",
    }
}

/// The tetrahedral tag a template atom writes outright: 1 for `@`, 2 for `@@`.
fn tetrahedral_of(expression: &AtomExpression) -> Option<u8> {
    asserted(expression, &|primitive| match primitive {
        AtomExpression::Chirality(ChiralClass::TH(tag @ 1..=2)) => Some(*tag),
        AtomExpression::Chirality(ChiralClass::Counterclockwise) => Some(1),
        AtomExpression::Chirality(ChiralClass::Clockwise) => Some(2),
        _ => None,
    })
}

/// An application condition: a pattern around the match that must (or must
/// not) be found.
#[derive(Clone)]
struct Condition {
    pattern: CxSMARTSPattern,
    /// condition atom -> the reactant map index it is pinned to, if any
    map_index_of: Vec<Option<usize>>,
    negative: bool,
}

/// A pattern that matches what `pattern` does in any Kekulé structure of a
/// molecule held in aromatic form: an element written in capitals matches
/// that element whether aromatic or not, and a `-` or `=` matches an aromatic
/// bond too. Whether the bonds such a match takes for single and double fit
/// one Kekulé structure is left to [`ReactionSmarts::kekule_for`].
///
/// Lower-case symbols and `a`/`A` still test aromaticity; a negation is left
/// as written, since widening what it excludes would narrow what it matches.
fn kekule_tolerant(pattern: &CxSMARTSPattern) -> CxSMARTSPattern {
    fn atom(expression: &AtomExpression) -> AtomExpression {
        match expression {
            AtomExpression::Element(symbol) if symbol.starts_with(char::is_uppercase) => {
                use chemistry_consts::ElementProperties;
                match symbol.as_str().atomic_number() {
                    Some(atomic_number) => AtomExpression::AtomicNumber(atomic_number),
                    None => expression.clone(),
                }
            }
            AtomExpression::Logical {
                operator:
                    operator
                    @ (LogicalOperator::HighAnd | LogicalOperator::LowAnd | LogicalOperator::Or),
                operands,
            } => AtomExpression::Logical {
                operator: operator.clone(),
                operands: operands.iter().map(atom).collect(),
            },
            other => other.clone(),
        }
    }
    fn bond(expression: &BondExpression) -> BondExpression {
        match expression {
            BondExpression::Single | BondExpression::Double => BondExpression::Logical {
                operator: LogicalOperator::Or,
                operands: vec![expression.clone(), BondExpression::Aromatic],
            },
            BondExpression::Logical {
                operator: operator @ (LogicalOperator::HighAnd | LogicalOperator::LowAnd),
                operands,
            } => BondExpression::Logical {
                operator: operator.clone(),
                operands: operands.iter().map(bond).collect(),
            },
            other => other.clone(),
        }
    }
    let mut tolerant = pattern.clone();
    tolerant.atoms = pattern.atoms.iter().map(atom).collect();
    for (_, _, expression) in &mut tolerant.bonds {
        *expression = bond(expression);
    }
    tolerant
}

/// A pattern that matches what `pattern` does with its stereo ignored:
/// every chirality test true, every directional bond a plain single bond.
fn without_stereo(pattern: &CxSMARTSPattern) -> CxSMARTSPattern {
    fn atom(expression: &AtomExpression) -> AtomExpression {
        match expression {
            AtomExpression::Chirality(_) => AtomExpression::Wildcard,
            // Negating a stereo test negates nothing once the test is gone.
            AtomExpression::Logical {
                operator: LogicalOperator::Not,
                operands,
            } if operands.iter().any(|operand| atom(operand) != *operand) => {
                AtomExpression::Wildcard
            }
            AtomExpression::Logical { operator, operands } => AtomExpression::Logical {
                operator: operator.clone(),
                operands: operands.iter().map(atom).collect(),
            },
            other => other.clone(),
        }
    }
    fn bond(expression: &BondExpression) -> BondExpression {
        match expression {
            BondExpression::DirectionalUp
            | BondExpression::DirectionalDown
            | BondExpression::DirectionalUpOrUnspec
            | BondExpression::DirectionalDownOrUnspec => BondExpression::Single,
            BondExpression::Not(inner) => BondExpression::Not(Box::new(bond(inner))),
            BondExpression::Logical { operator, operands } => BondExpression::Logical {
                operator: operator.clone(),
                operands: operands.iter().map(bond).collect(),
            },
            other => other.clone(),
        }
    }
    let mut stripped = pattern.clone();
    stripped.atoms = pattern.atoms.iter().map(atom).collect();
    for (_, _, expression) in &mut stripped.bonds {
        *expression = bond(expression);
    }
    stripped
}

/// Whether a template atom states any stereo.
pub(crate) fn states_stereo(expression: &AtomExpression) -> bool {
    asserted(expression, &|primitive| {
        matches!(primitive, AtomExpression::Chirality(_)).then_some(())
    })
    .is_some()
}

/// Makes the hydrogen change a stereocentre's new bonding calls for, so the
/// centre's neighbours change by one swap rather than by two steps:
///
/// - It gained a neighbour and now has a bond too many: its one hydrogen the
///   rule does not map is detached, and returned for removal.
/// - It lost a neighbour and is a bond short: a hydrogen is attached.
fn settle_hydrogen(
    product: &mut MolecularGraph,
    centre: usize,
    before: &[usize],
    after: &[usize],
    mapped: &[usize],
) -> Option<usize> {
    let gained = after.iter().filter(|n| !before.contains(n)).count();
    let lost = before.iter().filter(|n| !after.contains(n)).count();
    let surplus = product.get_atom_degree(centre)?;
    match (gained, lost) {
        (1, 0) if surplus > 0 => {
            let hydrogen = before.iter().copied().rev().find(|&neighbor| {
                product.atomic_numbers()[neighbor] == 1 && !mapped.contains(&neighbor)
            })?;
            remove_bond(product, centre, hydrogen);
            Some(hydrogen)
        }
        (0, 1) if surplus < 0 => {
            let hydrogen = product.atomic_numbers().len();
            product.add_atom(Atom::new(1));
            product.add_bond(centre, hydrogen, BondOrder::Single);
            None
        }
        _ => None,
    }
}

/// A centre's old neighbours with the one the rule removed replaced by the one
/// it added, so the newcomer stands where its predecessor stood — if exactly
/// one was swapped for exactly one.
fn replaced_in_place(before: &[usize], after: &[usize]) -> Option<Vec<usize>> {
    let lost: Vec<usize> = before
        .iter()
        .copied()
        .filter(|n| !after.contains(n))
        .collect();
    let gained: Vec<usize> = after
        .iter()
        .copied()
        .filter(|n| !before.contains(n))
        .collect();
    let ([lost], [gained]) = (lost.as_slice(), gained.as_slice()) else {
        return None;
    };
    Some(
        before
            .iter()
            .map(|&neighbor| if neighbor == *lost { *gained } else { neighbor })
            .collect(),
    )
}

/// The order of the bond between two atoms, if they are bonded.
fn bond_order_in(molecule: &MolecularGraph, first: usize, second: usize) -> Option<BondOrder> {
    molecule.atom_bonds()[first]
        .iter()
        .find(|bond| bond.target() == second)
        .map(|bond| bond.bond_order())
}

/// A parsed, validated reaction SMARTS.
pub struct ReactionSmarts {
    reactants: Side,
    products: Side,
    semantics: Semantics,
    /// Application conditions, checked against each match before it is
    /// rewritten — see [`ReactionSmarts::forbid`].
    conditions: Vec<Condition>,
    /// The reactant pattern as matched. Under [`Semantics::Rdkit`] it is the
    /// reactant pattern without its stereo, which RDKit's `RunReactants`
    /// does not check by default: the tags there only say what to do with a
    /// centre (see [`ReactionSmarts::update_stereo`]). Under
    /// [`Semantics::Dpo`] a reactant label is a predicate like any other, and
    /// the pattern is matched as written.
    query: CxSMARTSPattern,
}

/// How a reaction's two templates are read when it is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Semantics {
    /// RDKit's `RunReactants`. A consumed atom takes all its bonds and
    /// hydrogens with it, and the hydrogens of every atom whose bonding changed
    /// are re-derived afterwards, so a template need say nothing about them.
    #[default]
    Rdkit,
    /// The double pushout. The reactant side is L, the product side R, and the
    /// atoms both sides map are the interface K. Nothing happens that the rule
    /// does not write:
    ///
    /// - A match whose consumed atom has a bond L does not account for is
    ///   rejected rather than having that bond cut (the *dangling condition*).
    ///   Hydrogens are explicit atoms here, so a rule that consumes an atom
    ///   has to consume its hydrogens too.
    /// - No hydrogen is added or removed; a product with an atom off its
    ///   element's valences is rejected instead.
    /// - The product side has to assign rather than test, which
    ///   [`ReactionSmarts::parse_dpo`] checks along with the valence balance
    ///   of every atom the rule keeps.
    ///
    /// A rule may be written against a molecule's Kekulé structure: `=` and
    /// `-` match aromatic bonds wherever some Kekulé structure agrees with
    /// every such bond the match uses, and the rewrite happens in that
    /// structure. Products are then brought back to the form a SMILES parse
    /// gives — Kekulé structure checked, aromaticity perceived — so the next
    /// rule matches them as it matches parsed molecules. For the same reason
    /// an element written in capitals matches aromatic atoms of that element
    /// too; lower case, `a` and `A` still ask about aromaticity.
    Dpo,
}

/// Whether an atom of this element can go from one of its valences to
/// another by changing its bonding by `change`, its charge going from
/// `before` to `after` meanwhile.
fn valences_allow(atomic_number: u8, before: i8, after: i8, change: i8) -> bool {
    use chemistry_consts::ElementProperties;
    let Some(valences) = atomic_number.valencies() else {
        // Nothing tabulated to hold the atom to.
        return true;
    };
    let valences: Vec<i8> = valences.filter(|valence| *valence > 0).collect();
    if valences.is_empty() {
        return true;
    }
    let shift_before = charge_valency_shift(atomic_number, before);
    let shift_after = charge_valency_shift(atomic_number, after);
    valences.iter().any(|&from| {
        valences
            .iter()
            .any(|&to| (to + shift_after) - (from + shift_before) == change)
    })
}

/// Bond orders in half-units, the measure under which an aromatic bond is a
/// whole number.
fn half_units(order: BondOrder) -> i8 {
    match order {
        BondOrder::Double => 4,
        BondOrder::Triple => 6,
        BondOrder::Quadruple => 8,
        BondOrder::Aromatic => 3,
        _ => 2,
    }
}

/// Why a product label is not an assignment, if it is not one.
///
/// An element, a charge, an isotope and the map class are values an atom can
/// be given; `*` gives nothing and so keeps what was matched. Everything else
/// — alternatives, negations, hydrogen counts, ring and degree tests,
/// recursion — asks a question.
fn query_in_atom(expression: &AtomExpression) -> Option<String> {
    match expression {
        AtomExpression::Element(_)
        | AtomExpression::AtomicNumber(_)
        | AtomExpression::Charge(_)
        | AtomExpression::Isotope(_)
        | AtomExpression::AtomClass(_)
        | AtomExpression::Chirality(_)
        | AtomExpression::Wildcard => None,
        AtomExpression::Logical {
            operator: LogicalOperator::HighAnd | LogicalOperator::LowAnd,
            operands,
        } => operands.iter().find_map(query_in_atom),
        other => Some(format!("{other:?}")),
    }
}

/// Why a product bond is not an assignment, if it is not one. `~` and the
/// unwritten bond keep what was matched, so they are allowed on a bond both
/// sides write; a bond only the product side writes needs an order.
fn query_in_bond(expression: &BondExpression, created: bool) -> Option<String> {
    if simple_order(expression).is_some() && !matches!(expression, BondExpression::Logical { .. }) {
        return None;
    }
    match expression {
        BondExpression::SingleOrAromatic => None,
        BondExpression::Any if !created => None,
        other => Some(format!("{other:?}")),
    }
}

impl ReactionSmarts {
    /// Parses a reaction SMARTS to be applied as a double-pushout rule — see
    /// [`Semantics::Dpo`].
    ///
    /// On top of what [`ReactionSmarts::parse`] checks, every product label
    /// has to be an assignment, and every atom both sides keep has to change
    /// its bonding by an amount its element's valences allow. That second
    /// check is where most mistakes surface: `[C:1]-[O:2]>>[C:1]=[O:2]` is
    /// rejected, because with explicit hydrogens the carbon and oxygen each
    /// gain a bond and nothing gives one up. The balanced rule names the two
    /// hydrogens that leave.
    ///
    /// ```rust
    /// # use oxidane::prelude::*;
    /// use oxidane::molecule::reaction::{ReactionError, ReactionSmarts};
    ///
    /// assert!(matches!(
    ///     ReactionSmarts::parse_dpo("[C:1]-[O:2]>>[C:1]=[O:2]"),
    ///     Err(ReactionError::ValenceNotBalanced { .. })
    /// ));
    /// let oxidation = ReactionSmarts::parse_dpo(
    ///     "[C:1](-[#1:3])-[O:2]-[#1:4]>>[C:1]=[O:2].[#1:3]-[#1:4]",
    /// )
    /// .unwrap();
    /// let products = oxidation.apply(&MolecularGraph::from_smiles_single("CCO").unwrap());
    /// // Acetaldehyde, and the two hydrogens as H2.
    /// assert!(!products.is_empty());
    /// assert!(products.iter().all(|p| {
    ///     p.contains_smarts("[CH3][CH1]=O").unwrap() && p.contains_smarts("[#1]-[#1]").unwrap()
    /// }));
    /// ```
    pub fn parse_dpo(smarts: &str) -> Result<Self, ReactionError> {
        let mut reaction = Self::parse_opts(smarts, false)?;
        reaction.semantics = Semantics::Dpo;
        reaction.query = kekule_tolerant(&reaction.reactants.pattern);
        reaction.check_dpo_typing()?;
        Ok(reaction)
    }

    /// How this reaction is applied.
    pub fn semantics(&self) -> Semantics {
        self.semantics
    }

    /// Whether the rule carries application conditions.
    pub(crate) fn has_conditions(&self) -> bool {
        !self.conditions.is_empty()
    }

    /// The rule run backwards: L and R swapped, K unchanged.
    ///
    /// A double-pushout derivation is symmetric, so the inverse undoes any
    /// derivation of the rule. The reactant labels become product labels,
    /// so they have to be assignments too, and the inverse is checked as
    /// [`ReactionSmarts::parse_dpo`] checks any rule.
    pub fn inverse(&self) -> Result<Self, ReactionError> {
        if self.semantics != Semantics::Dpo {
            return Err(ReactionError::NotInvertible);
        }
        if !self.conditions.is_empty() {
            return Err(ReactionError::ConditionsNotInvertible);
        }
        let inverse = ReactionSmarts {
            query: kekule_tolerant(&self.products.pattern),
            conditions: Vec::new(),
            reactants: self.products.clone(),
            products: self.reactants.clone(),
            semantics: Semantics::Dpo,
        };
        for (&index, &atom) in &inverse.products.by_map_index {
            if !inverse.reactants.by_map_index.contains_key(&index)
                && element_of(&inverse.products.pattern.atoms[atom]).is_none()
            {
                return Err(ReactionError::NewAtomWithoutElement { index });
            }
        }
        inverse.check_dpo_typing()?;
        Ok(inverse)
    }

    /// The reactant pattern, product pattern, and their map-index lookups,
    /// for writers that need the rule as data.
    pub(crate) fn sides(
        &self,
    ) -> (
        &CxSMARTSPattern,
        &HashMap<usize, usize>,
        &CxSMARTSPattern,
        &HashMap<usize, usize>,
    ) {
        (
            &self.reactants.pattern,
            &self.reactants.by_map_index,
            &self.products.pattern,
            &self.products.by_map_index,
        )
    }

    /// The parse-time half of [`Semantics::Dpo`]: product labels assign, and
    /// every kept atom's bonding changes by an amount its valences allow.
    fn check_dpo_typing(&self) -> Result<(), ReactionError> {
        for (&index, &atom) in &self.products.by_map_index {
            if let Some(what) = query_in_atom(&self.products.pattern.atoms[atom]) {
                return Err(ReactionError::ProductQuery { index, what });
            }
        }
        let indices = self.map_indices();
        for (position, &first) in indices.iter().enumerate() {
            for &second in &indices[position + 1..] {
                let before = self.reactants.bond_between(first, second);
                let Some(after) = self.products.bond_between(first, second) else {
                    continue;
                };
                if let Some(what) = query_in_bond(after, before.is_none()) {
                    return Err(ReactionError::ProductQuery { index: first, what });
                }
            }
        }

        for &index in &indices {
            let (Some(&left), Some(&right)) = (
                self.reactants.by_map_index.get(&index),
                self.products.by_map_index.get(&index),
            ) else {
                continue;
            };
            let left_label = &self.reactants.pattern.atoms[left];
            let right_label = &self.products.pattern.atoms[right];
            let Some(atomic_number) = element_of(right_label).or(element_of(left_label)) else {
                continue;
            };
            // A charge only the product states is a change from an unknown
            // starting point; the check after the rewrite has to judge it.
            let left_charge = charge_of(left_label);
            let right_charge = charge_of(right_label);
            let (left_charge, right_charge) = match (left_charge, right_charge) {
                (Some(left), Some(right)) => (left, right),
                (Some(left), None) => (left, left),
                (None, None) => (0, 0),
                (None, Some(_)) => continue,
            };
            let Some(change) = self.bonding_change(index, &indices) else {
                continue;
            };
            if !valences_allow(atomic_number, left_charge, right_charge, change) {
                return Err(ReactionError::ValenceNotBalanced { index, change });
            }
        }
        Ok(())
    }

    /// How many bonds' worth an atom's bonding changes by, if every bond the
    /// rule changes around it has an order on both sides to compare.
    fn bonding_change(&self, index: usize, indices: &[usize]) -> Option<i8> {
        let mut change = 0;
        for &other in indices {
            if other == index {
                continue;
            }
            let before = self.reactants.bond_between(index, other);
            let after = self.products.bond_between(index, other);
            change += match (before, after) {
                (None, None) => 0,
                (Some(before), None) => -half_units(simple_order(before)?),
                (None, Some(after)) => half_units(simple_order(after).unwrap_or(BondOrder::Single)),
                (Some(before), Some(after)) => match product_order(before, after) {
                    None => 0,
                    Some(order) => half_units(order) - half_units(simple_order(before)?),
                },
            };
        }
        // An odd number of half-units is an aromatic bond made or broken
        // without the rest of its ring, which only perception can judge.
        (change % 2 == 0).then_some(change / 2)
    }
    /// Parses and validates a reaction SMARTS.
    ///
    /// Fails if any atom on either side lacks a map index, or if an index is
    /// reused within a side — see the module docs for why that is required.
    pub fn parse(smarts: &str) -> Result<Self, ReactionError> {
        Self::parse_opts(smarts, false)
    }

    /// Parses a reaction SMARTS, automatically assigning map indices to any unmapped atoms.
    pub fn parse_with_auto_mapping(smarts: &str) -> Result<Self, ReactionError> {
        Self::parse_opts(smarts, true)
    }

    fn parse_opts(smarts: &str, auto_map: bool) -> Result<Self, ReactionError> {
        let pattern = parse_cxsmarts(smarts)?;
        if !pattern.is_reaction {
            return Err(ReactionError::NotAReaction);
        }
        let (Some(reactants), Some(products)) = (pattern.reactants, pattern.products) else {
            return Err(ReactionError::NotAReaction);
        };

        let mut next_free = reactants
            .atom_classes
            .values()
            .chain(products.atom_classes.values())
            .max()
            .map_or(1, |&highest| highest + 1);
        let reactants = Side::build(*reactants, "reactants", auto_map.then_some(&mut next_free))?;
        let products = Side::build(*products, "products", auto_map.then_some(&mut next_free))?;

        // An atom the product invents must say what it is.
        for (&index, &atom) in &products.by_map_index {
            if !reactants.by_map_index.contains_key(&index)
                && element_of(&products.pattern.atoms[atom]).is_none()
            {
                return Err(ReactionError::NewAtomWithoutElement { index });
            }
        }

        Ok(ReactionSmarts {
            semantics: Semantics::Rdkit,
            conditions: Vec::new(),
            query: without_stereo(&reactants.pattern),
            reactants,
            products,
        })
    }

    /// Checks whether this reaction SMARTS conserves mass.
    ///
    /// Mass conservation requires that:
    /// 1. Every atom on the reactant side appears in the products (no atoms destroyed).
    /// 2. Every atom on the product side appears in the reactants (no atoms created from nothing).
    /// 3. Every mapped atom has identical chemical element on both sides (no transmutation).
    pub fn check_mass_conservation(&self) -> Result<(), MassConservationError> {
        // 1. Atoms destroyed
        for (&index, &r_atom) in &self.reactants.by_map_index {
            if !self.products.by_map_index.contains_key(&index) {
                let elem = element_of(&self.reactants.pattern.atoms[r_atom]);
                return Err(MassConservationError::AtomDestroyed {
                    map_index: index,
                    element: elem,
                });
            }
        }

        // 2. Atoms created
        for (&index, &p_atom) in &self.products.by_map_index {
            if !self.reactants.by_map_index.contains_key(&index) {
                let elem = element_of(&self.products.pattern.atoms[p_atom]);
                return Err(MassConservationError::AtomCreated {
                    map_index: index,
                    element: elem,
                });
            }
        }

        // 3. Element consistency
        for (&index, &r_atom) in &self.reactants.by_map_index {
            let &p_atom = self.products.by_map_index.get(&index).unwrap();
            let r_elem = element_of(&self.reactants.pattern.atoms[r_atom]);
            let p_elem = element_of(&self.products.pattern.atoms[p_atom]);
            if let (Some(z_r), Some(z_p)) = (r_elem, p_elem) {
                if z_r != z_p {
                    return Err(MassConservationError::ElementTransmutation {
                        map_index: index,
                        reactant_element: z_r,
                        product_element: z_p,
                    });
                }
            } else if r_elem.is_none() && p_elem.is_some() {
                return Err(MassConservationError::ElementTransmutation {
                    map_index: index,
                    reactant_element: 0,
                    product_element: p_elem.unwrap(),
                });
            }
        }

        Ok(())
    }

    /// Returns true if this reaction SMARTS conserves mass.
    pub fn is_mass_conserved(&self) -> bool {
        self.check_mass_conservation().is_ok()
    }

    /// The map indices the reaction mentions, on either side.
    fn map_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .reactants
            .by_map_index
            .keys()
            .chain(self.products.by_map_index.keys())
            .copied()
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Adds a negative application condition: a match is rejected wherever
    /// `condition` can be found around it.
    ///
    /// `condition` is a SMARTS that shares map indices with the reactant side:
    /// an atom carrying one must be the atom the reactant side matched under
    /// that index, and an atom carrying none must be some other atom. That is
    /// the double-pushout reading of a negative condition — L extended to a
    /// larger graph that must not occur — and it says what a SMARTS
    /// recursion cannot: `[C:1]-[O:2]` that is not part of an ester is
    /// `forbid("[C:1](=O)-[O:2]")`, the carbonyl oxygen being something the
    /// rule itself never names.
    ///
    /// ```rust
    /// # use oxidane::prelude::*;
    /// use oxidane::molecule::reaction::ReactionSmarts;
    ///
    /// let hydrolysis = ReactionSmarts::parse("[C:1]-[O:2]>>[C:1].[O:2]")
    ///     .unwrap()
    ///     .forbid("[C:1]=O")
    ///     .unwrap();
    /// let ester = MolecularGraph::from_smiles_single("CC(=O)OC").unwrap();
    /// // Only the methyl side is cut; the acyl C-O is part of the ester.
    /// assert_eq!(hydrolysis.apply(&ester).len(), 1);
    /// ```
    pub fn forbid(self, condition: &str) -> Result<Self, ReactionError> {
        self.with_condition(condition, true)
    }

    /// Adds a positive application condition: a match is kept only where
    /// `condition` can be found around it. Read as [`ReactionSmarts::forbid`]
    /// reads its pattern.
    pub fn require(self, condition: &str) -> Result<Self, ReactionError> {
        self.with_condition(condition, false)
    }

    fn with_condition(mut self, condition: &str, negative: bool) -> Result<Self, ReactionError> {
        let pattern = parse_cxsmarts(condition)?;
        if pattern.is_reaction {
            return Err(ReactionError::Smarts(CxSMARTSError::InvalidSyntax(
                "an application condition is a pattern, not a reaction".into(),
            )));
        }
        let mut map_index_of = vec![None; pattern.atoms.len()];
        for (&atom, &index) in &pattern.atom_classes {
            if !self.reactants.by_map_index.contains_key(&index) {
                return Err(ReactionError::UnknownConditionIndex { index });
            }
            map_index_of[atom] = Some(index);
        }
        let pattern = match self.semantics {
            Semantics::Rdkit => without_stereo(&pattern),
            Semantics::Dpo => kekule_tolerant(&pattern),
        };
        self.conditions.push(Condition {
            pattern,
            map_index_of,
            negative,
        });
        Ok(self)
    }

    /// The matches of the reactant side that every application condition
    /// allows.
    fn matches(&self, molecule: &MolecularGraph) -> Vec<HashMap<usize, usize>> {
        let embeddings = find_smarts_matches(&self.query, molecule);
        if self.conditions.is_empty() {
            return embeddings;
        }
        let found: Vec<Vec<HashMap<usize, usize>>> = self
            .conditions
            .iter()
            .map(|condition| find_smarts_matches(&condition.pattern, molecule))
            .collect();
        embeddings
            .into_iter()
            .filter(|embedding| {
                let atom_of_index: HashMap<usize, usize> = embedding
                    .iter()
                    .map(|(&pattern_atom, &atom)| (self.reactants.map_index_of[pattern_atom], atom))
                    .collect();
                self.conditions
                    .iter()
                    .zip(&found)
                    .all(|(condition, found)| {
                        let extends = found.iter().any(|occurrence| {
                            occurrence.iter().all(|(&condition_atom, &atom)| {
                                match condition.map_index_of[condition_atom] {
                                    Some(index) => atom_of_index.get(&index) == Some(&atom),
                                    None => !embedding.values().any(|&matched| matched == atom),
                                }
                            })
                        });
                        extends != condition.negative
                    })
            })
            .collect()
    }

    /// Applies the reaction everywhere it matches, returning one product
    /// molecule per match.
    ///
    /// A symmetric molecule yields the same product more than once — ethane
    /// under `[C:1][C:2]>>[C:1]=[C:2]` matches both ways round — because each
    /// match is reported separately rather than being deduplicated by
    /// structure. Canonize and dedupe if that matters.
    ///
    /// Under [`Semantics::Dpo`] a derivation is counted once however many
    /// matches lead to it: two matches that differ by a symmetry of the rule
    /// rewrite the same atoms the same way and give the same graph, atom for
    /// atom, and only one is kept. Different sites still give different
    /// products even where those are isomorphic.
    pub fn apply(&self, molecule: &MolecularGraph) -> Vec<MolecularGraph> {
        let products = self
            .matches(molecule)
            .into_iter()
            .filter_map(|embedding| self.rewrite(molecule, &embedding))
            .collect();
        self.derivations_once(products)
    }

    /// Under [`Semantics::Dpo`], drops every product identical, atom for atom,
    /// to one before it.
    fn derivations_once(&self, products: Vec<MolecularGraph>) -> Vec<MolecularGraph> {
        if self.semantics != Semantics::Dpo {
            return products;
        }
        let key = |molecule: &MolecularGraph| {
            let bonds: Vec<Vec<(usize, BondOrder)>> = molecule
                .atom_bonds()
                .iter()
                .map(|bonds| {
                    let mut bonds: Vec<(usize, BondOrder)> = bonds
                        .iter()
                        .map(|bond| (bond.target(), bond.bond_order()))
                        .collect();
                    bonds.sort_unstable_by_key(|&(target, _)| target);
                    bonds
                })
                .collect();
            (
                molecule.atomic_numbers().to_vec(),
                molecule.charges().to_vec(),
                bonds,
            )
        };
        let mut seen = Vec::new();
        products
            .into_iter()
            .filter(|product| {
                let key = key(product);
                let new = !seen.contains(&key);
                if new {
                    seen.push(key);
                }
                new
            })
            .collect()
    }

    /// Applies the reaction to several reactant molecules at once.
    ///
    /// They are combined into one graph first, so a template written across a
    /// `.` — `[C:1]-[O:2].[N:3]>>...` — can reach into both.
    ///
    /// When multiple reactants are passed, matches that do not touch every reactant
    /// are filtered out so that only genuine multi-reactant interactions are returned.
    pub fn apply_to(&self, reactants: &[&MolecularGraph]) -> Vec<MolecularGraph> {
        if reactants.is_empty() {
            return Vec::new();
        }
        if reactants.len() == 1 {
            return self.apply(reactants[0]);
        }
        let combined = MolecularGraph::combine(reactants);
        let mut offsets = Vec::with_capacity(reactants.len() + 1);
        offsets.push(0);
        for r in reactants {
            offsets.push(offsets.last().unwrap() + r.atomic_numbers().len());
        }

        let products = self
            .matches(&combined)
            .into_iter()
            .filter(|embedding| {
                (0..reactants.len()).all(|r_idx| {
                    let start = offsets[r_idx];
                    let end = offsets[r_idx + 1];
                    embedding.values().any(|&atom| atom >= start && atom < end)
                })
            })
            .filter_map(|embedding| self.rewrite(&combined, &embedding))
            .collect();
        self.derivations_once(products)
    }

    /// Rewrites one match into a product molecule.
    fn rewrite(
        &self,
        molecule: &MolecularGraph,
        embedding: &HashMap<usize, usize>,
    ) -> Option<MolecularGraph> {
        // map index -> the molecule atom it matched
        let mut atom_of_index: HashMap<usize, usize> = HashMap::new();
        for (&pattern_atom, &molecule_atom) in embedding {
            let index = *self.reactants.map_index_of.get(pattern_atom)?;
            atom_of_index.insert(index, molecule_atom);
        }
        if self.semantics == Semantics::Dpo && !self.no_dangling_bonds(molecule, embedding) {
            return None;
        }
        let kekule;
        let molecule = match self.semantics {
            Semantics::Rdkit => molecule,
            Semantics::Dpo => {
                kekule = self.kekule_for(molecule, embedding)?;
                &kekule
            }
        };

        let mut product = molecule.clone();
        let mut touched: Vec<usize> = Vec::new();

        // Atoms the product invents, created before any bond is wired so that
        // both ends of every new bond exist.
        for &index in &self.map_indices() {
            if self.reactants.by_map_index.contains_key(&index) {
                continue;
            }
            let &pattern_atom = self.products.by_map_index.get(&index)?;
            let atomic_number = element_of(&self.products.pattern.atoms[pattern_atom])?;
            let new_atom = product.atomic_numbers().len();
            product.add_atom(Atom::new(atomic_number));
            atom_of_index.insert(index, new_atom);
            touched.push(new_atom);
        }

        // Bonds: compare what each side asks for, pair by pair.
        let indices = self.map_indices();
        for (position, &first_index) in indices.iter().enumerate() {
            for &second_index in &indices[position + 1..] {
                let before = self.reactants.bond_between(first_index, second_index);
                let after = self.products.bond_between(first_index, second_index);
                let (Some(&first), Some(&second)) = (
                    atom_of_index.get(&first_index),
                    atom_of_index.get(&second_index),
                ) else {
                    continue;
                };
                match (before, after) {
                    (Some(before), Some(after)) => {
                        let Some(order) = product_order(before, after) else {
                            continue;
                        };
                        if bond_order_in(&product, first, second) == Some(order) {
                            continue;
                        }
                        product.change_bond_order(first, second, order);
                    }
                    (Some(_), None) => remove_bond(&mut product, first, second),
                    (None, Some(after)) => {
                        let order = simple_order(after).unwrap_or(BondOrder::Single);
                        product.add_bond(first, second, order);
                    }
                    (None, None) => continue,
                }
                touched.push(first);
                touched.push(second);
            }
        }

        // Element and charge changes stated by the product template.
        for (&index, &pattern_atom) in &self.products.by_map_index {
            let Some(&atom) = atom_of_index.get(&index) else {
                continue;
            };
            let expression = &self.products.pattern.atoms[pattern_atom];
            if let Some(atomic_number) = element_of(expression) {
                if product.atomic_numbers()[atom] != atomic_number {
                    product.atomic_numbers_mut()[atom] = atomic_number;
                    touched.push(atom);
                }
            }
            if let Some(charge) = charge_of(expression) {
                if product.charges()[atom] != charge {
                    product.charges_mut()[atom] = charge;
                    touched.push(atom);
                }
            }
            if let Some(isotope) = isotope_of(expression) {
                let atom_count = product.atomic_numbers().len();
                product
                    .isotopes_mut()
                    .get_or_insert_with(|| vec![0; atom_count])[atom] = isotope;
            }
        }

        let detached = self.update_stereo(molecule, &mut product, &atom_of_index, &touched);

        // Atoms the reaction consumes. Removing shifts every later index, so
        // this goes last and in descending order, and the atoms whose hydrogen
        // count has to be revisited are recorded as atom *identities* first.
        let mut doomed: Vec<usize> = self
            .reactants
            .by_map_index
            .keys()
            .filter(|index| !self.products.by_map_index.contains_key(index))
            .filter_map(|index| atom_of_index.get(index).copied())
            .collect();
        doomed.extend(detached);
        // A consumed atom's hydrogens go with it, unless the template maps
        // them and so has its own say in their fate.
        let mapped: Vec<usize> = atom_of_index.values().copied().collect();
        let hydrogens: Vec<usize> = doomed
            .iter()
            .filter(|&&atom| product.atomic_numbers()[atom] != 1)
            .flat_map(|&atom| product.atom_bonds()[atom].iter().map(|bond| bond.target()))
            .filter(|&target| product.atomic_numbers()[target] == 1 && !mapped.contains(&target))
            .collect();
        doomed.extend(hydrogens);
        for &atom in &doomed {
            for bond in product.atom_bonds()[atom].iter() {
                touched.push(bond.target());
            }
        }

        touched.sort_unstable();
        touched.dedup();
        doomed.sort_unstable();
        doomed.dedup();

        // Re-index `touched` around the removals, then drop the atoms.
        let mut survivors: Vec<usize> = touched
            .iter()
            .filter(|atom| !doomed.contains(atom))
            .map(|&atom| atom - doomed.iter().filter(|&&gone| gone < atom).count())
            .collect();
        for &atom in doomed.iter().rev() {
            product.remove_atom(atom);
        }
        survivors.sort_unstable();
        survivors.dedup();

        match self.semantics {
            Semantics::Rdkit => {
                rebalance_hydrogens(&mut product, &survivors);
                // As RDKit sanitizes a product, though leniently: a product
                // with no Kekulé structure is still returned, as RDKit's
                // `RunReactants` returns it unsanitized.
                if product.localize_aromatic_bonds().is_ok() {
                    perceive_aromaticity(&mut product);
                }
            }
            Semantics::Dpo => normalize_product(&mut product, &survivors)?,
        }
        Some(product)
    }

    /// Brings the stereo tags of the atoms the rule keeps or creates in line
    /// with what it did to them, as RDKit does.
    ///
    /// Tags are stored against a centre's neighbours sorted by index, so one
    /// whose neighbours the rule leaves alone needs nothing. Otherwise, for a
    /// tetrahedral centre:
    ///
    /// - A product template that writes `@` or `@@` and names every neighbour
    ///   but hydrogens sets the tag from its own written order. That covers
    ///   creating a centre, and retaining or inverting one, exactly.
    /// - A reactant template that writes a tag the product template drops
    ///   removes it.
    /// - Otherwise the tag is carried over, a neighbour the rule replaced
    ///   taking the place of the one it lost, and inverted when the two
    ///   templates write opposite tags. A centre that gains or loses a
    ///   neighbour without replacing one is no longer the same centre and
    ///   loses its tag.
    ///
    /// A cis/trans or allene tag is dropped when the rule changes a bond of its
    /// atom or of a neighbour, since what it describes may no longer exist.
    ///
    /// Under [`Semantics::Rdkit`] the hydrogens are re-derived afterwards,
    /// which would swap a centre's neighbours behind this function's back:
    /// a centre that gains a bond loses a hydrogen, one that loses a bond
    /// gains one. So the swap is made here, where it can be tracked — the new
    /// neighbour taking the hydrogen's place, or a new hydrogen the lost
    /// neighbour's — and the configuration is kept, as it is under
    /// [`Semantics::Dpo`] where the hydrogens are explicit in the rule. RDKit
    /// itself returns the mirror image in these two cases, an artifact of how
    /// it numbers an implicit hydrogen. Returns the hydrogens detached, for
    /// the caller to remove.
    fn update_stereo(
        &self,
        molecule: &MolecularGraph,
        product: &mut MolecularGraph,
        atom_of_index: &HashMap<usize, usize>,
        rebonded: &[usize],
    ) -> Vec<usize> {
        let mut detached = Vec::new();
        let mapped: Vec<usize> = atom_of_index.values().copied().collect();
        let original_count = molecule.atomic_numbers().len();
        for &index in &self.map_indices() {
            let (Some(&atom), Some(&right)) = (
                atom_of_index.get(&index),
                self.products.by_map_index.get(&index),
            ) else {
                continue;
            };
            let left_tag = self
                .reactants
                .by_map_index
                .get(&index)
                .and_then(|&left| tetrahedral_of(&self.reactants.pattern.atoms[left]));
            let right_tag = tetrahedral_of(&self.products.pattern.atoms[right]);
            let before = if atom < original_count {
                index_sorted_neighbors(molecule, atom)
            } else {
                Vec::new()
            };
            let mut after = index_sorted_neighbors(&*product, atom);
            let stored = product.get_chiral_class(atom);
            let determined =
                right_tag.and_then(|tag| self.tag_from_product(product, atom_of_index, right, tag));
            if determined.is_none()
                && !(left_tag.is_some() && right_tag.is_none())
                && matches!(stored, ChiralClass::TH(_))
                && self.semantics == Semantics::Rdkit
            {
                detached.extend(settle_hydrogen(product, atom, &before, &after, &mapped));
                after = index_sorted_neighbors(&*product, atom);
            }

            let class = if let Some(tag) = determined {
                ChiralClass::TH(tag)
            } else if left_tag.is_some() && right_tag.is_none() {
                ChiralClass::None
            } else if let ChiralClass::TH(tag) = stored {
                let carried = if before == after {
                    Some(tag)
                } else {
                    replaced_in_place(&before, &after)
                        .map(|reference| tetrahedral_tag_for_order(tag, &reference, &after))
                };
                match (carried, left_tag, right_tag) {
                    (Some(tag), Some(left), Some(right)) if left != right => {
                        ChiralClass::TH(3 - tag)
                    }
                    (Some(tag), _, _) => ChiralClass::TH(tag),
                    (None, _, _) => ChiralClass::None,
                }
            } else if stored != ChiralClass::None
                && (rebonded.contains(&atom)
                    || after
                        .iter()
                        .chain(&before)
                        .any(|neighbor| rebonded.contains(neighbor)))
            {
                ChiralClass::None
            } else {
                continue;
            };
            if class == stored {
                continue;
            }
            if product.chiral_classes().is_none() {
                product.initialize_chiral_classes();
            }
            if let Some(classes) = product.chiral_classes_mut() {
                classes[atom] = class;
            }
        }
        detached
    }

    /// The tag a product template's `@` or `@@` gives `atom`, re-expressed
    /// against its sorted neighbours — if the template names every neighbour
    /// the atom has, hydrogens aside, which it reads as the matcher does: in
    /// the order written, a hydrogen it leaves out in the place an implicit
    /// one would take.
    fn tag_from_product(
        &self,
        product: &MolecularGraph,
        atom_of_index: &HashMap<usize, usize>,
        pattern_atom: usize,
        tag: u8,
    ) -> Option<u8> {
        let pattern = &self.products.pattern;
        let atom = *atom_of_index.get(&self.products.map_index_of[pattern_atom])?;
        let mut written: Vec<usize> = pattern
            .neighbor_order
            .get(pattern_atom)?
            .iter()
            .map(|&neighbor| {
                atom_of_index
                    .get(&self.products.map_index_of[neighbor])
                    .copied()
            })
            .collect::<Option<_>>()?;
        let sorted = index_sorted_neighbors(product, atom);
        let mut missing: Vec<usize> = sorted
            .iter()
            .copied()
            .filter(|neighbor| !written.contains(neighbor))
            .collect();
        if missing
            .iter()
            .any(|&neighbor| product.atomic_numbers()[neighbor] != 1)
            || written.len() + missing.len() != sorted.len()
            || !(3..=4).contains(&sorted.len())
        {
            return None;
        }
        missing.sort_unstable();
        let insertion = usize::from(
            pattern
                .incoming
                .get(pattern_atom)
                .copied()
                .flatten()
                .is_some(),
        );
        written.splice(insertion..insertion, missing);
        Some(tetrahedral_tag_for_order(tag, &written, &sorted))
    }

    /// The Kekulé structure of `molecule` a double-pushout match is made in,
    /// if it has one.
    ///
    /// A molecule is held in its aromatic form, but a double-pushout rule
    /// rewrites a graph, and the graph of benzene is one of its Kekulé
    /// structures. So a rule's `-` or `=` matches an aromatic bond (see
    /// [`kekule_tolerant`]), and here the match is kept only if some Kekulé
    /// structure has every such bond as the rule writes it — which is the
    /// structure the rule then rewrites. `[C:1]=[C:2]` matches every bond of
    /// benzene; `[C:1]=[C:2]=[C:3]` matches nothing there, since no Kekulé
    /// structure gives one carbon two double bonds.
    fn kekule_for<'a>(
        &self,
        molecule: &'a MolecularGraph,
        embedding: &HashMap<usize, usize>,
    ) -> Option<std::borrow::Cow<'a, MolecularGraph>> {
        let forced: Vec<(usize, usize, BondOrder)> = self
            .reactants
            .pattern
            .bonds
            .iter()
            .filter_map(|(a, b, expression)| {
                let (first, second) = (embedding[a], embedding[b]);
                let order = simple_order(expression)?;
                (matches!(order, BondOrder::Single | BondOrder::Double)
                    && bond_order_in(molecule, first, second) == Some(BondOrder::Aromatic))
                .then_some((first, second, order))
            })
            .collect();
        if forced.is_empty() {
            return Some(std::borrow::Cow::Borrowed(molecule));
        }
        let mut kekule = molecule.clone();
        for &(first, second, order) in &forced {
            kekule.change_bond_order(first, second, order);
        }
        kekule.localize_aromatic_bonds().ok()?;
        forced
            .iter()
            .all(|&(first, second, order)| bond_order_in(&kekule, first, second) == Some(order))
            .then_some(std::borrow::Cow::Owned(kekule))
    }

    /// The dangling condition: every bond of an atom the rule consumes is a
    /// bond L matched, so deleting the atom leaves no bond without an end.
    fn no_dangling_bonds(
        &self,
        molecule: &MolecularGraph,
        embedding: &HashMap<usize, usize>,
    ) -> bool {
        embedding.iter().all(|(&pattern_atom, &molecule_atom)| {
            let index = self.reactants.map_index_of[pattern_atom];
            if self.products.by_map_index.contains_key(&index) {
                return true;
            }
            let in_rule = self
                .reactants
                .pattern
                .bonds
                .iter()
                .filter(|&&(a, b, _)| a == pattern_atom || b == pattern_atom)
                .count();
            molecule.atom_bonds()[molecule_atom].len() == in_rule
        })
    }
}

/// Perceives aromaticity with the model a SMILES parse uses, so a product
/// answers aromaticity questions as a parsed molecule does.
fn perceive_aromaticity(molecule: &mut MolecularGraph) {
    if let Some(model) = SmilesParseOptions::default().aromaticity {
        let _ = molecule.aromatize(model, None, false);
    }
}

/// Brings a double-pushout product to the form a SMILES parse gives, or
/// rejects it as no molecule: its aromatic bonds must have a Kekulé
/// structure, every atom the rule touched must sit on one of its element's
/// valences in that structure, and aromaticity is then perceived afresh.
fn normalize_product(molecule: &mut MolecularGraph, touched: &[usize]) -> Option<()> {
    molecule.localize_aromatic_bonds().ok()?;
    for &atom in touched {
        let bonded = molecule.actual_valency(atom)
            + molecule.radical_states().get(atom).copied().unwrap_or(0) as i8;
        let atomic_number = molecule.atomic_numbers()[atom];
        let charge = molecule.charges()[atom];
        match saturating_valency(atomic_number, charge, bonded) {
            Some(valence) if valence != bonded => return None,
            // No valence reaches this far: more bonds than the element forms.
            None if chemistry_consts::ElementProperties::valencies(&atomic_number).is_some() => {
                return None;
            }
            _ => {}
        }
    }
    perceive_aromaticity(molecule);
    Some(())
}

fn remove_bond(molecule: &mut MolecularGraph, first: usize, second: usize) {
    molecule.atom_bonds_mut()[first].retain(|bond| bond.target() != second);
    molecule.atom_bonds_mut()[second].retain(|bond| bond.target() != first);
}

/// Brings the hydrogen count of every atom whose bonding changed back in line
/// with its valence.
///
/// A template says which heavy-atom bonds to make and break and nothing about
/// hydrogens, so `[C:1]-[O:2]>>[C:1]=[O:2]` on ethanol would otherwise leave a
/// carbon with five bonds. Excess hydrogens are dropped first, then
/// [`MoleculeCore::add_hydrogens`] fills whatever is still short.
fn rebalance_hydrogens(molecule: &mut MolecularGraph, atoms: &[usize]) {
    let mut excess: Vec<usize> = Vec::new();
    for &atom in atoms {
        if atom >= molecule.atomic_numbers().len() || molecule.atomic_numbers()[atom] == 1 {
            continue;
        }
        let surplus = molecule.get_atom_degree(atom).unwrap_or(0);
        if surplus <= 0 {
            continue;
        }
        let hydrogens: Vec<usize> = molecule.atom_bonds()[atom]
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| molecule.atomic_numbers()[target] == 1)
            .take(surplus as usize)
            .collect();
        excess.extend(hydrogens);
    }

    excess.sort_unstable();
    excess.dedup();
    for &hydrogen in excess.iter().rev() {
        molecule.remove_atom(hydrogen);
    }

    molecule.add_hydrogens();
}
