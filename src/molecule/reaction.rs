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
    AtomExpression, BondExpression, CxSMARTSError, CxSMARTSPattern, find_smarts_matches,
    parse_cxsmarts,
};
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::molecular_graph::MolecularGraph;

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
}

impl From<CxSMARTSError> for ReactionError {
    fn from(error: CxSMARTSError) -> Self {
        ReactionError::Smarts(error)
    }
}

/// One side of a reaction, indexed by atom map.
struct Side {
    pattern: CxSMARTSPattern,
    /// map index -> pattern atom
    by_map_index: HashMap<usize, usize>,
    /// pattern atom -> map index
    map_index_of: Vec<usize>,
}

impl Side {
    fn build(pattern: CxSMARTSPattern, side: &'static str) -> Result<Self, ReactionError> {
        if pattern.atoms.is_empty() {
            return Err(ReactionError::EmptySide(side));
        }
        let mut by_map_index = HashMap::new();
        let mut map_index_of = vec![0usize; pattern.atoms.len()];
        for atom in 0..pattern.atoms.len() {
            let Some(&index) = pattern.atom_classes.get(&atom) else {
                return Err(ReactionError::UnmappedAtom { side, atom });
            };
            if by_map_index.insert(index, atom).is_some() {
                return Err(ReactionError::DuplicateMapIndex { side, index });
            }
            map_index_of[atom] = index;
        }
        Ok(Side { pattern, by_map_index, map_index_of })
    }

    /// The bond order this side asks for between two map indices, if it joins
    /// them at all.
    fn bond_between(&self, first_index: usize, second_index: usize) -> Option<BondOrder> {
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
            .map(|(_, _, expression)| bond_order_of(expression))
    }
}

/// The bond order a product template's expression asks for.
///
/// A product bond has to be a definite bond, so anything that only *tests* a
/// bond — `~`, `!-`, an alternation — is read as a single bond, which is what
/// an unwritten bond means in SMILES too.
fn bond_order_of(expression: &BondExpression) -> BondOrder {
    match expression {
        BondExpression::Double => BondOrder::Double,
        BondExpression::Triple => BondOrder::Triple,
        BondExpression::Aromatic => BondOrder::Aromatic,
        _ => BondOrder::Single,
    }
}

/// The element a product template names, if it names one outright.
fn element_of(expression: &AtomExpression) -> Option<u8> {
    match expression {
        AtomExpression::AtomicNumber(atomic_number) => Some(*atomic_number),
        AtomExpression::Element(symbol) => {
            use chemistry_consts::ElementProperties;
            symbol.as_str().atomic_number()
        }
        AtomExpression::Logical { operands, .. } => operands.iter().find_map(element_of),
        _ => None,
    }
}

/// The formal charge a product template states, if it states one.
fn charge_of(expression: &AtomExpression) -> Option<i8> {
    match expression {
        AtomExpression::Charge(charge) => Some(*charge),
        AtomExpression::Logical { operands, .. } => operands.iter().find_map(charge_of),
        _ => None,
    }
}

/// A parsed, validated reaction SMARTS.
pub struct ReactionSmarts {
    reactants: Side,
    products: Side,
}

impl ReactionSmarts {
    /// Parses and validates a reaction SMARTS.
    ///
    /// Fails if any atom on either side lacks a map index, or if an index is
    /// reused within a side — see the module docs for why that is required.
    pub fn parse(smarts: &str) -> Result<Self, ReactionError> {
        let pattern = parse_cxsmarts(smarts)?;
        if !pattern.is_reaction {
            return Err(ReactionError::NotAReaction);
        }
        let (Some(reactants), Some(products)) = (pattern.reactants, pattern.products) else {
            return Err(ReactionError::NotAReaction);
        };

        let reactants = Side::build(*reactants, "reactants")?;
        let products = Side::build(*products, "products")?;

        // An atom the product invents must say what it is.
        for (&index, &atom) in &products.by_map_index {
            if !reactants.by_map_index.contains_key(&index)
                && element_of(&products.pattern.atoms[atom]).is_none()
            {
                return Err(ReactionError::NewAtomWithoutElement { index });
            }
        }

        Ok(ReactionSmarts { reactants, products })
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

    /// Applies the reaction everywhere it matches, returning one product
    /// molecule per match.
    ///
    /// A symmetric molecule yields the same product more than once — ethane
    /// under `[C:1][C:2]>>[C:1]=[C:2]` matches both ways round — because each
    /// match is reported separately rather than being deduplicated by
    /// structure. Canonize and dedupe if that matters.
    pub fn apply(&self, molecule: &MolecularGraph) -> Vec<MolecularGraph> {
        find_smarts_matches(&self.reactants.pattern, molecule)
            .into_iter()
            .filter_map(|embedding| self.rewrite(molecule, &embedding))
            .collect()
    }

    /// Applies the reaction to several reactant molecules at once.
    ///
    /// They are combined into one graph first, so a template written across a
    /// `.` — `[C:1]-[O:2].[N:3]>>...` — can reach into both.
    pub fn apply_to(&self, reactants: &[&MolecularGraph]) -> Vec<MolecularGraph> {
        self.apply(&MolecularGraph::combine(reactants))
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
                if before == after {
                    continue;
                }
                let (Some(&first), Some(&second)) = (
                    atom_of_index.get(&first_index),
                    atom_of_index.get(&second_index),
                ) else {
                    continue;
                };
                match (before, after) {
                    (Some(_), Some(order)) => product.change_bond_order(first, second, order),
                    (Some(_), None) => remove_bond(&mut product, first, second),
                    (None, Some(order)) => product.add_bond(first, second, order),
                    (None, None) => continue,
                }
                touched.push(first);
                touched.push(second);
            }
        }

        // Element and charge changes stated by the product template.
        for (&index, &pattern_atom) in &self.products.by_map_index {
            let Some(&atom) = atom_of_index.get(&index) else { continue };
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
        }

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

        rebalance_hydrogens(&mut product, &survivors);
        Some(product)
    }
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
