//! Stereo predicates evaluated against a complete query-to-target mapping.

use super::{AtomExpression as Atom, BondExpression as Bond, CxSMARTSPattern, LogicalOperator};
use crate::molecule::bond::BondOrder;
use crate::molecule::chirality::{
    ChiralClass, index_sorted_allene_carriers, index_sorted_neighbors, remap_oh_number,
    remap_sp_number, remap_tb_number, tetrahedral_tag_for_order,
};
use crate::prelude::*;

fn logical(operator: &LogicalOperator, values: impl Iterator<Item = bool>) -> bool {
    let mut values = values;
    match operator {
        LogicalOperator::Not => values.next().is_some_and(|v| !v) && values.next().is_none(),
        LogicalOperator::HighAnd | LogicalOperator::LowAnd => values.all(|v| v),
        LogicalOperator::Or => values.any(|v| v),
    }
}

pub(super) fn has_atom_stereo(expr: &Atom) -> bool {
    match expr {
        Atom::Chirality(_) => true,
        Atom::Logical { operands, .. } => operands.iter().any(has_atom_stereo),
        _ => false,
    }
}

pub(super) fn has_stereo(pattern: &CxSMARTSPattern) -> bool {
    pattern.atoms.iter().any(has_atom_stereo)
        || pattern.bonds.iter().any(|(_, _, expr)| has_direction(expr))
}

pub(super) fn atom_matches(
    expr: &Atom,
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    atom: usize,
    assignment: &[usize],
    ordinary: &impl Fn(&Atom) -> bool,
) -> bool {
    match expr {
        Atom::Chirality(class) => chiral_matches(*class, pattern, molecule, atom, assignment),
        Atom::Logical { operator, operands } => logical(
            operator,
            operands
                .iter()
                .map(|expr| atom_matches(expr, pattern, molecule, atom, assignment, ordinary)),
        ),
        _ => ordinary(expr),
    }
}

/// Complete the written carrier order with hydrogens that are implicit in the
/// query but explicit in the target. A missing heavy ligand is not a hydrogen.
fn written_neighbors(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    atom: usize,
    assignment: &[usize],
) -> Option<Vec<usize>> {
    let query = pattern.neighbor_order.get(atom)?;
    let mut order: Vec<_> = query
        .iter()
        .map(|&q| assignment.get(q).copied())
        .collect::<Option<_>>()?;
    let target = assignment[atom];
    let mut missing = Vec::new();
    for bond in &molecule.atom_bonds()[target] {
        if !order.contains(&bond.target()) {
            if molecule.atomic_numbers()[bond.target()] != 1 {
                return None;
            }
            missing.push(bond.target());
        }
    }
    missing.sort_unstable();
    let insertion = usize::from(pattern.incoming.get(atom).copied().flatten().is_some());
    order.splice(insertion..insertion, missing);
    Some(order)
}

fn chiral_matches(
    mut query: ChiralClass,
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    atom: usize,
    assignment: &[usize],
) -> bool {
    let target = assignment[atom];
    let stored = molecule.get_chiral_class(target);
    if matches!(
        query,
        ChiralClass::Clockwise | ChiralClass::Counterclockwise
    ) {
        let number = if query == ChiralClass::Clockwise {
            2
        } else {
            1
        };
        query = match stored {
            ChiralClass::TH(_) => ChiralClass::TH(number),
            ChiralClass::AL(_) => ChiralClass::AL(number),
            ChiralClass::SP(_) => ChiralClass::SP(number),
            ChiralClass::TB(_) => ChiralClass::TB(number),
            ChiralClass::OH(_) => ChiralClass::OH(number),
            _ => return false,
        };
    }
    let Some(written) = written_neighbors(pattern, molecule, atom, assignment) else {
        return false;
    };
    let sorted = index_sorted_neighbors(molecule, target);
    match (query, stored) {
        (ChiralClass::TH(q), ChiralClass::TH(t))
            if (1..=2).contains(&q) && (3..=4).contains(&written.len()) =>
        {
            // The query is read with the SMILES reader's convention, so a
            // string means the same centre whether it is parsed as a query or
            // as a molecule: a three-coordinate centre written with one ring
            // closure has its parity inverted, as RDKit does.
            let closures = pattern
                .ring_closures
                .values()
                .flatten()
                .filter(|&&written_at| written_at == atom)
                .count();
            let q = if crate::molecule::chirality::tetrahedral_ring_closure_inverts(
                molecule, target, closures,
            ) {
                3 - q
            } else {
                q
            };
            tetrahedral_tag_for_order(q, &written, &sorted) == t
        }
        (ChiralClass::AL(q), ChiralClass::AL(t)) if (1..=2).contains(&q) => {
            let Some(ends) = pattern.neighbor_order.get(atom) else {
                return false;
            };
            if ends.len() != 2 {
                return false;
            }
            let mut carriers = Vec::new();
            for &end in ends {
                let Some(mut order) = written_neighbors(pattern, molecule, end, assignment) else {
                    return false;
                };
                order.retain(|&neighbor| neighbor != target);
                if order.len() != 2 {
                    return false;
                }
                carriers.extend(order);
            }
            index_sorted_allene_carriers(molecule, target)
                .is_some_and(|sorted| tetrahedral_tag_for_order(q, &carriers, &sorted) == t)
        }
        (ChiralClass::SP(q), ChiralClass::SP(t)) => {
            remap_sp_number(q, &written, &sorted) == Some(t)
        }
        (ChiralClass::TB(q), ChiralClass::TB(t)) => {
            remap_tb_number(q, &written, &sorted) == Some(t)
        }
        (ChiralClass::OH(q), ChiralClass::OH(t)) => {
            remap_oh_number(q, &written, &sorted) == Some(t)
        }
        _ => false,
    }
}

fn has_direction(expr: &Bond) -> bool {
    match expr {
        Bond::DirectionalUp
        | Bond::DirectionalDown
        | Bond::DirectionalUpOrUnspec
        | Bond::DirectionalDownOrUnspec => true,
        Bond::Not(inner) => has_direction(inner),
        Bond::Logical { operands, .. } => operands.iter().any(has_direction),
        _ => false,
    }
}

fn bond_matches(expr: &Bond, direction: Option<bool>, ordinary: &impl Fn(&Bond) -> bool) -> bool {
    match expr {
        Bond::DirectionalUp => ordinary(&Bond::Single) && direction == Some(true),
        Bond::DirectionalDown => ordinary(&Bond::Single) && direction == Some(false),
        Bond::DirectionalUpOrUnspec => ordinary(&Bond::Single) && direction != Some(false),
        Bond::DirectionalDownOrUnspec => ordinary(&Bond::Single) && direction != Some(true),
        Bond::Not(inner) => !bond_matches(inner, direction, ordinary),
        Bond::Logical { operator, operands } => logical(
            operator,
            operands
                .iter()
                .map(|expr| bond_matches(expr, direction, ordinary)),
        ),
        _ => ordinary(expr),
    }
}

pub(super) fn bond_candidate(expr: &Bond, ordinary: &impl Fn(&Bond) -> bool) -> bool {
    [None, Some(false), Some(true)]
        .into_iter()
        .any(|direction| bond_matches(expr, direction, ordinary))
}

pub(super) fn reverse_bond(expr: Bond) -> Bond {
    match expr {
        Bond::DirectionalUp => Bond::DirectionalDown,
        Bond::DirectionalDown => Bond::DirectionalUp,
        Bond::DirectionalUpOrUnspec => Bond::DirectionalDownOrUnspec,
        Bond::DirectionalDownOrUnspec => Bond::DirectionalUpOrUnspec,
        Bond::Not(inner) => Bond::Not(Box::new(reverse_bond(*inner))),
        Bond::Logical { operator, operands } => Bond::Logical {
            operator,
            operands: operands.into_iter().map(reverse_bond).collect(),
        },
        other => other,
    }
}

pub(super) fn bonds_match(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    assignment: &[usize],
    ordinary: &impl Fn(&Bond, usize, usize) -> bool,
) -> bool {
    for &(a, b, _) in &pattern.bonds {
        let (ta, tb) = (assignment[a], assignment[b]);
        if !molecule.atom_bonds()[ta]
            .iter()
            .any(|bond| bond.target() == tb && bond.bond_order() == BondOrder::Double)
        {
            continue;
        }
        let sides: Vec<Vec<_>> = [a, b]
            .into_iter()
            .map(|center| {
                pattern
                    .bonds
                    .iter()
                    .filter(|(x, y, expr)| (*x == center || *y == center) && has_direction(expr))
                    .collect()
            })
            .collect();
        // One slash alone specifies no relative geometry. Its bond order has
        // already been checked by the structural search.
        if sides.iter().any(Vec::is_empty) {
            continue;
        }
        let classes = [molecule.get_chiral_class(ta), molecule.get_chiral_class(tb)];
        let specified = classes.iter().all(|class| {
            matches!(
                class,
                ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
            )
        });
        let valid = [false, true].into_iter().any(|flip| {
            [a, b].into_iter().enumerate().all(|(side, center)| {
                let target = assignment[center];
                let first = molecule.atom_bonds()[target]
                    .iter()
                    .find(|bond| bond.bond_order() != BondOrder::Double)
                    .map(|bond| bond.target());
                sides[side].iter().all(|&&(x, y, ref expr)| {
                    let other = if x == center { y } else { x };
                    let direction = if specified {
                        Some(
                            (classes[side] == ChiralClass::CisTransClockwise)
                                ^ (first != Some(assignment[other]))
                                ^ (x != center)
                                ^ flip,
                        )
                    } else {
                        None
                    };
                    bond_matches(expr, direction, &|expr| {
                        ordinary(expr, assignment[x], assignment[y])
                    })
                })
            })
        });
        if !valid {
            return false;
        }
    }
    true
}
