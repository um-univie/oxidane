//! Distance-bounds matrix construction and triangle-inequality smoothing.
//!
//! Follows the Crippen/Havel convention: for a pair `(i, j)` with `i < j`,
//! the upper bound is stored at `[i, j]` and the lower bound at `[j, i]`.

use super::EmbedError;
use crate::molecule::{base::MoleculeCore, bond::BondOrder, chirality::ChiralClass};
use bit_set::BitSet;
use chemistry_consts::ElementProperties;
use ndarray::Array2;
use rustc_hash::FxHashSet;

/// A generous finite placeholder for pairs not otherwise constrained, before
/// triangle-inequality smoothing tightens it. Must stay finite so smoothing
/// arithmetic never produces `inf`.
const DEFAULT_UPPER: f64 = 100.0;
const BOND_TOLERANCE: f64 = 0.02;
const VDW_LOWER_SCALE: f64 = 0.8;

/// Bondi/Alvarez-style van der Waals radii (Å), keyed by atomic number.
const VDW_RADII: &[(u8, f64)] = &[
    (1, 1.10),  // H
    (6, 1.70),  // C
    (7, 1.55),  // N
    (8, 1.52),  // O
    (9, 1.47),  // F
    (15, 1.80), // P
    (16, 1.80), // S
    (17, 1.75), // Cl
    (35, 1.85), // Br
    (53, 1.98), // I
];

pub(super) fn vdw_radius(atomic_number: u8) -> f64 {
    if let Some(&(_, radius)) = VDW_RADII.iter().find(|&&(z, _)| z == atomic_number) {
        return radius;
    }
    atomic_number
        .covalent_radius()
        .map(|r| r * 1.6)
        .unwrap_or(1.7)
}

/// Multiplicative bond-order correction applied to the sum of single-bond
/// covalent radii (Pyykkoe-style). Calibrated against typical C-C bond
/// lengths (single 1.54 A, aromatic 1.39 A, double 1.33 A, triple 1.20 A)
/// using this crate's covalent-radius table (C = 0.76 A); a known
/// simplification since real increments are element-pair-specific.
fn bond_order_factor(order: BondOrder) -> f64 {
    match order {
        BondOrder::Single | BondOrder::Up | BondOrder::Down | BondOrder::Coordinate => 1.01,
        BondOrder::Aromatic => 0.915,
        BondOrder::Double => 0.87,
        BondOrder::Triple => 0.78,
        BondOrder::Quadruple => 0.70,
    }
}

/// Returns `(min, max)` bond angle in radians expected at `atom`, based on a
/// simple hybridization heuristic derived from its neighbor bond orders.
/// No lone-pair model exists in this crate, so atoms with fewer than 4
/// explicit substituents get a wider tolerance.
pub(super) fn estimate_hybridization_angle<T: MoleculeCore>(mol: &T, atom: usize) -> (f64, f64) {
    let Some(bonds) = mol.get_atom_bonds(atom) else {
        return angle_bounds_deg(109.5, 15.0);
    };
    let degree = bonds.len();
    let double_count = bonds
        .iter()
        .filter(|b| b.bond_order == BondOrder::Double)
        .count();
    let has_triple = bonds.iter().any(|b| b.bond_order == BondOrder::Triple);
    let has_aromatic_or_double = bonds
        .iter()
        .any(|b| matches!(b.bond_order, BondOrder::Aromatic | BondOrder::Double));

    if has_triple || double_count >= 2 {
        angle_bounds_deg(180.0, 10.0) // sp: triple bond, or allene-type cumulated double bonds
    } else if has_aromatic_or_double {
        angle_bounds_deg(120.0, 10.0) // sp2
    } else if degree == 4 {
        angle_bounds_deg(109.5, 5.0) // sp3, fully substituted
    } else {
        angle_bounds_deg(109.5, 15.0) // sp3-ish default, looser (no lone-pair model)
    }
}

fn angle_bounds_deg(center_deg: f64, tol_deg: f64) -> (f64, f64) {
    let min_deg = (center_deg - tol_deg).max(0.0);
    let max_deg = (center_deg + tol_deg).min(180.0);
    (min_deg.to_radians(), max_deg.to_radians())
}

/// Interior angle (radians) of a regular polygon with `ring_size` vertices.
fn ring_interior_angle(ring_size: usize) -> f64 {
    let n = ring_size as f64;
    ((n - 2.0) * 180.0 / n).to_radians()
}

/// The two trans (180 deg) pairs among a square-planar centre's index-sorted
/// neighbors `n`, per the OpenSMILES `SP1`(U)/`SP2`(4)/`SP3`(Z) shape codes.
/// Calibrated empirically against RDKit's `AssignStereochemistryFrom3D`
/// (`tests/embed_oracle.py`): a simple hand-built square arrangement with
/// substituents placed at consecutive corners in written order reads back as
/// `SP1`; swapping the 2nd/3rd corners gives `SP2`; swapping the 1st/4th
/// corners gives `SP3` - i.e. `SP1` pairs neighbors (0,2) and (1,3) trans,
/// `SP2` pairs (0,1) and (2,3), `SP3` pairs (0,3) and (1,2). Any pair not
/// named here is one of the four *adjacent* (90 deg) pairs.
fn square_planar_trans_pairs(sp_number: u8, n: [usize; 4]) -> Option<[(usize, usize); 2]> {
    match sp_number {
        1 => Some([(n[0], n[2]), (n[1], n[3])]),
        2 => Some([(n[0], n[1]), (n[2], n[3])]),
        3 => Some([(n[0], n[3]), (n[1], n[2])]),
        _ => None,
    }
}

/// `Some((min, max))` 1-3 angle bounds for `(i, k, j)` if `k` is tagged as a
/// square-planar (`SP`) centre: 180 deg for the tag's two trans pairs, 90 deg
/// for every other (adjacent) pair. Unlike tetrahedral/allene chirality,
/// square-planar geometry has no separate handedness once the trans-pairing
/// is fixed - a plain 4-point distance arrangement (via these angle bounds)
/// fully determines the shape, so no additional stereo-consistency energy
/// term (unlike [`super::chirality`]'s `TH`/cis-trans terms) is needed.
fn square_planar_angle<T: MoleculeCore>(mol: &T, k: usize, i: usize, j: usize) -> Option<(f64, f64)> {
    let ChiralClass::SP(sp_number) = *mol.chiral_classes()?.get(k)? else {
        return None;
    };
    let mut neighbors: Vec<usize> = mol.get_atom_bonds(k)?.iter().map(|b| b.target()).collect();
    if neighbors.len() != 4 {
        return None;
    }
    neighbors.sort_unstable();
    let n = [neighbors[0], neighbors[1], neighbors[2], neighbors[3]];
    let trans_pairs = square_planar_trans_pairs(sp_number, n)?;
    let is_trans = trans_pairs.iter().any(|&(a, b)| (a == i && b == j) || (a == j && b == i));

    let tol = 5.0_f64.to_radians();
    Some(if is_trans {
        (std::f64::consts::PI - tol, std::f64::consts::PI)
    } else {
        (90.0_f64.to_radians() - tol, 90.0_f64.to_radians() + tol)
    })
}

/// Angle bounds for the 1-3 pair `(i, j)` through center `k`. Overrides the
/// generic hybridization estimate with the tight interior angle of the
/// smallest 3- or 4-membered ring containing all three atoms, since strained
/// rings are the one place the generic heuristic is badly wrong.
pub(super) fn angle_bounds_for_triple<T: MoleculeCore>(
    mol: &T,
    i: usize,
    k: usize,
    j: usize,
    cycles: &FxHashSet<BitSet>,
) -> (f64, f64) {
    if let Some(bounds) = square_planar_angle(mol, k, i, j) {
        return bounds;
    }

    let smallest_small_ring = cycles
        .iter()
        .filter(|cycle| cycle.len() == 3 || cycle.len() == 4)
        .filter(|cycle| cycle.contains(i) && cycle.contains(k) && cycle.contains(j))
        .map(|cycle| cycle.len())
        .min();

    if let Some(ring_size) = smallest_small_ring {
        let theta = ring_interior_angle(ring_size);
        let tol = 3.0_f64.to_radians();
        return ((theta - tol).max(0.0), (theta + tol).min(std::f64::consts::PI));
    }

    estimate_hybridization_angle(mol, k)
}

fn order(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

pub(crate) fn get_upper(bounds: &Array2<f64>, a: usize, b: usize) -> f64 {
    let (i, j) = order(a, b);
    bounds[[i, j]]
}

pub(crate) fn get_lower(bounds: &Array2<f64>, a: usize, b: usize) -> f64 {
    let (i, j) = order(a, b);
    bounds[[j, i]]
}

fn set_bounds(bounds: &mut Array2<f64>, a: usize, b: usize, lower: f64, upper: f64) {
    let (i, j) = order(a, b);
    bounds[[i, j]] = upper;
    bounds[[j, i]] = lower;
}

fn intersect_bounds(bounds: &mut Array2<f64>, a: usize, b: usize, lower: f64, upper: f64) {
    let cur_lower = get_lower(bounds, a, b);
    let cur_upper = get_upper(bounds, a, b);
    set_bounds(bounds, a, b, cur_lower.max(lower), cur_upper.min(upper));
}

fn law_of_cosines_distance(d1: f64, d2: f64, theta: f64) -> f64 {
    (d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * theta.cos())
        .max(0.0)
        .sqrt()
}

/// Builds the initial (pre-smoothing) distance-bounds matrix for `mol`:
/// exact-ish 1-2 (bonded) bounds, 1-3 (angle-derived) bounds, and loose
/// van-der-Waals-based 1-4+ bounds.
pub(crate) fn build_bounds<T: MoleculeCore>(mol: &T) -> Result<Array2<f64>, EmbedError> {
    let n = mol.len();
    if n < 2 {
        return Err(EmbedError::TooFewAtoms);
    }

    let mut bounds = Array2::<f64>::from_elem((n, n), DEFAULT_UPPER);
    // Tracks which pairs have received a real 1-2/1-3 geometric constraint,
    // as opposed to still sitting at the generic van-der-Waals default. Only
    // specific-vs-specific bounds get intersected; a 1-3 estimate must never
    // be intersected against the unrelated (and often stricter-looking, but
    // inapplicable) VDW default, or it can spuriously force lower > upper.
    let mut is_specific = Array2::<bool>::from_elem((n, n), false);
    for i in 0..n {
        for j in (i + 1)..n {
            let lower = VDW_LOWER_SCALE
                * (vdw_radius(mol.atomic_numbers()[i]) + vdw_radius(mol.atomic_numbers()[j]));
            set_bounds(&mut bounds, i, j, lower, DEFAULT_UPPER);
        }
    }

    // 1-2 (bonded) bounds.
    for i in 0..n {
        let Some(bonds_i) = mol.get_atom_bonds(i) else {
            continue;
        };
        for bond in bonds_i {
            let k = bond.target();
            if k <= i {
                continue;
            }
            let nominal = (mol.atomic_numbers()[i].covalent_radius().unwrap_or(0.77)
                + mol.atomic_numbers()[k].covalent_radius().unwrap_or(0.77))
                * bond_order_factor(bond.bond_order());
            set_bounds(
                &mut bounds,
                i,
                k,
                nominal - BOND_TOLERANCE,
                nominal + BOND_TOLERANCE,
            );
            is_specific[[i, k]] = true;
        }
    }

    // 1-3 (angle-derived) bounds, intersected across every shared neighbor
    // (fused/bridged rings can give the same pair multiple paths) - but only
    // once a specific bound already exists for that pair; the first specific
    // bound for a pair always overwrites the generic default outright.
    let cycles = mol.compute_fundamental_cycles();
    for k in 0..n {
        let Some(bonds_k) = mol.get_atom_bonds(k) else {
            continue;
        };
        let neighbors: Vec<usize> = bonds_k.iter().map(|b| b.target()).collect();
        for a in 0..neighbors.len() {
            for b in (a + 1)..neighbors.len() {
                let i = neighbors[a];
                let j = neighbors[b];
                let (theta_min, theta_max) = angle_bounds_for_triple(mol, i, k, j, &cycles);
                let l_ik = get_lower(&bounds, i, k);
                let u_ik = get_upper(&bounds, i, k);
                let l_kj = get_lower(&bounds, k, j);
                let u_kj = get_upper(&bounds, k, j);

                let new_lower = law_of_cosines_distance(l_ik, l_kj, theta_min);
                let new_upper = law_of_cosines_distance(u_ik, u_kj, theta_max);
                let (oi, oj) = order(i, j);
                if is_specific[[oi, oj]] {
                    intersect_bounds(&mut bounds, i, j, new_lower, new_upper);
                } else {
                    set_bounds(&mut bounds, i, j, new_lower, new_upper);
                    is_specific[[oi, oj]] = true;
                }
            }
        }
    }

    Ok(bounds)
}

/// Havel's triangle-inequality bounds smoothing: a single interleaved triple
/// loop where, within each `k`, upper-bound tightening happens before the
/// lower-bound update (the lower update consumes the just-tightened uppers).
/// One-time O(n^3) preprocessing, reused across every trial embedding.
pub(crate) fn smooth_bounds(bounds: &mut Array2<f64>) -> Result<(), EmbedError> {
    let n = bounds.nrows();
    for k in 0..n {
        for i in 0..n {
            if i == k {
                continue;
            }
            for j in (i + 1)..n {
                if j == k {
                    continue;
                }
                let u_ik = get_upper(bounds, i, k);
                let u_kj = get_upper(bounds, k, j);
                let u_ij = get_upper(bounds, i, j);
                let candidate_upper = u_ik + u_kj;
                if u_ij > candidate_upper {
                    set_bounds(bounds, i, j, get_lower(bounds, i, j), candidate_upper);
                }

                let l_ik = get_lower(bounds, i, k);
                let l_kj = get_lower(bounds, k, j);
                let u_ik = get_upper(bounds, i, k);
                let u_kj = get_upper(bounds, k, j);
                let l_ij = get_lower(bounds, i, j);
                let candidate_lower = (l_ik - u_kj).max(l_kj - u_ik);
                if l_ij < candidate_lower {
                    set_bounds(bounds, i, j, candidate_lower, get_upper(bounds, i, j));
                }
            }
        }
    }

    // Feasibility check: repair small inconsistencies with a tiny buffer,
    // otherwise report infeasibility rather than silently embedding garbage.
    let mut fixups = 0usize;
    let mut total = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += 1;
            let lower = get_lower(bounds, i, j);
            let upper = get_upper(bounds, i, j);
            if lower > upper {
                fixups += 1;
                set_bounds(bounds, i, j, lower, lower * 1.02);
            }
        }
    }
    if total > 0 && (fixups as f64) / (total as f64) > 0.01 {
        return Err(EmbedError::BoundsInfeasible);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::molecular_graph::MolecularGraph;

    fn mol(smiles: &str) -> MolecularGraph {
        let mut m = MolecularGraph::from_smiles_single(smiles).unwrap();
        m.add_hydrogens();
        m
    }

    #[test]
    fn ethane_cc_bond_length() {
        let m = mol("CC");
        let bounds = build_bounds(&m).unwrap();
        let lower = get_lower(&bounds, 0, 1);
        let upper = get_upper(&bounds, 0, 1);
        assert!(
            lower <= 1.54 && upper >= 1.54,
            "expected ~1.54 A C-C bond, got [{lower}, {upper}]"
        );
    }

    #[test]
    fn benzene_bond_length_and_1_3_angle() {
        let m = mol("c1ccccc1");
        let mut bounds = build_bounds(&m).unwrap();
        smooth_bounds(&mut bounds).unwrap();

        let lower = get_lower(&bounds, 0, 1);
        let upper = get_upper(&bounds, 0, 1);
        assert!(
            lower <= 1.39 && upper >= 1.39,
            "expected ~1.39 A aromatic C-C bond, got [{lower}, {upper}]"
        );

        // 1-3 pair (atoms 0 and 2, through ring neighbor 1): for a ~120 deg
        // angle with ~1.39 A bonds, the opposite-side distance is
        // 1.39 * sqrt(3) =~ 2.408 A.
        let lower13 = get_lower(&bounds, 0, 2);
        let upper13 = get_upper(&bounds, 0, 2);
        let expected = 2.408;
        assert!(
            lower13 <= expected && upper13 >= expected,
            "1-3 bound [{lower13}, {upper13}] should bracket {expected}"
        );
    }

    #[test]
    fn cyclobutane_ring_angle_override() {
        // 4-membered ring: the (0,2) and (1,3) pairs are genuine 1-3 pairs
        // (not directly bonded), so the ring-size override (90 deg interior
        // angle) should apply instead of the generic sp3 default (109.5 deg).
        let m = mol("C1CCC1");
        let bounds = build_bounds(&m).unwrap();

        let bond_lower = get_lower(&bounds, 0, 1);
        let bond_upper = get_upper(&bounds, 0, 1);

        let _lower13 = get_lower(&bounds, 0, 2);
        let upper13 = get_upper(&bounds, 0, 2);

        // 90 deg (+-3 deg ring-override tolerance) -> diagonal close to
        // bond * sqrt(2) =~ 1.414 * bond. Compare against the exact
        // law-of-cosines value at the tolerance edge (93 deg) rather than a
        // naive sqrt(2) approximation, since the +-3 deg band alone shifts
        // the upper bound measurably.
        let ring_tol = 3.0_f64.to_radians();
        let expected_upper = law_of_cosines_distance(
            bond_upper,
            bond_upper,
            std::f64::consts::FRAC_PI_2 + ring_tol,
        );
        // 109.5 deg (generic sp3 default) would give =~ 1.633 * bond instead
        // of ~1.41-1.45 * bond for the 90 deg ring override - a clearly
        // distinguishable difference.
        let sp3_default_diagonal = bond_lower * 1.633;
        assert!(
            (upper13 - expected_upper).abs() < 1e-6,
            "expected ring override upper bound {expected_upper} (90+3 deg), got {upper13}"
        );
        assert!(
            upper13 < sp3_default_diagonal - 0.1,
            "ring override should be clearly tighter than the generic sp3 (109.5 deg) diagonal estimate {sp3_default_diagonal}, got upper={upper13}"
        );
    }

    #[test]
    fn allene_center_is_linear() {
        let m = mol("C=C=C");
        let bounds = build_bounds(&m).unwrap();

        let l_ik = get_lower(&bounds, 0, 1);
        let u_ik = get_upper(&bounds, 0, 1);
        let l_kj = get_lower(&bounds, 1, 2);
        let u_kj = get_upper(&bounds, 1, 2);
        let u13 = get_upper(&bounds, 0, 2);

        // 180 deg -> the opposite-side distance is exactly the sum of the two
        // bond upper bounds (law of cosines collapses to d1+d2 at theta=pi).
        let expected_upper = u_ik + u_kj;
        assert!(
            (u13 - expected_upper).abs() < 1e-6,
            "expected allene center near-linear (180 deg): got 1-3 upper={u13}, expected {expected_upper} (bonds [{l_ik},{u_ik}] [{l_kj},{u_kj}])"
        );
    }

    #[test]
    fn smoothing_keeps_fused_ring_bounds_feasible() {
        let m = mol("C1CC2CCCCC2C1");
        let mut bounds = build_bounds(&m).unwrap();
        smooth_bounds(&mut bounds).unwrap();

        let n = bounds.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    get_lower(&bounds, i, j) <= get_upper(&bounds, i, j) + 1e-6,
                    "bounds inconsistent for pair ({i}, {j})"
                );
            }
        }
    }
}

