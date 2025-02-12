//! Planarity restraint (Stage 2): nudges sp2 aromatic/conjugated atoms
//! toward coplanarity.
//!
//! Two mechanisms, both formulated as a harmonic restraint on the *signed
//! out-of-plane height* of an atom above a *fixed* reference plane (so
//! `dE/d(atom) = 2*K*height*normal` is exactly parallel to `normal` - the
//! atom can only ever move perpendicular to the reference plane, never
//! laterally within it):
//!
//! - [`RingPlane`]: every atom of a small (3-8 membered), fully-sp2
//!   fundamental ring shares *one* common best-fit plane (centroid + Newell's
//!   method normal, recomputed fresh each gradient evaluation - i.e. a
//!   frozen-and-reiterated reference, not differentiated through). This is
//!   what actually keeps a ring like benzene flat *and* regular: an earlier
//!   version gave each ring atom its own *local* plane from just its own 3
//!   neighbors, which let adjacent atoms disagree on "flat" and, combined
//!   with the bounds energy being exactly zero (zero gradient) anywhere
//!   inside its +/-10 deg angle tolerance, measurably distorted a regular
//!   hexagon into an irregular one (110-130 deg -> 105-131 deg) while barely
//!   reducing its out-of-plane RMSD.
//! - [`PlanarCenter`]: a local 3-neighbor plane restraint for sp2 atoms *not*
//!   covered by any such ring (e.g. a standalone amide/ketone carbon) -
//!   there's no ring-wide consistency to lose for an isolated centre.

use crate::molecule::{base::MoleculeCore, bond::BondOrder};
use crate::vector::Vector3D;
use std::collections::HashSet;

/// Harmonic force constant on `height^2`. Kept soft relative to the
/// bounds/chirality margins so planarity nudges rings flat without fighting
/// genuine bond-length/angle bounds when a ring is bridged or fused and
/// geometrically can't be perfectly flat.
const PLANARITY_K: f64 = 0.3;

pub(super) struct PlanarCenter {
    centre: usize,
    n0: usize,
    n1: usize,
    n2: usize,
}

/// A fundamental ring, all of whose atoms are sp2, in cyclic (bond-adjacency)
/// order - required for Newell's method to produce a meaningful normal.
pub(super) struct RingPlane {
    atoms: Vec<usize>,
}

impl RingPlane {
    /// Ring membership, in cyclic order - exposed so [`super::forcefield`]
    /// can skip generating a redundant torsion restraint for bonds this
    /// (stronger, whole-ring) restraint already governs.
    pub(super) fn atoms(&self) -> &[usize] {
        &self.atoms
    }
}

fn is_sp2_atom<T: MoleculeCore>(mol: &T, atom: usize) -> bool {
    let Some(bonds) = mol.get_atom_bonds(atom) else {
        return false;
    };
    if bonds.len() != 3 {
        return false;
    }
    let double_count = bonds.iter().filter(|b| b.bond_order() == BondOrder::Double).count();
    let has_triple = bonds.iter().any(|b| b.bond_order() == BondOrder::Triple);
    let has_aromatic_or_double = bonds
        .iter()
        .any(|b| matches!(b.bond_order(), BondOrder::Aromatic | BondOrder::Double));
    !has_triple && double_count < 2 && has_aromatic_or_double
}

/// Collects every sp2 atom's local-plane restraint, for use by callers that
/// only want the per-atom mechanism (see [`collect_planarity`] for the
/// combined, ring-aware collection normally used by the embedder).
fn collect_planar_centers<T: MoleculeCore>(mol: &T) -> Vec<PlanarCenter> {
    let mut centers = Vec::new();
    for centre in 0..mol.len() {
        if !is_sp2_atom(mol, centre) {
            continue;
        }
        let neighbors: Vec<usize> = mol.get_atom_bonds(centre).unwrap().iter().map(|b| b.target()).collect();
        centers.push(PlanarCenter { centre, n0: neighbors[0], n1: neighbors[1], n2: neighbors[2] });
    }
    centers
}

/// Walks `ring`'s bond-adjacency to recover a cyclic atom ordering. Every
/// member of a genuine simple cycle (which is what
/// [`MoleculeCore::compute_fundamental_cycles`] produces - spanning-tree path
/// plus one back edge) has exactly 2 neighbors that are themselves ring
/// members, so a greedy walk that never revisits an atom always closes back
/// onto the start.
fn ring_cycle_order<T: MoleculeCore>(mol: &T, ring: &bit_set::BitSet) -> Option<Vec<usize>> {
    let n = ring.len();
    let start = ring.iter().next()?;
    let mut order = Vec::with_capacity(n);
    order.push(start);
    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(start);
    let mut prev = None;
    let mut current = start;

    while order.len() < n {
        let next = mol.get_atom_bonds(current)?.iter().map(|b| b.target()).find(|&t| {
            ring.contains(t) && Some(t) != prev && !visited.contains(&t)
        })?;
        order.push(next);
        visited.insert(next);
        prev = Some(current);
        current = next;
    }

    let closes = mol
        .get_atom_bonds(current)?
        .iter()
        .any(|b| b.target() == start && Some(b.target()) != prev);
    if closes { Some(order) } else { None }
}

/// Collects fundamental rings of size 3-8 whose every member is sp2, in
/// cyclic order, plus a local [`PlanarCenter`] restraint for every sp2 atom
/// not covered by one of those rings.
pub(super) fn collect_planarity<T: MoleculeCore>(mol: &T) -> (Vec<RingPlane>, Vec<PlanarCenter>) {
    let cycles = mol.compute_fundamental_cycles();
    let mut rings = Vec::new();
    let mut in_a_ring = HashSet::new();

    for cycle in &cycles {
        if cycle.len() < 3 || cycle.len() > 8 {
            continue;
        }
        if !cycle.iter().all(|atom| is_sp2_atom(mol, atom)) {
            continue;
        }
        let Some(order) = ring_cycle_order(mol, cycle) else {
            continue;
        };
        in_a_ring.extend(order.iter().copied());
        rings.push(RingPlane { atoms: order });
    }

    let centers = collect_planar_centers(mol)
        .into_iter()
        .filter(|c| !in_a_ring.contains(&c.centre))
        .collect();

    (rings, centers)
}

/// Newell's method centroid + normal: robust for a near-planar (or even
/// mildly non-convex) polygon and cheap (`O(ring size)`, no eigen-solve).
/// `normal_i = (p_i - p_{i+1}) x (p_i + p_{i+1})` summed over cyclic edges;
/// this is `2x` the standard Newell sum but the extra constant factor is
/// irrelevant once the result is normalized.
fn ring_plane(positions: &[Vector3D], ring: &RingPlane) -> Option<(Vector3D, Vector3D)> {
    let n = ring.atoms.len();
    let mut centroid = Vector3D::default();
    let mut normal = Vector3D::default();
    for i in 0..n {
        let p = positions[ring.atoms[i]];
        let q = positions[ring.atoms[(i + 1) % n]];
        centroid += p;
        normal += Vector3D::new(
            (p.y - q.y) * (p.z + q.z),
            (p.z - q.z) * (p.x + q.x),
            (p.x - q.x) * (p.y + q.y),
        );
    }
    if normal.length() < 1e-6 {
        return None; // degenerate (collinear/coincident) ring positions
    }
    Some((centroid / n as f64, normal.normalize()))
}

fn out_of_plane_height(positions: &[Vector3D], c: &PlanarCenter) -> Option<(f64, Vector3D)> {
    let n0 = positions[c.n0];
    let n1 = positions[c.n1];
    let n2 = positions[c.n2];
    let normal = (n1 - n0).cross(&(n2 - n0));
    if normal.length() < 1e-6 {
        return None; // neighbors (near-)collinear: no well-defined plane
    }
    let normal = normal.normalize();
    let centroid = (n0 + n1 + n2) / 3.0;
    Some((normal.dot(&(positions[c.centre] - centroid)), normal))
}

/// Adds `PLANARITY_K * height^2` (and its gradient, applied only to the atom
/// being restrained toward its reference plane - the plane's own defining
/// points are treated as fixed for this term, see module docs) into `grad`
/// for every ring and local centre, returning the summed energy.
pub(super) fn planarity_energy_and_gradient(
    positions: &[Vector3D],
    rings: &[RingPlane],
    centers: &[PlanarCenter],
    grad: &mut [Vector3D],
) -> f64 {
    let mut energy = 0.0;
    for ring in rings {
        let Some((centroid, normal)) = ring_plane(positions, ring) else {
            continue;
        };
        for &atom in &ring.atoms {
            let height = normal.dot(&(positions[atom] - centroid));
            energy += PLANARITY_K * height * height;
            grad[atom] += normal * (2.0 * PLANARITY_K * height);
        }
    }
    for c in centers {
        let Some((height, normal)) = out_of_plane_height(positions, c) else {
            continue;
        };
        energy += PLANARITY_K * height * height;
        grad[c.centre] += normal * (2.0 * PLANARITY_K * height);
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_centre_gradient_descent_flattens_a_bent_centre() {
        let centers = vec![PlanarCenter { centre: 0, n0: 1, n1: 2, n2: 3 }];
        let mut positions = vec![
            Vector3D::new(0.0, 0.0, 0.3), // centre, pushed out of the n0/n1/n2 plane
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(-0.5, 0.87, 0.0),
            Vector3D::new(-0.5, -0.87, 0.0),
        ];

        let (h0, _) = out_of_plane_height(&positions, &centers[0]).unwrap();
        assert!(h0.abs() > 1e-3, "fixture should start non-planar");

        for _ in 0..200 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            planarity_energy_and_gradient(&positions, &[], &centers, &mut grad);
            for i in 0..positions.len() {
                positions[i] += -grad[i] * (0.5 / PLANARITY_K);
            }
        }

        let (h1, _) = out_of_plane_height(&positions, &centers[0]).unwrap();
        assert!(h1.abs() < 1e-3, "expected gradient descent to flatten the centre, |height|={}", h1.abs());
        // Only `centre` (index 0) should have moved - n0/n1/n2 are fixed by construction.
        assert_eq!(positions[1], Vector3D::new(1.0, 0.0, 0.0));
        assert_eq!(positions[2], Vector3D::new(-0.5, 0.87, 0.0));
        assert_eq!(positions[3], Vector3D::new(-0.5, -0.87, 0.0));
    }

    #[test]
    fn ring_gradient_descent_flattens_a_puckered_hexagon() {
        // A hexagon with alternating +/-0.3 z-displacement ("chair"-like pucker).
        let radius = 1.4;
        let mut positions: Vec<Vector3D> = (0..6)
            .map(|i| {
                let theta = std::f64::consts::PI / 3.0 * i as f64;
                let z = if i % 2 == 0 { 0.3 } else { -0.3 };
                Vector3D::new(radius * theta.cos(), radius * theta.sin(), z)
            })
            .collect();
        let ring = RingPlane { atoms: (0..6).collect() };

        let (_, normal0) = ring_plane(&positions, &ring).unwrap();
        let rmsd_before: f64 = {
            let (centroid, _) = ring_plane(&positions, &ring).unwrap();
            (positions.iter().map(|&p| normal0.dot(&(p - centroid)).powi(2)).sum::<f64>() / 6.0).sqrt()
        };
        assert!(rmsd_before > 0.2, "fixture should start meaningfully puckered");

        for _ in 0..500 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            planarity_energy_and_gradient(&positions, std::slice::from_ref(&ring), &[], &mut grad);
            for i in 0..positions.len() {
                positions[i] += -grad[i] * (0.5 / PLANARITY_K);
            }
        }

        let (centroid, normal) = ring_plane(&positions, &ring).unwrap();
        let rmsd_after = (positions.iter().map(|&p| normal.dot(&(p - centroid)).powi(2)).sum::<f64>() / 6.0).sqrt();
        assert!(rmsd_after < 1e-2, "expected the ring to flatten, RMSD={rmsd_after}");
    }
}
