//! Planarity restraint (Stage 2): nudges sp2 aromatic/conjugated atoms
//! toward coplanarity.
//!
//! Two mechanisms, both a harmonic restraint on the *signed out-of-plane
//! height* of an atom above a plane its neighbours define:
//!
//! - [`RingPlane`]: every atom of a small (3-8 membered), fully-sp2 ring
//!   shares *one* common best-fit plane (centroid + Newell's method normal).
//!   This is what actually keeps a ring like benzene flat *and* regular: an
//!   earlier version gave each ring atom its own *local* plane from just its
//!   own 3 neighbors, which let adjacent atoms disagree on "flat" and,
//!   combined with the bounds energy being exactly zero (zero gradient)
//!   anywhere inside its +/-10 deg angle tolerance, measurably distorted a
//!   regular hexagon into an irregular one (110-130 deg -> 105-131 deg)
//!   while barely reducing its out-of-plane RMSD.
//! - [`PlanarCenter`]: a local 3-neighbor plane restraint for sp2 atoms *not*
//!   covered by any such ring (e.g. a standalone amide/ketone carbon) -
//!   there's no ring-wide consistency to lose for an isolated centre.
//!
//! Both gradients are exact, plane included. They used to treat the plane
//! as fixed -- a frozen reference, recomputed at every evaluation but not
//! differentiated through -- which is fine for the plain gradient descent
//! that was written for, and wrong for the quasi-Newton minimiser that runs
//! it now: the line search reads its slope off the gradient and its
//! sufficient-decrease test off the energy, and BFGS builds its curvature
//! from gradient differences, so a gradient that is not the energy's
//! derivative misleads all three. For the ring term the missing part is not
//! small -- a Newell normal is not the least-squares normal, so the energy
//! is not stationary in it -- and for a centre it was the whole of the
//! neighbours' share.
//!
//! The rings are [`crate::molecule::topology::smallest_rings`], the same set
//! the bounds, the force field and the torsion search read. They were the
//! spanning-tree cycle basis, which for a peri-fused system holds the
//! perimeter rather than the small rings: pyrene's basis is 6, 6, 12 and 14,
//! so two of its four aromatic rings had no planarity term at all, and
//! coronene lost four of seven.

use crate::molecule::{base::MoleculeCore, bond::BondOrder};
use crate::vector::Vector3D;
use std::collections::HashSet;

/// Harmonic force constant on `height^2`. Kept soft relative to the
/// bounds/chirality margins so planarity nudges rings flat without fighting
/// genuine bond-length/angle bounds when a ring is bridged or fused and
/// geometrically can't be perfectly flat.
const PLANARITY_K: f64 = 0.3;

/// Force constant for an aromatic ring, which unlike the general sp2 case is
/// planar by definition rather than by tendency -- so the reason
/// [`PLANARITY_K`] is soft (not fighting genuine bond/angle bounds in a
/// bridged or fused system) does not apply, and there is nothing legitimate
/// for it to lose an argument with.
///
/// Distance bounds alone do not get there. `bounds::apply_aromatic_ring_bounds`
/// pins the chords across the ring, but a real aromatic ring is not a
/// regular polygon, and the slack its bonds and angles genuinely need is
/// also enough slack to pucker: with the chords pinned and this constant
/// left at 0.3, a biphenyl ring still came out 0.11 A out of plane against
/// RDKit ETKDG's 0.0002.
const AROMATIC_PLANARITY_K: f64 = 30.0;

pub(super) struct PlanarCenter {
    centre: usize,
    n0: usize,
    n1: usize,
    n2: usize,
}

/// A small ring, all of whose atoms are sp2, in cyclic (bond-adjacency)
/// order - required for Newell's method to produce a meaningful normal.
pub(super) struct RingPlane {
    atoms: Vec<usize>,
    /// Force constant for this particular ring. Aromatic rings get
    /// [`AROMATIC_PLANARITY_K`]; everything else keeps [`PLANARITY_K`].
    stiffness: f64,
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

/// Collects the smallest rings of size 3-8 whose every member is sp2 (or
/// that are aromatic), in cyclic order, plus a local [`PlanarCenter`]
/// restraint for every sp2 atom not covered by one of those rings.
pub(super) fn collect_planarity<T: MoleculeCore>(mol: &T) -> (Vec<RingPlane>, Vec<PlanarCenter>) {
    let mut rings = Vec::new();
    let mut in_a_ring = HashSet::new();

    // Already in bond order, which is what Newell's method needs.
    for order in crate::molecule::topology::smallest_rings(mol) {
        if order.len() < 3 || order.len() > 8 {
            continue;
        }
        // Aromatic first, and on its own terms. [`is_sp2_atom`] demands three
        // bonds, which the two-coordinate heteroatom of a pyridine, furan or
        // thiophene does not have -- so requiring every ring atom to pass it
        // left those rings with no planarity restraint whatsoever, and
        // pyridine embedded 0.4 A out of plane. Aromaticity already implies
        // planarity without asking about coordination.
        let aromatic = cycle_is_aromatic(mol, &order);
        if !aromatic && !order.iter().all(|&atom| is_sp2_atom(mol, atom)) {
            continue;
        }
        in_a_ring.extend(order.iter().copied());
        rings.push(RingPlane {
            atoms: order,
            stiffness: if aromatic { AROMATIC_PLANARITY_K } else { PLANARITY_K },
        });
    }

    let centers = collect_planar_centers(mol)
        .into_iter()
        .filter(|c| !in_a_ring.contains(&c.centre))
        .collect();

    (rings, centers)
}

/// Whether every bond around `ring` (given in cyclic order) is aromatic.
fn cycle_is_aromatic<T: MoleculeCore>(mol: &T, ring: &[usize]) -> bool {
    ring.iter().enumerate().all(|(index, &atom)| {
        let next = ring[(index + 1) % ring.len()];
        mol.get_atom_bonds(atom).is_some_and(|bonds| {
            bonds
                .iter()
                .any(|bond| bond.target() == next && bond.bond_order() == BondOrder::Aromatic)
        })
    })
}

/// Newell's method centroid + normal: robust for a near-planar (or even
/// mildly non-convex) polygon and cheap (`O(ring size)`, no eigen-solve).
///
/// The raw normal is `N = sum_i (p_i - c) x (p_{i+1} - c)` over cyclic
/// edges -- the shoelace form of Newell's sum, which it equals for a closed
/// polygon, taken about the centroid so the cross products stay small. It
/// comes back unit, with `|N|`, which the gradient needs.
fn ring_plane(positions: &[Vector3D], ring: &RingPlane) -> Option<(Vector3D, Vector3D, f64)> {
    let n = ring.atoms.len();
    let centroid = ring
        .atoms
        .iter()
        .fold(Vector3D::default(), |sum, &atom| sum + positions[atom])
        / n as f64;
    let mut normal = Vector3D::default();
    for i in 0..n {
        let p = positions[ring.atoms[i]] - centroid;
        let q = positions[ring.atoms[(i + 1) % n]] - centroid;
        normal += p.cross(&q);
    }
    let length = normal.length();
    if length < 1e-6 {
        return None; // degenerate (collinear/coincident) ring positions
    }
    Some((centroid, normal * (1.0 / length), length))
}

#[cfg(test)]
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

/// `K * height^2` for every ring atom and local centre, and its gradient,
/// into `grad`, returning the summed energy.
///
/// The gradient includes the plane's own dependence on the atoms that
/// define it. Through the centroid that dependence vanishes -- the heights
/// about a centroid sum to zero -- but through the normal it does not. With
/// `n = N / |N|`, `dE/dN` is `(I - n n^T) v / |N|` where `v` is `dE/dn`, and
/// `N` is a sum of cross products whose derivative is one cross product per
/// atom.
pub(super) fn planarity_energy_and_gradient(
    positions: &[Vector3D],
    rings: &[RingPlane],
    centers: &[PlanarCenter],
    grad: &mut [Vector3D],
) -> f64 {
    let mut energy = 0.0;
    for ring in rings {
        let Some((centroid, normal, length)) = ring_plane(positions, ring) else {
            continue;
        };
        let mut along_normal = Vector3D::default();
        for &atom in &ring.atoms {
            let offset = positions[atom] - centroid;
            let height = normal.dot(&offset);
            energy += ring.stiffness * height * height;
            grad[atom] += normal * (2.0 * ring.stiffness * height);
            along_normal += offset * (2.0 * ring.stiffness * height);
        }
        // `N = sum_i p_i x p_{i+1}`, so `d(w . N)/dp_i = (p_{i+1} - p_{i-1}) x w`.
        let tilt = (along_normal - normal * along_normal.dot(&normal)) * (1.0 / length);
        let n = ring.atoms.len();
        for i in 0..n {
            let next = positions[ring.atoms[(i + 1) % n]];
            let previous = positions[ring.atoms[(i + n - 1) % n]];
            grad[ring.atoms[i]] += (next - previous).cross(&tilt);
        }
    }
    for c in centers {
        let (n0, n1, n2) = (positions[c.n0], positions[c.n1], positions[c.n2]);
        let (first, second) = (n1 - n0, n2 - n0);
        let raw = first.cross(&second);
        let length = raw.length();
        if length < 1e-6 {
            continue; // neighbors (near-)collinear: no well-defined plane
        }
        let normal = raw * (1.0 / length);
        let offset = positions[c.centre] - (n0 + n1 + n2) / 3.0;
        let height = normal.dot(&offset);
        energy += PLANARITY_K * height * height;

        let d_height = 2.0 * PLANARITY_K * height;
        // The centre directly, and each neighbour through the centroid.
        grad[c.centre] += normal * d_height;
        let through_centroid = normal * (-d_height / 3.0);
        // And each neighbour through the normal: `N = first x second`.
        let tilt = (offset - normal * height) * (d_height / length);
        let d_first = second.cross(&tilt);
        let d_second = tilt.cross(&first);
        grad[c.n1] += d_first + through_centroid;
        grad[c.n2] += d_second + through_centroid;
        grad[c.n0] += -(d_first + d_second) + through_centroid;
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central differences of the summed energy, against the analytic
    /// gradient, at every coordinate.
    fn assert_gradient_matches(positions: &[Vector3D], rings: &[RingPlane], centers: &[PlanarCenter]) {
        let energy = |p: &[Vector3D]| {
            let mut discard = vec![Vector3D::default(); p.len()];
            planarity_energy_and_gradient(p, rings, centers, &mut discard)
        };
        let mut analytic = vec![Vector3D::default(); positions.len()];
        planarity_energy_and_gradient(positions, rings, centers, &mut analytic);

        const STEP: f64 = 1e-6;
        for atom in 0..positions.len() {
            for axis in 0..3 {
                let nudged = |delta: f64| {
                    let mut p = positions.to_vec();
                    match axis {
                        0 => p[atom].x += delta,
                        1 => p[atom].y += delta,
                        _ => p[atom].z += delta,
                    }
                    energy(&p)
                };
                let numeric = (nudged(STEP) - nudged(-STEP)) / (2.0 * STEP);
                let exact = [analytic[atom].x, analytic[atom].y, analytic[atom].z][axis];
                assert!(
                    (numeric - exact).abs() < 1e-5 * numeric.abs().max(1.0),
                    "atom {atom} axis {axis}: analytic {exact}, numeric {numeric}"
                );
            }
        }
    }

    #[test]
    fn local_centre_gradient_matches_finite_difference() {
        let centers = vec![PlanarCenter { centre: 0, n0: 1, n1: 2, n2: 3 }];
        let positions = vec![
            Vector3D::new(0.1, -0.2, 0.35),
            Vector3D::new(1.3, 0.1, -0.1),
            Vector3D::new(-0.6, 1.1, 0.2),
            Vector3D::new(-0.4, -1.2, 0.15),
        ];
        assert_gradient_matches(&positions, &[], &centers);
    }

    #[test]
    fn ring_gradient_matches_finite_difference() {
        // An irregular, puckered hexagon away from the origin, so that
        // neither the centroid nor the normal is special.
        let positions: Vec<Vector3D> = (0..6)
            .map(|i| {
                let theta = std::f64::consts::PI / 3.0 * i as f64 + 0.1 * (i * i) as f64;
                let z = [0.3, -0.25, 0.1, -0.35, 0.2, -0.05][i];
                Vector3D::new(3.0 + 1.4 * theta.cos(), -2.0 + 1.3 * theta.sin(), 1.0 + z)
            })
            .collect();
        let ring = RingPlane { atoms: vec![0, 1, 2, 3, 4, 5], stiffness: AROMATIC_PLANARITY_K };
        assert_gradient_matches(&positions, std::slice::from_ref(&ring), &[]);
    }

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
            // A restraint on a shape exerts no net force.
            let net = grad.iter().fold(Vector3D::default(), |sum, &g| sum + g);
            assert!(net.length() < 1e-12, "net force {net:?}");
            for i in 0..positions.len() {
                positions[i] += -grad[i] * (0.25 / PLANARITY_K);
            }
        }

        let (h1, _) = out_of_plane_height(&positions, &centers[0]).unwrap();
        assert!(h1.abs() < 1e-3, "expected gradient descent to flatten the centre, |height|={}", h1.abs());
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
        let ring = RingPlane { atoms: (0..6).collect(), stiffness: PLANARITY_K };

        let rms_height = |positions: &[Vector3D]| {
            let (centroid, normal, _) = ring_plane(positions, &ring).unwrap();
            (positions.iter().map(|&p| normal.dot(&(p - centroid)).powi(2)).sum::<f64>() / 6.0).sqrt()
        };
        let rmsd_before = rms_height(&positions);
        assert!(rmsd_before > 0.2, "fixture should start meaningfully puckered");

        for _ in 0..500 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            planarity_energy_and_gradient(&positions, std::slice::from_ref(&ring), &[], &mut grad);
            for i in 0..positions.len() {
                positions[i] += -grad[i] * (0.5 / PLANARITY_K);
            }
        }

        let rmsd_after = rms_height(&positions);
        assert!(rmsd_after < 1e-2, "expected the ring to flatten, RMSD={rmsd_after}");
    }

    /// Every aromatic ring of a peri-fused system gets its own plane, which a
    /// spanning-tree cycle basis does not give: pyrene's is 6, 6, 12, 14.
    #[test]
    fn every_ring_of_pyrene_is_restrained() {
        let mut pyrene =
            crate::molecule::molecular_graph::MolecularGraph::from_smiles_single("c1cc2ccc3cccc4ccc(c1)c2c34")
                .expect("pyrene parses");
        pyrene.add_hydrogens();
        let (rings, centers) = collect_planarity(&pyrene);
        assert_eq!(rings.len(), 4, "pyrene has four aromatic rings");
        assert!(rings.iter().all(|ring| ring.atoms.len() == 6));
        assert!(centers.is_empty(), "every sp2 carbon is in a restrained ring");
    }
}
