//! UFF/MMFF-lite refinement terms (angle-bend, torsion) for the final
//! touch-up pass (see [`super`]'s module docs). Bond-stretch and van-der-
//! Waals-style non-bonded repulsion are *not* duplicated here - both are
//! already enforced by the bounds-violation energy in [`super::refine`] (a
//! tight +/-0.02 A flat bottom for 1-2 pairs, and a VDW-radius-based lower
//! bound for every non-bonded pair) - so this module only adds the two
//! terms distance geometry alone cannot express: a real restoring force
//! toward a *single* target angle (rather than "anywhere inside a wide
//! tolerance band is free", which is what let the cocaine phenyl ring in
//! testing relax into an irregular hexagon), and a torsion-periodicity
//! preference (staggered-vs-eclipsed, planar-vs-twisted) that a pairwise
//! distance bound has no way to represent at all.

use super::bounds::angle_bounds_for_triple;
use crate::molecule::utils::order_tuple;
use crate::molecule::{base::MoleculeCore, bond::BondOrder, chirality::ChiralClass};
use crate::vector::Vector3D;

/// Harmonic-in-cosine force constant, soft relative to the bounds/chirality
/// margins for the same reason `planarity::PLANARITY_K` is - see that
/// module's doc comment.
const K_ANGLE: f64 = 0.4;

/// Torsion barrier heights (soft, arbitrary energy units - *not* calibrated
/// to real kcal/mol values, chosen only to produce a stable, gentle
/// preference without dominating the bounds/stereo/planarity terms it runs
/// alongside in the final touch-up pass). See [`collect_torsions`] for the
/// classification these correspond to.
const V_CONJUGATED: f64 = 0.5;
const V_STAGGERED: f64 = 0.15;
const V_NEARFREE: f64 = 0.05;

/// Step size for the central finite-difference torsion gradient - see
/// [`torsion_energy_and_grad`] for why this term uses finite differences
/// instead of an analytic derivative.
const FD_STEP: f64 = 1e-5;

pub(super) struct AngleBend {
    i: usize,
    k: usize,
    j: usize,
    cos_theta0: f64,
}

pub(super) struct Torsion {
    i: usize,
    b: usize,
    c: usize,
    j: usize,
    n: f64,
    barrier: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hyb {
    Sp,
    Sp2,
    Sp3,
}

/// Same triple/double/aromatic bond-order classification tree as
/// `bounds::estimate_hybridization_angle` and `planarity::is_sp2_atom`, kept
/// as its own small local classifier here (each of those two modules
/// already does the same, for their own narrower needs) rather than a third
/// shared function.
fn hybridization<T: MoleculeCore>(mol: &T, atom: usize) -> Hyb {
    let Some(bonds) = mol.get_atom_bonds(atom) else {
        return Hyb::Sp3;
    };
    let double_count = bonds.iter().filter(|b| b.bond_order() == BondOrder::Double).count();
    let has_triple = bonds.iter().any(|b| b.bond_order() == BondOrder::Triple);
    let has_aromatic_or_double = bonds
        .iter()
        .any(|b| matches!(b.bond_order(), BondOrder::Aromatic | BondOrder::Double));
    if has_triple || double_count >= 2 {
        Hyb::Sp
    } else if has_aromatic_or_double {
        Hyb::Sp2
    } else {
        Hyb::Sp3
    }
}

/// Collects a harmonic-in-cosine angle-bend restraint for every bonded 1-3
/// triple `(i, k, j)` - the same triples `bounds::build_bounds` iterates -
/// targeting the midpoint of `angle_bounds_for_triple`'s `(min, max)`, which
/// already folds in the hybridization heuristic, ring-size override, and
/// square-planar trans/adjacent override.
pub(super) fn collect_angle_bends<T: MoleculeCore>(mol: &T) -> Vec<AngleBend> {
    let cycles = mol.compute_fundamental_cycles();
    let mut angles = Vec::new();
    for k in 0..mol.len() {
        let Some(bonds_k) = mol.get_atom_bonds(k) else {
            continue;
        };
        let neighbors: Vec<usize> = bonds_k.iter().map(|b| b.target()).collect();
        for a in 0..neighbors.len() {
            for b in (a + 1)..neighbors.len() {
                let i = neighbors[a];
                let j = neighbors[b];
                let (theta_min, theta_max) = angle_bounds_for_triple(mol, i, k, j, &cycles);
                let theta0 = 0.5 * (theta_min + theta_max);
                angles.push(AngleBend { i, k, j, cos_theta0: theta0.cos() });
            }
        }
    }
    angles
}

/// A bond whose ends are both tagged `CisTransClockwise`/
/// `CisTransCounterclockwise` already has a dedicated, tuned
/// `chirality::CisTransBond` restraint (see [`super::chirality`]) driving it
/// to a *specific* side (E or Z) - this module's generic, side-agnostic
/// torsion term (minima at both 0 and 180 deg) has nothing to add there, and
/// summing 4 redundant quadruples' worth of extra gradient on top of that
/// bond risks overpowering/fighting the tuned term instead of reinforcing
/// it (observed during development: it could pull a `F/C=C/F` fixture's
/// dihedral to a worse, near-90-deg saddle than leaving the bond to
/// `CisTransBond` alone).
fn is_explicit_cis_trans_bond<T: MoleculeCore>(mol: &T, b: usize, c: usize) -> bool {
    let Some(classes) = mol.chiral_classes() else {
        return false;
    };
    let is_tag = |class: ChiralClass| {
        matches!(class, ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise)
    };
    is_tag(classes[b]) && is_tag(classes[c])
}

/// Collects a periodic torsion restraint for every `(i, b, c, j)` quadruple
/// around each bond `(b, c)` - `i` ranges over `b`'s other neighbors, `j`
/// over `c`'s other neighbors. Skipped entirely when either bond end is `Sp`
/// (linear, no well-defined torsion angle to restrain around a linear
/// centre), when the bond already has an explicit
/// [`super::chirality::CisTransBond`] restraint (see
/// [`is_explicit_cis_trans_bond`]), or when the bond lies in *any* ring: a
/// ring bond is already geometrically constrained all the way around by
/// bond-length/angle bounds (plus, for small enough all-sp2 rings,
/// `planarity::RingPlane`'s stronger whole-ring shared-plane restraint), and
/// a rotation preference isn't meaningful the way it is for an acyclic bond
/// (you can't freely rotate a ring bond without breaking the ring) - adding
/// one anyway was observed during development to sometimes pull a fused
/// polycyclic aromatic's ring bond length below its bounds' lower limit.
/// Ring membership here comes from `MoleculeCore::find_cyclic_bonds_and_atoms`
/// directly (every bond in every cycle, via DFS back-edges) rather than
/// `RingPlane`'s membership list, since `RingPlane` only covers the
/// *fundamental*-cycle-basis rings up to 8 atoms - for a fused bicyclic,
/// that basis is not guaranteed to align with the two chemical rings (it
/// could just as well return one chemical ring plus the larger
/// whole-perimeter cycle), which would leave some genuine ring bonds
/// unprotected. See the module doc comment for why `phi0` is fixed at 0 for
/// every term (only periodicity is modeled, not a specific rotamer).
pub(super) fn collect_torsions<T: MoleculeCore>(mol: &T) -> Vec<Torsion> {
    let (_, cyclic_bonds) = mol.find_cyclic_bonds_and_atoms();
    let mut torsions = Vec::new();
    for b in 0..mol.len() {
        let Some(bonds_b) = mol.get_atom_bonds(b) else {
            continue;
        };
        for bond_bc in bonds_b {
            let c = bond_bc.target();
            if c <= b {
                continue; // visit each central bond once, from its lower-indexed end
            }
            if is_explicit_cis_trans_bond(mol, b, c) || cyclic_bonds.contains(&order_tuple(b, c)) {
                continue;
            }
            let hyb_b = hybridization(mol, b);
            let hyb_c = hybridization(mol, c);
            if hyb_b == Hyb::Sp || hyb_c == Hyb::Sp {
                continue;
            }
            let (n, barrier) = match (hyb_b, hyb_c) {
                (Hyb::Sp2, Hyb::Sp2) => (2.0, V_CONJUGATED),
                (Hyb::Sp3, Hyb::Sp3) => (3.0, V_STAGGERED),
                _ => (6.0, V_NEARFREE),
            };
            let Some(bonds_c) = mol.get_atom_bonds(c) else {
                continue;
            };
            for i_bond in bonds_b {
                let i = i_bond.target();
                if i == c {
                    continue;
                }
                for j_bond in bonds_c {
                    let j = j_bond.target();
                    if j == b || j == i {
                        continue;
                    }
                    torsions.push(Torsion { i, b, c, j, n, barrier });
                }
            }
        }
    }
    torsions
}

fn angle_bend_energy_and_grad(positions: &[Vector3D], a: &AngleBend, grad: &mut [Vector3D]) -> f64 {
    let u = positions[a.i] - positions[a.k];
    let v = positions[a.j] - positions[a.k];
    let ru = u.length();
    let rv = v.length();
    if ru < 1e-8 || rv < 1e-8 {
        return 0.0; // degenerate (coincident points): no well-defined angle
    }

    let cos_theta = (u.dot(&v) / (ru * rv)).clamp(-1.0, 1.0);
    let delta = cos_theta - a.cos_theta0;
    let energy = 0.5 * K_ANGLE * delta * delta;

    // d(cos theta)/du = v/(|u||v|) - cos(theta)*u/|u|^2, symmetric for dv;
    // p_k appears (with a factor of -1) in both u=p_i-p_k and v=p_j-p_k, so
    // its gradient is the negated sum of the other two.
    let dcos_du = v * (1.0 / (ru * rv)) - u * (cos_theta / (ru * ru));
    let dcos_dv = u * (1.0 / (ru * rv)) - v * (cos_theta / (rv * rv));

    let coeff = K_ANGLE * delta;
    let grad_i = dcos_du * coeff;
    let grad_j = dcos_dv * coeff;
    let grad_k = -(grad_i + grad_j);

    grad[a.i] += grad_i;
    grad[a.j] += grad_j;
    grad[a.k] += grad_k;

    energy
}

/// The `i-b-c-j` dihedral angle in `(-pi, pi]`, via the numerically robust
/// "praxeolitic" formula (projects the `i-b` and `c-j` bond vectors into the
/// plane perpendicular to the central `b-c` bond, then takes the angle
/// between those projections). `None` if any of the three bonds involved is
/// degenerate (near-zero length) or the `i-b-c` / `b-c-j` angle is near
/// linear (the projection is then ill-defined) - both cases have no
/// meaningful torsion angle to restrain.
fn dihedral_angle(positions: &[Vector3D], i: usize, b: usize, c: usize, j: usize) -> Option<f64> {
    let b0 = positions[i] - positions[b];
    let b1 = positions[c] - positions[b];
    let b2 = positions[j] - positions[c];
    let b1_len = b1.length();
    if b1_len < 1e-8 {
        return None;
    }
    let b1_hat = b1 * (1.0 / b1_len);

    let v = b0 - b1_hat * b0.dot(&b1_hat);
    let w = b2 - b1_hat * b2.dot(&b1_hat);
    if v.length() < 1e-8 || w.length() < 1e-8 {
        return None;
    }

    let x = v.dot(&w);
    let y = b1_hat.cross(&v).dot(&w);
    Some(y.atan2(x))
}

fn torsion_energy(phi: f64, t: &Torsion) -> f64 {
    0.5 * t.barrier * (1.0 - (t.n * phi).cos())
}

fn nudge(v: Vector3D, axis: usize, delta: f64) -> Vector3D {
    match axis {
        0 => Vector3D::new(v.x + delta, v.y, v.z),
        1 => Vector3D::new(v.x, v.y + delta, v.z),
        _ => Vector3D::new(v.x, v.y, v.z + delta),
    }
}

/// Energy plus a central-finite-difference gradient (in `[grad_i, grad_b,
/// grad_c, grad_j]` order) for one torsion. The gradient is computed by
/// finite difference rather than the analytic 4-body dihedral derivative
/// (one of the most notoriously sign-error-prone formulas in force-field
/// code) since this term only runs in the final O(iterations) touch-up
/// pass, which is already dominated by the O(n^2) bounds/planarity pair
/// loop - a few hundred torsions x 12 cheap extra scalar evaluations each is
/// negligible next to that, so there is no performance reason to take on the
/// derivation risk.
fn torsion_energy_and_grad(positions: &[Vector3D], t: &Torsion) -> Option<(f64, [Vector3D; 4])> {
    let phi = dihedral_angle(positions, t.i, t.b, t.c, t.j)?;
    let energy = torsion_energy(phi, t);

    let atoms = [t.i, t.b, t.c, t.j];
    let mut grad = [Vector3D::default(); 4];
    let mut perturbed = positions.to_vec();

    for (slot, &atom) in atoms.iter().enumerate() {
        let original = positions[atom];
        let mut g = Vector3D::default();
        for axis in 0..3 {
            perturbed[atom] = nudge(original, axis, FD_STEP);
            let e_plus = dihedral_angle(&perturbed, t.i, t.b, t.c, t.j).map(|p| torsion_energy(p, t));
            perturbed[atom] = nudge(original, axis, -FD_STEP);
            let e_minus = dihedral_angle(&perturbed, t.i, t.b, t.c, t.j).map(|p| torsion_energy(p, t));
            perturbed[atom] = original;

            let (Some(ep), Some(em)) = (e_plus, e_minus) else {
                continue;
            };
            let d = (ep - em) / (2.0 * FD_STEP);
            match axis {
                0 => g.x += d,
                1 => g.y += d,
                _ => g.z += d,
            }
        }
        grad[slot] = g;
    }

    Some((energy, grad))
}

/// Adds the angle-bend and torsion energy (and gradient) into `grad` for
/// every collected restraint, returning the summed energy - the combined
/// entry point [`super::refine::refine`] threads through alongside
/// `stereo`/`planarity`.
pub(super) fn forcefield_energy_and_gradient(
    positions: &[Vector3D],
    angles: &[AngleBend],
    torsions: &[Torsion],
    grad: &mut [Vector3D],
) -> f64 {
    let mut energy = 0.0;
    for a in angles {
        energy += angle_bend_energy_and_grad(positions, a, grad);
    }
    for t in torsions {
        if let Some((e, atom_grad)) = torsion_energy_and_grad(positions, t) {
            energy += e;
            let atoms = [t.i, t.b, t.c, t.j];
            for (slot, &atom) in atoms.iter().enumerate() {
                grad[atom] += atom_grad[slot];
            }
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_bend_gradient_descent_relaxes_to_target() {
        // Distorted ~100 deg angle, target 109.5 deg (tetrahedral).
        let angle = AngleBend { i: 1, k: 0, j: 2, cos_theta0: 109.5_f64.to_radians().cos() };
        let mut positions = vec![
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(-0.17, 0.98, 0.0), // ~100 deg from the first bond
        ];

        for _ in 0..500 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            angle_bend_energy_and_grad(&positions, &angle, &mut grad);
            for i in 0..positions.len() {
                positions[i] += -grad[i] * (1.0 / K_ANGLE);
            }
        }

        let u = positions[1] - positions[0];
        let v = positions[2] - positions[0];
        let cos_theta = u.dot(&v) / (u.length() * v.length());
        assert!(
            (cos_theta - 109.5_f64.to_radians().cos()).abs() < 1e-3,
            "expected angle to relax to 109.5 deg, cos(theta)={cos_theta}"
        );
    }

    #[test]
    fn angle_bend_gradient_matches_finite_difference() {
        let angle = AngleBend { i: 0, k: 1, j: 2, cos_theta0: 100.0_f64.to_radians().cos() };
        let positions = vec![
            Vector3D::new(1.3, -0.2, 0.4),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(-0.5, 1.1, 0.3),
        ];

        let mut analytic = vec![Vector3D::default(); positions.len()];
        angle_bend_energy_and_grad(&positions, &angle, &mut analytic);

        let h = 1e-6;
        for atom in 0..positions.len() {
            for axis in 0..3 {
                let mut plus = positions.clone();
                plus[atom] = nudge(positions[atom], axis, h);
                let mut minus = positions.clone();
                minus[atom] = nudge(positions[atom], axis, -h);

                let mut scratch = vec![Vector3D::default(); positions.len()];
                let e_plus = angle_bend_energy_and_grad(&plus, &angle, &mut scratch);
                let mut scratch = vec![Vector3D::default(); positions.len()];
                let e_minus = angle_bend_energy_and_grad(&minus, &angle, &mut scratch);

                let fd = (e_plus - e_minus) / (2.0 * h);
                let analytic_component = match axis {
                    0 => analytic[atom].x,
                    1 => analytic[atom].y,
                    _ => analytic[atom].z,
                };
                assert!(
                    (fd - analytic_component).abs() < 1e-4,
                    "atom {atom} axis {axis}: analytic={analytic_component}, fd={fd}"
                );
            }
        }
    }

    #[test]
    fn dihedral_angle_matches_known_configurations() {
        // i-b-c-j forming a known 90 deg dihedral: i,b,c in the xy-plane,
        // j displaced purely along z from c relative to the b-c direction.
        let i = Vector3D::new(1.0, 0.0, 0.0);
        let b = Vector3D::new(0.0, 0.0, 0.0);
        let c = Vector3D::new(0.0, 1.0, 0.0);
        let j_cis = Vector3D::new(1.0, 1.0, 0.0); // eclipsed with i -> phi = 0
        let j_trans = Vector3D::new(-1.0, 1.0, 0.0); // phi = 180 deg
        let j_perp = Vector3D::new(0.0, 1.0, 1.0); // phi = +/-90 deg

        let positions = |j: Vector3D| vec![i, b, c, j];

        let phi_cis = dihedral_angle(&positions(j_cis), 0, 1, 2, 3).unwrap();
        assert!(phi_cis.abs() < 1e-6, "expected ~0 deg, got {} deg", phi_cis.to_degrees());

        let phi_trans = dihedral_angle(&positions(j_trans), 0, 1, 2, 3).unwrap();
        assert!(
            (phi_trans.abs() - std::f64::consts::PI).abs() < 1e-6,
            "expected ~180 deg, got {} deg",
            phi_trans.to_degrees()
        );

        let phi_perp = dihedral_angle(&positions(j_perp), 0, 1, 2, 3).unwrap();
        assert!(
            (phi_perp.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "expected ~90 deg, got {} deg",
            phi_perp.to_degrees()
        );
    }

    #[test]
    fn torsion_gradient_descent_reaches_a_minimum() {
        // `phi0` is fixed at 0 for every torsion term (see the module doc
        // comment) - the potential's minima are 3 evenly spaced angles, not
        // one specific "staggered" rotamer, so the meaningful invariant to
        // test is "energy reaches ~0 (its global minimum) from an arbitrary
        // non-minimum start", not "converges to a particular angle".
        let t = Torsion { i: 0, b: 1, c: 2, j: 3, n: 3.0, barrier: V_STAGGERED };
        let mut positions = vec![
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.5, 1.0, 0.5), // arbitrary, not at a minimum or maximum
        ];

        let phi_start = dihedral_angle(&positions, t.i, t.b, t.c, t.j).unwrap();
        let energy_start = torsion_energy(phi_start, &t);
        assert!(energy_start > 1e-3, "fixture should start away from a minimum");

        for _ in 0..2000 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            if let Some((_, atom_grad)) = torsion_energy_and_grad(&positions, &t) {
                let atoms = [t.i, t.b, t.c, t.j];
                for (slot, &atom) in atoms.iter().enumerate() {
                    grad[atom] += atom_grad[slot];
                }
            }
            for k in 0..positions.len() {
                positions[k] += -grad[k] * 0.05;
            }
        }

        let phi_end = dihedral_angle(&positions, t.i, t.b, t.c, t.j).unwrap();
        let energy_end = torsion_energy(phi_end, &t);
        assert!(
            energy_end < 1e-6,
            "expected energy to reach ~0 (a minimum), got {energy_end} (phi={} deg)",
            phi_end.to_degrees()
        );
    }
}
