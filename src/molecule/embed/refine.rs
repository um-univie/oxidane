//! Gradient-based refinement against a bounds-violation energy.
//!
//! `E = sum_{i<j} [ max(0, d^2 - u^2)^2 / u^2 + max(0, l^2 - d^2)^2 / l^2 ]`
//! (squared-distance formulation, avoids a `sqrt` in the O(n^2) inner loop).
//! Stage 1 refines in plain 3D: the classical MDS step embeds into 4D for a
//! better initial fit, but the 4th dimension only earns its keep once a
//! chirality-consistency term exists to escape into it (the ETKDG "4D
//! flatten" trick, Stage 2) - without that term the 4th coordinate has
//! nothing driving it to zero, so keeping it around and then silently
//! dropping it before returning `Molecule3D` would leave the 3D projection
//! satisfying bounds only in 4D, not in 3D. The `w` component from MDS is
//! discarded by the caller before refinement starts.
//!
//! Steepest descent with Armijo backtracking line search is used rather than
//! CG/L-BFGS: no optimizer infrastructure exists yet in this crate, and
//! steepest descent's monotonic-decrease invariant is trivially testable.
//! Upgrading the search direction to Polak-Ribiere CG later is a near
//! drop-in change reusing the same gradient and line search.

use super::bounds;
use super::chirality::StereoConstraints;
use super::forcefield::{AngleBend, Torsion};
use super::planarity::{PlanarCenter, RingPlane};
use crate::vector::Vector3D;
use ndarray::Array2;

const ARMIJO_C1: f64 = 1e-4;
const MAX_LINE_SEARCH_SHRINKS: usize = 20;
const GRADIENT_NORM_TOLERANCE: f64 = 1e-8;

/// Runs gradient-descent refinement in place, returning the final energy
/// (bounds-violation energy plus the [`chirality`] consistency term).
pub(super) fn refine(
    positions: &mut [Vector3D],
    bounds_matrix: &Array2<f64>,
    stereo: &StereoConstraints,
    rings: &[RingPlane],
    planar: &[PlanarCenter],
    angles: &[AngleBend],
    torsions: &[Torsion],
    max_iterations: usize,
) -> f64 {
    let (mut energy, mut grad_pos) =
        energy_and_gradient(positions, bounds_matrix, stereo, rings, planar, angles, torsions);

    for _ in 0..max_iterations {
        let grad_norm_sq: f64 = grad_pos.iter().map(|g| g.dot(g)).sum::<f64>();
        if grad_norm_sq < GRADIENT_NORM_TOLERANCE {
            break;
        }

        let mut alpha = 1.0;
        let mut accepted = false;
        let mut trial_positions = positions.to_vec();

        for _ in 0..MAX_LINE_SEARCH_SHRINKS {
            for i in 0..positions.len() {
                trial_positions[i] = positions[i] + (-grad_pos[i] * alpha);
            }
            let trial_energy =
                energy_only(&trial_positions, bounds_matrix, stereo, rings, planar, angles, torsions);
            if trial_energy <= energy - ARMIJO_C1 * alpha * grad_norm_sq {
                positions.copy_from_slice(&trial_positions);
                energy = trial_energy;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            break;
        }
        let (next_energy, next_grad_pos) =
            energy_and_gradient(positions, bounds_matrix, stereo, rings, planar, angles, torsions);
        energy = next_energy;
        grad_pos = next_grad_pos;
    }

    energy
}

fn energy_only(
    positions: &[Vector3D],
    bounds_matrix: &Array2<f64>,
    stereo: &StereoConstraints,
    rings: &[RingPlane],
    planar: &[PlanarCenter],
    angles: &[AngleBend],
    torsions: &[Torsion],
) -> f64 {
    let n = positions.len();
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let (e, _) = pair_energy_and_grad(positions, bounds_matrix, i, j);
            energy += e;
        }
    }
    let mut scratch = vec![Vector3D::default(); n];
    energy += stereo.energy_and_gradient(positions, &mut scratch);
    energy += super::planarity::planarity_energy_and_gradient(positions, rings, planar, &mut scratch);
    energy += super::forcefield::forcefield_energy_and_gradient(positions, angles, torsions, &mut scratch);
    energy
}

fn energy_and_gradient(
    positions: &[Vector3D],
    bounds_matrix: &Array2<f64>,
    stereo: &StereoConstraints,
    rings: &[RingPlane],
    planar: &[PlanarCenter],
    angles: &[AngleBend],
    torsions: &[Torsion],
) -> (f64, Vec<Vector3D>) {
    let n = positions.len();
    let mut energy = 0.0;
    let mut grad_pos = vec![Vector3D::default(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            let (e, g_xyz) = pair_energy_and_grad(positions, bounds_matrix, i, j);
            energy += e;
            grad_pos[i] += g_xyz;
            grad_pos[j] += -g_xyz;
        }
    }

    energy += stereo.energy_and_gradient(positions, &mut grad_pos);
    energy += super::planarity::planarity_energy_and_gradient(positions, rings, planar, &mut grad_pos);
    energy += super::forcefield::forcefield_energy_and_gradient(positions, angles, torsions, &mut grad_pos);

    (energy, grad_pos)
}

/// Energy contribution of pair `(i, j)` plus `dE/dp_i`. `dE/dp_j` is always
/// the negation of `dE/dp_i` by symmetry of `d^2`.
fn pair_energy_and_grad(
    positions: &[Vector3D],
    bounds_matrix: &Array2<f64>,
    i: usize,
    j: usize,
) -> (f64, Vector3D) {
    let lower = bounds::get_lower(bounds_matrix, i, j);
    let upper = bounds::get_upper(bounds_matrix, i, j);
    let diff = positions[i] - positions[j];
    let d2 = diff.dot(&diff);
    let u2 = upper * upper;
    let l2 = lower * lower;

    let mut energy = 0.0;
    let mut grad_xyz = Vector3D::default();

    if d2 > u2 {
        let excess = d2 - u2;
        energy += (excess * excess) / u2;
        let coeff = 4.0 * excess / u2;
        grad_xyz += diff * coeff;
    }
    if d2 < l2 {
        let deficit = l2 - d2;
        energy += (deficit * deficit) / l2;
        let coeff = 4.0 * deficit / l2;
        grad_xyz += -(diff * coeff);
    }

    (energy, grad_xyz)
}
