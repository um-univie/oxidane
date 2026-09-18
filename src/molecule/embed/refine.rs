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
//! Polak-Ribiere conjugate gradient with an Armijo backtracking line
//! search, falling back to steepest descent whenever the conjugate
//! direction stops pointing downhill -- so the monotonic-decrease invariant
//! steepest descent gave is still what the line search enforces.
//!
//! It replaced plain steepest descent because that was where essentially
//! all of an embedding's time went, and it was not converging: on a 76-atom
//! molecule the touch-up pass ran its full 1500-iteration budget and still
//! ended at an energy of 0.37. Measured over ethanol, benzene, caffeine,
//! cocaine, atorvastatin and a 90-atom peptide, the change lands 2-8x
//! faster and at a lower final energy on five of the six (0.13 rather than
//! 0.37 on the 76-atom case); caffeine is the exception, ending about 11%
//! higher for 7x less time.
//!
//! Two things beyond the search direction mattered as much:
//!
//! * the step size was carried between iterations rather than restarting the
//!   line search at a full step each time, which cut energy evaluations per
//!   iteration from ~5 to ~2. That belonged to the conjugate-gradient
//!   version and did not survive the move to [`crate::molecule::minimize`],
//!   which proposes a full step every iteration; putting it back was
//!   measured on 2026-10-02 and costs more quality than it buys time. See
//!   that module's comment.;
//! * a line search that fails is a reason to restart the direction, not to
//!   stop. Treating failure as convergence -- as an intermediate version of
//!   this did -- abandoned structures that were still improvable and gave up
//!   most of the accuracy the method had gained.
//!
//! This could not land before the stereo double-bond restraint in
//! [`super::forcefield`] did. Conjugate gradient reaches a lower total
//! energy but distributes what is left differently, and the cis/trans term
//! was at that point a hinge that stopped pulling roughly 105 degrees short
//! of planar -- slack enough that the redistribution pushed a tagged bond
//! over the line its test drew. Fixing the restraint to aim at the planar
//! dihedral removed the slack, and this change then made no test worse.

//! The pairs come in as a flat slice of [`bounds::Pair`], built once per
//! molecule, rather than as the bounds matrix itself. Every iteration reads
//! every pair, so what an `Array2` lookup costs -- an index computation and
//! a bounds check for each of the two bounds -- is paid `n^2/2` times per
//! energy evaluation; the squares and reciprocals the energy wants are
//! precomputed in the same move, which turns its two divisions into
//! multiplications. Worth 4% of an embedding, and it is also what lets a
//! pass be handed a subset of the pairs (see [`bounds::Pairs`]).

use super::bounds;
use super::chirality::StereoConstraints;
use super::forcefield::{AngleBend, Torsion};
use super::planarity::{PlanarCenter, RingPlane};
use crate::molecule::minimize::{Energy, minimize};
use crate::vector::Vector3D;

const GRADIENT_NORM_TOLERANCE: f64 = 1e-4;

/// Everything the embedder restrains at once: the distance bounds, the
/// stereo terms, and -- in the final touch-up only -- planarity, angle bend
/// and torsion.
struct Restraints<'a> {
    pairs: &'a [bounds::Pair],
    stereo: &'a StereoConstraints,
    rings: &'a [RingPlane],
    planar: &'a [PlanarCenter],
    angles: &'a [AngleBend],
    torsions: &'a [Torsion],
    /// Somewhere for the terms that only offer a combined energy-and-gradient
    /// entry point to write a gradient nobody wants, during a line search.
    discard: std::cell::RefCell<Vec<Vector3D>>,
}

impl Energy for Restraints<'_> {
    fn energy(&self, positions: &[Vector3D]) -> f64 {
        let mut energy = bounds::pair_violations(self.pairs, positions, None);
        let mut discard = self.discard.borrow_mut();
        discard.fill(Vector3D::default());
        energy += self.stereo.energy_and_gradient(positions, &mut discard);
        energy += super::planarity::planarity_energy_and_gradient(
            positions,
            self.rings,
            self.planar,
            &mut discard,
        );
        energy += super::forcefield::forcefield_energy(positions, self.angles, self.torsions);
        energy
    }

    fn energy_and_gradient(&self, positions: &[Vector3D], gradient: &mut [Vector3D]) -> f64 {
        gradient.fill(Vector3D::default());
        let mut energy = bounds::pair_violations(self.pairs, positions, Some(gradient));

        energy += self.stereo.energy_and_gradient(positions, gradient);
        energy += super::planarity::planarity_energy_and_gradient(
            positions,
            self.rings,
            self.planar,
            gradient,
        );
        energy += super::forcefield::forcefield_energy_and_gradient(
            positions,
            self.angles,
            self.torsions,
            gradient,
        );
        energy
    }
}

/// Runs gradient-descent refinement in place, returning the final energy.
#[allow(clippy::too_many_arguments)]
pub(super) fn refine(
    positions: &mut [Vector3D],
    pairs: &[bounds::Pair],
    stereo: &StereoConstraints,
    rings: &[RingPlane],
    planar: &[PlanarCenter],
    angles: &[AngleBend],
    torsions: &[Torsion],
    max_iterations: usize,
) -> f64 {
    let restraints = Restraints {
        pairs,
        stereo,
        rings,
        planar,
        angles,
        torsions,
        discard: std::cell::RefCell::new(vec![Vector3D::default(); positions.len()]),
    };
    minimize(positions, &restraints, max_iterations, GRADIENT_NORM_TOLERANCE)
}
