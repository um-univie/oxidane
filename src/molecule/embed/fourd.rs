//! Distance geometry's fourth dimension: getting the handedness right while
//! there is still room to be wrong.
//!
//! A trial embedding starts from a distance matrix, and a distance matrix
//! says nothing about handedness -- it is the same for a molecule and its
//! mirror image, and the same for any one centre and its inversion. So
//! roughly half of the stereocentres in a fresh trial come out wrong, and
//! something has to put them right.
//!
//! In three dimensions the only way to invert a centre is to drag one
//! substituent through the plane of the other three. That path is exactly
//! where the bounds are most violated, so the chirality term has to win a
//! fight against every other restraint at once; what it gets instead is a
//! compromise, a centre left near its own plane with the geometry around it
//! distorted to suit. `relax_positions` then finds that the nearest
//! force-field minimum is the *inverted* one, and either follows it -- to
//! the wrong molecule -- or holds the centre at a barrier, 36 kcal/mol above
//! where it should be. Glucose and cholesterol were both that.
//!
//! In four dimensions there is no plane to cross. The same inversion is a
//! rotation through `w`, and nothing is in the way: the chirality term is
//! satisfied almost for free, and everything else keeps its distances. The
//! fourth coordinate is then squeezed out -- an increasing penalty on `w`,
//! which flattens the structure back into three dimensions with the
//! handedness already decided. This is the standard trick, the "4D anneal"
//! of ETKDG and of distance geometry before it; `mds` has always embedded
//! into four dimensions and this is what it embedded into four dimensions
//! *for*.
//!
//! What runs here is only the bounds, the chirality terms and the `w`
//! penalty. Planarity, angle bend and torsion belong to the touch-up, which
//! happens afterwards in three dimensions: they describe a shape, and while
//! `w` is still open there is no shape to describe.

use super::bounds;
use super::chirality::StereoConstraints;
use crate::molecule::minimize::{Energy, minimize};
use crate::vector::{Vector3D, Vector4D};

const GRADIENT_NORM_TOLERANCE: f64 = 1e-4;

/// The `w` penalty, stage by stage.
///
/// The first stage is free: the structure can use the fourth dimension for
/// nothing, and that is when a wrong centre turns over. Each stage after it
/// costs ten times more, so `w` is squeezed out gradually rather than
/// snapped flat -- a single large penalty from the start is a three
/// dimensional refinement with extra steps, and fixes nothing.
const FLATTENING_SCHEDULE: [f64; 5] = [0.0, 0.01, 0.1, 1.0, 10.0];

/// How much the chirality terms outweigh the bounds here.
///
/// Turning a centre over is not enough; it has to end well clear of its own
/// plane, because everything downstream reads its sign off a signed volume
/// and a centre left near zero is a centre the force field can take either
/// way. At equal weight the fourth-dimension stage hands cholesterol's
/// ring-fusion quaternary carbons back at 0.68 cubic Angstroms and the
/// touch-up erodes that to 0.24, which RDKit then reads as the wrong
/// isomer; at ten they come back at 0.98 and stay there, and the molecule
/// finishes 59 kcal/mol lower. Thirty and a hundred are worse than ten --
/// past some point the chirality terms are just distorting the structure
/// the bounds are trying to build.
const CHIRALITY_WEIGHT: f64 = 10.0;

/// Iterations per stage, as a fraction of the caller's budget. Five stages
/// of the full budget would cost five times a refinement; the early stages
/// have little to do beyond turning centres over.
const STAGE_BUDGET: usize = 3;

/// Turns the trial's wrong stereocentres over in four dimensions, then
/// flattens it back into three.
///
/// `positions` comes in as the three real coordinates of the MDS embedding
/// and `w` as its fourth; `positions` goes out flattened, ready for the
/// ordinary three-dimensional refinement. Returns how many stereocentres
/// are still wrong, which is the caller's reason to have called this.
pub(super) fn resolve_chirality(
    positions: &mut [Vector3D],
    w: &[f64],
    pairs: &[bounds::Pair],
    stereo: &StereoConstraints,
    max_iterations: usize,
) -> usize {
    let mut points: Vec<Vector4D> = positions
        .iter()
        .zip(w.iter())
        .map(|(point, &fourth)| Vector4D::from_xyz(*point, fourth))
        .collect();

    let restraints = Restraints {
        pairs,
        stereo,
        flattening: std::cell::Cell::new(0.0),
        projection: std::cell::RefCell::new(vec![Vector3D::default(); points.len()]),
        scratch: std::cell::RefCell::new(vec![Vector3D::default(); points.len()]),
    };

    for weight in FLATTENING_SCHEDULE {
        restraints.flattening.set(weight);
        minimize(
            &mut points,
            &restraints,
            max_iterations / STAGE_BUDGET,
            GRADIENT_NORM_TOLERANCE,
        );
    }

    for (slot, point) in positions.iter_mut().zip(points.iter()) {
        *slot = point.xyz();
    }
    stereo.count_mismatches(positions)
}

/// The bounds, the chirality terms and the penalty on `w`.
struct Restraints<'a> {
    pairs: &'a [bounds::Pair],
    stereo: &'a StereoConstraints,
    /// The current stage's penalty on the fourth coordinate.
    flattening: std::cell::Cell<f64>,
    /// The three real coordinates, which is all the chirality terms see.
    projection: std::cell::RefCell<Vec<Vector3D>>,
    /// Where those terms write their gradient, to be lifted back into four
    /// dimensions with a zero fourth component.
    scratch: std::cell::RefCell<Vec<Vector3D>>,
}

impl Restraints<'_> {
    /// The chirality terms, evaluated on the projection. They are written
    /// for three dimensions and stay that way: handedness is a property of
    /// the structure this will become, not of the one it is passing
    /// through, and a signed volume that included `w` would be satisfied by
    /// a centre that is only correct while the fourth dimension is open.
    fn stereo_energy(&self, points: &[Vector4D], gradient: Option<&mut [Vector4D]>) -> f64 {
        let mut projection = self.projection.borrow_mut();
        for (slot, point) in projection.iter_mut().zip(points.iter()) {
            *slot = point.xyz();
        }
        let mut scratch = self.scratch.borrow_mut();
        scratch.fill(Vector3D::default());
        let energy = CHIRALITY_WEIGHT * self.stereo.energy_and_gradient(&projection, &mut scratch);
        for slot in scratch.iter_mut() {
            *slot = *slot * CHIRALITY_WEIGHT;
        }
        if let Some(gradient) = gradient {
            for (slot, part) in gradient.iter_mut().zip(scratch.iter()) {
                *slot += Vector4D::from_xyz(*part, 0.0);
            }
        }
        energy
    }
}

impl Energy<Vector4D> for Restraints<'_> {
    fn energy(&self, points: &[Vector4D]) -> f64 {
        let mut energy = bounds::pair_violations(self.pairs, points, None);
        energy += self.stereo_energy(points, None);
        let flattening = self.flattening.get();
        energy + flattening * points.iter().map(|point| point.w * point.w).sum::<f64>()
    }

    fn energy_and_gradient(&self, points: &[Vector4D], gradient: &mut [Vector4D]) -> f64 {
        gradient.fill(Vector4D::default());
        let mut energy = bounds::pair_violations(self.pairs, points, Some(&mut *gradient));
        energy += self.stereo_energy(points, Some(gradient));

        let flattening = self.flattening.get();
        for (slot, point) in gradient.iter_mut().zip(points.iter()) {
            slot.w += 2.0 * flattening * point.w;
            energy += flattening * point.w * point.w;
        }
        energy
    }
}
