//! Greedy search over the torsions distance geometry left behind.
//!
//! Distance geometry plus relaxation answers "is this a valid structure", and
//! the force field then walks downhill from wherever the trial happened to
//! land. Neither step can climb: a chain whose central bond came out *gauche*
//! stays *gauche*, however much better *anti* would be, because the barrier
//! between them is several kcal/mol and every step of a minimisation goes
//! down. More trials do find the better basin eventually -- ours did, at four
//! times the budget -- but paying for a whole extra embedding to re-roll one
//! torsion is a poor trade when the torsion can simply be turned.
//!
//! So: take the conformer the trials chose, turn each rotatable bond to the
//! other staggered positions in turn, relax, and keep the turn if it came out
//! lower. Repeat while a pass is still finding something. This is the
//! cheapest possible conformer search -- a coordinate descent in torsion
//! space -- and on `tests/embed_budget_benchmark.py`, eight trials a side, it
//! is worth far more than the same time spent on further trials. MMFF94
//! kcal/mol, against what RDKit's ETKDGv3 plus its own MMFF94 cleanup reaches
//! from the same eight:
//!
//! | | trials alone | with this | RDKit |
//! |---|---|---|---|
//! | peptide | 68.8 | **61.7** | 70.6 |
//! | atorvastatin | 41.3 | **35.3** | 38.5 |
//! | cocaine | 30.1 | **27.9** | 27.9 |
//! | biphenyl-chain | 90.5 | **88.5** | 89.7 |
//! | erythromycin-frag | 28.8 | **18.1** | 23.8 |
//!
//! It turns a corpus this crate lost 5 of 20 molecules on into one it wins or
//! ties on all 20. The macrocycle is the last row and needs a move of its
//! own: see [`corner_flip_moves`].
//!
//! It is not free: a molecule with twenty-two rotatable bonds has a lot of
//! turns, and each one costs a relaxation to reject. One trial of the peptide
//! goes from 350 ms to 1.0 s and atorvastatin from 290 ms to 770 ms, with
//! `EmbedConfig`'s `torsion_search` to turn it off.
//!
//! Three filters keep that from being three times worse, each cheaper than
//! the one it guards and each measured on the peptide, where they take the
//! search from 1.47 s to 990 ms with the same answer to two decimal places:
//!
//! 1. the rigid rotation's own energy ([`RIGID_ALLOWANCE`]), one energy
//!    evaluation, which throws out 60 of 132 turns;
//! 2. a short relaxation ([`SCREEN_ITERATIONS`]), forty gradients, which
//!    throws out 52 of the remaining 72 -- for a turned bond now twelve
//!    iterations in torsion space instead ([`TORSION_SCREEN_ITERATIONS`]),
//!    which judges as well and lets the stages after it halve;
//! 3. a third of the ranking relaxation ([`PARTIAL_ITERATIONS`]), after
//!    which a candidate that is not winning yet is abandoned, since a
//!    truncated relaxation can only overstate the final energy.

/// Minimum norm for a usable rotation axis or reflection normal.
/// Applied in the coordinate units of the construction being normalized.
const MIN_TRANSFORM_AXIS_NORM: f64 = 1e-9;
/// Dimensionless sine threshold for treating directions as parallel.
const PARALLEL_DIRECTION_SINE: f64 = 1e-9;

use super::chirality::StereoConstraints;
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::forcefield::Field;
use crate::molecule::minimize::{Energy, minimize};
use crate::molecule::molecule3d::Molecule3D;
use crate::vector::Vector3D;

/// Turns tried at each bond, in degrees. Three staggered positions for an
/// sp3-sp3 bond; 180 is also the one that matters for a conjugated or
/// amide-like bond, where the other two are uphill and simply get rejected.
/// Three staggered positions for an sp3-sp3 bond, and for a conjugated or
/// amide-like one the half turn is the move that matters -- *cis* against
/// *trans*, the two ways a biaryl sits.
///
/// Giving sp3-sp3 bonds only the two thirds of a turn looks right on paper
/// and costs the peptide 1.6 kcal/mol for 4% of the search: a starting
/// torsion is rarely exactly staggered, so the half turn lands in a
/// different well than either third does.
const TURNS: &[f64] = &[120.0, -120.0, 180.0];

/// Iterations for the screening relaxation that decides whether a turn is
/// worth relaxing properly. Enough to undo the clashes a rigid rotation
/// makes, not enough to reach a minimum.
///
/// This is where the search's time is: of 639 candidates over twelve
/// molecules, 36% die on the rigid energy for one evaluation each, so the
/// three relaxation gates are 99.7% of the cost and the rigid screens 0.3%.
///
/// Cocaine, the peptide and the cyclic peptide are identical at 20, 25 and
/// 40 (measured 2026-10-02). The budget still matters, just not to those
/// three:
///
/// | iterations | curated 72, default | PubChem 287, default |
/// |---|---|---|
/// | 20 | 1855 ms | 251 of 287, -1.78 kcal/mol |
/// | 30 | 2083 ms | 253 of 287, -1.90 |
/// | **40** | 2168 ms | **254 of 287, -1.93** |
///
/// A smooth dial with no free point, and the curated corpus reads 68 of 72
/// at -1.09 at every setting -- it would have sold the 14% as free, which is
/// what the second corpus is for. Declined, with the frozen-screen attempt
/// that tried to get the same time for nothing; `NEXT.md` has both.
const SCREEN_ITERATIONS: usize = 40;

/// Iterations of the ranking relaxation after which a candidate has to be
/// winning already. See the third filter in [`refine_torsions`].
const PARTIAL_ITERATIONS: usize = 100;

/// Iterations for the relaxation of a turn that survived screening. The
/// accepted structure is taken to convergence at the end, so this only has
/// to rank candidates, and 500 ranks them no better than 300 at 1.7x the
/// cost of the whole search.
const FULL_ITERATIONS: usize = 300;

/// Iterations of the screen a turned *bond* gets instead of the Cartesian
/// one, and the partial and full budgets after it. A bond turn is screened
/// in torsion space ([`relax_torsions`]); ring flips, ring inversions and
/// nitrogen inversions, which move atoms no torsion reaches, keep the
/// Cartesian screen and the budgets above.
///
/// A rigid turn leaves bond lengths and angles as they were and swings one
/// side into new contacts, and what undoes those contacts is the soft
/// modes -- the other torsions -- which is exactly what a Cartesian
/// minimiser is slowest to find among 3N coordinates dominated by stiff
/// stretches. Over 752 turns on 29 PubChem molecules, twelve torsion-space
/// iterations ended *lower* than forty Cartesian ones (median 1.7 against
/// 2.2 kcal/mol above the current structure), ranked the turns as well
/// against their fully relaxed energies (Spearman 0.61 against 0.60), and
/// kept as many of the eventual winners past the allowance (68 against 69
/// of 86). The turn also arrives at the Cartesian stages near its minimum,
/// which is what lets their budgets halve.
///
/// One start, against forty Cartesian iterations and 100/300:
///
/// | screen | partial / full | work | PubChem 287, better / worse | held-out 1600 |
/// |---|---|---|---|---|
/// | 12 torsion | 100 / 300 | -15% | 50 / 46, +0.05 | |
/// | 40 Cartesian | 50 / 150 | -16% | 19 / 32, +0.09 | |
/// | **12 torsion** | **50 / 150** | **-31%** | **46 / 45, +0.05** | **274 / 272, -0.12** |
/// | 12 torsion | 30 / 100 | -38% | 43 / 51, +0.08 | |
/// | 8 torsion | 50 / 150 | -35% | 41 / 53, +0.17 | |
///
/// (better / worse by 0.05 kcal/mol, then the mean change.) Halving the
/// Cartesian budgets alone is the usual dial; after a torsion-space screen
/// it is free. At the default five starts the search is a smaller share,
/// and the same change is 17-19% of the work at -0.05 kcal/mol on both
/// PubChem and 1200 held-out molecules.
const TORSION_SCREEN_ITERATIONS: usize = 12;
const TURN_PARTIAL_ITERATIONS: usize = 50;
const TURN_FULL_ITERATIONS: usize = 150;

/// The largest turn one torsion-space step may make at any bond, in
/// radians. The first step's direction is the raw torque, whose scale is
/// the strain of whatever contact the rigid turn made, so it needs a cap;
/// later ones come from the curvature and rarely meet it.
const MAX_TORSION_STEP: f64 = 0.35;

/// Armijo's sufficient-decrease constant, probes per line search, and the
/// smallest curvature `s . y` worth a BFGS update, for [`relax_torsions`].
const TORSION_ARMIJO: f64 = 1e-4;
const MAX_TORSION_PROBES: usize = 12;
const MIN_TORSION_CURVATURE: f64 = 1e-10;

/// How far above the current energy a screened turn may sit and still be
/// worth relaxing properly, in kcal/mol. A rigid rotation lands in contacts
/// the screening pass has not finished opening up, so a turn is not judged
/// on where it sits after forty iterations. Judging it strictly -- half a
/// kcal/mol -- costs the peptide fourteen.
const SCREEN_ALLOWANCE: f64 = 2.0;

/// Improvement, in kcal/mol, below which a turn is not worth taking. Keeps
/// the search from walking sideways through numerically indistinguishable
/// structures.
const ACCEPT: f64 = 0.05;

/// Passes over the bond list. The first finds most of it; a second pays when
/// turning one bond opened room for another, and is worth 3.8 kcal/mol on
/// atorvastatin for 300 ms. A third is not: it found nothing the second had
/// not on any molecule in the corpus.
const MAX_PASSES: usize = 2;

/// Improvement over a whole pass below which another pass is not worth it.
const PASS_TOLERANCE: f64 = 0.5;

/// Force threshold, matching the rest of relaxation.
const GRADIENT_TOLERANCE: f64 = 1e-4;

/// How far above the current energy the *unrelaxed* rotation may sit and
/// still be worth screening, in kcal/mol.
///
/// The cheapest of the three filters: one energy evaluation against the
/// forty gradients of the screening relaxation and the three hundred of the
/// ranking one. On the peptide it throws out 60 of 132 turns and takes 35%
/// off the search. Ten is the floor -- at five, atorvastatin loses 6 kcal/mol
/// and the peptide 14, because a turn that starts in a contact can still
/// relax into the best conformer there is.
const RIGID_ALLOWANCE: f64 = 25.0;

/// Turns `positions`' rotatable bonds while that lowers the energy, in place.
/// Returns the final energy.
///
/// `energy` is the field's energy at `positions`, which the caller already
/// has. Stereochemistry is checked rather than assumed: rotating a rigid
/// subtree about a bond axis cannot change a centre, but the relaxation after
/// it can, and a turn that costs a centre is rejected like any other.
pub(super) fn refine_torsions(
    molecule: &Molecule3D,
    positions: &mut Vec<Vector3D>,
    field: &Field,
    stereo: &StereoConstraints,
    mut energy: f64,
    seed: u64,
) -> f64 {
    // Order matters, because this is a coordinate descent and every move it
    // accepts is the structure the next move starts from. The moves that
    // reposition heavy atoms go first, so that they see the structure the
    // trials produced rather than one a hydrogen has already been turned on
    // -- interleaving the hydrogen rotors by atom index cost the
    // erythromycin fragment a corner flip it used to find. Hydrogens are a
    // polish and go last.
    let mut moves: Vec<Move> = Vec::new();
    let (hydrogen, heavy): (Vec<Move>, Vec<Move>) = torsion_moves(molecule)
        .into_iter()
        .partition(|candidate| candidate.turns_hydrogens_only(molecule));
    moves.extend(heavy);
    moves.extend(corner_flip_moves(molecule));
    moves.extend(inversion_moves(molecule));
    moves.extend(ring_inversion_moves(molecule));
    moves.extend(hydrogen);
    if moves.is_empty() {
        return energy;
    }
    let mismatches = stereo.count_mismatches(positions);
    let start_energy = energy;
    let bonds: Vec<RotatableBond> = moves
        .iter()
        .filter(|candidate| candidate.about_bond)
        .map(|candidate| RotatableBond {
            pivot: candidate.pivot,
            axis_end: candidate.axis_end,
            side: candidate.side.clone(),
        })
        .collect();

    // A candidate is a function of the structure it starts from and the
    // energy it has to beat, nothing else, so one tried since the last
    // accepted move would come out exactly as it did then. Counting
    // evaluations lets a later pass skip those: on PubChem's 287 at one
    // trial that is 1363 candidates and 5% of the search, with the same
    // structures to the bit.
    let mut tried_at: Vec<Vec<usize>> = moves
        .iter()
        .map(|candidate_move| vec![0; candidate_move.angles().len()])
        .collect();
    let (mut tried, mut accepted_at) = (0usize, 0usize);

    for _ in 0..MAX_PASSES {
        let before_pass = energy;
        for (candidate_move, tried_at) in moves.iter().zip(tried_at.iter_mut()) {
            for (&angle, tried_at) in candidate_move.angles().iter().zip(tried_at.iter_mut()) {
                if *tried_at > accepted_at {
                    continue;
                }
                tried += 1;
                *tried_at = tried;
                let mut trial = positions.clone();
                candidate_move.apply(&mut trial, angle);

                // The rigid rotation itself, before any relaxation. A turn
                // that swings half the molecule into the other half shows up
                // here for one energy evaluation, where the screening
                // relaxation below costs forty gradients.
                if candidate_move.rigid_allowance.is_finite()
                    && !(field.energy(&trial) < energy + candidate_move.rigid_allowance)
                {
                    continue;
                }
                let screen = if candidate_move.about_bond {
                    Screen::Torsions(&bonds)
                } else {
                    Screen::Cartesian(candidate_move.screen_iterations)
                };
                let Some(candidate) = relax_candidate(&mut trial, field, screen, energy) else {
                    continue;
                };
                if candidate < energy - ACCEPT
                    && candidate.is_finite()
                    && trial
                        .iter()
                        .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
                    && stereo.count_mismatches(&trial) <= mismatches
                {
                    // Taken at once rather than after comparing this bond's
                    // other turns: the next turn then starts from the better
                    // structure, and the walk compounds. Keeping the best of
                    // the three instead, which relaxes fewer candidates,
                    // costs the peptide 14 kcal/mol.
                    *positions = trial;
                    energy = candidate;
                    accepted_at = tried;
                }
            }
        }
        if before_pass - energy < PASS_TOLERANCE {
            break;
        }
    }

    energy = shake(positions, field, stereo, &moves, energy, mismatches, seed);

    // Every candidate above was relaxed on a ranking budget, so the structure
    // that won is near its minimum rather than at it. Finish it properly --
    // it is one structure, already close, and this is what the caller returns.
    if energy < start_energy {
        let polished = super::minimize_to_convergence(field, positions);
        if polished.is_finite() && stereo.count_mismatches(positions) <= mismatches {
            return polished;
        }
    }
    energy
}

/// Relaxes one candidate through the screening and ranking stages, giving up
/// as soon as a stage says it cannot win: `None` if it was abandoned, its
/// relaxed energy otherwise. `energy` is the energy it has to beat.
///
/// Each stage is a fresh minimisation. Continuing one minimisation across the
/// gates instead keeps the curvature built so far, and was measured: 3% slower
/// on the PubChem corpus, no better on average, and the erythromycin fragment
/// in `a_corner_flip_finds_a_better_macrocycle_conformer` stopped improving.
/// `minimize` answers with the energy it stopped at, so each gate reads that
/// rather than paying for it again.
fn relax_candidate(
    trial: &mut [Vector3D],
    field: &Field,
    screen: Screen,
    energy: f64,
) -> Option<f64> {
    let (screened, partial, full) = match screen {
        Screen::Cartesian(iterations) => (
            minimize(trial, field, iterations, GRADIENT_TOLERANCE),
            PARTIAL_ITERATIONS,
            FULL_ITERATIONS,
        ),
        Screen::Torsions(bonds) => (
            relax_torsions(trial, field, bonds, TORSION_SCREEN_ITERATIONS),
            TURN_PARTIAL_ITERATIONS,
            TURN_FULL_ITERATIONS,
        ),
    };
    if !(screened < energy + SCREEN_ALLOWANCE) {
        return None;
    }
    // Third filter: a truncated relaxation can only overstate the final
    // energy, so a candidate that is not yet winning after a third of the
    // budget usually never does.
    if !(minimize(trial, field, partial, GRADIENT_TOLERANCE) < energy) {
        return None;
    }
    Some(minimize(trial, field, full - partial, GRADIENT_TOLERANCE))
}

/// How a candidate's first relaxation is done: Cartesian for this many
/// iterations, or in the torsion space of these bonds.
#[derive(Clone, Copy)]
enum Screen<'a> {
    Cartesian(usize),
    Torsions(&'a [RotatableBond]),
}

/// One coordinate of torsion space: turning `side` about the
/// `pivot -> axis_end` bond.
struct RotatableBond {
    pivot: usize,
    axis_end: usize,
    side: Vec<usize>,
}

/// `dE/dtheta` for every bond at `positions`, from the Cartesian gradient:
/// the torque of the turned side about the bond axis. A small turn `d` of
/// bond `k` moves each side atom by `u x (r - p) d`, so its slope is
/// `u . sum (r - p) x g` -- exact at any structure, whatever turns came
/// before, which is why the steps below are taken about the current axes.
fn torsion_gradient(
    positions: &[Vector3D],
    cartesian: &[Vector3D],
    bonds: &[RotatableBond],
    torsional: &mut [f64],
) {
    for (slot, bond) in torsional.iter_mut().zip(bonds) {
        let pivot = positions[bond.pivot];
        let axis = positions[bond.axis_end] - pivot;
        let length = axis.length();
        if length < MIN_TRANSFORM_AXIS_NORM {
            *slot = 0.0;
            continue;
        }
        let mut torque = Vector3D::default();
        for &atom in &bond.side {
            torque += (positions[atom] - pivot).cross(&cartesian[atom]);
        }
        *slot = torque.dot(&axis) / length;
    }
}

/// Relaxes `positions` over the rotatable bonds alone, every bond length and
/// angle held where it is, returning the energy reached. See
/// [`TORSION_SCREEN_ITERATIONS`] for why a bond turn is screened this way.
///
/// BFGS with a dense inverse Hessian -- a molecule has tens of rotatable
/// bonds, not hundreds of coordinates -- scaled after the first step as
/// Nocedal and Wright 6.20 have it, which suits a space with no stiff
/// stretches in it. The line search is Armijo backtracking with a quadratic
/// model, and every probe takes the gradient: in a space this small the
/// full step is nearly always taken. A step is a set of turns applied in
/// order about the current axes, so the slope it was chosen on is exact.
fn relax_torsions(
    positions: &mut [Vector3D],
    field: &Field,
    bonds: &[RotatableBond],
    max_iterations: usize,
) -> f64 {
    let m = bonds.len();
    let mut cartesian = vec![Vector3D::default(); positions.len()];
    let mut energy = field.energy_and_gradient(positions, &mut cartesian);
    if m == 0 {
        return energy;
    }
    let identity = |inverse: &mut [f64], scale: f64| {
        inverse.fill(0.0);
        for k in 0..m {
            inverse[k * m + k] = scale;
        }
    };
    let mut inverse = vec![0.0; m * m];
    identity(&mut inverse, 1.0);
    let mut scaled = false;
    let mut gradient = vec![0.0; m];
    torsion_gradient(positions, &cartesian, bonds, &mut gradient);
    let (mut direction, mut next_gradient) = (vec![0.0; m], vec![0.0; m]);
    let (mut step, mut change, mut curved) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
    let mut trial = positions.to_vec();
    let mut trial_cartesian = cartesian.clone();

    for _ in 0..max_iterations {
        let norm_sq: f64 = gradient.iter().map(|g| g * g).sum();
        if norm_sq < GRADIENT_TOLERANCE {
            break;
        }
        for (i, slot) in direction.iter_mut().enumerate() {
            *slot = -(0..m)
                .map(|j| inverse[i * m + j] * gradient[j])
                .sum::<f64>();
        }
        let mut slope: f64 = direction.iter().zip(&gradient).map(|(d, g)| d * g).sum();
        if !(slope < 0.0) {
            identity(&mut inverse, 1.0);
            scaled = false;
            for (slot, g) in direction.iter_mut().zip(&gradient) {
                *slot = -g;
            }
            slope = -norm_sq;
        }
        let largest = direction.iter().fold(0.0f64, |most, d| most.max(d.abs()));
        let mut length = if largest > MAX_TORSION_STEP {
            MAX_TORSION_STEP / largest
        } else {
            1.0
        };
        let mut accepted = false;
        for _ in 0..MAX_TORSION_PROBES {
            trial.copy_from_slice(positions);
            for (bond, turn) in bonds.iter().zip(&direction) {
                rotate(
                    &mut trial,
                    bond.pivot,
                    bond.axis_end,
                    &bond.side,
                    turn * length,
                );
            }
            let probe = field.energy_and_gradient(&trial, &mut trial_cartesian);
            if probe.is_finite() && probe <= energy + TORSION_ARMIJO * length * slope {
                energy = probe;
                accepted = true;
                break;
            }
            let next = if probe.is_finite() {
                -slope * length * length / (2.0 * (probe - energy - slope * length))
            } else {
                0.5 * length
            };
            length = next.clamp(0.1 * length, 0.5 * length);
        }
        if !accepted {
            break;
        }
        positions.copy_from_slice(&trial);
        std::mem::swap(&mut cartesian, &mut trial_cartesian);
        torsion_gradient(positions, &cartesian, bonds, &mut next_gradient);

        for k in 0..m {
            step[k] = direction[k] * length;
            change[k] = next_gradient[k] - gradient[k];
        }
        let curvature: f64 = step.iter().zip(&change).map(|(s, y)| s * y).sum();
        if curvature > MIN_TORSION_CURVATURE {
            if !scaled {
                let change_sq: f64 = change.iter().map(|y| y * y).sum();
                identity(&mut inverse, curvature / change_sq);
                scaled = true;
            }
            // H += -rho (Hy s' + s y'H) + (rho^2 y'Hy + rho) s s'
            for (i, slot) in curved.iter_mut().enumerate() {
                *slot = (0..m).map(|j| inverse[i * m + j] * change[j]).sum();
            }
            let along: f64 = change.iter().zip(&curved).map(|(y, hy)| y * hy).sum();
            let rho = 1.0 / curvature;
            let outer = rho * rho * along + rho;
            for i in 0..m {
                for j in 0..m {
                    inverse[i * m + j] += -rho * (curved[i] * step[j] + step[i] * curved[j])
                        + outer * step[i] * step[j];
                }
            }
        }
        std::mem::swap(&mut gradient, &mut next_gradient);
    }
    energy
}

/// What a move does to the atoms it names.
enum Operation {
    /// Turn `side` about the `pivot -> axis_end` line.
    Turn,
    /// Turn a saturated ring's pucker inside out while leaving every
    /// stereocentre as it was: a chair inversion.
    ///
    /// Reflecting the ring through its own mean plane is the easy half -- a
    /// reflection is an isometry, so the inverted ring keeps every bond
    /// length and ring angle exactly -- and on its own it is wrong twice
    /// over. Reflecting the substituents with it inverts every centre in the
    /// molecule, and translating them instead (which was tried) leaves a
    /// structure 437 kcal/mol up whose downhill path runs straight back to
    /// the chair it came from: two hundred iterations of relaxation returned
    /// glucose to its starting energy to the decimal, every time.
    ///
    /// What makes it admissible is the second half: at each ring atom the
    /// two exocyclic substituents *change places*. A reflection inverts the
    /// configuration at a centre and swapping two of its substituents
    /// inverts it again, so the pair of operations leaves it alone -- which
    /// is exactly what a real chair inversion does, moving every substituent
    /// between axial and equatorial while the molecule stays the same
    /// molecule. Each branch is placed by a proper rotation about its ring
    /// atom, never reflected, so whatever chirality it contains survives
    /// too.
    ///
    /// Without the swap the move is a corner flip, legal only where the two
    /// substituents are identical. That is why [`corner_flip_moves`] can
    /// turn a macrocycle's CH2 over and cannot touch a sugar: the flip
    /// reaches lower energies on glucose and every one of them is the wrong
    /// diastereomer, which the stereo check in [`refine_torsions`] throws
    /// out.
    InvertRing {
        ring: Vec<usize>,
        /// Per ring atom, the exocyclic branches hanging off it, each as
        /// `(attachment, atoms)`. Two of them are swapped; one is simply
        /// carried; none and there is nothing to do.
        branches: Vec<Vec<(usize, Vec<usize>)>>,
    },
    /// Reflect `side` through the plane of `pivot` and the two atoms named
    /// here: the pyramidal inversion at a three-coordinate centre.
    ///
    /// A reflection is an isometry that fixes its plane, so every distance
    /// from the reflected atom to `pivot` and to the two in-plane atoms
    /// survives it exactly. The local geometry at the centre comes out its
    /// own mirror image with no strain introduced, and what changes is only
    /// how the reflected atom sits against the rest of the molecule --
    /// which is what the energy is being asked about.
    Invert { in_plane: (usize, usize) },
}

/// How many times to try turning several bonds at once, once turning them
/// one at a time has stopped paying.
///
/// Measured on the held-out 400 at the default preset, against the curated
/// corpus's single-threaded time:
///
/// | attempts | held out | time |
/// |---|---|---|
/// | 0 | 365 of 390, -1.78 | 4003 ms |
/// | **12** | **369, -1.85** | **4360 ms** |
/// | 24 | 370, -1.88 | 4735 ms |
///
/// Twelve buys four fifths of what twenty-four does for half the extra
/// time, and doubling again was not tried because the trend is already
/// flattening. The whole purchase is close to the going rate near the knee
/// of the budget curve -- about half what the fourth and fifth trials buy,
/// for about the same cost -- so it is kept small.
const SHAKE_ATTEMPTS: usize = 12;

/// How many moves one attempt makes at once. Two is the case the measurement
/// named; three is cheap to include and occasionally finds what two cannot.
const SHAKE_MOVES: [usize; 2] = [2, 3];

/// Turns several bonds at once, keeping only what comes out lower.
///
/// The greedy walk above is a coordinate descent, so it can only take a move
/// that is downhill *on its own*. Some conformers are not reachable that
/// way. Glucose is the measured case: against a 200-conformer MMFF
/// reference it sits 1.25 kcal/mol high, and the difference is one hydroxyl
/// at -52 degrees against our +44 *together with* the exocyclic O-C-C-O
/// torsion gauche at 56 against our anti at 171. The gauche arrangement is
/// what lets that hydroxyl reach an acceptor; turning either bond by itself
/// gains nothing, so no number of passes finds it.
///
/// Small, and run last. The budget work in `NEXT.md`
/// measured search time as the worst buy on the whole dial past three
/// trials, so this gets a fixed two dozen attempts after the descent has
/// converged rather than a role in the descent itself, and every attempt
/// faces the same screening gates as a single turn.
///
/// Seeded from the caller's seed so a given molecule and configuration
/// always makes the same attempts in the same order.
#[allow(clippy::too_many_arguments)]
fn shake(
    positions: &mut Vec<Vector3D>,
    field: &Field,
    stereo: &StereoConstraints,
    moves: &[Move],
    mut energy: f64,
    mismatches: usize,
    seed: u64,
) -> f64 {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    if moves.len() < 2 {
        return energy;
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5AFE_D00D);
    for attempt in 0..SHAKE_ATTEMPTS {
        let count = SHAKE_MOVES[attempt % SHAKE_MOVES.len()].min(moves.len());
        let mut trial = positions.clone();
        let mut touched = 0;
        for _ in 0..count {
            let candidate = &moves[rng.random_range(0..moves.len())];
            let angles = candidate.angles();
            let angle = angles[rng.random_range(0..angles.len())];
            candidate.apply(&mut trial, angle);
            touched += 1;
        }
        if touched < 2 {
            continue;
        }
        let Some(candidate) = relax_candidate(
            &mut trial,
            field,
            Screen::Cartesian(SCREEN_ITERATIONS),
            energy,
        ) else {
            continue;
        };
        if candidate < energy - ACCEPT
            && candidate.is_finite()
            && trial
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            && stereo.count_mismatches(&trial) <= mismatches
        {
            *positions = trial;
            energy = candidate;
        }
    }
    energy
}

/// One way to move part of the molecule: a rotation of `side` about the
/// `pivot -> axis_end` line, or a reflection of it through a plane.
///
/// `pivot`, `axis_end` and `side` mean what they say for a
/// [`Operation::Turn`] and for [`Operation::Invert`], which reflects `side`
/// through a plane through `pivot`. They mean nothing for
/// [`Operation::InvertRing`], which carries its own atoms and fills these
/// with the first two ring atoms: `side` is still every atom the move
/// touches, which is what the ordering in [`refine_torsions`] reads, but
/// `pivot` and `axis_end` are not to be used generically. A fourth move
/// type should put its geometry in its own variant rather than add more
/// fields here.
struct Move {
    pivot: usize,
    axis_end: usize,
    side: Vec<usize>,
    angles: &'static [f64],
    /// How far above the current energy this move's *unrelaxed* rotation
    /// may sit and still be screened. [`RIGID_ALLOWANCE`] for a turn of a
    /// bond; infinite for a corner flip, which always starts strained
    /// because it turns an atom through its own ring, so screening one on
    /// its unrelaxed energy rejects every flip there is; and zero for a
    /// symmetric top, whose whole point is that an eclipsed one goes
    /// downhill at once while a staggered one does not -- which keeps the
    /// peptide's ten methyls from costing 20% of its search for nothing.
    rigid_allowance: f64,
    /// Iterations for this move's screening relaxation. A flipped ring needs
    /// far longer to settle than a turned bond: at forty iterations, twelve
    /// of erythromycin's thirteen flips still look worse than the structure
    /// they came from, and the search takes none of them. At two hundred it
    /// takes four.
    screen_iterations: usize,
    /// Whether this turns one side of a bond about that bond -- a rotatable
    /// bond or a hydrogen rotor -- rather than a ring corner about the line
    /// through its two neighbours. Only these are screened in torsion space
    /// and serve as its coordinates; see [`TORSION_SCREEN_ITERATIONS`].
    about_bond: bool,
    operation: Operation,
}

impl Move {
    /// Whether this move swings nothing but hydrogens, which decides where
    /// it sits in the order the search tries them.
    fn turns_hydrogens_only(&self, molecule: &Molecule3D) -> bool {
        self.side
            .iter()
            .all(|&atom| molecule.get_atomic_number(atom) == 1)
    }

    fn angles(&self) -> &'static [f64] {
        self.angles
    }

    fn apply(&self, positions: &mut [Vector3D], degrees: f64) {
        match &self.operation {
            Operation::Turn => rotate(
                positions,
                self.pivot,
                self.axis_end,
                &self.side,
                degrees.to_radians(),
            ),
            Operation::Invert { in_plane } => reflect(positions, self.pivot, *in_plane, &self.side),
            Operation::InvertRing { ring, branches } => invert_ring(positions, ring, branches),
        }
    }
}

/// Turns a ring's pucker inside out, as [`Operation::InvertRing`] describes.
fn invert_ring(positions: &mut [Vector3D], ring: &[usize], branches: &[Vec<(usize, Vec<usize>)>]) {
    let centroid = ring
        .iter()
        .fold(Vector3D::default(), |sum, &atom| sum + positions[atom])
        * (1.0 / ring.len() as f64);
    // The mean plane's normal, by the Newell sum over the ring's perimeter,
    // so a puckered ring still gets the plane it is puckered about.
    let mut normal = Vector3D::default();
    for index in 0..ring.len() {
        let current = positions[ring[index]] - centroid;
        let next = positions[ring[(index + 1) % ring.len()]] - centroid;
        normal += current.cross(&next);
    }
    let length = normal.length();
    if length < MIN_TRANSFORM_AXIS_NORM {
        return;
    }
    let normal = normal * (1.0 / length);
    let reflect_point =
        |point: Vector3D| -> Vector3D { point - normal * (2.0 * (point - centroid).dot(&normal)) };

    // Where each substituent has to end up, worked out before anything
    // moves: the reflection of the *other* substituent's direction at the
    // same ring atom.
    let mut placements: Vec<(Vec<usize>, Vector3D, Vector3D, Vector3D)> = Vec::new();
    for (index, &atom) in ring.iter().enumerate() {
        let host_now = positions[atom];
        let host_then = reflect_point(host_now);
        let hanging = &branches[index];
        for (position, (attachment, carried)) in hanging.iter().enumerate() {
            let partner = if hanging.len() == 2 {
                hanging[1 - position].0
            } else {
                *attachment
            };
            let target = (reflect_point(positions[partner]) - host_then).normalize();
            let current = (positions[*attachment] - host_now).normalize();
            let mut atoms = vec![*attachment];
            atoms.extend(carried.iter().copied());
            placements.push((atoms, host_now, host_then, target));
            let _ = current;
        }
    }

    // The ring itself.
    for &atom in ring {
        positions[atom] = reflect_point(positions[atom]);
    }

    // Then every branch, by a proper rotation about its ring atom's new
    // place: the smallest rotation taking the bond where it was to where it
    // has to be, which preserves the branch's own geometry and handedness.
    for (atoms, host_now, host_then, target) in placements {
        let anchor = atoms[0];
        let current = (positions[anchor] - host_now).normalize();
        let axis = current.cross(&target);
        let sine = axis.length();
        let cosine = current.dot(&target).clamp(-1.0, 1.0);
        let turn: Box<dyn Fn(Vector3D) -> Vector3D> = if sine < PARALLEL_DIRECTION_SINE {
            if cosine > 0.0 {
                Box::new(|offset: Vector3D| offset)
            } else {
                // Antiparallel: any perpendicular axis will do for the half
                // turn that takes one to the other.
                let perpendicular = if current.x.abs() < 0.9 {
                    Vector3D::new(1.0, 0.0, 0.0)
                } else {
                    Vector3D::new(0.0, 1.0, 0.0)
                };
                let axis = current.cross(&perpendicular).normalize();
                Box::new(move |offset: Vector3D| axis * (2.0 * offset.dot(&axis)) - offset)
            }
        } else {
            let axis = axis * (1.0 / sine);
            let angle = sine.atan2(cosine);
            let (sin, cos) = angle.sin_cos();
            Box::new(move |offset: Vector3D| {
                offset * cos + axis.cross(&offset) * sin + axis * (axis.dot(&offset) * (1.0 - cos))
            })
        };
        for atom in atoms {
            let offset = positions[atom] - host_now;
            positions[atom] = host_then + turn(offset);
        }
    }
}

/// Reflects `side` through the plane containing `pivot` and `in_plane`.
fn reflect(positions: &mut [Vector3D], pivot: usize, in_plane: (usize, usize), side: &[usize]) {
    let origin = positions[pivot];
    let first = positions[in_plane.0] - origin;
    let second = positions[in_plane.1] - origin;
    let normal = first.cross(&second);
    let length = normal.length();
    if length < MIN_TRANSFORM_AXIS_NORM {
        // The three are collinear, so they name no plane and there is
        // nothing to invert through.
        return;
    }
    let normal = normal * (1.0 / length);
    for &atom in side {
        let offset = positions[atom] - origin;
        positions[atom] = positions[atom] - normal * (2.0 * offset.dot(&normal));
    }
}

/// Rotates `side` about the `first -> second` axis by `angle` radians.
fn rotate(positions: &mut [Vector3D], first: usize, second: usize, side: &[usize], angle: f64) {
    let axis = positions[second] - positions[first];
    let pivot = positions[first];
    for &atom in side {
        let offset = positions[atom] - pivot;
        positions[atom] = pivot + offset.rotate_around_axis(&axis, angle);
    }
}

/// The moves available at an end that carries nothing but hydrogens.
///
/// Two cases, and they are different moves for different reasons.
///
/// **Two or more hydrogens** is a symmetric top -- a methyl, an ammonium --
/// where turning it is a relabelling and the only thing worth doing is
/// taking an eclipsed arrangement to a staggered one, since distance
/// geometry can hand the relaxation an eclipsed top and a gradient cannot
/// leave a stationary point. Methylammonium came out 2.7 kcal/mol above
/// RDKit for exactly that. One sixth of a turn, at zero rigid allowance.
///
/// **Exactly one hydrogen** is a hydroxyl, a thiol, or an acyclic N-H, and
/// is not a relabelling at all: where that hydrogen points decides whether
/// the molecule can close a hydrogen bond, which is worth whole kcal/mol
/// and nothing else in the pipeline will find it. It was excluded until
/// `scratchpad/floor.py` compared our best of eight against 400
/// MMFF-minimised ETKDGv3 conformers and found glucose 1.25 kcal/mol
/// adrift with its five hydroxyls -- the same effect filed earlier in the
/// week as "RDKit folds into hydrogen bonds while we stay extended", which
/// was a missing move rather than a sampling property. It gets the full
/// [`TURNS`], since an acid's syn and anti O-H are a half turn apart and
/// the two gauche positions a third.
fn hydrogen_rotor(molecule: &Molecule3D, first: usize, second: usize) -> Option<Move> {
    const SIXTH_TURN: &[f64] = &[60.0];
    let hydrogens_of = |atom: usize| -> Vec<usize> {
        molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| molecule.get_atomic_number(target) == 1)
            .collect()
    };
    let neighbours = |atom: usize| molecule.get_atom_bonds(atom).unwrap_or_default().len();
    // Whichever end is the top turns; the other only has to have something
    // for it to be eclipsed against.
    let (pivot, top, hydrogens) = if !hydrogens_of(second).is_empty() && neighbours(first) >= 2 {
        (first, second, hydrogens_of(second))
    } else if !hydrogens_of(first).is_empty() && neighbours(second) >= 2 {
        (second, first, hydrogens_of(first))
    } else {
        return None;
    };
    // The top has to be *only* hydrogens, or turning it would swing a heavy
    // atom the caller has already decided not to swing.
    if hydrogens.len() + 1 != neighbours(top) {
        return None;
    }
    // Three hydrogens is a symmetric top -- a methyl, an ammonium -- where a
    // third of a turn is a relabelling and only the nudge off eclipse does
    // anything. *Two* is not: an amine's two hydrogens and its lone pair
    // make a three-fold rotor whose three positions are genuinely different
    // molecules to a hydrogen bond, and treating it as a top left a
    // dipeptide 2.29 kcal/mol short on the stress corpus.
    let symmetric = hydrogens.len() >= 3;
    Some(Move {
        pivot,
        axis_end: top,
        side: hydrogens,
        angles: if symmetric { SIXTH_TURN } else { TURNS },
        rigid_allowance: if symmetric { 0.0 } else { RIGID_ALLOWANCE },
        screen_iterations: SCREEN_ITERATIONS,
        about_bond: true,
        operation: Operation::Turn,
    })
}

/// Acyclic single bonds with something to swing at both ends, each with the
/// smaller of the two sides it separates.
///
/// Not [`crate::molecule::descriptors`]' rotatable-bond count:
/// that one answers a medicinal-chemistry question and excludes amides,
/// symmetric tops and the rest, because a `CF3` "rotation" is a relabelling
/// and an amide is not really single. Here the question is whether turning
/// the bond can reach a different conformer, so an amide stays in -- *cis*
/// and *trans* are different conformers, and the force field is what decides
/// between them -- while a terminal or one-heavy-neighbour end drops out,
/// since what it would swing is hydrogens.
fn torsion_moves(molecule: &Molecule3D) -> Vec<Move> {
    let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
    let mut bonds: Vec<Move> = Vec::new();
    for first in 0..molecule.len() {
        for bond in molecule.get_atom_bonds(first).unwrap_or_default() {
            let second = bond.target();
            if second <= first || bond.bond_order() != BondOrder::Single {
                continue;
            }
            if cyclic.contains(&(first.min(second), first.max(second))) {
                continue;
            }
            if heavy_degree(molecule, first) < 2 || heavy_degree(molecule, second) < 2 {
                // A methyl or an ammonium is a symmetric top, so turning it
                // is a relabelling -- except that a distance-geometry
                // structure can hand relaxation the eclipsed arrangement,
                // which is a stationary point, and a gradient cannot leave
                // one. Methylammonium came out 2.7 kcal/mol above RDKit for
                // exactly that reason. One sixtieth-of-a-turn move takes
                // eclipsed to staggered; three would be two relabellings and
                // a waste.
                if let Some(rotor) = hydrogen_rotor(molecule, first, second) {
                    bonds.push(rotor);
                }
                continue;
            }
            // Rotating either side about the bond gives the same conformer,
            // so move the smaller one and touch fewer atoms.
            let far = far_side(molecule, first, second);
            let (pivot, moving, side) = if far.len() * 2 <= molecule.len() {
                (first, second, far)
            } else {
                (second, first, far_side(molecule, second, first))
            };
            bonds.push(Move {
                pivot,
                axis_end: moving,
                side,
                angles: TURNS,
                rigid_allowance: RIGID_ALLOWANCE,
                screen_iterations: SCREEN_ITERATIONS,
                about_bond: true,
                operation: Operation::Turn,
            });
        }
    }
    bonds
}

/// Pyramidal inversion at a three-coordinate nitrogen that carries one
/// hydrogen, as a reflection of that hydrogen through the plane of the
/// nitrogen and its two heavy neighbours.
///
/// The move no rotation can make. Where an amine's N-H points relative to
/// its ring is decided by the pyramidalisation at the nitrogen, not by any
/// torsion, so the torsion search cannot reach it and a gradient cannot
/// cross the barrier between the two arrangements: whatever the distance
/// geometry produced is final. Measured against 400 MMFF-minimised ETKDGv3
/// conformers (`scratchpad/floor.py`), morpholine was the clearest case in
/// the corpus -- our eight trials all put the N-H axial, 88.9 degrees out of
/// the ring plane, at 57.63 kcal/mol, where the equatorial arrangement sits
/// at 56.21. Both are real minima and the ring torsions either side agree to
/// a degree; only the hydrogen differs.
///
/// The hydrogen is reflected rather than the nitrogen because the
/// nitrogen is in the ring, and moving it would take the ring with it, while
/// the hydrogen is a leaf and the reflection preserves its bond length and
/// both of its bond angles exactly (see [`Operation::Invert`]).
///
/// A nitrogen that is a genuine stereocentre is not excluded here, because
/// it does not need to be: [`refine_torsions`] rejects any move that leaves
/// more stereo mismatches than it found, so a centre that cannot invert in
/// reality has its inversion thrown out on the same test as everything else.
fn inversion_moves(molecule: &Molecule3D) -> Vec<Move> {
    const ONCE: &[f64] = &[0.0];
    let mut moves = Vec::new();
    for centre in 0..molecule.len() {
        if molecule.get_atomic_number(centre) != 7 {
            continue;
        }
        let neighbours = molecule.get_atom_bonds(centre).unwrap_or_default();
        if neighbours.len() != 3 {
            continue;
        }
        // A planar nitrogen has no pyramid to invert: its hydrogen is in the
        // plane already, so reflecting it through that plane is close to a
        // no-op that still costs an energy evaluation and a screening
        // relaxation to reject. Planar by its own bond orders -- an imine,
        // a nitro -- or by conjugation with a neighbour, which is what makes
        // an amide's nitrogen flat even though all three of its bonds are
        // single. A peptide has one such nitrogen per residue.
        if neighbours
            .iter()
            .any(|bond| bond.bond_order() != BondOrder::Single)
        {
            continue;
        }
        let conjugated = neighbours.iter().any(|bond| {
            let neighbour = bond.target();
            molecule.get_atomic_number(neighbour) != 1
                && molecule
                    .get_atom_bonds(neighbour)
                    .unwrap_or_default()
                    .iter()
                    .any(|further| {
                        further.target() != centre
                            && matches!(
                                further.bond_order(),
                                BondOrder::Double | BondOrder::Triple | BondOrder::Aromatic
                            )
                    })
        });
        if conjugated {
            continue;
        }
        let hydrogens: Vec<usize> = neighbours
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| molecule.get_atomic_number(target) == 1)
            .collect();
        let heavy: Vec<usize> = neighbours
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| molecule.get_atomic_number(target) != 1)
            .collect();
        if hydrogens.len() != 1 || heavy.len() != 2 {
            continue;
        }
        moves.push(Move {
            pivot: centre,
            axis_end: heavy[0],
            side: hydrogens,
            angles: ONCE,
            rigid_allowance: RIGID_ALLOWANCE,
            screen_iterations: SCREEN_ITERATIONS,
            about_bond: false,
            operation: Operation::Invert {
                in_plane: (heavy[0], heavy[1]),
            },
        });
    }
    moves
}

/// The pucker inversion of every small saturated ring that stands alone.
///
/// Five and six are the sizes [`corner_flip_moves`] refuses, on the grounds
/// that one flip of such a ring only ever makes a worse structure -- which
/// is true, and the reason this move exists: a chair has to be turned over
/// all at once, not a corner at a time.
///
/// Restricted to a ring that shares no atom with another ring. A fused or
/// bridged ring cannot be inverted on its own -- its shared atoms belong to
/// a ring that is not moving, and the branches hanging off them include that
/// ring -- so decalin, a steroid and a penicillin are all left alone. A
/// sugar's pyranose or furanose ring stands alone and is exactly the case
/// this is for: measured against a 200-conformer MMFF reference, galactose
/// was 1.76 kcal/mol short, glucose 1.25 and a nucleotide's ribose 1.17,
/// every one of them on which chair or pucker the ring took.
fn ring_inversion_moves(molecule: &Molecule3D) -> Vec<Move> {
    const ONCE: &[f64] = &[0.0];
    /// A puckered ring needs as long to settle as a flipped macrocycle: the
    /// move distorts every angle at the ring and the substituents have to
    /// travel to their new places.
    const RING_SCREEN_ITERATIONS: usize = 200;

    let rings = crate::molecule::topology::smallest_rings(molecule);
    let mut shared = vec![0usize; molecule.len()];
    for ring in &rings {
        for &atom in ring {
            shared[atom] += 1;
        }
    }

    let mut moves = Vec::new();
    for ring in &rings {
        if !(5..=6).contains(&ring.len()) || !saturated(molecule, ring) {
            continue;
        }
        if ring.iter().any(|&atom| shared[atom] > 1) {
            continue;
        }
        let in_ring: std::collections::HashSet<usize> = ring.iter().copied().collect();
        let mut branches = Vec::with_capacity(ring.len());
        for &atom in ring {
            let mut hanging = Vec::new();
            for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
                let attachment = bond.target();
                if in_ring.contains(&attachment) {
                    continue;
                }
                let mut carried = Vec::new();
                let mut seen = in_ring.clone();
                seen.insert(attachment);
                let mut stack = vec![attachment];
                while let Some(current) = stack.pop() {
                    for further in molecule.get_atom_bonds(current).unwrap_or_default() {
                        let next = further.target();
                        if seen.insert(next) {
                            carried.push(next);
                            stack.push(next);
                        }
                    }
                }
                hanging.push((attachment, carried));
            }
            // Three or more exocyclic neighbours is not a ring atom this
            // move understands: there is no pair to swap.
            if hanging.len() > 2 {
                hanging.clear();
            }
            branches.push(hanging);
        }
        let mut side: Vec<usize> = ring.clone();
        for hanging in &branches {
            for (attachment, carried) in hanging {
                side.push(*attachment);
                side.extend(carried.iter().copied());
            }
        }
        moves.push(Move {
            pivot: ring[0],
            axis_end: ring[1],
            side,
            angles: ONCE,
            rigid_allowance: f64::INFINITY,
            screen_iterations: RING_SCREEN_ITERATIONS,
            about_bond: false,
            operation: Operation::InvertRing {
                ring: ring.clone(),
                branches,
            },
        });
    }
    moves
}

/// Whether every bond of `ring` is a single bond between its own atoms.
fn saturated(molecule: &Molecule3D, ring: &[usize]) -> bool {
    ring.iter().all(|&atom| {
        molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .all(|bond| bond.bond_order() == BondOrder::Single || !ring.contains(&bond.target()))
    })
}

fn heavy_degree(molecule: &Molecule3D, atom: usize) -> usize {
    molecule
        .get_atom_bonds(atom)
        .unwrap_or_default()
        .iter()
        .filter(|bond| molecule.get_atomic_number(bond.target()) != 1)
        .count()
}

/// Every atom reachable from `second` without crossing the `first-second`
/// bond -- the half of the molecule that turning that bond moves.
fn far_side(molecule: &Molecule3D, first: usize, second: usize) -> Vec<usize> {
    let mut seen = vec![false; molecule.len()];
    seen[first] = true;
    seen[second] = true;
    let mut stack = vec![second];
    let mut side = vec![second];
    while let Some(atom) = stack.pop() {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let next = bond.target();
            if !seen[next] {
                seen[next] = true;
                side.push(next);
                stack.push(next);
            }
        }
    }
    side
}

/// The only move that reaches a ring: turn one ring atom over the line
/// joining its two ring neighbours, carrying whatever hangs off it.
///
/// A macrocycle's conformation is its ring torsions, and those cannot be
/// turned one at a time -- the ring has to close again afterwards. Rotating
/// a single corner by 180 degrees about the axis through its neighbours does
/// close: both of its ring bonds keep their lengths, and the angle between
/// them is preserved, because the rotation is a rigid motion of the corner
/// about a line through the two atoms it stays bonded to. Everything else
/// about the ring changes. This is the "corner flip" of
/// the macrocycle conformer-search literature, and it is the cheapest move
/// that crosses between ring conformations.
///
/// `erythromycin-frag` is what this is for: its bonds are all in the ring,
/// so the torsion moves above find nothing to turn, and its conformer came
/// out 3.3 kcal/mol above RDKit's with the whole difference in the MMFF
/// torsion term -- eight of its thirteen ring torsions in different
/// positions, not a clash.
fn corner_flip_moves(molecule: &Molecule3D) -> Vec<Move> {
    /// Rings smaller than this have essentially one or two conformations
    /// that the touch-up's planarity and angle terms already find, and
    /// flipping a corner of one only ever makes a worse structure that costs
    /// a relaxation to reject.
    const SMALLEST_FLIPPABLE_RING: usize = 7;
    const FLIP: &[f64] = &[180.0];
    /// See [`Move::screen_iterations`].
    const FLIP_SCREEN_ITERATIONS: usize = 200;

    let mut moves = Vec::new();
    for ring in crate::molecule::topology::smallest_rings(molecule) {
        if ring.len() < SMALLEST_FLIPPABLE_RING {
            continue;
        }
        // A ring with a double bond in it is held flat where the double
        // bond is, and flipping a corner of one is a relaxation spent to
        // reject it. Six-membered rings are out for a different reason:
        // their second chair is not one corner's flip away, so allowing
        // them made cyclohexane four times slower and morpholine no
        // better.
        if !saturated(molecule, &ring) {
            continue;
        }
        for index in 0..ring.len() {
            let before = ring[(index + ring.len() - 1) % ring.len()];
            let corner = ring[index];
            let after = ring[(index + 1) % ring.len()];
            let Some(side) = branch_between(molecule, corner, before, after) else {
                continue;
            };
            moves.push(Move {
                pivot: before,
                axis_end: after,
                side,
                angles: FLIP,
                rigid_allowance: f64::INFINITY,
                screen_iterations: FLIP_SCREEN_ITERATIONS,
                about_bond: false,
                operation: Operation::Turn,
            });
        }
    }
    moves
}

/// `corner` and everything hanging off it, reached without passing through
/// `before` or `after`.
///
/// `None` when that is not a branch at all: in a fused or bridged ring
/// system the search comes back around to the rest of the ring, and turning
/// a corner would then take most of the molecule with it. Half the atoms is
/// the cutoff, which no genuine substituent on a ring atom reaches.
fn branch_between(
    molecule: &Molecule3D,
    corner: usize,
    before: usize,
    after: usize,
) -> Option<Vec<usize>> {
    let mut seen = vec![false; molecule.len()];
    seen[before] = true;
    seen[after] = true;
    seen[corner] = true;
    let mut stack = vec![corner];
    let mut branch = vec![corner];
    while let Some(atom) = stack.pop() {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let next = bond.target();
            if !seen[next] {
                seen[next] = true;
                branch.push(next);
                stack.push(next);
            }
        }
    }
    (branch.len() * 2 <= molecule.len()).then_some(branch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::embed::Embed3D;
    use crate::molecule::molecular_graph::MolecularGraph;

    fn molecule(smiles: &str) -> Molecule3D {
        let mut graph = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        graph.add_hydrogens();
        super::super::to_molecule3d_topology(&graph)
    }

    /// A hydroxyl's hydrogen is a conformational degree of freedom, not a
    /// relabelling, so it gets the full set of turns rather than the
    /// methyl's single nudge off eclipse -- and it has to get them from the
    /// bond the heavy-atom path refuses, since the oxygen has one heavy
    /// neighbour.
    ///
    /// Worth testing at the move rather than through an energy: the
    /// relaxation can turn a hydroxyl on its own whenever no barrier is in
    /// the way, so the small molecules where this is easy to assert do not
    /// need the move, and the ones that need it are a peptide.
    #[test]
    fn a_hydroxyl_gets_a_single_hydrogen_rotor() {
        let ethanol = molecule("CCO");
        let oxygen = (0..ethanol.len())
            .find(|&atom| ethanol.get_atomic_number(atom) == 8)
            .expect("ethanol has an oxygen");
        let rotor = torsion_moves(&ethanol)
            .into_iter()
            .find(|candidate| candidate.axis_end == oxygen)
            .expect("the C-O bond should give a rotor");
        assert_eq!(rotor.side.len(), 1, "it turns the one hydroxyl hydrogen");
        assert_eq!(
            ethanol.get_atomic_number(rotor.side[0]),
            1,
            "and nothing but that hydrogen"
        );
        assert_eq!(rotor.angles(), TURNS, "with every turn, not just the nudge");
    }

    /// An amine is not a symmetric top, whatever its hydrogen count looks
    /// like: two hydrogens and a lone pair make a three-fold rotor whose
    /// three positions are different molecules to a hydrogen bond. Treating
    /// it as a top -- one nudge off eclipse, no rigid allowance -- left a
    /// dipeptide 2.29 kcal/mol short of a 200-conformer reference, and cost
    /// six molecules on the held-out corpus.
    #[test]
    fn an_amine_is_not_a_symmetric_top() {
        let amine = molecule("CCN");
        let nitrogen = (0..amine.len())
            .find(|&atom| amine.get_atomic_number(atom) == 7)
            .expect("ethylamine has a nitrogen");
        let rotor = torsion_moves(&amine)
            .into_iter()
            .find(|candidate| candidate.axis_end == nitrogen)
            .expect("the C-N bond should give a rotor");
        assert_eq!(rotor.side.len(), 2, "both amine hydrogens turn together");
        assert_eq!(rotor.angles(), TURNS, "with every turn, not one nudge");
        assert!(
            rotor.rigid_allowance > 0.0,
            "and a real conformational move gets a real allowance"
        );
    }

    /// A methyl is still a symmetric top: one nudge off eclipse, and no
    /// allowance for a rigid rotation that goes uphill.
    #[test]
    fn a_methyl_keeps_its_single_nudge() {
        let ethane = molecule("CC");
        let rotor = torsion_moves(&ethane)
            .into_iter()
            .next()
            .expect("ethane's one bond should give a rotor");
        assert_eq!(
            rotor.side.len(),
            3,
            "a methyl's three hydrogens turn together"
        );
        assert_eq!(rotor.angles().len(), 1, "one sixth of a turn, once");
        assert_eq!(rotor.rigid_allowance, 0.0);
    }

    /// Only a pyramidal nitrogen with a hydrogen to move gets an inversion:
    /// an amide's nitrogen is flat and has nothing to invert, and a
    /// tertiary amine's inversion would have to move three heavy atoms.
    #[test]
    fn inversion_is_offered_where_a_pyramid_can_turn_over() {
        let morpholine = molecule("C1COCCN1");
        assert_eq!(
            inversion_moves(&morpholine).len(),
            1,
            "morpholine's one N-H"
        );

        let amide = molecule("CC(=O)NC");
        assert!(
            inversion_moves(&amide).is_empty(),
            "an amide nitrogen is planar"
        );
        let tertiary = molecule("CN(C)C");
        assert!(
            inversion_moves(&tertiary).is_empty(),
            "a tertiary amine has no hydrogen to reflect"
        );
    }

    /// The reflection is an isometry that fixes its plane, so the inverted
    /// hydrogen keeps its bond length and both of its bond angles exactly.
    /// That is what makes the move cheap to judge: nothing local changed.
    #[test]
    fn inverting_a_nitrogen_preserves_its_local_geometry() {
        let mut graph = MolecularGraph::from_smiles_single("C1COCCN1").expect("should parse");
        graph.add_hydrogens();
        let embedded = graph.embed_3d().expect("should embed");
        let topology = molecule("C1COCCN1");
        let inversion = inversion_moves(&topology)
            .into_iter()
            .next()
            .expect("morpholine inverts");
        let before: Vec<Vector3D> = (0..embedded.len())
            .map(|atom| embedded.get_atom_position(atom).unwrap())
            .collect();
        let mut after = before.clone();
        inversion.apply(&mut after, 0.0);

        let hydrogen = inversion.side[0];
        let (centre, Operation::Invert { in_plane }) = (inversion.pivot, &inversion.operation)
        else {
            panic!("an inversion should carry a plane");
        };
        for anchor in [centre, in_plane.0, in_plane.1] {
            let was = before[hydrogen].distance(&before[anchor]);
            let now = after[hydrogen].distance(&after[anchor]);
            assert!(
                (was - now).abs() < 1e-9,
                "distance to {anchor} changed: {was:.6} to {now:.6}"
            );
        }
        assert!(
            before[hydrogen].distance(&after[hydrogen]) > 0.3,
            "the hydrogen should actually have moved"
        );
    }

    /// The ring inversion has to leave the ring a ring and every centre as
    /// it was. Both follow from how it is built -- the ring is reflected,
    /// which is an isometry, and each branch is placed by a proper rotation
    /// with the two at each atom swapped -- so this is the check that the
    /// construction does what the reasoning says.
    #[test]
    fn inverting_a_ring_keeps_its_bonds_and_its_centres() {
        let smiles = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O";
        let mut graph = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        graph.add_hydrogens();
        let embedded = graph.embed_3d().expect("should embed");
        let topology = molecule(smiles);
        let inversion = chair_or_envelope(&topology).expect("a pyranose inverts");

        let before: Vec<Vector3D> = (0..embedded.len())
            .map(|atom| embedded.get_atom_position(atom).unwrap())
            .collect();
        let mut after = before.clone();
        inversion.apply(&mut after, 0.0);

        // Every bond in the molecule keeps its length: the ring's because a
        // reflection preserves distances, the branches' because they move
        // rigidly.
        for atom in 0..topology.len() {
            for bond in topology.get_atom_bonds(atom).unwrap_or_default() {
                let was = before[atom].distance(&before[bond.target()]);
                let now = after[atom].distance(&after[bond.target()]);
                assert!(
                    (was - now).abs() < 0.05,
                    "bond {atom}-{} went from {was:.3} to {now:.3}",
                    bond.target()
                );
            }
        }

        // And the ring actually turned over: its atoms are on the other side
        // of their own mean plane.
        let Operation::InvertRing { ref ring, .. } = inversion.operation else {
            panic!("expected a ring inversion");
        };
        let moved = ring
            .iter()
            .map(|&atom| before[atom].distance(&after[atom]))
            .fold(0.0, f64::max);
        assert!(moved > 0.2, "no ring atom moved more than {moved:.2} A");
    }

    fn chair_or_envelope(topology: &Molecule3D) -> Option<Move> {
        ring_inversion_moves(topology).into_iter().next()
    }

    /// A fused ring cannot be inverted on its own: its shared atoms belong
    /// to a ring that is not moving.
    #[test]
    fn a_fused_ring_is_left_alone() {
        assert!(
            ring_inversion_moves(&molecule("C1CCC2CCCCC2C1")).is_empty(),
            "decalin's rings share two atoms"
        );
        assert_eq!(
            ring_inversion_moves(&molecule("C1CCCCC1")).len(),
            1,
            "cyclohexane stands alone"
        );
    }

    /// An embedded structure, its rotatable bonds as torsion-space
    /// coordinates, and the field the search would relax it in.
    fn torsion_space(smiles: &str) -> (Vec<Vector3D>, Vec<RotatableBond>, Field) {
        let mut graph = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        graph.add_hydrogens();
        let embedded = graph.embed_3d().expect("should embed");
        let positions = (0..embedded.len())
            .map(|atom| embedded.get_atom_position(atom).unwrap())
            .collect();
        let bonds = torsion_moves(&molecule(smiles))
            .into_iter()
            .filter(|candidate| candidate.about_bond)
            .map(|candidate| RotatableBond {
                pivot: candidate.pivot,
                axis_end: candidate.axis_end,
                side: candidate.side,
            })
            .collect();
        let field = Field::best_for(&embedded).expect("should type");
        (positions, bonds, field)
    }

    /// The torque is the derivative of the energy along a turn, checked
    /// against central differences at a structure away from any minimum.
    #[test]
    fn the_torsion_gradient_is_the_slope_of_a_turn() {
        let (mut positions, bonds, field) = torsion_space("CCCCOC(=O)N");
        assert!(bonds.len() >= 4, "the chain should have rotatable bonds");
        for (k, bond) in bonds.iter().enumerate() {
            rotate(
                &mut positions,
                bond.pivot,
                bond.axis_end,
                &bond.side,
                0.3 + 0.2 * k as f64,
            );
        }
        let mut cartesian = vec![Vector3D::default(); positions.len()];
        field.energy_and_gradient(&positions, &mut cartesian);
        let mut analytic = vec![0.0; bonds.len()];
        torsion_gradient(&positions, &cartesian, &bonds, &mut analytic);

        let step = 1e-5;
        for (bond, &slope) in bonds.iter().zip(&analytic) {
            let turned = |angle: f64| {
                let mut turned = positions.clone();
                rotate(&mut turned, bond.pivot, bond.axis_end, &bond.side, angle);
                field.energy(&turned)
            };
            let numeric = (turned(step) - turned(-step)) / (2.0 * step);
            assert!(
                (numeric - slope).abs() <= 1e-4 * numeric.abs().max(1.0),
                "bond {}-{}: analytic {slope:.6}, numeric {numeric:.6}",
                bond.pivot,
                bond.axis_end
            );
        }
    }

    /// Relaxing in torsion space lowers the energy and moves nothing else:
    /// every bond length and every distance across an angle survives, since
    /// each step is only turns about bonds.
    #[test]
    fn a_torsion_space_relaxation_only_turns_bonds() {
        let (mut positions, bonds, field) = torsion_space("CCCCCC(C)O");
        for bond in &bonds {
            rotate(&mut positions, bond.pivot, bond.axis_end, &bond.side, 2.0);
        }
        let topology = molecule("CCCCCC(C)O");
        let mut pairs = Vec::new();
        for atom in 0..topology.len() {
            let neighbours: Vec<usize> = topology
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .map(|bond| bond.target())
                .collect();
            for (index, &first) in neighbours.iter().enumerate() {
                pairs.push((atom, first));
                for &second in &neighbours[index + 1..] {
                    pairs.push((first, second));
                }
            }
        }
        let before = positions.clone();
        let start = field.energy(&positions);
        let relaxed = relax_torsions(&mut positions, &field, &bonds, 50);

        assert!(
            relaxed < start - 1.0,
            "{start:.3} should have relaxed, got {relaxed:.3}"
        );
        assert!((relaxed - field.energy(&positions)).abs() < 1e-9);
        for (a, b) in pairs {
            let was = before[a].distance(&before[b]);
            let now = positions[a].distance(&positions[b]);
            assert!(
                (was - now).abs() < 1e-9,
                "{a}-{b} changed: {was:.6} to {now:.6}"
            );
        }
    }
}
