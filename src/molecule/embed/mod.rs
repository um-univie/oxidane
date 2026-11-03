//! Distance-geometry ("ETKDG-lite") 3D coordinate embedding.
//!
//! Stage 1: bounds matrix -> triangle-inequality smoothing -> classical 4D MDS
//! -> for a molecule with stereochemistry, `fourd`'s four-dimensional
//! resolution of it -> gradient refinement against a bounds-violation
//! energy. Produces geometrically valid structures (correct bond
//! lengths/angles).
//!
//! Stage 2: `chirality` adds tetrahedral (`TH`), cis/trans double-bond, and
//! allene (`AL`) consistency terms plus a repair fallback; `bounds` adds
//! trans/adjacent angle bounds for square-planar (`SP`) centres (which,
//! unlike the others, has no separate handedness once the trans-pairing is
//! fixed, so needs no extra energy term); `planarity` adds a ring/local-
//! centre out-of-plane restraint applied as a touch-up on each trial.
//! With relaxation enabled, each trial is then minimized with MMFF94 or
//! UFF -- with the chirality terms still on as a barrier, see
//! `StereoRestrained` -- and ranked by that energy, prioritising
//! stereochemical correctness.
//! Disconnected components (salts) are embedded and relaxed independently,
//! then packed side by side (see `run`).
//!
//! `TH` and `SP` were cross-checked against RDKit's
//! `AssignStereochemistryFrom3D` (`tests/embed_oracle.py`) and, in `TH`'s
//! case, this caught a real sign inversion in
//! [`crate::molecule::chirality::ChiralClassifier::determine_tetrahedral_chirality`]
//! (the oracle used to *read back* chirality from a `Molecule3D`, e.g. when
//! writing SMILES for a structure read from XYZ/SDF) - since fixed there
//! too, see that function's doc comment.
//! `AL`'s absolute handedness could *not* be cross-checked the same way -
//! RDKit doesn't perceive allene stereo from 3D coordinates in the version
//! used here - so it is verified only to be internally self-consistent (see
//! `chirality::AllenCenter`'s doc comment).
//!
//! A UFF/MMFF-lite refinement pass (`forcefield`) adds harmonic-in-cosine
//! angle-bend and periodic-torsion restraints, applied in the same
//! touch-up-only style as `planarity` (see `embed_connected`), and limited
//! to those two terms: bond-stretch and van-der-Waals-style
//! non-bonded repulsion are already covered by the bounds-violation energy
//! itself (a tight flat bottom for 1-2 pairs, a VDW-radius-based lower bound
//! for every other pair), so adding them again would be redundant. The
//! touch-up runs for at most 300 L-BFGS iterations; force-field relaxation
//! then removes residual strain from each surviving trial.

pub(crate) mod bounds;
mod chirality;
mod forcefield;
mod mds;
mod planarity;
mod fourd;
mod torsion_preferences;
mod torsion_search;
mod refine;

#[cfg(test)]
mod tests;

use crate::{
    molecule::{
        base::MoleculeCore, chirality::MoleculeChirality, graph::MoleculeGraph,
        molecule3d::Molecule3D,
    },
    vector::Vector3D,
};
use rand::{SeedableRng, rngs::StdRng};

/// Configuration for [`Embed3D::embed_3d_with_config`].
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    /// Seed for the deterministic trial-distance sampler. Successive attempts
    /// derive their own seed from this one, so a fixed `seed` gives fully
    /// reproducible output.
    pub seed: u64,
    /// Number of independent embedding trials to attempt before giving up.
    /// The lowest-energy successful trial is returned, prioritising correct
    /// stereochemistry. With `relax` enabled, trials are compared after
    /// relaxation using the selected force field's energy.
    ///
    /// More than one trial runs on `rayon`'s thread pool, since the trials
    /// share nothing. The result does not depend on it: trials are
    /// collected in order and compared in that order, and the output is
    /// bit-identical to the same call with `RAYON_NUM_THREADS=1`. A caller
    /// that wants the embedding on one thread can set that, or embed a
    /// library of molecules in its own parallel loop and leave this at one.
    pub max_attempts: usize,
    /// Maximum number of initial refinement (L-BFGS) iterations per trial.
    pub max_iterations: usize,
    /// Whether to finish by relaxing against
    /// [`crate::molecule::forcefield::Field::best_for`] -- MMFF94 where it
    /// types the molecule, UFF where it does not.
    /// On by default.
    ///
    /// Distance geometry answers a different question than "what does this
    /// molecule look like": it produces coordinates consistent with a set of
    /// distances, which leaves bonds and angles a little off and torsions
    /// wherever the distances left them. Every toolkit closes that gap the
    /// same way -- RDKit's users call `MMFFOptimizeMoleculeConfs`, Open
    /// Babel does it inside `--gen3d`.
    ///
    /// Measured against RDKit's ETKDGv3 on `tests/embed_benchmark.py`, it
    /// moves this crate from beating it on 4 of 20 molecules to 20 of 20.
    /// Each surviving trial is relaxed, so cost grows with `max_attempts`.
    /// It is still a flag rather than
    /// unconditional because relaxation is the step that assumes the force
    /// field has parameters for the molecule; anything it cannot type is
    /// returned unrelaxed rather than refused. Turning it off also gives
    /// the distance-geometry structure on its own, which is what
    /// `crate::tests` asserts the bounds against -- relaxation exists to
    /// overrule those bounds where they are approximate.
    pub relax: bool,
    /// Whether to finish by turning each rotatable bond to its other
    /// staggered positions, keeping the turns that lower the force-field
    /// energy. On by default, and only reachable when `relax` is on, since
    /// it is the force field that judges each turn.
    ///
    /// Distance geometry and relaxation together cannot cross a torsional
    /// barrier: whichever rotamer a trial happened to land in is the one it
    /// keeps. Extra trials re-roll every torsion at once and so find better
    /// rotamers only by luck -- this turns them one at a time and keeps what
    /// helps, which is both cheaper and more thorough. A macrocycle, which
    /// has no rotatable bonds at all, gets corner flips instead.
    ///
    /// Measured on `tests/embed_budget_benchmark.py` at eight trials a side,
    /// it takes the peptide from 68.8 to 61.7 kcal/mol, atorvastatin from
    /// 41.3 to 35.3 and the erythromycin fragment from 28.8 to 18.1, all
    /// below anything RDKit's ETKDGv3 reached from the same budget, and moves
    /// the corpus from 15 of 20 molecules at least matching RDKit to 20 of
    /// 20. It costs most of a second per trial on the large flexible ones and
    /// nothing measurable on anything under about forty atoms.
    pub torsion_search: bool,
    /// Whether the experimental torsion preferences written for rings of
    /// eight atoms or fewer are in the rule set. Off by default.
    ///
    /// RDKit makes the same split and the same default: its ETKDGv3 has the
    /// macrocycle rules on -- so this crate does too, unconditionally -- and
    /// keeps the small-ring ones for a separate `srETKDGv3` preset. Measured
    /// here they are a wash in aggregate and a real gain on exactly the
    /// molecules this embedder is weakest on: morpholine's chair goes from
    /// 1.4 kcal/mol behind RDKit to a tie, and a pentacyclic triterpene from
    /// 5.5 behind to 2.4. Against that, the corpora lose a row each -- 251
    /// of 287 PubChem molecules at least matching RDKit rather than 249,
    /// while the mean improves from -2.74 to -2.87 kcal/mol.
    ///
    /// So: available, documented, and not the default, because the
    /// aggregate does not ask for it.
    pub small_ring_torsions: bool,
    /// Whether each trial is refined against every restraint at once, or
    /// against the distance bounds with a touch-up to follow. Off by
    /// default; [`EmbedConfig::fast`] turns it on.
    ///
    /// RDKit's ETKDG does the former -- its first minimisation carries the
    /// experimental torsion terms and the basic-knowledge terms from the
    /// start -- and where nothing downstream fixes torsions, so should
    /// this: at one trial with no torsion search it takes 56 of 72 corpus
    /// molecules at least matching RDKit to 62, and the mean from +0.51 to
    /// -0.02 kcal/mol, for the same total iteration budget.
    ///
    /// It is off by default because the torsion search reaches the same
    /// torsions by turning bonds, so with the search on this buys a wash
    /// -- 246 of 287 PubChem molecules against 250, at a mean of -2.89
    /// against -2.73 -- and costs about 40% of an embedding, since every
    /// iteration of every trial then evaluates every term.
    pub combined_refinement: bool,
}

impl EmbedConfig {
    /// One trial, no torsion search: the cheapest structure this crate will
    /// produce, and the one that costs what another toolkit's does.
    ///
    /// The default spends five trials and then searches torsion space,
    /// which is what wins the flexible molecules and is also about four
    /// fifths of the time. Measured over 287 PubChem molecules against
    /// RDKit's ETKDGv3 plus an MMFF94 minimisation -- the same job -- this
    /// preset runs at 0.86x their total time, 0.84x over the curated corpus
    /// and 1.03x over 1634 molecules stratified by size, where the default
    /// is 5.5x. Most of the speed comes from the minimiser's bookkeeping --
    /// see the `DenseHessian` and `wide_dot` in [`crate::molecule::minimize`]. What is left is at
    /// the ends of the size range, and `NEXT.md` has the curve.
    ///
    /// It is also the configuration whose trials are shaped by every
    /// restraint at once rather than by the bounds with a touch-up to
    /// follow, since nothing downstream will fix a torsion for it: that is
    /// worth 0.5 kcal/mol of mean difference against RDKit and most of the
    /// 1.1x this preset used to cost.
    ///
    /// Use it for a library of molecules, where the per-conformer quality
    /// matters less than getting through them; use the default when the
    /// conformer is the answer.
    pub fn fast() -> Self {
        EmbedConfig {
            max_attempts: 1,
            torsion_search: false,
            combined_refinement: true,
            ..Default::default()
        }
    }
}

impl Default for EmbedConfig {
    fn default() -> Self {
        EmbedConfig {
            seed: 0xE7D6_5EED,
            max_attempts: 5,
            max_iterations: 300,
            relax: true,
            torsion_search: true,
            small_ring_torsions: false,
            combined_refinement: false,
        }
    }
}

/// Errors that can occur while embedding a molecule into 3D coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedError {
    /// Fewer than 2 atoms - nothing meaningful to embed.
    TooFewAtoms,
    /// The distance bounds derived from the molecular graph are internally
    /// inconsistent (a lower bound exceeds its matching upper bound) and
    /// could not be repaired by a small tolerance widening.
    BoundsInfeasible,
    /// No trial embedding converged to an acceptable bounds-violation energy
    /// within `max_attempts`.
    MaxAttemptsExceeded,
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::TooFewAtoms => write!(f, "molecule has too few atoms to embed"),
            EmbedError::BoundsInfeasible => {
                write!(f, "distance bounds are infeasible (lower bound exceeds upper bound)")
            }
            EmbedError::MaxAttemptsExceeded => {
                write!(f, "no embedding trial converged within max_attempts")
            }
        }
    }
}

impl std::error::Error for EmbedError {}

/// Generates 3D coordinates for a molecule via distance geometry.
///
/// Implemented generically for any type that exposes the molecular graph
/// (bonds, atomic numbers, chirality) needed to build a distance-bounds
/// matrix. The result is always a fresh [`Molecule3D`] with explicit
/// hydrogens added (via [`MoleculeCore::add_hydrogens`]) before embedding.
pub trait Embed3D: MoleculeCore + MoleculeGraph + MoleculeChirality {
    fn embed_3d(&self) -> Result<Molecule3D, EmbedError>
    where
        Self: Sized,
    {
        self.embed_3d_with_config(&EmbedConfig::default())
    }

    fn embed_3d_with_config(&self, config: &EmbedConfig) -> Result<Molecule3D, EmbedError>
    where
        Self: Sized,
    {
        run(self, config)
    }
}

/// Copies the atoms/bonds/charges/etc. of `mol` into a fresh [`Molecule3D`]
/// with zeroed positions, preserving bond topology exactly (unlike
/// `Molecule3D::from_atoms`, which would try to re-derive bonds from
/// positions via `identify_bonds`).
fn to_molecule3d_topology<T: MoleculeCore>(mol: &T) -> Molecule3D {
    let n = mol.len();
    Molecule3D {
        atomic_numbers: mol.atomic_numbers().to_vec(),
        atom_classes: mol.atom_classes().clone(),
        charges: mol.charges().to_vec(),
        chiral_classes: mol.chiral_classes().map(|c| c.to_vec()),
        isotopes: mol.isotopes().cloned(),
        positions: vec![Vector3D::default(); n],
        radical_states: mol.radical_states().to_vec(),
        atom_bonds: mol.atom_bonds().clone(),
    }
}

/// Just the SMARTS of the experimental torsion rules, so tests can check
/// the whole set is readable without exposing the parameters.
#[cfg(test)]
pub(crate) fn torsion_preference_smarts() -> Vec<&'static str> {
    torsion_preferences::TORSION_PREFERENCES
        .iter()
        .chain(torsion_preferences::MACROCYCLE_PREFERENCES)
        .chain(torsion_preferences::SMALL_RING_PREFERENCES)
        .map(|&(smarts, _)| smarts)
        .collect()
}

/// The smoothed bounds matrix for a molecule, as `(atoms, [(i, j, lower,
/// upper)])`, with hydrogens added exactly as an embedding would.
///
/// For `examples/bounds_dump.rs` and the bounds-quality probe beside it: a
/// bound is what the embedder knows before it has any coordinates, and
/// measuring a change to one through an end-to-end conformer energy cannot
/// resolve it -- the refinement and the force field between them absorb most
/// of the difference.
/// One `(i, j, lower, upper)` row per pair, as [`bounds_for_dump`] reports
/// them.
pub type BoundsListing = Vec<(usize, usize, f64, f64)>;

pub fn bounds_for_dump<T: MoleculeCore + MoleculeGraph + MoleculeChirality>(
    molecule: &T,
) -> Result<(usize, BoundsListing), EmbedError> {
    let mut working = to_molecule3d_topology(molecule);
    working.add_hydrogens();
    let rings = bounds::ring_set(&working);
    let geometry = bounds::Geometry::for_molecule(&working, &rings);
    let mut matrix = bounds::build_bounds(&working, &geometry, &rings)?;
    bounds::smooth_bounds(&mut matrix)?;
    let atoms = working.len();
    let mut pairs = Vec::with_capacity(atoms * atoms / 2);
    for i in 0..atoms {
        for j in (i + 1)..atoms {
            pairs.push((i, j, bounds::get_lower(&matrix, i, j), bounds::get_upper(&matrix, i, j)));
        }
    }
    Ok((atoms, pairs))
}

fn run<T: MoleculeCore + MoleculeGraph + MoleculeChirality>(
    mol: &T,
    config: &EmbedConfig,
) -> Result<Molecule3D, EmbedError> {
    let mut working = to_molecule3d_topology(mol);
    working.add_hydrogens();

    // A lone atom has no distance to satisfy: it is already embedded, and
    // the origin is where it goes. PubChem holds thousands of these --
    // `[La+3]`, `[177Ta]` -- and refusing them made 16 of a random 6000
    // fail for no geometric reason, where RDKit returns the one-atom
    // conformer. Only an empty molecule is left to refuse.
    if working.len() == 1 {
        working.positions = vec![Vector3D::default()];
        return Ok(working);
    }
    if working.is_empty() {
        return Err(EmbedError::TooFewAtoms);
    }

    let components = working.get_components();
    if components.len() <= 1 {
        let positions = embed_connected(&working, config)?;
        working.positions = positions;
        return Ok(working);
    }

    // Disconnected input (e.g. a salt: "CC(=O)[O-].[Na+]" parsed - unusually
    // for this crate, whose SMILES reader normally splits fragments into
    // separate MolecularGraphs - into one multi-fragment MolecularGraph/
    // Molecule3D by hand). Each component has no distance-bounds relationship
    // to any other (nothing in `bounds::build_bounds` constrains cross-
    // component pairs), so embed every component independently and then pack
    // them side by side along the x axis, spaced by their bounding-sphere
    // radii plus a fixed van-der-Waals-scale gap - not a tight/physical
    // packing, just enough to guarantee components don't overlap.
    const PACKING_GAP: f64 = 3.0;
    let mut positions = vec![Vector3D::default(); working.len()];
    let mut cursor_x = 0.0;
    for atoms in &components {
        // A lone atom (e.g. a bare counter-ion like [Na+]) has no bonds to
        // build a distance-bounds matrix from and needs no embedding - place
        // it directly and let the packing loop below give it a slot.
        let mut sub_positions = if atoms.len() == 1 {
            vec![Vector3D::default()]
        } else {
            let sub = extract_component(&working, atoms);
            embed_connected(&sub, config)?
        };

        let centroid = sub_positions.iter().fold(Vector3D::default(), |acc, &p| acc + p) / sub_positions.len() as f64;
        let radius = sub_positions.iter().map(|&p| (p - centroid).length()).fold(0.0_f64, f64::max);

        let offset_x = cursor_x + radius;
        for p in &mut sub_positions {
            *p = *p - centroid + Vector3D::new(offset_x, 0.0, 0.0);
        }
        cursor_x = offset_x + radius + PACKING_GAP;

        for (local, &global) in atoms.iter().enumerate() {
            positions[global] = sub_positions[local];
        }
    }

    working.positions = positions;
    Ok(working)
}

/// Minimise until the energy stops moving, rather than until a fixed
/// iteration count runs out.
///
/// A fixed budget is a budget per molecule size, which is the wrong shape:
/// benzene is finished in tens of iterations and a ninety-atom peptide is
/// nowhere near. At the 500 iterations this used to spend, one peptide trial
/// came back at 82.5 kcal/mol and was still descending -- it reaches 68.8,
/// which is *lower than anything RDKit's ETKDGv3 found in eight tries*, so
/// the conformer was being generated and then thrown away by a stopwatch.
/// Worse, the trial ranking below compares these numbers: an under-relaxed
/// trial loses to a smaller molecule's fully relaxed one on an energy it does
/// not actually have.
///
/// Successive calls rather than one large budget because a fresh call also
/// discards the curvature history, which is what lets the next round move
/// again after a stall. Rounds stop paying off quickly -- the peptide's third
/// round buys 0.002 kcal/mol -- so a round that improves less than
/// `RELAX_ROUND_TOLERANCE` ends it, and a molecule that converges inside the
/// first round pays one extra near-empty call for the check.
pub(super) fn minimize_to_convergence<E: crate::molecule::minimize::Energy + ?Sized>(
    field: &E,
    positions: &mut [Vector3D],
) -> f64 {
    /// Iterations per round. Large enough that a small molecule finishes
    /// inside the first one.
    const RELAX_ITERATIONS: usize = 500;
    /// Rounds before giving up on a structure that will not settle.
    ///
    /// Eight is near the knee, and both sides of it were measured on thirty
    /// molecules of 130 to 390 atoms, where this is the constant that binds
    /// -- a 248-atom molecule spends all eight rounds and is still
    /// descending, where anything under about a hundred atoms converges
    /// inside the first one and never sees this number at all.
    ///
    /// Two rounds: 2.8x faster and **+12.5 kcal/mol** on the mean, worse on
    /// 30 of 30. Sixteen rounds: 24% slower for **-0.57 kcal/mol**, better
    /// on some and worse on none. So the grinding is buying real energy
    /// right up to about here and very little past it.
    const RELAX_ROUNDS: usize = 8;
    /// Energy improvement, in kcal/mol, below which another round is waste.
    ///
    /// Not 1e-3: at that setting the peptide spends all eight rounds, and
    /// the last five of them together buy 0.015 kcal/mol for 80% of the
    /// relaxation's time. A minimiser grinding towards a tight gradient
    /// tolerance is the most expensive part of an embedding and the least
    /// valuable.
    const RELAX_ROUND_TOLERANCE: f64 = 0.05;

    let mut energy = crate::molecule::minimize::minimize(
        positions,
        field,
        RELAX_ITERATIONS,
        RELAX_GRADIENT_TOLERANCE,
    );
    for _ in 1..RELAX_ROUNDS {
        let next = crate::molecule::minimize::minimize(
            positions,
            field,
            RELAX_ITERATIONS,
            RELAX_GRADIENT_TOLERANCE,
        );
        if !next.is_finite() || energy - next < RELAX_ROUND_TOLERANCE {
            return next;
        }
        energy = next;
    }
    energy
}

/// What the trial refinement gets on top of `EmbedConfig::max_iterations`.
///
/// The budget is spent in one refinement with every restraint active. Three
/// hundred more changes nothing measurable, and three hundred fewer gives
/// back most of what the change bought.
const EXTRA_REFINEMENT_ITERATIONS: usize = 300;

/// Force threshold for relaxation, matching the `GRADIENT_TOLERANCE` in
/// [`crate::molecule::forcefield`], which is the number a caller minimising
/// against the same field by hand would get.
///
/// What [`crate::molecule::minimize`] tests is not a norm but the largest
/// single scaled component, `max_i |g_i| * max(|x_i|, 1)`, in kcal/mol per
/// Angstrom. This asked for 1e-6 of that -- a hundred times tighter than
/// the field's own default, and four orders tighter than the effective
/// threshold RDKit's relative test works out to on a molecule whose energy
/// is around a hundred kcal/mol. It was not reached: relaxation ended on
/// the step-size test or the iteration budget instead, having spent the
/// tail of every descent grinding at a tolerance nothing asked for.
///
/// Loosening it to the field's own number is worth 3% of an embedding at
/// one trial and no torsion search (805 ms to 779 over the corpus) and
/// changes no conformer worth the name: 62 of 72 molecules still at least
/// match RDKit, at a mean of -0.05 rather than -0.04 kcal/mol. 1e-2 buys
/// another 2% on the same numbers and is not here because there is no
/// principle behind it; this one is the same tolerance as the rest of the
/// crate.
///
/// [`minimize_to_convergence`]'s `RELAX_ROUND_TOLERANCE` was swept the same
/// way and does nothing: 0.2 and 0.5 measure as 781 and 777 ms against
/// 0.05's 779, because a relaxation that needs a second round is rare once
/// the first one is allowed to stop.
const RELAX_GRADIENT_TOLERANCE: f64 = 1e-4;

/// The force field with the embedder's stereochemistry restraints still on.
///
/// A distance-geometry structure is strained -- 160 kcal/mol for glucose,
/// 500 for cholesterol, where the relaxed answer is 85 and 110 -- and a
/// stereocentre in a ring full of other strain is not yet firmly on one side
/// of its plane. Relaxing such a structure against the force field alone
/// walks it downhill through whichever inversion is nearest: glucose came
/// back with up to four of its five centres wrong by RDKit's reading, which
/// `relax_positions` then had to throw away, returning the strained
/// structure it started from. Both of the corpus's worst molecules were that,
/// and both were stereo-dense saturated rings.
///
/// So the first descent keeps the chirality terms that got the structure
/// right, and only once the strain is gone does the field have the last
/// word. The terms are the embedder's own, so this costs no new machinery
/// and nothing at all on a molecule with no stereocentres, where
/// [`StereoConstraints`] is empty.
struct StereoRestrained<'a> {
    field: &'a crate::molecule::forcefield::Field,
    stereo: &'a chirality::StereoConstraints,
    /// Somewhere for the stereo terms, which only offer a combined
    /// energy-and-gradient entry point, to write a gradient nobody wants.
    discard: std::cell::RefCell<Vec<Vector3D>>,
}

/// What one unit of stereo deficit is worth in kcal/mol.
///
/// The restraints are written for distance geometry, where the energy scale
/// is whatever the bounds violations make it; against a force field their
/// natural size -- one, at a flattened centre -- is nothing next to the
/// hundreds of kcal/mol of strain being relieved around them, and the
/// centre inverts anyway. A hundred makes inverting a centre cost more than
/// any strain relief on offer, and costs nothing at a centre that is on the
/// right side, where the term is flat.
const STEREO_RESTRAINT_WEIGHT: f64 = 100.0;

/// How much of the embedder's own margin the barrier keeps.
///
/// A correct centre's signed volume is about 0.92, so at the full margin
/// the term leans on every centre in the molecule at once and the
/// relaxation stalls 40 kcal/mol short of the minimum. Seven tenths is the
/// most that still leaves a healthy centre in the flat region -- and it
/// has to be that high, because wherever the barrier sits is where
/// relaxation leaves a centre it wants to invert. At a quarter,
/// cholesterol's ring-fusion carbon came to rest at 0.24, flat enough that
/// RDKit read it as the wrong isomer.
const STEREO_BARRIER_MARGIN: f64 = 0.7;

impl crate::molecule::minimize::Energy for StereoRestrained<'_> {
    fn energy(&self, positions: &[Vector3D]) -> f64 {
        let mut discard = self.discard.borrow_mut();
        discard.fill(Vector3D::default());
        self.field.energy(positions)
            + STEREO_RESTRAINT_WEIGHT
                * self.stereo.energy_and_gradient_scaled(
                    positions,
                    &mut discard,
                    STEREO_BARRIER_MARGIN,
                )
    }

    fn energy_and_gradient(&self, positions: &[Vector3D], gradient: &mut [Vector3D]) -> f64 {
        let energy = self.field.energy_and_gradient(positions, gradient);
        let mut penalty_gradient = self.discard.borrow_mut();
        penalty_gradient.fill(Vector3D::default());
        let penalty = self.stereo.energy_and_gradient_scaled(
            positions,
            &mut penalty_gradient,
            STEREO_BARRIER_MARGIN,
        );
        for (slot, extra) in gradient.iter_mut().zip(penalty_gradient.iter()) {
            *slot += *extra * STEREO_RESTRAINT_WEIGHT;
        }
        energy + STEREO_RESTRAINT_WEIGHT * penalty
    }
}

/// Relax one trial without losing stereochemistry. All candidates are scored
/// with the same field, including candidates restored after a stereo failure.
/// The UFF fallback is constructed only if needed and reused across trials.
fn relax_positions(
    molecule: &Molecule3D,
    positions: &mut Vec<Vector3D>,
    field: &crate::molecule::forcefield::Field,
    stereo: &chirality::StereoConstraints,
    fallback: &std::sync::OnceLock<Option<crate::molecule::forcefield::ForceField>>,
) -> f64 {
    let mismatches = stereo.count_mismatches(positions);
    let embedded = positions.clone();

    // Relax against the field with the chirality terms still on. They are
    // flat above their margin, so a structure whose centres are on the
    // right side of their planes sees exactly the field's own minimum; what
    // they cost is the one thing worth paying for, which is that the
    // strain cannot come out by inverting a centre.
    if stereo.is_empty() {
        let energy = minimize_to_convergence(field, positions);
        if energy.is_finite() && positions_are_finite(positions) {
            return energy;
        }
    } else {
        let restrained = StereoRestrained {
            field,
            stereo,
            discard: std::cell::RefCell::new(vec![Vector3D::default(); positions.len()]),
        };
        minimize_to_convergence(&restrained, positions);
    }
    let energy = field.energy(positions);
    if energy.is_finite() && positions_are_finite(positions)
        && stereo.count_mismatches(positions) <= mismatches
    {
        return energy;
    }

    // Neither field restrains the twist about a linear allene centre.
    // MMFF can flatten it; UFF may preserve it from the same starting point.
    positions.clone_from(&embedded);
    if matches!(field, crate::molecule::forcefield::Field::Mmff(_))
        && let Some(uff) = fallback.get_or_init(|| {
            crate::molecule::forcefield::ForceField::new(molecule).ok()
        })
    {
        let energy = minimize_to_convergence(uff, positions);
        if !energy.is_finite() || !positions_are_finite(positions)
            || stereo.count_mismatches(positions) > mismatches
        {
            *positions = embedded;
        }
    }
    field.energy(positions)
}

fn positions_are_finite(positions: &[Vector3D]) -> bool {
    positions.iter().all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
}

/// Extracts the induced subgraph on `atoms` (a connected component of
/// `mol`) as a fresh, locally-0-indexed [`Molecule3D`] - mirrors
/// [`to_molecule3d_topology`] but subsets and remaps bond targets instead of
/// copying everything 1:1.
fn extract_component(mol: &Molecule3D, atoms: &[usize]) -> Molecule3D {
    let index_map: std::collections::HashMap<usize, usize> =
        atoms.iter().enumerate().map(|(local, &global)| (global, local)).collect();

    let atom_bonds = atoms
        .iter()
        .map(|&global| {
            mol.atom_bonds()[global]
                .iter()
                .map(|b| crate::molecule::bond::BondTarget::new(index_map[&b.target()], b.bond_order()))
                .collect()
        })
        .collect();

    Molecule3D {
        atomic_numbers: atoms.iter().map(|&a| mol.atomic_numbers()[a]).collect(),
        atom_classes: mol.atom_classes().as_ref().map(|classes| atoms.iter().map(|&a| classes[a]).collect()),
        charges: atoms.iter().map(|&a| mol.charges()[a]).collect(),
        chiral_classes: mol.chiral_classes().map(|cc| atoms.iter().map(|&a| cc[a]).collect()),
        isotopes: mol.isotopes().map(|isotopes| atoms.iter().map(|&a| isotopes[a]).collect()),
        positions: vec![Vector3D::default(); atoms.len()],
        radical_states: atoms.iter().map(|&a| mol.radical_states()[a]).collect(),
        atom_bonds,
    }
}

/// Embeds a single connected, already-hydrogenated [`Molecule3D`], returning
/// its positions (indexed the same as `working`). This is the Stage 1 + 2
/// core pipeline: bounds -> smoothing -> multi-attempt MDS+refine (ranked by
/// stereo correctness, then energy) -> stereo repair fallback -> planarity
/// touch-up -> optional force-field relaxation and ranking.
fn embed_connected(working: &Molecule3D, config: &EmbedConfig) -> Result<Vec<Vector3D>, EmbedError> {
    // One MMFF typing pass for the whole embedding. The field is what the
    // relaxation minimises against, and its equilibrium geometry is what
    // the bounds and the angle restraints are built from, so it is built
    // first and read twice rather than typed twice: see
    // [`bounds::Geometry`].
    //
    // Construction reads topology only; reuse it for all conformer trials.
    // Unsupported molecules still get distance geometry without relaxation.
    let field = crate::molecule::forcefield::Field::best_for(working).ok();
    // The rings, once: the geometry, the bounds and the angle restraints all
    // read them.
    let cycles = bounds::ring_set(working);
    let lengths = field
        .as_ref()
        .and_then(|field| field.mmff())
        .map(|mmff| bounds::Geometry::from_field(mmff, working, &cycles))
        .unwrap_or_else(|| bounds::Geometry::covalent_only(working, &cycles));
    let field = if config.relax { field } else { None };
    let mut bounds_matrix = bounds::build_bounds(working, &lengths, &cycles)?;
    bounds::smooth_bounds(&mut bounds_matrix)?;
    // Every pair the refinement will read, hoisted out of the matrix once:
    // see [`bounds::Pairs`] for the list and for which passes get the short
    // one.
    let pairs = bounds::Pairs::collect(&bounds_matrix);
    let stereo = chirality::StereoConstraints::collect(working);
    let (ring_planes, planar_centers) = planarity::collect_planarity(working);
    let angle_bends = forcefield::collect_angle_bends(working, &lengths, &cycles);
    let torsions = forcefield::collect_torsions(working, config.small_ring_torsions);
    let fallback = std::sync::OnceLock::new();

    // Two stages, because the per-trial bounds energy cannot rank trials.
    //
    // It is a flat-bottomed hinge: exactly zero for any structure that
    // satisfies its bounds, and most trials do. Ranking on it does not
    // choose the best trial, it chooses the first one that scored zero --
    // raising `max_attempts` from 5 to 40 once left the output
    // bit-identical, trial 2 winning a tie against every later one. So the
    // loop below only *screens*, discarding trials that came out
    // non-finite, degenerate, or geometrically impossible; every survivor
    // is then given the full touch-up and optional relaxation. Ranking uses
    // the final force-field energy, or the touch-up energy when unrelaxed.
    //
    // This is what makes `max_attempts` mean something: it now buys
    // proportionally more touch-ups, which is also what it costs.
    let shaping_terms = config.combined_refinement;

    // One trial, from its own seed, answering `None` when it comes out
    // unusable. Written as a closure because the trials are independent and
    // the loop below may run them on several threads.
    let trial_from = |attempt: usize| -> Option<(Vec<Vector3D>, usize, f64)> {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(attempt as u64));
        let trial = mds::sample_trial_distances(&bounds_matrix, &mut rng);
        let (mut positions, w) = mds::classical_mds_4d(&trial);

        // A distance matrix cannot tell a molecule from its mirror image,
        // so about half of a fresh trial's stereocentres are wrong. Turn
        // them over in the fourth dimension, where the inversion costs
        // nothing, before flattening back to three -- see [`fourd`]. A
        // molecule with no stereochemistry skips it and keeps the plain
        // three-dimensional path.
        if !stereo.is_empty() {
            fourd::resolve_chirality(
                &mut positions,
                &w,
                pairs.tight(),
                &stereo,
                config.max_iterations,
            );
        }

        // Which restraints shape the trial, and for how long.
        //
        // The choice is between the bounds alone and then a touch-up that
        // adds planarity, angle bend and the torsion preferences, or every
        // restraint at once for the same total budget. RDKit does the
        // latter: ETKDG's first minimisation carries the experimental
        // torsion terms and the basic-knowledge terms from the start, and
        // measured on the corpus at one trial and no torsion search, so
        // should we -- 56 of 72 molecules at least matching RDKit becomes
        // 62, and the mean +0.51 becomes -0.02 kcal/mol. A bounds-only pass
        // leaves a geometry that satisfies every distance and sits wherever
        // the torsion terms would not have put it, and the second stage
        // then has to argue it out of there. Both halves matter: all the
        // terms for only 300 iterations is +0.54, and a separate
        // 900-iteration touch-up is +0.47.
        //
        // But it costs about 40% of an embedding, because every iteration
        // of every trial now evaluates every term, and when the torsion
        // search follows it buys nothing: the search reaches the same
        // torsions by turning bonds, and the corpora come out a wash
        // (PubChem 246 of 287 against 250, at a mean of -2.89 against
        // -2.73). So the combined refinement is for the configuration that
        // has nothing downstream to fix torsions -- which is
        // `EmbedConfig::fast`, and any other caller who turns the search
        // off.
        let energy = if shaping_terms {
            refine::refine(
                &mut positions,
                pairs.all(),
                &stereo,
                &ring_planes,
                &planar_centers,
                &angle_bends,
                &torsions,
                config.max_iterations + EXTRA_REFINEMENT_ITERATIONS,
            )
        } else {
            // The bounds-only trial pass, on the pairs whose window says
            // something: RDKit's `basinThresh`, and its first minimisation
            // reads exactly this subset. The touch-up below restores every
            // pair, so this only changes what a fresh trial is shaped by.
            // Worth 4% of the default configuration and nothing to the
            // corpora -- 70 of 72 molecules at a mean of -1.10 kcal/mol
            // either way.
            refine::refine(
                &mut positions,
                pairs.tight(),
                &stereo,
                &[],
                &[],
                &[],
                &[],
                config.max_iterations,
            )
        };

        if !energy.is_finite() || !positions_are_finite(&positions) {
            return None;
        }
        // Guard against degenerate near-zero-length pairs before accepting a trial.
        if has_degenerate_distances(&positions, &bounds_matrix) {
            return None;
        }

        let mismatches = stereo.count_mismatches(&positions);
        Some((positions, mismatches, energy))
    };

    // Trials in parallel, because they share nothing: each draws its own
    // distances from its own seed and refines its own copy of the
    // coordinates. `collect` keeps them in order, so which trial wins is
    // still decided by the same comparison in the same sequence as before
    // and the answer does not depend on how the work was spread.
    //
    // Only when there is more than one, so that a single-trial embedding --
    // `EmbedConfig::fast`, half a millisecond on a small molecule -- never
    // pays to enter a thread pool it has no use for.
    let screened: Vec<(Vec<Vector3D>, usize, f64)> = if config.max_attempts > 1 {
        use rayon::prelude::*;
        (0..config.max_attempts)
            .into_par_iter()
            .filter_map(trial_from)
            .collect()
    } else {
        (0..config.max_attempts).filter_map(trial_from).collect()
    };

    if screened.is_empty() {
        return Err(EmbedError::MaxAttemptsExceeded);
    }

    // The same argument for the second stage, which is the expensive one:
    // every survivor gets a touch-up and a force-field relaxation, and
    // those are independent too. The winner is chosen afterwards, from the
    // finished list, in order.
    let relax_trial = |(mut positions, mismatches, mut energy): (Vec<Vector3D>, usize, f64)|
        -> Option<(Vec<Vector3D>, usize, f64)> {
        // Last-resort fallback: the energy term alone didn't flip every
        // stereocenter. Force the remaining ones via a local point
        // reflection, then relax the resulting local bond-length distortions
        // with one more short refinement pass.
        if mismatches > 0 && stereo.repair(&mut positions) {
            // The reflection leaves local bond lengths distorted; settle
            // them against the same restraints the trial was refined with.
            energy = refine::refine(
                &mut positions,
                pairs.all(),
                &stereo,
                &ring_planes,
                &planar_centers,
                &angle_bends,
                &torsions,
                config.max_iterations,
            );
        }

        // The touch-up, for the trials the loop above refined against the
        // bounds alone: planarity, angle bend and the torsion preferences,
        // on a structure whose distances already work. Where the trial was
        // shaped with every term from the start there is nothing left for
        // it to add.
        if !shaping_terms {
            energy = refine::refine(
                &mut positions,
                pairs.all(),
                &stereo,
                &ring_planes,
                &planar_centers,
                &angle_bends,
                &torsions,
                EXTRA_REFINEMENT_ITERATIONS,
            );
        }

        // Rank the geometries that will actually be returned. A low
        // restraint energy can relax into a worse force-field minimum than
        // another trial; selecting before relaxation discards that trial.
        let energy = if let Some(field) = &field {
            relax_positions(working, &mut positions, field, &stereo, &fallback)
        } else {
            energy
        };
        if !energy.is_finite() || !positions_are_finite(&positions) {
            return None;
        }
        let mismatches = stereo.count_mismatches(&positions);
        Some((positions, mismatches, energy))
    };

    let relaxed: Vec<(Vec<Vector3D>, usize, f64)> = if screened.len() > 1 {
        use rayon::prelude::*;
        screened.into_par_iter().filter_map(relax_trial).collect()
    } else {
        screened.into_iter().filter_map(relax_trial).collect()
    };

    let mut best: Option<(Vec<Vector3D>, usize, f64)> = None;
    for (positions, mismatches, energy) in relaxed {
        let is_better = best.as_ref().is_none_or(|(_, best_mismatches, best_energy)| {
            (mismatches, energy) < (*best_mismatches, *best_energy)
        });
        if is_better {
            best = Some((positions, mismatches, energy));
        }
    }

    let (mut positions, _, energy) = best.ok_or(EmbedError::MaxAttemptsExceeded)?;

    // Only the trial that won, not every trial: the search costs about as
    // much as an embedding does on a large flexible molecule, and running it
    // on a trial that is about to be discarded buys nothing. It cannot lose,
    // either -- every turn is judged by the same field that ranked the
    // trials, and one that does not lower the energy is not taken.
    if let Some(field) = &field
        && config.torsion_search
    {
        torsion_search::refine_torsions(working, &mut positions, field, &stereo, energy, config.seed);
    }
    Ok(positions)
}

fn has_degenerate_distances(positions: &[Vector3D], bounds_matrix: &ndarray::Array2<f64>) -> bool {
    const MIN_SEPARATION: f64 = 1e-3;
    let n = positions.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let lower = bounds::get_lower(bounds_matrix, i, j);
            if lower > MIN_SEPARATION && positions[i].distance(&positions[j]) < MIN_SEPARATION {
                return true;
            }
        }
    }
    false
}
