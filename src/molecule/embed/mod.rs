//! Distance-geometry ("ETKDG-lite") 3D coordinate embedding.
//!
//! Stage 1: bounds matrix -> triangle-inequality smoothing -> classical 4D MDS
//! -> gradient refinement against a bounds-violation energy. Produces
//! geometrically valid structures (correct bond lengths/angles).
//!
//! Stage 2: [`chirality`] adds tetrahedral (`TH`), cis/trans double-bond, and
//! allene (`AL`) consistency terms plus a repair fallback; [`bounds`] adds
//! trans/adjacent angle bounds for square-planar (`SP`) centres (which,
//! unlike the others, has no separate handedness once the trans-pairing is
//! fixed, so needs no extra energy term); [`planarity`] adds a ring/local-
//! centre out-of-plane restraint applied as a touch-up on the winning trial;
//! disconnected components (salts) are embedded independently and packed
//! side by side (see [`run`]).
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
//! used here - so it's a considered best effort, verified only to be
//! internally self-consistent (see [`chirality::AllenCenter`]'s doc
//! comment).
//!
//! A UFF/MMFF-lite refinement pass ([`forcefield`]) adds harmonic-in-cosine
//! angle-bend and periodic-torsion restraints, applied in the same
//! touch-up-only style as `planarity` (see [`embed_connected`]) - deliberately
//! scoped to just those two terms: bond-stretch and van-der-Waals-style
//! non-bonded repulsion are already covered by the bounds-violation energy
//! itself (a tight flat bottom for 1-2 pairs, a VDW-radius-based lower bound
//! for every other pair), so adding them again would be redundant. Unlike
//! `planarity`'s original 300-iteration touch-up, the combined touch-up here
//! runs for `TOUCH_UP_ITERATIONS` (1500, see [`embed_connected`]) - adding
//! angle-bend/torsion on top of planarity meaningfully slows how many
//! iterations a single-step-size steepest descent needs to fully resolve any
//! one restraint on a real (not toy-sized) molecule; verified empirically
//! that 300 was not enough (see the constant's doc comment).

pub(crate) mod bounds;
mod chirality;
mod forcefield;
mod mds;
mod planarity;
mod refine;

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
    /// The lowest-energy successful trial is returned.
    pub max_attempts: usize,
    /// Maximum number of refinement (gradient descent) iterations per trial.
    pub max_iterations: usize,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        EmbedConfig {
            seed: 0xE7D6_5EED,
            max_attempts: 5,
            max_iterations: 300,
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

fn run<T: MoleculeCore + MoleculeGraph + MoleculeChirality>(
    mol: &T,
    config: &EmbedConfig,
) -> Result<Molecule3D, EmbedError> {
    let mut working = to_molecule3d_topology(mol);
    working.add_hydrogens();

    if working.len() < 2 {
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
        atom_classes: mol.atom_classes().clone(),
        charges: atoms.iter().map(|&a| mol.charges()[a]).collect(),
        chiral_classes: mol.chiral_classes().map(|cc| atoms.iter().map(|&a| cc[a]).collect()),
        isotopes: mol.isotopes().cloned(),
        positions: vec![Vector3D::default(); atoms.len()],
        radical_states: atoms.iter().map(|&a| mol.radical_states()[a]).collect(),
        atom_bonds,
    }
}

/// Embeds a single connected, already-hydrogenated [`Molecule3D`], returning
/// its positions (indexed the same as `working`). This is the Stage 1 + 2
/// core pipeline: bounds -> smoothing -> multi-attempt MDS+refine (ranked by
/// stereo correctness, then energy) -> stereo repair fallback -> planarity
/// touch-up.
fn embed_connected(working: &Molecule3D, config: &EmbedConfig) -> Result<Vec<Vector3D>, EmbedError> {
    let mut bounds_matrix = bounds::build_bounds(working)?;
    bounds::smooth_bounds(&mut bounds_matrix)?;
    let stereo = chirality::StereoConstraints::collect(working);
    let (ring_planes, planar_centers) = planarity::collect_planarity(working);
    let angle_bends = forcefield::collect_angle_bends(working);
    let torsions = forcefield::collect_torsions(working);

    // Ranked by (mismatched stereocenters, energy): a trial with fewer wrong
    // tetrahedral tags always beats one with lower bounds-violation energy
    // but worse chirality, since the chirality term only bends the raw energy
    // landscape - it doesn't guarantee the global optimum is stereo-correct.
    let mut best: Option<(Vec<Vector3D>, usize, f64)> = None;

    for attempt in 0..config.max_attempts {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(attempt as u64));
        let trial = mds::sample_trial_distances(&bounds_matrix, &mut rng);
        // The 4th (`w`) coordinate from MDS is discarded here: refinement
        // operates directly in 3D (see refine.rs) rather than the full
        // ETKDG 4D-anneal trick, so keeping `w` around would leave the 3D
        // projection satisfying bounds only in 4D.
        let (mut positions, _w) = mds::classical_mds_4d(&trial);

        // Planarity is deliberately left out of both the per-trial
        // optimization and the cross-trial ranking below: bounds energy is
        // exactly zero (zero gradient too) anywhere inside its +/-10 deg
        // angle tolerance, so once several trials tie at zero bounds+stereo
        // energy, ranking by "total energy including planarity" degenerates
        // into "pick whichever trial happened to end up flattest" with no
        // regard for angle regularity - empirically this relaxed a regular
        // hexagon into an irregular (still bounds-valid) one. Planarity is
        // instead applied once, below, as a touch-up on the already-selected
        // winning trial.
        let energy =
            refine::refine(&mut positions, &bounds_matrix, &stereo, &[], &[], &[], &[], config.max_iterations);

        if !energy.is_finite() || positions.iter().any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite()) {
            continue;
        }
        // Guard against degenerate near-zero-length pairs before accepting a trial.
        if has_degenerate_distances(&positions, &bounds_matrix) {
            continue;
        }

        let mismatches = stereo.count_mismatches(&positions);
        let is_better = best
            .as_ref()
            .map(|(_, best_mismatches, best_energy)| {
                (mismatches, energy) < (*best_mismatches, *best_energy)
            })
            .unwrap_or(true);
        if is_better {
            best = Some((positions, mismatches, energy));
        }
    }

    let Some((mut positions, mismatches, _energy)) = best else {
        return Err(EmbedError::MaxAttemptsExceeded);
    };

    // Last-resort fallback: the energy term alone didn't flip every
    // stereocenter within max_attempts. Force the remaining ones via a local
    // point-reflection, then relax the resulting local bond-length
    // distortions with one more short refinement pass.
    if mismatches > 0 && stereo.repair(&mut positions) {
        refine::refine(
            &mut positions,
            &bounds_matrix,
            &stereo,
            &[],
            &[],
            &[],
            &[],
            config.max_iterations,
        );
    }

    // Combined touch-up on the single winning trial (see comment above):
    // starting from an already bounds-satisfying, regular geometry, planarity,
    // angle-bend, and torsion only need to nudge out residual pucker/
    // irregularity/eclipsing, not reshape the structure - see
    // `forcefield`'s module doc comment for why angle-bend/torsion are
    // introduced only here rather than in the main per-trial loop above.
    //
    // `TOUCH_UP_ITERATIONS` is deliberately higher than `config.max_iterations`
    // (default 300, which was sufficient when this pass only carried the
    // planarity term): steepest descent with a single Armijo step per
    // iteration shares one step size across *every* active term's combined
    // gradient, so adding angle-bend/torsion on top of planarity - many more
    // simultaneous restraints on a real, larger molecule - measurably slows
    // how many iterations it takes to fully resolve any one of them. Verified
    // empirically on cocaine (43 atoms, 4 fused/bridged rings): the phenyl
    // ring's out-of-plane RMSD only drops back below its pre-angle/torsion
    // baseline (~0.135 A) somewhere between 1000-1500 iterations (300: 0.201
    // A, worse than baseline; 1500: 0.121 A, better).
    const TOUCH_UP_ITERATIONS: usize = 1500;
    refine::refine(
        &mut positions,
        &bounds_matrix,
        &stereo,
        &ring_planes,
        &planar_centers,
        &angle_bends,
        &torsions,
        TOUCH_UP_ITERATIONS,
    );

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

