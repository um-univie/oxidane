//! Limited-memory BFGS minimisation of a function of atomic positions.
//!
//! Quasi-Newton with a backtracking line search: the last `HISTORY`
//! position and gradient differences stand in for the inverse Hessian, via
//! the usual two-loop recursion, and that path scales the identity it starts
//! from by `(s . y) / (y . y)` so that a full step is the natural first thing
//! to try. The dense path beside it does not -- see below.
//!
//! The line search is Numerical Recipes' `lnsrch`, which is RDKit's too:
//! Armijo's sufficient-decrease test alone, with a quadratic model for the
//! first backtrack and a cubic one after that, and the gradient taken only
//! at the point it accepts. Measured over twelve molecules it spends 2.77
//! energy evaluations and one gradient per accepted step, accepting the
//! proposed full step 32% of the time. Where the full step is usually
//! taken -- short limited-memory runs -- the first probe asks for the
//! gradient too; see [`backtracking_line_search`].
//!
//! That ratio looks like waste and is not. Two standard ways of cutting it
//! were measured and both cost more than they saved:
//!
//! * **Starting the next search near the last accepted length** (twice it,
//!   capped at the full step) takes 2.77 probes to 1.66 and the corpus from
//!   587 to 454 ms -- and costs 0.76 kcal/mol, 61 of 72 molecules at -0.28
//!   becoming 58 at +0.48. Starting lower means accepting an
//!   Armijo-sufficient step instead of the better one the interpolation
//!   would have found. Growing by three instead of two moves along the same
//!   line, 499 ms at +0.14: a speed-for-quality dial, not a win. Growing by
//!   one is degenerate -- the guess can then only shrink, and it collapses to
//!   zero.
//! * **Scaling the initial inverse Hessian** by `(s . y) / (y . y)`, which
//!   the limited-memory path beside this one does every iteration and which
//!   is the textbook remedy (Nocedal and Wright 6.20), takes the probes to
//!   1.06 and the accepted step to 0.98 -- the Newton step becomes the right
//!   length, exactly as advertised -- and then needs **twice** the iterations
//!   for it, 892 ms against 587, at 151 of 287 PubChem molecules against 162.
//!   Measured, that scale is about 7e-4: the first curvature pair of a
//!   distance-geometry structure is dominated by bond stretches, so a single
//!   scalar tells the minimiser to creep when what the structure needs is
//!   large coordinated motion. `a_flat_energy_does_not_stop_a_remaining_soft_mode`
//!   fails under it for the same reason, and that test's gamma is 1e-3 --
//!   indistinguishable from the molecular median, so no clamp separates them.
//!
//! So the unscaled identity and a long proposed step are not a defect here.
//! Proposing far and letting the search interpolate is what suits a problem
//! whose stiff and soft modes are orders of magnitude apart, and the extra
//! probes are what buy the step.
//! Armijo alone accepts the first step that goes down at all, which is
//! usually far shorter than the curvature wants; a short step gives a small
//! `s` and a small `y`, and a correction pair built from those says almost
//! nothing about the Hessian. On that diet a ninety-atom peptide's
//! relaxation ran 500 iterations and 2109 energy evaluations per round and
//! still did not converge. With the curvature condition it reaches a lower
//! energy in half the iterations, and because the search evaluates the
//! gradient at every trial point, the accepted one is the iteration's
//! gradient -- so an iteration costs no evaluation beyond its line search.
//!
//! It replaced Polak-Ribiere conjugate gradient, which was converging
//! linearly and therefore never converging at all inside a fixed budget:
//! every touch-up ran its full 300 iterations and every force-field
//! relaxation its full 500, and the tail bought almost nothing. Caffeine's
//! UFF energy went 72.30 at iteration 100, 70.51 at 200, 69.97 at 500 --
//! three hundred iterations for half a kcal/mol. Curvature information is
//! what closes that, and it is also why RDKit's embedder, which minimises
//! with BFGS, was three times quicker on the same molecules.
//!
//! Three details matter as much as the direction does:
//!
//! * a line search that fails is a reason to drop the curvature history and
//!   retry along the plain gradient, not to stop. Treating failure as
//!   convergence abandons structures that are still improvable.
//! * the first step, before there is any history, is scaled to a unit
//!   displacement. A raw gradient can be thousands of kcal/mol/A on a fresh
//!   distance-geometry structure, and a full step along it is a step across
//!   the molecule.
//! * a curvature pair is only kept when `s . y` is positive. Armijo alone
//!   does not guarantee it, and a non-positive pair makes the implied
//!   Hessian indefinite.
//!
//! Shared rather than written twice: the distance-geometry refinement and
//! [`crate::molecule::forcefield`] minimise different energies over the same
//! variables, and the part that is the same for both is all of this.

use crate::vector::{Vector3D, Vector4D};

/// What the minimiser needs of a coordinate: the vector-space operations,
/// an inner product, and a zero.
///
/// Implemented for [`Vector3D`] and, for the embedder's four-dimensional
/// stage, [`Vector4D`]. Everything above this trait -- the two-loop
/// recursion, the line search, the convergence tests -- is the same
/// algorithm in either dimension, and writing it twice is how the two would
/// drift apart.
pub trait Coordinate:
    Copy
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<f64, Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
{
    /// How many numbers one of these is, for the dense inverse Hessian,
    /// which indexes a flat vector of them.
    const DIMENSIONS: usize;
    fn dot(&self, other: &Self) -> f64;
    fn axis(&self, axis: usize) -> f64;
    fn axis_mut(&mut self, axis: usize) -> &mut f64;
}

impl Coordinate for Vector3D {
    const DIMENSIONS: usize = 3;
    fn dot(&self, other: &Self) -> f64 {
        Vector3D::dot(self, other)
    }
    fn axis(&self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
    fn axis_mut(&mut self, axis: usize) -> &mut f64 {
        match axis {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

impl Coordinate for Vector4D {
    const DIMENSIONS: usize = 4;
    fn dot(&self, other: &Self) -> f64 {
        Vector4D::dot(self, other)
    }
    fn axis(&self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => self.w,
        }
    }
    fn axis_mut(&mut self, axis: usize) -> &mut f64 {
        match axis {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.w,
        }
    }
}

/// The largest problem that gets a dense inverse Hessian, in variables.
///
/// Nine hundred is three hundred atoms in three dimensions. Below it a real
/// inverse Hessian is worth several times its cost in iterations saved --
/// 1896 against 389 on a fifty-atom molecule -- and above it the matrix
/// stops fitting any cache and the limited-memory recursion is the better
/// approximation per unit time.
///
/// Six hundred was tried, on the argument that the crossover was measured
/// when the dense path cost three times what it now does, and it is twice
/// as fast: 50.1 s against 105.8 for thirty molecules of 130 to 390 atoms,
/// 20.1 against 26.9 for forty of 90 to 400 (minimum of three runs, one
/// trial, no torsion search). Thirty of those molecules also came back with
/// the same energy to three decimals, which is what made it look free.
///
/// It is not free. On the 100-to-150-heavy-atom bucket of the stratified
/// corpus -- 200 to 300 atoms with hydrogens, which is exactly the range
/// the change moves off the dense path -- the mean difference against
/// RDKit goes from +13.48 to +27.19 kcal/mol and the median from +7.80 to
/// +15.05. The limited-memory recursion does not find the same minima
/// there, and it is not the iteration budget: sixteen relaxation rounds
/// instead of eight recovers 1 kcal/mol of the 14, and a history of 24
/// correction pairs instead of 8 recovers none of it.
///
/// So the number stays where the quality is, and the large-molecule
/// slowness stays with it. There is no knob for the other trade; this
/// constant is where one would go.
const DENSE_HESSIAN_MAX_DIMENSION: usize = 900;

/// The shortest run that gets a dense inverse Hessian.
///
/// Building one costs a matrix-vector product and a rank-two update per
/// iteration -- for a ninety-atom molecule that is about as much
/// arithmetic as the force-field gradient itself, and it only repays that
/// over a long descent. The torsion search runs hundreds of forty- and
/// three-hundred-iteration screens where it never would: with no budget
/// test, those screens made a peptide's eight-start search 2.5x slower
/// than the limited-memory recursion had.
const DENSE_HESSIAN_MIN_BUDGET: usize = 400;

/// An explicit inverse Hessian, BFGS-updated, as Numerical Recipes 10.7
/// has it and as RDKit's minimiser uses.
///
/// The limited-memory recursion this sits beside keeps the last
/// [`HISTORY`] correction pairs and reconstructs a direction from them.
/// That is the right thing for a problem too big to hold a matrix for, and
/// on a molecule it is a poor trade: eight pairs over 150 coupled variables
/// is a thin approximation, and the iteration count shows it.
///
/// Everything here reads and writes flat `f64` slices, and the step and the
/// gradient change are flattened into scratch buffers before either of the
/// `O(dimension^2)` loops touches them. That is not a style preference. The
/// version that indexed the coordinates in place -- `vectors[index /
/// DIMENSIONS].axis(index % DIMENSIONS)`, inside both loops -- paid an
/// integer division, a remainder, a match on the axis and a bounds check
/// per matrix element, and it was **62% of a ninety-atom embedding's
/// instructions**: 107 instructions per element where the arithmetic is
/// four. RDKit's equivalent (`BFGSOpt.h`) is the same algorithm over the
/// same size of matrix at about six. Flattening took it to 36, the
/// row-order write below to less, and the embedder from 1.5x RDKit's total
/// time to 0.93x; nothing about the mathematics changed.
struct DenseHessian {
    /// Row-major `dimension x dimension`.
    inverse: Vec<f64>,
    dimension: usize,
    /// `H y`, kept across the update to avoid reallocating.
    product: Vec<f64>,
    /// The BFGS update's third vector.
    combination: Vec<f64>,
    /// The step, the gradient change and the gradient, flattened to one
    /// number per variable.
    step_flat: Vec<f64>,
    change_flat: Vec<f64>,
    gradient_flat: Vec<f64>,
    /// `-H g`, flattened, before it is spread back over the atoms.
    direction_flat: Vec<f64>,
    /// No update has been applied since the last reset. Tracked rather than
    /// read off the matrix, which every iteration asks and which is a
    /// `dimension^2` scan to answer.
    identity: bool,
}

/// One number per variable, from one vector per atom.
fn flatten<C: Coordinate>(vectors: &[C], flat: &mut [f64]) {
    for (index, slot) in flat.iter_mut().enumerate() {
        *slot = vectors[index / C::DIMENSIONS].axis(index % C::DIMENSIONS);
    }
}

/// [`flatten`] of `new - old`, without the intermediate vector.
fn flatten_difference<C: Coordinate>(new: &[C], old: &[C], flat: &mut [f64]) {
    for (index, slot) in flat.iter_mut().enumerate() {
        let (atom, axis) = (index / C::DIMENSIONS, index % C::DIMENSIONS);
        *slot = new[atom].axis(axis) - old[atom].axis(axis);
    }
}

/// A dot product over four running sums rather than one.
///
/// `f64` addition is not associative, so one accumulator is a dependency
/// chain the compiler may not break: the adder's latency per element, where
/// the multiply is one cycle and four lanes of it fit in a register. Four
/// chains measure at 21.2 microseconds against 9.8 for a 270-wide
/// matrix-vector product -- and the two the BFGS bookkeeping does per
/// iteration are the largest single cost in an embedding after the force
/// field itself. The split changes the answer only by rounding.
///
/// Eight chains are slower than four here (14.1 microseconds), and so is
/// gathering each lane into a `[f64; 4]` first. A real SIMD type does
/// better -- `wide::f64x4` reaches 7.6 and `faer`'s dispatching kernel 5.5
/// -- which is why the two matrix-vector products now go through
/// [`faer_gemv`] and this is left for the short dot products beside them.
/// `out = M x` for a row-major `dimension x dimension` matrix, through
/// faer's dispatching kernel.
///
/// The BFGS bookkeeping does two of these per iteration -- `H y` for the
/// update and `-H g` for the direction -- and they are the largest single
/// cost in an embedding after the force field itself. faer's kernel
/// dispatches on the instruction set at run time and reaches 5.5
/// microseconds at `dimension = 270` where four hand-rolled accumulator
/// chains reach 9.8 and one reaches 21.2.
///
/// End to end it is worth about 7% on molecules of 130 to 390 atoms (99.3 s
/// against 106.4 for thirty of them) and nothing measurable at drug size,
/// where the force field rather than the matrix is the cost. The rank-two
/// update below stays hand-written: it is a `saxpy` over the whole matrix,
/// already memory-bound, and measured at 128 microseconds against faer's
/// 129 for the same work.
///
/// It obliges the crate to thin LTO -- see the note in `Cargo.toml`.
fn faer_gemv(matrix: &[f64], x: &[f64], out: &mut [f64], dimension: usize) {
    let m = faer::MatRef::from_row_major_slice(matrix, dimension, dimension);
    let v = faer::MatRef::from_column_major_slice(x, dimension, 1);
    let mut o = faer::MatMut::from_column_major_slice_mut(out, dimension, 1);
    faer::linalg::matmul::matmul(o.as_mut(), faer::Accum::Replace, m, v, 1.0, faer::Par::Seq);
}

fn wide_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sums = [0.0f64; 4];
    let mut left = a.chunks_exact(4);
    let mut right = b.chunks_exact(4);
    for (x, y) in left.by_ref().zip(right.by_ref()) {
        sums[0] += x[0] * y[0];
        sums[1] += x[1] * y[1];
        sums[2] += x[2] * y[2];
        sums[3] += x[3] * y[3];
    }
    let mut total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for (x, y) in left.remainder().iter().zip(right.remainder()) {
        total += x * y;
    }
    total
}

impl DenseHessian {
    fn identity(dimension: usize) -> Self {
        let mut inverse = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            inverse[index * dimension + index] = 1.0;
        }
        DenseHessian {
            inverse,
            dimension,
            product: vec![0.0; dimension],
            combination: vec![0.0; dimension],
            step_flat: vec![0.0; dimension],
            change_flat: vec![0.0; dimension],
            gradient_flat: vec![0.0; dimension],
            direction_flat: vec![0.0; dimension],
            identity: true,
        }
    }

    /// Whether this is still the identity, i.e. the first direction will be
    /// a plain gradient and needs scaling like one.
    fn is_identity(&self) -> bool {
        self.identity
    }

    fn reset(&mut self) {
        if self.identity {
            return;
        }
        self.identity = true;
        self.inverse.fill(0.0);
        for index in 0..self.dimension {
            self.inverse[index * self.dimension + index] = 1.0;
        }
    }

    /// Writes `-H g` into `direction`.
    fn direction<C: Coordinate>(&mut self, gradient: &[C], direction: &mut [C]) {
        let dimension = self.dimension;
        flatten(gradient, &mut self.gradient_flat);
        faer_gemv(&self.inverse, &self.gradient_flat, &mut self.direction_flat, dimension);
        for (index, &value) in self.direction_flat.iter().enumerate() {
            *direction[index / C::DIMENSIONS].axis_mut(index % C::DIMENSIONS) = -value;
        }
    }

    /// The BFGS update for the step from `previous_positions` to
    /// `positions` and the gradient change over it, both flattened straight
    /// into scratch. Skipped when the curvature is too small to trust, which
    /// leaves the matrix as it was rather than making it indefinite.
    fn update<C: Coordinate>(
        &mut self,
        positions: &[C],
        previous_positions: &[C],
        gradient: &[C],
        previous_gradient: &[C],
    ) {
        let DenseHessian {
            inverse,
            dimension,
            product,
            combination,
            step_flat,
            change_flat,
            identity,
            ..
        } = self;
        let dimension = *dimension;
        flatten_difference(positions, previous_positions, step_flat);
        flatten_difference(gradient, previous_gradient, change_flat);
        let mut curvature = 0.0;
        let mut change_squared = 0.0;
        let mut step_squared = 0.0;
        for (&s, &y) in step_flat.iter().zip(change_flat.iter()) {
            curvature += s * y;
            change_squared += y * y;
            step_squared += s * s;
        }
        if !(curvature > (MIN_CURVATURE * change_squared * step_squared).sqrt()) {
            return;
        }

        *identity = false;
        faer_gemv(inverse, change_flat, product, dimension);
        let change_product = wide_dot(change_flat, product);

        let inverse_curvature = 1.0 / curvature;
        let inverse_product = 1.0 / change_product;
        for index in 0..dimension {
            combination[index] =
                inverse_curvature * step_flat[index] - inverse_product * product[index];
        }
        // The whole row, not the triangle and a mirror. The update is
        // symmetric either way, so half the arithmetic is redundant -- but
        // the mirror store writes down a column, a `dimension`-stride hop
        // per element that misses cache on every one of them. Writing both
        // triangles in row order is 8% of an embedding cheaper than being
        // clever about it. The cost is that the two triangles are now
        // computed rather than copied, so they agree only to rounding.
        for (row, cells) in inverse.chunks_exact_mut(dimension).enumerate() {
            let step_row = inverse_curvature * step_flat[row];
            let product_row = inverse_product * product[row];
            let combination_row = change_product * combination[row];
            for (column, cell) in cells.iter_mut().enumerate() {
                *cell += step_row * step_flat[column] - product_row * product[column]
                    + combination_row * combination[column];
            }
        }
    }
}

/// Armijo's sufficient-decrease constant: how much of the decrease the
/// slope promises a step actually has to deliver.
const ARMIJO_C1: f64 = 1e-4;

/// Trial points per line search.
const MAX_LINE_SEARCH_STEPS: usize = 20;

/// Step length below which the line search gives up rather than keep
/// interpolating into the noise.
const MINIMUM_STEP: f64 = 1e-12;

/// How many correction pairs stand in for the inverse Hessian. Eight is the
/// usual choice and measurably enough here: sixteen changed no benchmark
/// result and cost memory proportional to the molecule.
const HISTORY: usize = 8;

/// Below this, a correction pair is discarded rather than stored -- see the
/// module comment on `s . y`.
const MIN_CURVATURE: f64 = 1e-12;

/// How many times a stalled line search may throw the curvature away and
/// start again from the gradient before the minimisation calls it a day.
const MAX_CURVATURE_RESTARTS: usize = 2;

/// Step below which the minimisation has stopped moving, as a fraction of
/// each coordinate's own size. Numerical Recipes' `TOLX`.
///
/// A relative test rather than an absolute one because an absolute test
/// depends on where the molecule happens to sit in space, and a relative
/// one does not.
const POSITION_TOLERANCE: f64 = 1.2e-7;

/// An energy over atomic positions, and its gradient.
///
/// `energy_and_gradient` writes `dE/dp` for every atom into `gradient` --
/// which it must overwrite rather than accumulate into -- and returns the
/// energy. `energy` returns the same number without the gradient, and exists
/// because the line search evaluates several times per iteration and never
/// looks at one.
pub trait Energy<C: Coordinate = Vector3D> {
    fn energy(&self, positions: &[C]) -> f64;
    fn energy_and_gradient(&self, positions: &[C], gradient: &mut [C]) -> f64;
}

/// One L-BFGS correction pair: how far the last accepted step moved, how much
/// the gradient changed over it, and `1 / (s . y)`.
struct Correction<C> {
    step: Vec<C>,
    gradient_change: Vec<C>,
    inverse_curvature: f64,
}

/// Scratch reused across every iteration and every line-search probe.
struct Workspace<C> {
    gradient: Vec<C>,
    previous_gradient: Vec<C>,
    direction: Vec<C>,
    trial: Vec<C>,
    trial_gradient: Vec<C>,
    previous_positions: Vec<C>,
    history: std::collections::VecDeque<Correction<C>>,
    /// The two-loop recursion's inner coefficients, one per stored pair.
    coefficients: Vec<f64>,
    /// Present for a problem small enough to hold a real inverse Hessian,
    /// which is most molecules; `history` is what serves the rest.
    dense: Option<DenseHessian>,
}

impl<C: Coordinate> Workspace<C> {
    fn new(n: usize, budget: usize) -> Self {
        Workspace {
            gradient: vec![C::default(); n],
            previous_gradient: vec![C::default(); n],
            direction: vec![C::default(); n],
            trial: vec![C::default(); n],
            trial_gradient: vec![C::default(); n],
            previous_positions: vec![C::default(); n],
            history: std::collections::VecDeque::with_capacity(HISTORY),
            coefficients: Vec::with_capacity(HISTORY),
            dense: (n * C::DIMENSIONS <= DENSE_HESSIAN_MAX_DIMENSION
                && budget >= DENSE_HESSIAN_MIN_BUDGET)
                .then(|| DenseHessian::identity(n * C::DIMENSIONS)),
        }
    }

    /// Writes `-H g` into `direction`, where `H` is the inverse Hessian the
    /// stored corrections imply. With no history that is just `-g`.
    fn set_quasi_newton_direction(&mut self) {
        {
            // Field by field, so the matrix can be written into the
            // direction while the gradient is read, which one `&mut self`
            // would not allow.
            let Workspace { dense, gradient, direction, .. } = self;
            if let Some(dense) = dense {
                dense.direction(gradient, direction);
                return;
            }
        }
        for (slot, gradient) in self.direction.iter_mut().zip(&self.gradient) {
            *slot = *gradient;
        }
        self.coefficients.clear();
        self.coefficients.resize(self.history.len(), 0.0);

        // Newest correction first, peeling each one off the gradient.
        for (index, correction) in self.history.iter().enumerate().rev() {
            let coefficient = correction.inverse_curvature * dot(&correction.step, &self.direction);
            self.coefficients[index] = coefficient;
            for (slot, change) in self.direction.iter_mut().zip(&correction.gradient_change) {
                *slot = *slot - *change * coefficient;
            }
        }

        // The scale of the identity the recursion starts from, taken from
        // the newest pair -- without it every direction would be a gradient
        // in the units of the gradient rather than of a displacement.
        if let Some(newest) = self.history.back() {
            let scale = 1.0
                / (newest.inverse_curvature
                    * dot(&newest.gradient_change, &newest.gradient_change));
            for slot in &mut self.direction {
                *slot = *slot * scale;
            }
        }

        // Oldest correction first, putting back what each one contributes.
        for (index, correction) in self.history.iter().enumerate() {
            let beta =
                correction.inverse_curvature * dot(&correction.gradient_change, &self.direction);
            let coefficient = self.coefficients[index] - beta;
            for (slot, step) in self.direction.iter_mut().zip(&correction.step) {
                *slot = *slot + *step * coefficient;
            }
        }

        for slot in &mut self.direction {
            *slot = -*slot;
        }
    }

    fn set_steepest_descent_direction(&mut self) {
        self.history.clear();
        if let Some(dense) = &mut self.dense {
            dense.reset();
        }
        for (slot, gradient) in self.direction.iter_mut().zip(&self.gradient) {
            *slot = -*gradient;
        }
    }

    /// Stores the step just taken, unless it carries no usable curvature.
    fn record(&mut self, positions: &[C]) {
        if let Some(dense) = &mut self.dense {
            dense.update(positions, &self.previous_positions, &self.gradient, &self.previous_gradient);
            return;
        }
        let curvature: f64 = positions
            .iter()
            .zip(&self.previous_positions)
            .zip(self.gradient.iter().zip(&self.previous_gradient))
            .map(|((new, old), (new_gradient, old_gradient))| {
                (*new - *old).dot(&(*new_gradient - *old_gradient))
            })
            .sum();
        if !(curvature > MIN_CURVATURE) {
            return;
        }
        // A full history hands its oldest pair's buffers to the new one, so
        // only the first `HISTORY` stored steps allocate.
        let mut correction = if self.history.len() == HISTORY {
            self.history.pop_front().expect("history is full")
        } else {
            Correction {
                step: vec![C::default(); positions.len()],
                gradient_change: vec![C::default(); positions.len()],
                inverse_curvature: 0.0,
            }
        };
        for ((slot, new), old) in correction.step.iter_mut().zip(positions).zip(&self.previous_positions) {
            *slot = *new - *old;
        }
        for ((slot, new), old) in correction
            .gradient_change
            .iter_mut()
            .zip(&self.gradient)
            .zip(&self.previous_gradient)
        {
            *slot = *new - *old;
        }
        correction.inverse_curvature = 1.0 / curvature;
        self.history.push_back(correction);
    }
}

fn dot<C: Coordinate>(a: &[C], b: &[C]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.dot(y)).sum()
}

/// The result of one line search: how far it went and the energy there.
struct Step {
    length: f64,
    energy: f64,
    /// Whether `work.trial_gradient` already holds the gradient at the
    /// accepted point, because the probe that found it asked for one.
    with_gradient: bool,
}

/// A step along `direction` that lowers the energy by enough, found by
/// evaluating the *energy* alone and interpolating.
///
/// On success `work.trial` holds the positions at that step; the caller
/// evaluates the gradient there once, which is the iteration's only
/// gradient -- unless `gradient_first` asked for it with the first probe,
/// in which case it is already in `work.trial_gradient`.
///
/// That is a bet on the first probe being accepted, placed whenever the
/// previous search's was. Short limited-memory runs, which is every
/// torsion-search relaxation, take their full step about four times in
/// five, and each of those used to cost an energy evaluation followed by
/// an energy-and-gradient at the same point. A lost bet costs one gradient
/// thrown away. Measured on PubChem's 287 at one trial: energy-only
/// evaluations 1449k to 435k, gradients 1156k to 1207k, 28% less work, and
/// the dense path -- which takes its first probe a third of the time --
/// mostly never places the bet. Quality is unchanged within noise: the two
/// energies agree to 1e-9 kcal/mol, so what moves is rounding, and rounding
/// sends a few large flexible molecules to different minima in both
/// directions (78 better and 65 worse by 0.05 kcal/mol on 1600 held-out
/// molecules at one trial, 59 and 70 at five).
///
/// This replaced a strong-Wolfe search, and the reason is arithmetic. Wolfe
/// needs the slope at every trial point, so every probe is a gradient --
/// two and a half of them per iteration on average, four in the tail. The
/// curvature condition buys better-conditioned correction pairs, but not
/// two and a half times better: measured against RDKit on the same
/// structure and the same force field, we were taking 4480 gradient
/// evaluations where they take a few hundred.
///
/// So: backtrack on the energy, and model what backtracking learns. The
/// first retry fits a quadratic through the energy, its slope at zero and
/// the failed point; later ones fit a cubic through the two most recent
/// failures. That is Numerical Recipes' `lnsrch`, which is what RDKit uses,
/// and it lands in one or two probes where halving takes five or six.
fn backtracking_line_search<C: Coordinate, E: Energy<C> + ?Sized>(
    terms: &E,
    positions: &[C],
    work: &mut Workspace<C>,
    energy: f64,
    slope: f64,
    first_step: f64,
    gradient_first: bool,
) -> Option<Step> {
    debug_assert!(slope < 0.0);
    let mut length = first_step;
    let (mut previous_length, mut previous_energy) = (0.0f64, 0.0f64);

    for attempt in 0..MAX_LINE_SEARCH_STEPS {
        for ((slot, position), along) in work
            .trial
            .iter_mut()
            .zip(positions.iter())
            .zip(work.direction.iter())
        {
            *slot = *position + *along * length;
        }
        let with_gradient = gradient_first && attempt == 0;
        let trial = if with_gradient {
            terms.energy_and_gradient(&work.trial, &mut work.trial_gradient)
        } else {
            terms.energy(&work.trial)
        };
        if trial.is_finite() && trial <= energy + ARMIJO_C1 * length * slope {
            return Some(Step { length, energy: trial, with_gradient });
        }

        // Where to try next. A non-finite energy means the step left the
        // region where the field is defined at all, and nothing can be
        // modelled from it -- halve and move on.
        let mut next = if !trial.is_finite() {
            0.5 * length
        } else if attempt == 0 {
            // One failed point plus the slope: a quadratic.
            -slope / (2.0 * (trial - energy - slope * length) / length)
        } else {
            // Two failed points: a cubic, solved for its minimum.
            let first = (trial - energy - length * slope) / (length * length);
            let second =
                (previous_energy - energy - previous_length * slope) / (previous_length * previous_length);
            let a = (first - second) / (length - previous_length);
            let b = (-previous_length * first + length * second) / (length - previous_length);
            if a == 0.0 {
                -slope / (2.0 * b)
            } else {
                let discriminant = b * b - 3.0 * a * slope;
                if discriminant < 0.0 {
                    0.5 * length
                } else if b <= 0.0 {
                    (-b + discriminant.sqrt()) / (3.0 * a)
                } else {
                    -slope / (b + discriminant.sqrt())
                }
            }
        };
        // Bound the interpolation: never more than half the current
        // step, never less than a tenth of it.
        if !(next.is_finite()) || next > 0.5 * length {
            next = 0.5 * length;
        }
        previous_length = length;
        previous_energy = trial;
        length = next.max(0.1 * length);
        if length < MINIMUM_STEP {
            break;
        }
    }
    None
}

/// Minimises `terms` over `positions` in place, returning the final energy.
///
/// Stops when the squared gradient norm is below `gradient_tolerance`, or
/// when the energy is stable and every atom's squared gradient norm is below
/// that threshold. Also stops when `max_iterations` is spent or no downhill
/// step can be found along either the quasi-Newton or plain gradient direction.
pub fn minimize<C: Coordinate, E: Energy<C> + ?Sized>(
    positions: &mut [C],
    terms: &E,
    max_iterations: usize,
    gradient_tolerance: f64,
) -> f64 {
    let mut work = Workspace::new(positions.len(), max_iterations);
    let mut energy = terms.energy_and_gradient(positions, &mut work.gradient);
    let mut restarts = 0usize;
    // Whether the last line search took its first probe, which is the bet
    // that the next one will too -- see `backtracking_line_search`.
    let mut first_probe_accepted = true;

    for _ in 0..max_iterations {
        let gradient_norm_sq = dot(&work.gradient, &work.gradient);
        if gradient_norm_sq < gradient_tolerance {
            break;
        }

        // Pass 0 tries the quasi-Newton direction; pass 1 throws the
        // curvature history away and falls back to the plain gradient. A
        // stored Hessian is only ever an approximation, and one the line
        // search cannot move along is evidence about the approximation, not
        // about the structure.
        let mut accepted = false;
        for pass in 0..2 {
            if pass == 0 {
                work.set_quasi_newton_direction();
            } else {
                work.set_steepest_descent_direction();
            }

            let slope = dot(&work.direction, &work.gradient);
            if !(slope < 0.0) {
                continue;
            }

            // A quasi-Newton direction is already a displacement, so a full
            // step is the right thing to try. A raw gradient is not, so the
            // very first step is cut to a unit displacement instead.
            let fresh = work
                .dense
                .as_ref()
                .map_or(work.history.is_empty(), |dense| dense.is_identity());
            let first_step = if fresh {
                (1.0 / gradient_norm_sq.sqrt()).min(1.0)
            } else {
                1.0
            };

            let Some(step) = backtracking_line_search(
                terms,
                positions,
                &mut work,
                energy,
                slope,
                first_step,
                first_probe_accepted,
            ) else {
                continue;
            };
            work.previous_positions.copy_from_slice(positions);
            positions.copy_from_slice(&work.trial);
            // The iteration's one gradient, at the point the search chose.
            std::mem::swap(&mut work.gradient, &mut work.previous_gradient);
            if step.with_gradient {
                std::mem::swap(&mut work.gradient, &mut work.trial_gradient);
                energy = step.energy;
            } else {
                energy = terms.energy_and_gradient(positions, &mut work.gradient);
            }
            first_probe_accepted = step.length == first_step;
            accepted = true;
            break;
        }
        if !accepted {
            break;
        }

        work.record(positions);

        // Two convergence tests, each scaled by the size of what it
        // measures, after Numerical Recipes 10.7.
        //
        // Not quite theirs: Numerical Recipes, and RDKit after it, divide
        // the force test by the energy, which makes convergence depend on
        // an arbitrary additive offset. A soft double well sitting at 1e6
        // instead of 25 is the same problem with the same forces, and that
        // form declares it converged at the saddle -- which is
        // `a_flat_energy_does_not_stop_a_remaining_soft_mode`, a test that
        // exists because tetramethylphosphonium once stopped 11 kcal/mol
        // above its minimum. The force test here is absolute, in kcal/mol
        // per Angstrom, scaled only by each coordinate's own size.
        //
        // This replaced an absolute energy-change test with a
        // clear-the-curvature-and-retry when the energy went flat but forces
        // remained. That retry is what the tests below are for, and pairing
        // it with a real inverse Hessian was a disaster: it fired 263 times
        // in 1500 iterations, throwing the matrix away faster than BFGS
        // could build it, and the minimisation never converged at all.
        let mut movement: f64 = 0.0;
        let mut force: f64 = 0.0;
        for (position, previous) in positions.iter().zip(&work.previous_positions) {
            for axis in 0..C::DIMENSIONS {
                let scale = position.axis(axis).abs().max(1.0);
                movement = movement.max((position.axis(axis) - previous.axis(axis)).abs() / scale);
            }
        }
        for (gradient, position) in work.gradient.iter().zip(positions.iter()) {
            for axis in 0..C::DIMENSIONS {
                force = force.max(gradient.axis(axis).abs() * position.axis(axis).abs().max(1.0));
            }
        }
        if force < gradient_tolerance {
            break;
        }
        if movement < POSITION_TOLERANCE {
            // The step became too small to matter while forces remain. That
            // is usually a line search stalled against a stale curvature
            // estimate rather than a minimum, and one fresh start from the
            // gradient gets it moving again -- the peptide finishes 5.4
            // kcal/mol lower for it. Bounded, because the version of this
            // that restarted every time it saw a flat iteration is what
            // stopped the inverse Hessian ever being worth building.
            if restarts >= MAX_CURVATURE_RESTARTS {
                break;
            }
            restarts += 1;
            work.history.clear();
            if let Some(dense) = &mut work.dense {
                dense.reset();
            }
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A soft double well coupled only through the optimizer to a stiff
    /// coordinate. Once the stiff coordinate settles, energy-only stopping
    /// mistakes the slow escape from the double well's saddle for convergence.
    struct StiffAndSoft {
        offset: f64,
    }

    impl Energy for StiffAndSoft {
        fn energy(&self, p: &[Vector3D]) -> f64 {
            self.offset + (p[0].x * p[0].x - 1.0).powi(2) + 500.0 * p[0].y.powi(2)
        }

        fn energy_and_gradient(&self, p: &[Vector3D], gradient: &mut [Vector3D]) -> f64 {
            gradient[0] =
                Vector3D::new(4.0 * p[0].x * (p[0].x * p[0].x - 1.0), 1000.0 * p[0].y, 0.0);
            self.energy(p)
        }
    }

    #[test]
    fn a_flat_energy_does_not_stop_a_remaining_soft_mode() {
        for offset in [25.0, 1e6] {
            let mut positions = [Vector3D::new(0.001, 1.0, 0.0)];
            minimize(&mut positions, &StiffAndSoft { offset }, 500, 1e-6);
            assert!(
                (positions[0].x.abs() - 1.0).abs() < 1e-3,
                "{offset}: {:?}",
                positions[0]
            );
            assert!(positions[0].y.abs() < 1e-5);
        }
    }
}
