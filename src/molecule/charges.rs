//! Two independent ways to estimate per-atom partial charge: Gasteiger-Marsili
//! (iterative, topology only) and QEq (charge equilibration, needs a 3D
//! embedding). Neither is a substitute for real quantum chemistry, but both
//! give a signal a purely categorical graph can't -- how electron density is
//! actually pulled around a molecule -- which a static per-element
//! electronegativity lookup or a formal-charge one-hot cannot.
//!
//! Gasteiger's `(a, b, c)` polynomial-electronegativity parameters come
//! straight from [`chemistry_consts::get_gasteiger_params`], itself sourced
//! from RDKit's `GasteigerParams.cpp`; the iteration here (charge transfer
//! per bond, damping schedule) mirrors RDKit's `GasteigerCharges.cpp`.
//!
//! QEq mirrors the `cheq` crate (<https://github.com/caltechmsc/cheq>), a
//! from-scratch, independently-tested Rust implementation of Rappe &
//! Goddard (*J. Phys. Chem.* 1991, 95, 3358) -- specifically `cheq`'s
//! *default* configuration, which is what its own checked-in expected
//! values were produced with:
//!
//! * the same per-element `chi0`/`J0`/covalent-radius/principal-quantum-number
//!   table (`QEQ_PARAMETERS`, from `cheq`'s `resources/qeq.data.toml`),
//! * the same Slater exponent `zeta = lambda*(2n+1)/(2*r_cov)` with
//!   `lambda = 0.5` (`LAMBDA_SCALE`),
//! * the *exact* two-center Coulomb integral over ns Slater orbitals, via
//!   the same `sto_ns` crate `cheq` delegates to -- not the cheaper
//!   Gaussian-fit approximation `cheq` also offers as a non-default option,
//! * hydrogen's charge-dependent hardness, refined by the same damped SCF
//!   loop (`hydrogen_hardness`, [`qeq_charges`]).
//!
//! Getting all four right matters more than it looks. Two of them
//! individually shift hydrogen's charge by ~0.1 e in *opposite* directions,
//! so a version of this module that took the Gaussian approximation and
//! skipped the hydrogen SCF landed within 0.03 of `cheq`'s published HF
//! number by cancellation while being 3.8x off on CH4 -- an agreement that
//! looked like verification and wasn't. The reference test below now pins
//! all four of `cheq`'s checked-in molecules (`tests/hydrogen.rs`) at
//! `cheq`'s own experimental geometries, to 1e-3, precisely so that a
//! single flattering data point can't stand in for the others.
//!
//! Sourcing both methods from working, tested implementations, rather than
//! re-deriving them from the papers, is the same reasoning behind
//! cross-checking [`crate::molecule::embed`] against RDKit in
//! `tests/embed_oracle.py`: an independently-verified reference is worth
//! more than a from-scratch derivation nobody has run -- and, as the
//! cancellation above shows, verified against the reference's *whole* set
//! of concrete numbers, not whichever one happens to agree.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::descriptors::MolecularDescriptors;
//!
//! let water = MolecularGraph::from_smiles_single("O").unwrap();
//! let charges = water.gasteiger_charges();
//! // Oxygen (atom 0) pulls electron density from both hydrogens.
//! assert!(charges[0] < 0.0);
//! assert!(charges[1] > 0.0 && charges[2] > 0.0);
//! ```

/// Initial fraction of a Gasteiger charge transfer applied per iteration.
const GASTEIGER_INITIAL_DAMPING: f64 = 0.5;
/// Geometric decay of Gasteiger charge transfers on successive passes.
const GASTEIGER_DAMPING_DECAY: f64 = 0.5;
/// QEq SCF damping limits and adaptation factors (dimensionless).
const QEQ_MIN_DAMPING: f64 = 0.001;
const QEQ_DAMPING_SHRINK: f64 = 0.5;
const QEQ_DAMPING_GROWTH: f64 = 1.1;
/// Grow the mixing fraction only after a charge-change reduction of at least 10%.
const QEQ_DAMPING_IMPROVEMENT: f64 = 0.9;
/// Absolute pivot threshold for the augmented QEq linear system.
const QEQ_MIN_PIVOT: f64 = 1e-12;

use crate::molecule::base::MoleculeCore;
use crate::molecule::descriptors::MolecularDescriptors;
use crate::molecule::featurize::{Hybridization, hybridization};
use crate::vector::Vector3D;

/// Six iterations, the original Gasteiger-Marsili (1980) count: `chi` values
/// are already stable to three decimal places by then. RDKit defaults to
/// twelve; the extra six buy essentially nothing here.
const GASTEIGER_ITERATIONS: usize = 6;

use chemistry_consts::{
    BOHR_RADIUS_ANGSTROM as BOHR_TO_ANGSTROM, HARTREE_EV as HARTREE_TO_EV,
    QEQ_HYDROGEN_CHARGE_DEPENDENCE as HYDROGEN_CHARGE_DEPENDENCE,
    QEQ_SLATER_EXPONENT_SCALE as LAMBDA_SCALE, get_qeq_params as qeq_parameters,
};

/// How far hydrogen's charge is allowed to push [`hydrogen_hardness`] before
/// it stops moving. `cheq`'s clamp: past +-0.95 e the linear model is well
/// outside anything it was fitted on, and letting it through would let one
/// wild intermediate charge drive the hardness negative and the next solve
/// somewhere worse.
const HYDROGEN_CHARGE_CLAMP: f64 = 0.95;

/// Largest per-atom charge change between two SCF passes that still counts as
/// converged, in elementary charge units. `cheq`'s `SolverOptions` default.
const QEQ_TOLERANCE: f64 = 1.0e-6;

/// SCF passes after which [`qeq_charges`] gives up. `cheq`'s `SolverOptions`
/// default; real molecules converge in tens, not thousands.
const QEQ_MAX_ITERATIONS: usize = 2000;

/// How much of each new solve is mixed into the running charges to start
/// with. `cheq`'s `DampingStrategy::Auto { initial: 0.4 }`; the schedule in
/// [`qeq_charges`] adapts it from there.
const QEQ_INITIAL_DAMPING: f64 = 0.4;

/// Hydrogen's hardness at charge `q`, in eV, from its `q=0` value. Every
/// other element uses its tabulated `J0` unchanged.
fn hydrogen_hardness(hardness_at_zero: f64, q: f64) -> f64 {
    let clamped = q.clamp(-HYDROGEN_CHARGE_CLAMP, HYDROGEN_CHARGE_CLAMP);
    hardness_at_zero * (1.0 + clamped * HYDROGEN_CHARGE_DEPENDENCE)
}

/// An atom's Slater exponent, in inverse Bohr, from its covalent radius (in
/// Bohr) and principal quantum number. `cheq`'s
/// `shielding::sto::compute_slater_exponent`.
fn slater_exponent(n: u8, radius_bohr: f64) -> f64 {
    LAMBDA_SCALE * (2.0 * n as f64 + 1.0) / (2.0 * radius_bohr)
}

/// The exact two-center Coulomb integral between two spherical (`ns`) Slater
/// orbitals -- the shielding QEq needs, and the only piece of quantum
/// chemistry in this module.
///
/// This is a transcription of the `sto-ns` crate (`sto_coulomb_integral`),
/// which is what `cheq` delegates to and therefore what `cheq`'s published
/// charges were computed with. It is here rather than in `Cargo.toml`
/// because the whole kernel is a few dozen lines of closed-form algebra with
/// no dependencies and no state, `oxidane` uses exactly one function from it
/// at `n` no greater than 5, and a library that publishes a dependency
/// imposes it on everyone downstream. The tests hold it to the same numbers
/// either way: `src/tests/charges_tests.rs` pins whole-molecule charges
/// against `cheq`, and pins this kernel on its own against the closed forms
/// it has to reproduce -- the analytic 1s-1s integral, the point-charge
/// limit, orbital-exchange symmetry, and smoothness as two exponents
/// converge. Transcribed, it agreed with `sto-ns` to 8e-15 relative over
/// 2.4 million points spanning every `n` pair from 1 to 5, exponents from
/// 0.15 to 4.8, and separations up to 8.85 Bohr. It now deliberately
/// disagrees in one band, where `sto-ns` is unstable and this is not: see
/// [`slater::RECURRENCE_THRESHOLD`].
///
/// The method is the standard ellipsoidal-coordinate expansion: the
/// potential of charge distribution A at distance `r` ([`potential_of`]),
/// minus its overlap with B's distribution, expanded over the auxiliary
/// integrals `A_k` and `B_k` ([`fill_auxiliary_a`], [`fill_auxiliary_b`]).
pub(crate) mod slater {
    use std::f64::consts::PI;

    /// Below this internuclear distance, in Bohr, the two-center expansion
    /// divides by a vanishing `r`; the one-center limit is exact there and
    /// continuous with it.
    const SINGULARITY_THRESHOLD: f64 = 1e-7;

    /// The `|q|` above which [`fill_auxiliary_b`] switches from its series
    /// to its recurrence. `q` is the separation times the difference of the
    /// two orbital exponents, so small `q` covers both atoms of the same
    /// element -- where it is exactly zero -- and any two atoms whose
    /// exponents happen to be close.
    ///
    /// The recurrence divides by `q` at every step, so below this it loses
    /// more digits than it has, and the loss compounds with `k`: measured
    /// against Gauss-Legendre quadrature at `q = 0.01`, its relative error
    /// is 2.4e-10 at `k = 1`, 41 at `k = 6`, and 4.1e45 at `k = 20`.
    ///
    /// Both halves of that are reachable here. `k` runs to
    /// `2*(n_a + n_b) - 2`, so anything past the second row climbs into the
    /// bad range, and `q` is `r * (zeta_a - zeta_b)`, which is small
    /// whenever two elements happen to sit close in
    /// [`chemistry_consts::QEQ_PARAMETERS`] -- O and Se differ by 0.0035, F and Cl by
    /// 0.0053, N and Cl by 0.0065. A molecule holding both, `C(F)(Cl)(Br)I`
    /// for one, came out with fluorine -- the most electronegative element
    /// in the table -- at -0.008 e and bromine at +0.0001 e. It now gives
    /// -0.50 and -0.14, in the order the halogens' electronegativities put
    /// them.
    ///
    /// 6.0 is where the two methods cross over cleanly: below it the series
    /// holds to 8e-14, above it the recurrence holds to 3e-11, and both
    /// improve away from the boundary.
    ///
    /// `sto-ns` (and so `cheq`, and so an earlier version of this module)
    /// puts this at 1e-5, which leaves the whole unstable band to the
    /// recurrence.
    const RECURRENCE_THRESHOLD: f64 = 6.0;

    /// Terms of the series for `B_k`. Enough to be exact in `f64` out to
    /// [`RECURRENCE_THRESHOLD`], where the recurrence takes over; the term
    /// `q^i / i!` is what has to have decayed by then.
    const SERIES_TERMS: usize = 32;

    /// Factorials `0!` through `31!`. The length is set by
    /// [`SERIES_TERMS`], which divides by `i!` for every term it takes; the
    /// integral's own combinatorics reach only `2*(n_a + n_b) - 2`, which
    /// is 18 at iodine. Everything here is far inside `f64`'s range --
    /// `31!` is 8.2e33 -- though past `22!` the values are rounded rather
    /// than exact, which the series tolerates because those terms are
    /// already negligible where it is used.
    const FACTORIAL: [f64; 32] = {
        let mut table = [1.0; 32];
        let mut n = 1;
        while n < 32 {
            table[n] = table[n - 1] * n as f64;
            n += 1;
        }
        table
    };

    /// Length of the auxiliary-integral scratch arrays. The largest index
    /// [`coulomb_integral`] asks for is `(2*n_a - 1) + (2*n_b - 1)`, which
    /// is 18 at the `n = 5` of iodine, the heaviest row
    /// [`chemistry_consts::QEQ_PARAMETERS`] carries.
    const AUXILIARY_SIZE: usize = 20;

    fn binomial(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        FACTORIAL[n] / (FACTORIAL[k] * FACTORIAL[n - k])
    }

    /// The squared normalisation of an `ns` Slater orbital's charge density.
    fn norm_squared(n: u8, zeta: f64) -> f64 {
        let m = (2 * n) as usize;
        (2.0 * zeta).powi((m + 1) as i32) / (FACTORIAL[m] * 4.0 * PI)
    }

    /// The electrostatic potential of one `ns` Slater charge distribution at
    /// distance `r`: the bare `1/r` a point charge would give, damped by the
    /// fraction of the cloud that has not yet been enclosed.
    fn potential_of(r: f64, n: u8, alpha: f64) -> f64 {
        let m = (2 * n) as usize;
        let ar = alpha * r;
        let mut polynomial = 0.0;
        for k in (0..m).rev() {
            polynomial = polynomial * ar + (m - k) as f64 / (m as f64 * FACTORIAL[k]);
        }
        (1.0 - (-ar).exp() * polynomial) / r
    }

    /// `A_k(p) = integral from 1 to infinity of x^k e^(-p x) dx`, by upward
    /// recurrence from `A_0 = e^-p / p` -- stable in this direction.
    fn fill_auxiliary_a(values: &mut [f64], max_k: usize, p: f64) {
        let exp_neg_p = (-p).exp();
        let inverse_p = 1.0 / p;
        let mut value = exp_neg_p * inverse_p;
        values[0] = value;
        for (index, slot) in values[1..=max_k].iter_mut().enumerate() {
            value = ((index + 1) as f64 * value + exp_neg_p) * inverse_p;
            *slot = value;
        }
    }

    /// `B_k(q) = integral from -1 to 1 of x^k e^(-q x) dx`, by expanding
    /// `e^(-q x)` and integrating term by term for small `q`, and by upward
    /// recurrence for large `q` where the series would need too many terms.
    /// Neither method works over the whole range; see
    /// [`RECURRENCE_THRESHOLD`] for where they hand over and why the
    /// recurrence cannot simply be used throughout.
    fn fill_auxiliary_b(values: &mut [f64], max_k: usize, q: f64) {
        if q.abs() < RECURRENCE_THRESHOLD {
            // Every `B_k` sums the same `(-q)^i / i!`, differing only in
            // which terms parity admits and what each is divided by, so the
            // chain is built once rather than per `k`.
            let mut weight = [0.0; SERIES_TERMS];
            let mut q_power = 1.0;
            for (slot, &factorial) in weight.iter_mut().zip(FACTORIAL.iter()) {
                *slot = q_power / factorial;
                q_power *= -q;
            }

            for (k, slot) in values[..=max_k].iter_mut().enumerate() {
                let mut sum = 0.0;
                // `(k + i)` odd contributes nothing, so only every second
                // term is visited at all.
                let mut i = k % 2;
                while i < SERIES_TERMS {
                    sum += weight[i] * 2.0 / ((k + i + 1) as f64);
                    i += 2;
                }
                *slot = sum;
            }
            return;
        }

        let (exp_q, exp_neg_q, inverse_q) = (q.exp(), (-q).exp(), 1.0 / q);
        let mut value = (exp_q - exp_neg_q) * inverse_q;
        values[0] = value;
        for (index, slot) in values[1..=max_k].iter_mut().enumerate() {
            let k = index + 1;
            let signed = if k % 2 == 0 { exp_q } else { -exp_q };
            value = (k as f64 * value + signed - exp_neg_q) * inverse_q;
            *slot = value;
        }
    }

    /// The overlap of `x^pa e^(-alpha_a x)` on one center with
    /// `x^pb e^(-alpha_b x)` on the other, in ellipsoidal coordinates, where
    /// it separates into the `A_k`/`B_k` pair above.
    ///
    /// Those two only ever depend on the exponents and the separation, so
    /// [`coulomb_integral`] fills them once per atom pair and hands them
    /// down; `pb` is the only thing that moves across the calls it makes.
    fn overlap(rho: f64, pa: i32, pb: i32, a_values: &[f64], b_values: &[f64]) -> f64 {
        let (a_exponent, b_exponent) = ((pa + 1) as usize, (pb + 1) as usize);

        let mut sum = 0.0;
        for k in 0..=a_exponent {
            for l in 0..=b_exponent {
                let sign = if (b_exponent - l) % 2 == 1 { -1.0 } else { 1.0 };
                sum += sign
                    * binomial(a_exponent, k)
                    * binomial(b_exponent, l)
                    * a_values[k + l]
                    * b_values[(a_exponent - k) + (b_exponent - l)];
            }
        }
        2.0 * PI * rho.powi(pa + pb + 3) * sum
    }

    /// Both distributions on the same center, where the angular part
    /// integrates out and only radial moments are left.
    fn one_center_limit(n_a: u8, zeta_a: f64, n_b: u8, zeta_b: f64) -> f64 {
        let (alpha_a, alpha_b) = (2.0 * zeta_a, 2.0 * zeta_b);
        let density_power = (2 * n_a) as usize - 2;
        let self_term =
            4.0 * PI * FACTORIAL[density_power + 1] / alpha_a.powi((density_power + 2) as i32);

        let m_b = (2 * n_b) as usize;
        let mut cross_term = 0.0;
        let mut alpha_b_power = 1.0;
        for (k, &factorial) in FACTORIAL.iter().take(m_b).enumerate() {
            let total = density_power + k + 1;
            cross_term +=
                (m_b - k) as f64 / (m_b as f64 * factorial) * alpha_b_power * FACTORIAL[total]
                    / (alpha_a + alpha_b).powi((total + 1) as i32);
            alpha_b_power *= alpha_b;
        }
        norm_squared(n_a, zeta_a) * (self_term - 4.0 * PI * cross_term)
    }

    /// `J_AB(r)`, in Hartree, for `ns` Slater orbitals of principal quantum
    /// number `n_a`/`n_b` and exponent `zeta_a`/`zeta_b` whose centers are
    /// `r` Bohr apart.
    pub fn coulomb_integral(r: f64, n_a: u8, zeta_a: f64, n_b: u8, zeta_b: f64) -> f64 {
        if r < SINGULARITY_THRESHOLD {
            return one_center_limit(n_a, zeta_a, n_b, zeta_b);
        }
        let (alpha_a, alpha_b) = (2.0 * zeta_a, 2.0 * zeta_b);
        let m_b = (2 * n_b) as usize;

        // `A_k` and `B_k` depend only on the two exponents and the
        // separation, all fixed here, so the loop below reuses one fill
        // instead of rebuilding them for each of its `m_b` terms. The
        // largest index any of those terms reaches is at `k = m_b - 1`.
        let rho = r / 2.0;
        let a_exponent = (2 * n_a - 1) as usize;
        let max_index = a_exponent + m_b - 1;
        debug_assert!(
            max_index < AUXILIARY_SIZE,
            "principal quantum number beyond what the auxiliary tables cover"
        );
        let mut a_values = [0.0; AUXILIARY_SIZE];
        let mut b_values = [0.0; AUXILIARY_SIZE];
        fill_auxiliary_a(&mut a_values, max_index, rho * (alpha_a + alpha_b));
        fill_auxiliary_b(&mut b_values, max_index, rho * (alpha_a - alpha_b));

        let mut overlap_sum = 0.0;
        let mut alpha_b_power = 1.0;
        for (k, &factorial) in FACTORIAL.iter().take(m_b).enumerate() {
            overlap_sum += (m_b - k) as f64 / (m_b as f64 * factorial)
                * alpha_b_power
                * overlap(rho, 2 * n_a as i32 - 2, k as i32 - 1, &a_values, &b_values);
            alpha_b_power *= alpha_b;
        }

        potential_of(r, n_a, alpha_a) - overlap_sum * norm_squared(n_a, zeta_a)
    }
}

/// The shielded Coulomb interaction between two atoms' ns-Slater-orbital
/// charge distributions, in eV. `r_bohr` is the distance between them, in
/// Bohr; `radius_bohr`/`n` are each atom's covalent radius (in Bohr) and
/// principal quantum number.
///
/// This is the exact two-center integral ([`slater::coulomb_integral`]).
/// `cheq` also offers a Gaussian-fit approximation to it, and it is
/// genuinely cheaper, but it is not what `cheq`'s published numbers were
/// computed with and it moves hydrogen's charge by ~0.1 e (see the module
/// doc comment).
fn shielded_coulomb_ev(
    r_bohr: f64,
    (n1, radius1_bohr): (u8, f64),
    (n2, radius2_bohr): (u8, f64),
) -> f64 {
    let integral_hartree = slater::coulomb_integral(
        r_bohr,
        n1,
        slater_exponent(n1, radius1_bohr),
        n2,
        slater_exponent(n2, radius2_bohr),
    );
    integral_hartree * HARTREE_TO_EV
}

/// This crate's own [`Hybridization`] mapped onto `chemistry_consts`'s
/// narrower one, for [`chemistry_consts::get_gasteiger_params`]. The
/// sulfoxide/sulfone (`So`/`So2`) distinction RDKit's table carries is not
/// reproduced: this crate's own `hybridization()` already reports sulfone
/// and sulfoxide sulfur as `Sp3` (see its doc comment), so folding both into
/// `Sp3` here is consistent with what the rest of the crate already believes
/// about that atom, not a separate simplification.
fn gasteiger_hybridization(mode: Hybridization) -> chemistry_consts::Hybridization {
    use chemistry_consts::Hybridization as G;
    match mode {
        Hybridization::Sp => G::Sp,
        Hybridization::Sp2 => G::Sp2,
        Hybridization::Sp3 | Hybridization::Sp3d | Hybridization::Sp3d2 => G::Sp3,
        Hybridization::S | Hybridization::Unspecified => G::Unspecified,
    }
}

/// Gasteiger-Marsili partial charges, one per atom, in elementary charge
/// units.
///
/// Topology only -- no 3D geometry needed, unlike [`qeq_charges`]. Each of
/// six iterations moves a little charge along every bond, away from the
/// less electronegative atom and toward the more electronegative one,
/// scaled by how far apart their electronegativities are and normalised by
/// the donor's own electronegativity as a cation (the classical PEOE
/// scheme), then damped by half each round. See the module doc comment for
/// exactly which reference implementation this mirrors.
///
/// Conserves total charge exactly: every bond's transfer is added at one
/// atom and subtracted at the other, so the sum never moves from where it
/// started -- each atom's formal charge (see
/// [`MolecularDescriptors::total_formal_charge`]).
///
/// An atom whose element/hybridization pair has no Gasteiger parameters
/// (selenium, among [`crate::molecule::featurize::FEATURED_ELEMENTS`]) never
/// gives or receives charge -- it contributes nothing, the same way an
/// unparametrised atom contributes nothing to
/// [`crate::molecule::crippen`]'s log P.
pub fn gasteiger_charges<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<f64> {
    let atom_count = molecule.len();
    let mut charges: Vec<f64> = (0..atom_count)
        .map(|atom| molecule.get_atom_charge(atom) as f64)
        .collect();
    if atom_count == 0 {
        return charges;
    }

    let params: Vec<Option<(f64, f64, f64)>> = (0..atom_count)
        .map(|atom| {
            let atomic_number = molecule.get_atomic_number(atom);
            let mode = gasteiger_hybridization(hybridization(molecule, atom));
            chemistry_consts::get_gasteiger_params(atomic_number, mode)
        })
        .collect();
    // `chi` at q=+1: the classical PEOE normaliser, "the electronegativity of
    // the corresponding cation" of whichever atom in a bond is donating.
    let cation_chi: Vec<f64> = params
        .iter()
        .map(|p| p.map_or(0.0, |(a, b, c)| a + b + c))
        .collect();

    let mut damp = GASTEIGER_INITIAL_DAMPING;
    for _ in 0..GASTEIGER_ITERATIONS {
        let chi: Vec<f64> = (0..atom_count)
            .map(|atom| {
                params[atom].map_or(0.0, |(a, b, c)| {
                    let q = charges[atom];
                    a + q * (b + c * q)
                })
            })
            .collect();

        let mut delta = vec![0.0; atom_count];
        for atom in 0..atom_count {
            if params[atom].is_none() {
                continue;
            }
            for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
                let neighbor = bond.target();
                // Each undirected bond is visited once, from its
                // lower-indexed atom -- the graph hands every atom the bond
                // from its own side, so processing both would double the
                // transfer.
                if neighbor <= atom || params[neighbor].is_none() {
                    continue;
                }
                let dx = chi[neighbor] - chi[atom];
                let donor = if dx >= 0.0 { atom } else { neighbor };
                let transfer = dx / cation_chi[donor];
                delta[atom] += transfer;
                delta[neighbor] -= transfer;
            }
        }
        for atom in 0..atom_count {
            charges[atom] += damp * delta[atom];
        }
        damp *= GASTEIGER_DAMPING_DECAY;
    }
    charges
}

/// Something [`qeq_charges`] cannot compute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChargeError {
    /// An atom's element has no row in `QEQ_PARAMETERS`. Every element
    /// [`crate::molecule::featurize::FEATURED_ELEMENTS`] covers does; this
    /// only fires for something outside that set.
    MissingParameters(u8),
    /// Hydrogen's charge-dependent hardness never settled: the SCF loop in
    /// [`qeq_charges`] ran `QEQ_MAX_ITERATIONS` passes and the largest
    /// per-atom charge was still moving by more than `QEQ_TOLERANCE`.
    /// Carries that last movement, so a caller can tell "nearly there" from
    /// "oscillating wildly".
    NotConverged(f64),
}

impl std::fmt::Display for ChargeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChargeError::MissingParameters(atomic_number) => {
                write!(f, "no QEq parameters for atomic number {atomic_number}")
            }
            ChargeError::NotConverged(delta) => write!(
                f,
                "QEq did not converge in {QEQ_MAX_ITERATIONS} iterations \
                 (largest charge still moving by {delta})"
            ),
        }
    }
}

impl std::error::Error for ChargeError {}

/// QEq (charge equilibration, Rappe & Goddard 1991) partial charges, one per
/// atom, in elementary charge units.
///
/// Needs a 3D embedding, unlike [`gasteiger_charges`]: `positions` is
/// indexed exactly like `molecule`'s own atoms, the same convention
/// [`crate::molecule::featurize::graph_features_with_geometry`] uses --
/// straight from what [`crate::molecule::embed::Embed3D::embed_3d`] returns.
/// Every atom's electronegativity is charge-dependent
/// (`chi_i(q) = chi0_i + J0_i*q_i`); the charges that equalise every atom's
/// `chi` -- the physical requirement that electron density stops moving once
/// nothing gains by moving further -- are the solution of an `(N+1)x(N+1)`
/// linear system (`N` unknowns plus a Lagrange multiplier enforcing that the
/// charges sum to the molecule's total formal charge). Off-diagonal entries
/// are a Slater-shielded Coulomb interaction between atoms
/// (`shielded_coulomb_ev`) rather than a bare `1/R`, which is what makes
/// two directly bonded atoms not blow up as `R` gets small.
///
/// One solve is enough only for a molecule with no hydrogen. Hydrogen's
/// hardness depends on the charge being solved for
/// (`hydrogen_hardness`), so the system is nonlinear there and is
/// iterated to self-consistency with adaptive damping -- typically tens of
/// passes. See the module doc comment for exactly which reference
/// implementation, and which of its options, this mirrors.
///
/// # Errors
///
/// [`ChargeError::MissingParameters`] if any atom's element has no QEq
/// parameters. A linear-algebra method silently zeroing one atom's row would
/// break the charge-conservation guarantee for every other atom in the same
/// molecule, so this refuses rather than guesses.
///
/// [`ChargeError::NotConverged`] if the hydrogen self-consistency loop is
/// still moving after `QEQ_MAX_ITERATIONS` passes -- likewise a refusal
/// rather than a half-equilibrated answer that would still look plausible.
pub fn qeq_charges<M: MolecularDescriptors + ?Sized>(
    molecule: &M,
    positions: &[Vector3D],
) -> Result<Vec<f64>, ChargeError> {
    let atom_count = molecule.len();
    if atom_count == 0 {
        return Ok(Vec::new());
    }

    let mut chi0 = vec![0.0; atom_count];
    let mut hardness = vec![0.0; atom_count];
    let mut radius_bohr = vec![0.0; atom_count];
    let mut principal_quantum_number = vec![0u8; atom_count];
    for atom in 0..atom_count {
        let atomic_number = molecule.get_atomic_number(atom);
        let (c0, j0, radius, n) =
            qeq_parameters(atomic_number).ok_or(ChargeError::MissingParameters(atomic_number))?;
        chi0[atom] = c0;
        hardness[atom] = j0;
        radius_bohr[atom] = radius / BOHR_TO_ANGSTROM;
        principal_quantum_number[atom] = n;
    }

    // Everything except hydrogen's diagonal is fixed by the geometry, so it
    // is built once and reused by every SCF pass below. Row-major in one
    // allocation: the SCF loop copies this matrix once per pass, and a
    // row-of-vectors would make that copy `n` separate allocations.
    let n = atom_count;
    let size = n + 1;
    let mut base = vec![0.0; size * size];
    let mut rhs = vec![0.0; size];
    for i in 0..n {
        base[i * size + i] = hardness[i];
        // Row i is `J0_i*q_i + sum_j J_ij*q_j - lambda = -chi0_i`, from
        // minimising `sum_i(chi0_i*q_i + J0_i*q_i^2/2) + sum_{i<j} J_ij*q_i*q_j`
        // subject to `sum(q_i) = Z` (Lagrange multiplier `lambda`) -- the
        // stationarity condition is that every atom's chemical potential
        // `chi0_i + J0_i*q_i + sum_j J_ij*q_j` equals the same `lambda`.
        base[i * size + n] = -1.0;
        base[n * size + i] = 1.0;
        rhs[i] = -chi0[i];
        for j in (i + 1)..n {
            let r_bohr = positions[i].distance(&positions[j]) / BOHR_TO_ANGSTROM;
            let interaction = shielded_coulomb_ev(
                r_bohr,
                (principal_quantum_number[i], radius_bohr[i]),
                (principal_quantum_number[j], radius_bohr[j]),
            );
            base[i * size + j] = interaction;
            base[j * size + i] = interaction;
        }
    }
    rhs[n] = molecule.total_formal_charge() as f64;

    // The SCF loop below solves the same shape repeatedly, so both the
    // matrix it eliminates in place and the right-hand side it consumes are
    // allocated once here and overwritten per pass.
    let mut working_matrix = vec![0.0; size * size];
    let mut working_rhs = vec![0.0; size];

    let hydrogens: Vec<usize> = (0..n)
        .filter(|&atom| molecule.get_atomic_number(atom) == 1)
        .collect();
    if hydrogens.is_empty() {
        // Nothing on the diagonal depends on charge: one solve is exact.
        working_matrix.copy_from_slice(&base);
        working_rhs.copy_from_slice(&rhs);
        assert!(
            solve_linear_system(&mut working_matrix, &mut working_rhs),
            "{SOLVABLE}"
        );
        return Ok(working_rhs[..n].to_vec());
    }

    // With hydrogen present the diagonal depends on the charges it is being
    // solved for, so the system is nonlinear and has to be iterated: solve,
    // mix the answer into the running charges, rebuild hydrogen's diagonal,
    // repeat. Damping starts at `QEQ_INITIAL_DAMPING` and adapts -- halved
    // whenever a pass moves the charges further than the last one did
    // (oscillation), eased back up while they are settling -- which is
    // `cheq`'s `DampingStrategy::Auto`.
    let mut charges = vec![0.0; n];
    let mut damping = QEQ_INITIAL_DAMPING;
    let mut previous_delta = f64::MAX;
    for _ in 0..QEQ_MAX_ITERATIONS {
        working_matrix.copy_from_slice(&base);
        for &atom in &hydrogens {
            working_matrix[atom * size + atom] = hydrogen_hardness(hardness[atom], charges[atom]);
        }
        working_rhs.copy_from_slice(&rhs);
        assert!(
            solve_linear_system(&mut working_matrix, &mut working_rhs),
            "{SOLVABLE}"
        );
        let solution = &working_rhs;

        let delta = (0..n)
            .map(|atom| (solution[atom] - charges[atom]).abs())
            .fold(0.0, f64::max);
        for atom in 0..n {
            charges[atom] = (1.0 - damping) * charges[atom] + damping * solution[atom];
        }
        if delta < QEQ_TOLERANCE {
            return Ok(charges);
        }

        if delta > previous_delta {
            damping = (damping * QEQ_DAMPING_SHRINK).max(QEQ_MIN_DAMPING);
        } else if delta < previous_delta * QEQ_DAMPING_IMPROVEMENT {
            damping = (damping * QEQ_DAMPING_GROWTH).min(1.0);
        }
        previous_delta = delta;
    }
    Err(ChargeError::NotConverged(previous_delta))
}

/// Why [`qeq_charges`] treats an unsolvable system as a bug rather than a
/// [`ChargeError`]: every row carries that element's chemical hardness on
/// the diagonal, which is positive for every row of [`chemistry_consts::QEQ_PARAMETERS`].
const SOLVABLE: &str = "QEq's hardness matrix has a nonzero diagonal (chemical hardness) \
                        on every row and is solvable for any real molecule";

/// Solves `matrix * x = rhs` by Gaussian elimination with partial pivoting,
/// leaving `x` in `rhs`. `false` if the matrix is (numerically) singular.
/// `matrix` is row-major and square with `rhs.len()` rows; both are
/// scratch, overwritten as elimination proceeds.
///
/// Plain dense LU, unblocked. At the sizes [`qeq_charges`] calls it with,
/// that is not the thing to fix. For a 76-atom molecule the whole matrix is
/// 47 KB and sits in L2, so tiling has little to recover, and the eleven
/// SCF passes together take 0.37 ms against 0.49 ms for the shielding
/// integrals that built the matrix in the first place.
///
/// Two things were measured and not taken, recorded here so nobody has to
/// measure them twice:
///
/// * `faer`, at `Par::Seq`, solves this 1.24x faster at 76 atoms and 1.64x
///   at 300 -- about 8% off a whole `qeq_charges` call at the first size,
///   and *slower* than this at 24 atoms, where its fixed overhead has
///   nothing to amortise against. Not worth the dependency. Note also that
///   its default parallel mode is catastrophic at this scale: 13.5 ms
///   against 0.37 ms here, thread-pool setup per factorization swamping the
///   work.
/// * The matrix is symmetric apart from the sign of its Lagrange row.
///   Negating that row makes it symmetric indefinite, and `LDL^T` would
///   then do roughly half the arithmetic of `LU` -- about 20% off the call,
///   more than `faer` offers and with nothing added to `Cargo.toml`. It
///   needs Bunch-Kaufman pivoting to be safe on an indefinite matrix, which
///   is real work and would want the same kind of verification the Slater
///   kernel got.
fn solve_linear_system(matrix: &mut [f64], rhs: &mut [f64]) -> bool {
    let n = rhs.len();
    for pivot in 0..n {
        let Some(best) = (pivot..n).max_by(|&a, &b| {
            matrix[a * n + pivot]
                .abs()
                .partial_cmp(&matrix[b * n + pivot].abs())
                .unwrap()
        }) else {
            return false;
        };
        if matrix[best * n + pivot].abs() < QEQ_MIN_PIVOT {
            return false;
        }
        if best != pivot {
            for col in 0..n {
                matrix.swap(pivot * n + col, best * n + col);
            }
            rhs.swap(pivot, best);
        }
        let diagonal = matrix[pivot * n + pivot];
        for row in (pivot + 1)..n {
            let factor = matrix[row * n + pivot] / diagonal;
            if factor == 0.0 {
                continue;
            }
            // Splitting gives the pivot row and the row being eliminated as
            // disjoint slices, which is what lets this inner loop drop its
            // bounds checks and vectorize.
            let (above, below) = matrix.split_at_mut(row * n);
            let pivot_row = &above[pivot * n..pivot * n + n];
            let target = &mut below[..n];
            for col in pivot..n {
                target[col] -= factor * pivot_row[col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    // Back-substitution in place: `rhs[row]` is still the right-hand side
    // when it is read, and every `rhs[col]` above it has already become
    // `x[col]`.
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for col in (row + 1)..n {
            sum -= matrix[row * n + col] * rhs[col];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
    true
}
