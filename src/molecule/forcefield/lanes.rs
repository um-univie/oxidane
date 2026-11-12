//! Four `f64` lanes for the force fields' pair loops, on whichever
//! instruction set the CPU offers.
//!
//! Both force fields' nonbonded sums are one long loop doing the same
//! arithmetic per pair, and both are written once over [`Lanes`]. Their
//! `energy` and `energy_and_gradient` go through [`dispatch`], which asks
//! [`pulp`] per call what the CPU has and runs the whole evaluation compiled
//! for it, so that the crate can keep building for each target's baseline.

/// How many pairs one pass of a vectorised loop covers.
pub(crate) const LANES: usize = 4;

/// Four `f64` lanes, and the few operations the pair loop needs beyond
/// arithmetic.
///
/// [`Pulp`] implements it on whatever [`pulp`] has confirmed the CPU offers,
/// available only inside code it has compiled for it; see [`dispatch`].
/// [`PORTABLE`] is the same on pulp's scalar backend, which runs anywhere.
/// Every backend does exactly the same IEEE operations in the same order, so
/// they agree to the bit and the choice between them changes speed, not
/// results.
pub(crate) trait Lanes: Copy {
    type Vector: Copy
        + core::ops::Add<Output = Self::Vector>
        + core::ops::Sub<Output = Self::Vector>
        + core::ops::Mul<Output = Self::Vector>
        + core::ops::Div<Output = Self::Vector>
        + core::ops::Neg<Output = Self::Vector>
        + core::ops::AddAssign;

    fn splat(self, value: f64) -> Self::Vector;
    fn from_array(self, values: [f64; LANES]) -> Self::Vector;
    fn to_array(self, vector: Self::Vector) -> [f64; LANES];
    fn sqrt(self, vector: Self::Vector) -> Self::Vector;
    /// `if_greater` where `left > right`, `otherwise` elsewhere.
    fn select_greater(
        self,
        left: Self::Vector,
        right: Self::Vector,
        if_greater: Self::Vector,
        otherwise: Self::Vector,
    ) -> Self::Vector;
}

/// [`Lanes`] on any instruction set [`pulp`] can dispatch to: the four lanes
/// held in `LANES / S::F64_LANES` registers of `S` -- one AVX2 register, two
/// NEON ones -- and each operation applied to each register in turn. Holding
/// an `S` is the proof that the CPU has its instructions, which is what makes
/// the operations sound.
///
/// The width of `S` changes how many registers a vector takes, never which
/// operations a lane sees, so every `S` agrees with [`PORTABLE`] to the bit.
#[derive(Clone, Copy)]
pub(crate) struct Pulp<S>(pub(crate) S);

/// [`Pulp`] on pulp's one-lane scalar backend: the four lanes as four plain
/// `f64`, on every target. On baseline x86-64 LLVM still packs them into
/// SSE2 pairs, and the MMFF pair loop runs as fast as on the `wide::f64x4`
/// this replaced, with the same bits.
pub(crate) const PORTABLE: Pulp<pulp::Scalar> = Pulp(pulp::Scalar);

/// One [`Pulp`] vector. It carries the token so the arithmetic operators can
/// reach the instructions. Only the first [`Self::PARTS`] registers are used;
/// the rest are never read and compile away.
#[derive(Clone, Copy)]
pub(crate) struct PulpVector<S: pulp::Simd>(S, [S::f64s; LANES]);

impl<S: pulp::Simd> PulpVector<S> {
    /// How many registers of `S` the four lanes take.
    const PARTS: usize = LANES / S::F64_LANES;

    #[inline(always)]
    fn map(self, operation: impl Fn(S, S::f64s) -> S::f64s) -> Self {
        let mut parts = self.1;
        for part in &mut parts[..Self::PARTS] {
            *part = operation(self.0, *part);
        }
        PulpVector(self.0, parts)
    }

    #[inline(always)]
    fn zip(self, other: Self, operation: impl Fn(S, S::f64s, S::f64s) -> S::f64s) -> Self {
        let mut parts = self.1;
        for (part, &right) in parts[..Self::PARTS].iter_mut().zip(&other.1) {
            *part = operation(self.0, *part, right);
        }
        PulpVector(self.0, parts)
    }
}

macro_rules! pulp_binary {
    ($trait:ident, $method:ident, $instruction:ident) => {
        impl<S: pulp::Simd> core::ops::$trait for PulpVector<S> {
            type Output = PulpVector<S>;
            #[inline(always)]
            fn $method(self, other: PulpVector<S>) -> PulpVector<S> {
                self.zip(other, |simd, left, right| simd.$instruction(left, right))
            }
        }
    };
}
pulp_binary!(Add, add, add_f64s);
pulp_binary!(Sub, sub, sub_f64s);
pulp_binary!(Mul, mul, mul_f64s);
pulp_binary!(Div, div, div_f64s);

impl<S: pulp::Simd> core::ops::Neg for PulpVector<S> {
    type Output = PulpVector<S>;
    /// `0 - x` rather than a sign flip: the two differ on a zero, and this
    /// keeps the results of the earlier `wide` backend to the bit.
    #[inline(always)]
    fn neg(self) -> PulpVector<S> {
        self.map(|simd, value| simd.sub_f64s(simd.splat_f64s(0.0), value))
    }
}

impl<S: pulp::Simd> core::ops::AddAssign for PulpVector<S> {
    #[inline(always)]
    fn add_assign(&mut self, other: PulpVector<S>) {
        *self = *self + other;
    }
}

impl<S: pulp::Simd> Lanes for Pulp<S> {
    type Vector = PulpVector<S>;

    #[inline(always)]
    fn splat(self, value: f64) -> PulpVector<S> {
        PulpVector(self.0, [self.0.splat_f64s(value); LANES])
    }
    #[inline(always)]
    fn from_array(self, values: [f64; LANES]) -> PulpVector<S> {
        let mut parts = [self.0.splat_f64s(0.0); LANES];
        let (registers, _) = S::as_simd_f64s(&values);
        parts[..PulpVector::<S>::PARTS].copy_from_slice(registers);
        PulpVector(self.0, parts)
    }
    #[inline(always)]
    fn to_array(self, vector: PulpVector<S>) -> [f64; LANES] {
        let mut values = [0.0; LANES];
        let (registers, _) = S::as_mut_simd_f64s(&mut values);
        registers.copy_from_slice(&vector.1[..PulpVector::<S>::PARTS]);
        values
    }
    #[inline(always)]
    fn sqrt(self, vector: PulpVector<S>) -> PulpVector<S> {
        vector.map(|simd, value| simd.sqrt_f64s(value))
    }
    #[inline(always)]
    fn select_greater(
        self,
        left: PulpVector<S>,
        right: PulpVector<S>,
        if_greater: PulpVector<S>,
        otherwise: PulpVector<S>,
    ) -> PulpVector<S> {
        let simd = self.0;
        let mut parts = otherwise.1;
        for part in 0..PulpVector::<S>::PARTS {
            let mask = simd.greater_than_f64s(left.1[part], right.1[part]);
            parts[part] = simd.select_f64s(mask, if_greater.1[part], otherwise.1[part]);
        }
        PulpVector(simd, parts)
    }
}

/// A computation written once over [`Lanes`], for [`dispatch`] to run on
/// whichever lanes the CPU offers.
pub(crate) trait LanesOp {
    type Output;
    /// Marked `#[inline(always)]` by implementors, so that inside
    /// [`pulp`]'s dispatch the whole computation is compiled for the
    /// instruction set it detected.
    fn run<L: Lanes>(self, lanes: L) -> Self::Output;
}

struct Dispatch<Op>(Op);

impl<Op: LanesOp> pulp::WithSimd for Dispatch<Op> {
    type Output = Op::Output;
    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Op::Output {
        // A width that holds the four lanes in whole registers runs on it
        // -- one lane included, where pulp found no vector unit; one wider
        // than four runs on the scalar backend.
        if LANES.is_multiple_of(S::F64_LANES) {
            self.0.run(Pulp(simd))
        } else {
            self.0.run(PORTABLE)
        }
    }
}

/// Runs `op` on the best lanes this CPU has: AVX2 on x86-64, NEON on
/// AArch64, [`PORTABLE`] where [`pulp`] finds nothing better. Detection is per call
/// and cheap; the answers are the same bits whichever is chosen.
#[inline]
pub(crate) fn dispatch<Op: LanesOp>(op: Op) -> Op::Output {
    pulp::Arch::new().dispatch(Dispatch(op))
}

/// The four lanes added up, pairwise so the order is at least fixed.
#[inline(always)]
pub(crate) fn horizontal_sum(lanes: [f64; LANES]) -> f64 {
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// `op` on [`PORTABLE`] -- the generic code, on every target -- and through
/// [`dispatch`] on whatever this CPU offers, for tests that the two agree to
/// the bit.
#[cfg(test)]
pub(crate) fn on_every_lanes<Op: LanesOp + Clone>(op: Op) -> [Op::Output; 2] {
    [op.clone().run(PORTABLE), dispatch(op)]
}
