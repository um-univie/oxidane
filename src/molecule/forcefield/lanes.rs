//! Four `f64` lanes for the force fields' pair loops, on whichever
//! instruction set the CPU offers.
//!
//! Both force fields' nonbonded sums are one long loop doing the same
//! arithmetic per pair, and both are written once over [`Lanes`]. Their
//! `energy` and `energy_and_gradient` check for AVX2 per call and run the
//! whole evaluation inside [`pulp`]'s dispatch when it is there, so that
//! the crate can keep building for baseline x86-64.

/// How many pairs one pass of a vectorised loop covers.
pub(crate) const LANES: usize = 4;

/// Four `f64` lanes, and the few operations the pair loop needs beyond
/// arithmetic.
///
/// Two of these exist. [`Portable`] is `wide::f64x4`, whose instruction set is
/// fixed when the crate is compiled -- for baseline x86-64 that is two SSE2
/// halves. [`Avx`] is the same four lanes in one AVX register, available only
/// once [`pulp`] has confirmed the CPU has AVX2 and only inside code it has
/// compiled for it; see `mmff::ForceField::energy_and_gradient`. Both do exactly
/// the same IEEE operations in the same order, so they agree to the bit and
/// the choice between them changes speed, not results.
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

/// [`Lanes`] on `wide`, for whatever the crate was compiled for.
#[derive(Clone, Copy)]
pub(crate) struct Portable;

impl Lanes for Portable {
    type Vector = wide::f64x4;

    #[inline(always)]
    fn splat(self, value: f64) -> wide::f64x4 {
        wide::f64x4::splat(value)
    }
    #[inline(always)]
    fn from_array(self, values: [f64; LANES]) -> wide::f64x4 {
        wide::f64x4::new(values)
    }
    #[inline(always)]
    fn to_array(self, vector: wide::f64x4) -> [f64; LANES] {
        vector.to_array()
    }
    #[inline(always)]
    fn sqrt(self, vector: wide::f64x4) -> wide::f64x4 {
        vector.sqrt()
    }
    #[inline(always)]
    fn select_greater(
        self,
        left: wide::f64x4,
        right: wide::f64x4,
        if_greater: wide::f64x4,
        otherwise: wide::f64x4,
    ) -> wide::f64x4 {
        wide::CmpGt::cmp_gt(left, right).blend(if_greater, otherwise)
    }
}

/// [`Lanes`] in one 256-bit register. Holding a `pulp::x86::V3` is the proof
/// that the CPU has AVX2, which is what makes the operations below sound.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct Avx(pub(crate) pulp::x86::V3);

/// One [`Avx`] vector. It carries the token so the arithmetic operators can
/// reach the instructions.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(crate) struct AvxVector(pulp::x86::V3, pulp::f64x4);

#[cfg(target_arch = "x86_64")]
macro_rules! avx_binary {
    ($trait:ident, $method:ident, $instruction:ident) => {
        impl core::ops::$trait for AvxVector {
            type Output = AvxVector;
            #[inline(always)]
            fn $method(self, other: AvxVector) -> AvxVector {
                AvxVector(self.0, self.0.$instruction(self.1, other.1))
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
avx_binary!(Add, add, add_f64x4);
#[cfg(target_arch = "x86_64")]
avx_binary!(Sub, sub, sub_f64x4);
#[cfg(target_arch = "x86_64")]
avx_binary!(Mul, mul, mul_f64x4);
#[cfg(target_arch = "x86_64")]
avx_binary!(Div, div, div_f64x4);

#[cfg(target_arch = "x86_64")]
impl core::ops::Neg for AvxVector {
    type Output = AvxVector;
    /// `0 - x`, as `wide` does it, rather than a sign flip: the two differ
    /// on a zero, and the paths have to agree to the bit.
    #[inline(always)]
    fn neg(self) -> AvxVector {
        AvxVector(self.0, self.0.sub_f64x4(self.0.splat_f64x4(0.0), self.1))
    }
}

#[cfg(target_arch = "x86_64")]
impl core::ops::AddAssign for AvxVector {
    #[inline(always)]
    fn add_assign(&mut self, other: AvxVector) {
        *self = *self + other;
    }
}

#[cfg(target_arch = "x86_64")]
impl Lanes for Avx {
    type Vector = AvxVector;

    #[inline(always)]
    fn splat(self, value: f64) -> AvxVector {
        AvxVector(self.0, self.0.splat_f64x4(value))
    }
    #[inline(always)]
    fn from_array(self, [a, b, c, d]: [f64; LANES]) -> AvxVector {
        AvxVector(self.0, pulp::f64x4(a, b, c, d))
    }
    #[inline(always)]
    fn to_array(self, vector: AvxVector) -> [f64; LANES] {
        let pulp::f64x4(a, b, c, d) = vector.1;
        [a, b, c, d]
    }
    #[inline(always)]
    fn sqrt(self, vector: AvxVector) -> AvxVector {
        AvxVector(self.0, self.0.sqrt_f64x4(vector.1))
    }
    #[inline(always)]
    fn select_greater(
        self,
        left: AvxVector,
        right: AvxVector,
        if_greater: AvxVector,
        otherwise: AvxVector,
    ) -> AvxVector {
        let mask = self.0.cmp_gt_f64x4(left.1, right.1);
        AvxVector(self.0, self.0.select_f64x4(mask, if_greater.1, otherwise.1))
    }
}

/// The four lanes added up, pairwise so the order is at least fixed.
#[inline(always)]
pub(crate) fn horizontal_sum(lanes: [f64; LANES]) -> f64 {
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}
