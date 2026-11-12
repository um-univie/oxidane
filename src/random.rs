//! A small, fast, seedable random number generator.
//!
//! The embedder draws random trial distances, start vectors and torsion
//! moves, and needs them reproducible from a seed. That takes a few dozen
//! lines, not a dependency: [`Rng`] is xoshiro256++ (Blackman and Vigna),
//! seeded through SplitMix64 as its authors recommend. It is not meant for
//! cryptography.

use std::hash::{BuildHasher, Hasher};

/// xoshiro256++: 256 bits of state, period 2^256 - 1.
///
/// ```
/// use oxidane::random::Rng;
///
/// let mut a = Rng::seed_from_u64(7);
/// let mut b = Rng::seed_from_u64(7);
/// assert_eq!(a.next_u64(), b.next_u64());
/// let x = a.uniform(-1.0, 1.0);
/// assert!((-1.0..1.0).contains(&x));
/// assert!(a.below(10) < 10);
/// ```
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// A generator whose sequence is fixed by `seed`.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut mix = seed;
        let mut next = || {
            mix = mix.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = mix;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // SplitMix64 never yields four zeros in a row, so the state is valid.
        Self { state: [next(), next(), next(), next()] }
    }

    /// A generator seeded differently on every call, from the per-process
    /// random keys of the standard library's hash maps and the clock.
    pub fn from_entropy() -> Self {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            hasher.write_u128(now.as_nanos());
        }
        Self::seed_from_u64(hasher.finish())
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = &mut self.state;
        let result = s0.wrapping_add(*s3).rotate_left(23).wrapping_add(*s0);
        let t = *s1 << 17;
        *s2 ^= *s0;
        *s3 ^= *s1;
        *s1 ^= *s2;
        *s0 ^= *s3;
        *s2 ^= t;
        *s3 = s3.rotate_left(45);
        result
    }

    /// A float in `[0, 1)`, from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A float in `[low, high)`; `low` itself when the two are equal.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        debug_assert!(low <= high, "empty range {low}..{high}");
        let value = low + (high - low) * self.next_f64();
        // Rounding can land exactly on `high`; keep the interval half-open.
        if value < high { value } else { low }
    }

    /// An integer in `[0, bound)`, without modulo bias (Lemire's method).
    ///
    /// # Panics
    ///
    /// If `bound` is zero.
    pub fn below(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "Rng::below(0)");
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u128::from(self.next_u64()) * u128::from(bound);
            if (product as u64) >= threshold {
                return (product >> 64) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_xoshiro256plusplus_sequence() {
        // State 1, 2, 3, 4 gives these outputs in the reference C code.
        let mut rng = Rng { state: [1, 2, 3, 4] };
        let expected = [41943041, 58720359, 3588806011781223, 3591011842654386];
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

    #[test]
    fn below_is_uniform_and_in_range() {
        let mut rng = Rng::seed_from_u64(1);
        let mut counts = [0usize; 7];
        for _ in 0..70_000 {
            counts[rng.below(7)] += 1;
        }
        for count in counts {
            assert!((count as f64 - 10_000.0).abs() < 400.0, "{counts:?}");
        }
    }

    #[test]
    fn uniform_stays_in_its_interval() {
        let mut rng = Rng::seed_from_u64(2);
        for _ in 0..10_000 {
            let x = rng.uniform(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&x));
        }
        assert_eq!(rng.uniform(3.5, 3.5), 3.5);
    }

    #[test]
    fn entropy_seeds_differ() {
        assert_ne!(Rng::from_entropy().next_u64(), Rng::from_entropy().next_u64());
    }
}
