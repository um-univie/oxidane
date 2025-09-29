//! MinHash sketches of a fingerprint.
//!
//! Tanimoto over a folded bit vector degrades badly as molecules grow: a
//! peptide or a natural product sets so many bits that two unrelated ones look
//! alike, and the identifiers that would have told them apart have collided.
//! A MinHash sketch keeps a fixed number of numbers whatever the molecule's
//! size, and the fraction of them that agree estimates the Jaccard index of the
//! identifier sets themselves -- the unfolded ones. That is why MHFP and MAP4
//! beat folded ECFP on exactly the molecules where folded ECFP gives up.
//!
//! The estimate is unbiased and its error falls as one over the square root of
//! the sketch length, so 128 permutations are worth about ±0.09 and 2048 about
//! ±0.02. Choosing the length is choosing how much precision to pay for.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::fingerprint::{MinHash, circular_fingerprint};
//!
//! let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
//! let salicylic = MolecularGraph::from_smiles_single("OC(=O)c1ccccc1O").unwrap();
//! let first = MinHash::of(&circular_fingerprint(&aspirin, 2).fingerprint, 256);
//! let second = MinHash::of(&circular_fingerprint(&salicylic, 2).fingerprint, 256);
//! assert_eq!(first.jaccard(&first), 1.0);
//! assert!(first.jaccard(&second) > 0.2);
//! ```

use super::{Fingerprint, mix64};

/// A fixed-length sketch of a set of identifiers.
///
/// Two sketches may only be compared when they are the same length: the
/// permutation a slot stands for is decided by the slot's number, so slot seven
/// of one sketch and slot seven of another are the same question asked of two
/// molecules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHash {
    values: Vec<u32>,
}

impl MinHash {
    /// Sketches a fingerprint with `permutations` independent hashes.
    ///
    /// A slot holds the smallest value any identifier took under that slot's
    /// hash, which is what makes two sets agree on a slot exactly when the
    /// identifier that minimises it is in both.
    pub fn of(fingerprint: &Fingerprint, permutations: usize) -> Self {
        // One thorough hash per identifier, then a multiplier and an offset per
        // slot: `a * h ^ b` is a permutation family good enough to sketch with
        // once `h` is already well mixed, and it costs one multiply and one xor
        // where mixing each identifier afresh for every slot costs five of each.
        //
        // The arithmetic is 32-bit and the running minimum is taken
        // branchlessly, both so that the inner loop vectorises: it is an
        // identifier count times a permutation count, which for a minhashed
        // pair fingerprint of a large molecule is half a million rounds, and it
        // is the whole cost of the sketch.
        let mut multipliers: Vec<u32> = Vec::with_capacity(permutations);
        let mut offsets: Vec<u32> = Vec::with_capacity(permutations);
        for slot in 0..permutations {
            // An even multiplier loses a bit of the hash on every round.
            multipliers.push((mix64(slot as u64 * 2 + 1) >> 32) as u32 | 1);
            offsets.push((mix64(slot as u64 * 2 + 2) >> 32) as u32);
        }

        let mut values = vec![u32::MAX; permutations];
        for id in fingerprint.ids() {
            let hashed = (mix64(id as u64) >> 32) as u32;
            for ((value, &multiplier), &offset) in values
                .iter_mut()
                .zip(multipliers.iter())
                .zip(offsets.iter())
            {
                *value = (*value).min(multiplier.wrapping_mul(hashed) ^ offset);
            }
        }
        Self { values }
    }

    /// The sketch's slots.
    pub fn values(&self) -> &[u32] {
        &self.values
    }

    /// How many slots it has.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The estimated Jaccard index: the fraction of slots that agree.
    ///
    /// Sketches of different lengths cannot be compared and give zero. Two
    /// sketches of nothing agree on everything they say.
    pub fn jaccard(&self, other: &Self) -> f64 {
        if self.values.len() != other.values.len() || self.values.is_empty() {
            return if self.values == other.values { 1.0 } else { 0.0 };
        }
        let agreeing = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(left, right)| left == right)
            .count();
        agreeing as f64 / self.values.len() as f64
    }

    /// The estimated Jaccard distance, one minus the index.
    pub fn distance(&self, other: &Self) -> f64 {
        1.0 - self.jaccard(other)
    }
}
