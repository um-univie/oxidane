//! Molecular fingerprints: what they are made of, and how they are compared.
//!
//! A fingerprint here is a multiset of 32-bit identifiers, each standing for
//! something the molecule contains -- an atom environment, a pair of atoms at a
//! distance, a substructure key. That sparse form is the one to keep: it is
//! exact, it can be counted rather than merely flagged, and an identifier can
//! be traced back to the atoms that produced it. Folding it down to a bit
//! vector loses all three and is a decision about a *query*, not about a
//! molecule, so it happens at the point of comparison rather than at the point
//! of generation.
//!
//! Identifiers are produced by [`StableHasher`], whose output is defined here
//! rather than borrowed from the standard library or from `rustc_hash`. A
//! fingerprint written to a file and compared against one generated a year
//! later by a different build has to agree bit for bit, and neither `SipHash`'s
//! random keys nor `FxHash`'s deliberately unspecified output can promise that.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::fingerprint::circular_fingerprint;
//!
//! let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
//! let written_differently =
//!     MolecularGraph::from_smiles_single("OC(=O)C1=CC=CC=C1OC(C)=O").unwrap();
//! let first = circular_fingerprint(&aspirin, 2);
//! let second = circular_fingerprint(&written_differently, 2);
//! assert_eq!(first.fingerprint, second.fingerprint);
//! assert_eq!(first.fingerprint.tanimoto(&second.fingerprint), 1.0);
//! ```

mod circular;
mod heavy;
mod maccs;
mod minhash;
mod pairs;
mod shingles;

pub use maccs::{MACCS_KEY_COUNT, maccs_keys};
pub use minhash::MinHash;

pub use shingles::{
    DEFAULT_MAP4_RADIUS, map4_fingerprint, map4_shingles, mhfp_fingerprint, shingle_id,
};

pub use pairs::{
    DEFAULT_MAX_DISTANCE, atom_pair_fingerprint, atom_pair_types, pair_id,
    topological_torsion_fingerprint, torsion_id,
};

pub use circular::{
    CircularFingerprint, Environment, circular_fingerprint, circular_fingerprint_with_invariants,
    feature_invariants, standard_invariants,
};

/// An arbitrary odd constant, so that a zero seed and a zero first value do not
/// leave the state at zero.
const SEED_OFFSET: u64 = 0x9e37_79b9_7f4a_7c15;

/// The finalising mix of SplitMix64.
///
/// One multiply-shift-xor round per word: enough to spread a small integer
/// across all sixty-four bits, which matters because folding a fingerprint into
/// a bit vector takes only the low ones.
pub const fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// A hash whose output is part of this crate's format rather than an
/// implementation detail.
///
/// Each word is folded in by [`mix64`], and the result is whatever the state
/// has become. Changing any of that invalidates every fingerprint anyone has
/// stored, so it is versioned by the crate's own version and must not be
/// changed casually. It is not a general-purpose hash and does not resist an
/// adversary; it guarantees that the same molecule gives the
/// same number on another machine, in another build, next year.
///
/// Width is not part of what is hashed -- writing `1u8` and writing `1u32` are
/// the same -- so a caller whose fields could otherwise be confused for one
/// another should pack them into one word rather than rely on their types.
#[derive(Debug, Clone, Copy)]
pub struct StableHasher {
    state: u64,
}

impl StableHasher {
    /// A hasher seeded with `seed`, which callers use to keep the identifier
    /// spaces of different fingerprint kinds apart.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed ^ SEED_OFFSET,
        }
    }

    /// Folds one word in.
    pub fn write_u64(&mut self, value: u64) {
        self.state = mix64(self.state ^ value);
    }

    /// Folds four bytes in.
    pub fn write_u32(&mut self, value: u32) {
        self.write_u64(value as u64);
    }

    /// Folds one byte in.
    pub fn write_u8(&mut self, value: u8) {
        self.write_u64(value as u64);
    }

    /// The 32 bits of the state that are best distributed.
    pub fn finish_u32(self) -> u32 {
        (self.state >> 32) as u32
    }

    /// The whole state.
    pub fn finish_u64(self) -> u64 {
        self.state
    }
}

/// A multiset of fingerprint identifiers.
///
/// Stored sorted by identifier and deduplicated, so comparing two fingerprints
/// is a merge rather than a lookup, and so the representation of a molecule is
/// unique for a given set of identifiers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Fingerprint {
    entries: Vec<(u32, u32)>,
}

impl Fingerprint {
    /// An empty fingerprint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts how often each identifier appears.
    pub fn from_ids<I: IntoIterator<Item = u32>>(ids: I) -> Self {
        // Sorted and run-length encoded rather than counted into a map: a
        // molecule contributes a hundred or so identifiers, and sorting that
        // many machine words costs less than hashing them.
        let mut sorted: Vec<u32> = ids.into_iter().collect();
        sorted.sort_unstable();
        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(sorted.len());
        for id in sorted {
            match entries.last_mut() {
                Some(last) if last.0 == id => last.1 += 1,
                _ => entries.push((id, 1)),
            }
        }
        Self { entries }
    }

    /// The identifiers and their counts, in identifier order.
    pub fn entries(&self) -> &[(u32, u32)] {
        &self.entries
    }

    /// The identifiers, in order, without their counts.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.iter().map(|&(id, _)| id)
    }

    /// How often `id` occurs, zero if it does not.
    pub fn count(&self, id: u32) -> u32 {
        match self.entries.binary_search_by_key(&id, |&(key, _)| key) {
            Ok(position) => self.entries[position].1,
            Err(_) => 0,
        }
    }

    /// How many distinct identifiers the fingerprint holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every occurrence of every identifier.
    pub fn total_count(&self) -> u64 {
        self.entries.iter().map(|&(_, count)| count as u64).sum()
    }

    /// Folds the identifiers into a bit vector of `length` bits.
    ///
    /// Two identifiers can land on the same bit, and the more that do the less
    /// a comparison means -- which is why this is not done until it has to be.
    pub fn fold(&self, length: usize) -> BitFingerprint {
        let mut folded = BitFingerprint::new(length);
        for &(id, _) in &self.entries {
            folded.set(id as usize % length.max(1));
        }
        folded
    }

    /// The Tanimoto coefficient over the identifiers present, ignoring counts.
    ///
    /// Two empty fingerprints are called identical: they agree on everything
    /// they say, which is nothing.
    pub fn tanimoto(&self, other: &Self) -> f64 {
        let (shared, union) = self.overlap(other);
        if union == 0 {
            return 1.0;
        }
        shared as f64 / union as f64
    }

    /// The Dice coefficient over the identifiers present.
    pub fn dice(&self, other: &Self) -> f64 {
        let (shared, _) = self.overlap(other);
        let both = self.len() + other.len();
        if both == 0 {
            return 1.0;
        }
        2.0 * shared as f64 / both as f64
    }

    /// The cosine (Ochiai) coefficient over the identifiers present.
    pub fn cosine(&self, other: &Self) -> f64 {
        let (shared, _) = self.overlap(other);
        if self.is_empty() && other.is_empty() {
            return 1.0;
        }
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }
        shared as f64 / ((self.len() as f64) * (other.len() as f64)).sqrt()
    }

    /// The Tversky coefficient, which weights what each side has and the other
    /// lacks. `alpha == beta == 1` is Tanimoto and `alpha == beta == 0.5` is
    /// Dice; an asymmetric pair asks how much of one molecule the other
    /// contains, which is what a substructure-like search wants.
    pub fn tversky(&self, other: &Self, alpha: f64, beta: f64) -> f64 {
        let (shared, _) = self.overlap(other);
        let only_self = self.len() - shared;
        let only_other = other.len() - shared;
        let denominator = shared as f64 + alpha * only_self as f64 + beta * only_other as f64;
        if denominator == 0.0 {
            return 1.0;
        }
        shared as f64 / denominator
    }

    /// The count-weighted Tanimoto (MinMax), which distinguishes a molecule
    /// with one benzene ring from one with four.
    pub fn count_tanimoto(&self, other: &Self) -> f64 {
        let mut minimum = 0u64;
        let mut maximum = 0u64;
        let mut left = self.entries.iter().peekable();
        let mut right = other.entries.iter().peekable();
        loop {
            match (left.peek(), right.peek()) {
                (Some(&&(id_a, count_a)), Some(&&(id_b, count_b))) => {
                    if id_a == id_b {
                        minimum += count_a.min(count_b) as u64;
                        maximum += count_a.max(count_b) as u64;
                        left.next();
                        right.next();
                    } else if id_a < id_b {
                        maximum += count_a as u64;
                        left.next();
                    } else {
                        maximum += count_b as u64;
                        right.next();
                    }
                }
                (Some(&&(_, count_a)), None) => {
                    maximum += count_a as u64;
                    left.next();
                }
                (None, Some(&&(_, count_b))) => {
                    maximum += count_b as u64;
                    right.next();
                }
                (None, None) => break,
            }
        }
        if maximum == 0 {
            return 1.0;
        }
        minimum as f64 / maximum as f64
    }

    /// How many identifiers the two share, and how many they have between them.
    fn overlap(&self, other: &Self) -> (usize, usize) {
        let mut shared = 0usize;
        let mut left = 0usize;
        let mut right = 0usize;
        let mut union = 0usize;
        while left < self.entries.len() && right < other.entries.len() {
            match self.entries[left].0.cmp(&other.entries[right].0) {
                std::cmp::Ordering::Equal => {
                    shared += 1;
                    left += 1;
                    right += 1;
                }
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
            }
            union += 1;
        }
        union += self.entries.len() - left + other.entries.len() - right;
        (shared, union)
    }
}

/// A fingerprint folded onto a fixed number of bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitFingerprint {
    words: Vec<u64>,
    length: usize,
}

impl BitFingerprint {
    /// A vector of `length` bits, all clear.
    pub fn new(length: usize) -> Self {
        Self {
            words: vec![0; length.div_ceil(64)],
            length,
        }
    }

    /// How many bits the vector holds.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Sets one bit.
    pub fn set(&mut self, bit: usize) {
        if bit < self.length {
            self.words[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    /// Whether one bit is set.
    pub fn get(&self, bit: usize) -> bool {
        bit < self.length && self.words[bit / 64] & (1u64 << (bit % 64)) != 0
    }

    /// How many bits are set.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// Every set bit, in order.
    pub fn set_bits(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.length).filter(move |&bit| self.get(bit))
    }

    /// The Tanimoto coefficient. Two all-zero vectors are called identical.
    pub fn tanimoto(&self, other: &Self) -> f64 {
        let union = self.union_with(other);
        if union == 0 {
            return 1.0;
        }
        self.intersection_with(other) as f64 / union as f64
    }

    /// The Dice coefficient.
    pub fn dice(&self, other: &Self) -> f64 {
        let total = self.count_ones() + other.count_ones();
        if total == 0 {
            return 1.0;
        }
        2.0 * self.intersection_with(other) as f64 / total as f64
    }

    /// The cosine (Ochiai) coefficient.
    pub fn cosine(&self, other: &Self) -> f64 {
        let (left, right) = (self.count_ones(), other.count_ones());
        if left == 0 && right == 0 {
            return 1.0;
        }
        if left == 0 || right == 0 {
            return 0.0;
        }
        self.intersection_with(other) as f64 / ((left as f64) * (right as f64)).sqrt()
    }

    /// How many bits both have set.
    pub fn intersection_with(&self, other: &Self) -> usize {
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(left, right)| (left & right).count_ones() as usize)
            .sum()
    }

    /// How many bits either has set.
    pub fn union_with(&self, other: &Self) -> usize {
        let mut total = 0usize;
        let shared = self.words.len().min(other.words.len());
        for index in 0..shared {
            total += (self.words[index] | other.words[index]).count_ones() as usize;
        }
        for word in self.words.iter().skip(shared) {
            total += word.count_ones() as usize;
        }
        for word in other.words.iter().skip(shared) {
            total += word.count_ones() as usize;
        }
        total
    }
}
