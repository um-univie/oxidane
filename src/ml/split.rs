//! Deciding which molecules a model gets to see.
//!
//! A random split of a chemical dataset flatters every model that runs on it.
//! Datasets are built series by series -- twenty analogues of one lead, thirty
//! of the next -- so a random split puts near-copies of the test molecules in
//! the training set, and the number that comes out is the model's ability to
//! interpolate within a series rather than to say anything about chemistry it
//! has not seen. [`scaffold_split`] is the usual correction: molecules sharing
//! a Bemis--Murcko scaffold go to the same side, so the test set is made of
//! skeletons the model has never met.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::scaffold::scaffold_key;
//! use oxidane::ml::split::{Fractions, scaffold_split};
//!
//! let smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "C1CCCCC1", "CCCC"];
//! let keys: Vec<String> = smiles
//!     .iter()
//!     .map(|s| scaffold_key(&MolecularGraph::from_smiles_single(s).unwrap()))
//!     .collect();
//!
//! let split = scaffold_split(&keys, Fractions::new(0.6, 0.2, 0.2));
//! // The two toluenes share a benzene scaffold, so they are never separated.
//! let side = |index: usize| split.side_of(index);
//! assert_eq!(side(0), side(1));
//! ```

use crate::ml::shuffle;
use crate::molecule::fingerprint::StableHasher;
use rustc_hash::FxHashMap;

/// Which of the three sets a molecule landed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Train,
    Valid,
    Test,
}

/// How much of the dataset each side gets.
///
/// The three are normalised, so `Fractions::new(8.0, 1.0, 1.0)` and
/// `Fractions::new(0.8, 0.1, 0.1)` are the same request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fractions {
    pub train: f64,
    pub valid: f64,
    pub test: f64,
}

impl Fractions {
    pub fn new(train: f64, valid: f64, test: f64) -> Self {
        Self { train, valid, test }
    }

    /// The three sizes for a dataset of `total` items, adding up to `total`.
    fn sizes(self, total: usize) -> (usize, usize, usize) {
        let sum = self.train + self.valid + self.test;
        if sum <= 0.0 || total == 0 {
            return (total, 0, 0);
        }
        let train = ((self.train / sum) * total as f64).round() as usize;
        let valid = ((self.valid / sum) * total as f64).round() as usize;
        let train = train.min(total);
        let valid = valid.min(total - train);
        (train, valid, total - train - valid)
    }
}

impl Default for Fractions {
    /// The split everybody uses: 80 / 10 / 10.
    fn default() -> Self {
        Self::new(0.8, 0.1, 0.1)
    }
}

/// Three disjoint lists of indices into the dataset, each ascending.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Split {
    pub train: Vec<usize>,
    pub valid: Vec<usize>,
    pub test: Vec<usize>,
}

impl Split {
    /// Which side an index is on. Panics if the index is in none of them,
    /// which cannot happen for a split this module produced.
    pub fn side_of(&self, index: usize) -> Side {
        if self.train.binary_search(&index).is_ok() {
            Side::Train
        } else if self.valid.binary_search(&index).is_ok() {
            Side::Valid
        } else {
            debug_assert!(self.test.binary_search(&index).is_ok());
            Side::Test
        }
    }

    pub fn len(&self) -> usize {
        self.train.len() + self.valid.len() + self.test.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn from_sides(sides: &[Side]) -> Self {
        let mut split = Split::default();
        for (index, side) in sides.iter().enumerate() {
            match side {
                Side::Train => split.train.push(index),
                Side::Valid => split.valid.push(index),
                Side::Test => split.test.push(index),
            }
        }
        split
    }
}

/// A split that ignores the molecules entirely.
///
/// Worth having as the baseline the scaffold split is compared against: the
/// gap between the two is the amount of the score that was series memorisation.
/// The shuffle is a seeded permutation, so the same seed gives the same split
/// on any machine.
pub fn random_split(count: usize, fractions: Fractions, seed: u64) -> Split {
    let mut order: Vec<usize> = (0..count).collect();
    shuffle(&mut order, seed);

    let (train, valid, _) = fractions.sizes(count);
    let mut sides = vec![Side::Test; count];
    for (rank, &index) in order.iter().enumerate() {
        sides[index] = if rank < train {
            Side::Train
        } else if rank < train + valid {
            Side::Valid
        } else {
            Side::Test
        };
    }
    Split::from_sides(&sides)
}

/// Molecules sharing a scaffold stay together, biggest group first.
///
/// `keys` is one scaffold key per molecule -- [`crate::molecule::scaffold::scaffold_key`]
/// is what produces them. Groups are placed largest first into whichever side
/// is furthest below its quota, which keeps the sizes close to the fractions
/// asked for while never splitting a group.
///
/// This is deterministic and takes no seed. Two runs over the same dataset give
/// the same split, which is what makes a published number checkable.
///
/// # The empty key
///
/// Acyclic molecules have no scaffold and all share the empty key, so they
/// would otherwise form one enormous group that lands entirely on one side.
/// Each is given a group of its own instead.
pub fn scaffold_split(keys: &[String], fractions: Fractions) -> Split {
    let mut groups: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    let mut loners: Vec<Vec<usize>> = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        if key.is_empty() {
            loners.push(vec![index]);
        } else {
            groups.entry(key.as_str()).or_default().push(index);
        }
    }

    let mut ordered: Vec<Vec<usize>> = groups.into_values().collect();
    ordered.append(&mut loners);
    // Largest first, and by first member where sizes tie, so the hash map's
    // iteration order cannot reach the result.
    ordered.sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));

    place_groups(&ordered, keys.len(), fractions)
}

/// [`scaffold_split`] with the group order shuffled, for repeating a
/// benchmark over several splits.
///
/// A deterministic split gives one number and no idea how much of it was luck.
/// Published scaffold results are means over three or more splits, and this is
/// how those splits differ from each other: the groups too large to fit in a
/// validation or test set are put in the training set first -- shuffled among
/// themselves -- and the rest are shuffled and placed greedily. Chemprop calls
/// this the balanced scaffold split, and it is what its reported numbers use.
///
/// Groups are still never split, so the guarantee [`scaffold_split`] makes
/// holds here too.
pub fn scaffold_split_balanced(keys: &[String], fractions: Fractions, seed: u64) -> Split {
    let mut groups: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
    let mut ordered: Vec<Vec<usize>> = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        if key.is_empty() {
            ordered.push(vec![index]);
        } else {
            groups.entry(key.as_str()).or_default().push(index);
        }
    }
    let mut from_map: Vec<Vec<usize>> = groups.into_values().collect();
    // The hash map's order is not a property of the data, so sort before
    // shuffling: the seed should be the only thing that decides the order.
    from_map.sort_unstable_by(|a, b| a[0].cmp(&b[0]));
    ordered.append(&mut from_map);
    ordered.sort_unstable_by(|a, b| a[0].cmp(&b[0]));

    let (_, valid_quota, test_quota) = fractions.sizes(keys.len());
    let threshold = valid_quota.max(test_quota) / 2;
    let (mut large, mut small): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
        ordered.into_iter().partition(|group| group.len() > threshold);

    shuffle_groups(&mut large, seed);
    shuffle_groups(&mut small, seed.wrapping_add(1));
    large.append(&mut small);

    place_groups(&large, keys.len(), fractions)
}

fn shuffle_groups(groups: &mut [Vec<usize>], seed: u64) {
    let mut order: Vec<usize> = (0..groups.len()).collect();
    shuffle(&mut order, seed);
    let mut taken: Vec<Option<Vec<usize>>> = groups.iter_mut().map(|g| Some(std::mem::take(g))).collect();
    for (slot, &from) in order.iter().enumerate() {
        groups[slot] = taken[from].take().unwrap_or_default();
    }
}

/// [`scaffold_split`] over groups the caller has worked out for itself.
///
/// Anything that names a series will do: a cluster label, an assay plate, a
/// patent. The rule is the same -- a group is never split -- and so is the
/// reason for it.
pub fn grouped_split(groups: &[Vec<usize>], count: usize, fractions: Fractions) -> Split {
    let mut ordered = groups.to_vec();
    ordered.sort_unstable_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| a.first().cmp(&b.first()))
    });
    place_groups(&ordered, count, fractions)
}

fn place_groups(groups: &[Vec<usize>], count: usize, fractions: Fractions) -> Split {
    let (train_quota, valid_quota, test_quota) = fractions.sizes(count);
    let quotas = [train_quota, valid_quota, test_quota];
    let sides = [Side::Train, Side::Valid, Side::Test];
    let mut filled = [0usize; 3];

    let mut assigned = vec![Side::Train; count];
    for group in groups {
        // Whichever side has the most room left. Filling train to its quota
        // before touching the others -- the other obvious rule -- puts every
        // large group in train and leaves the test set made of singletons.
        let choice = (0..3)
            .max_by(|&a, &b| {
                let room = |side: usize| quotas[side] as isize - filled[side] as isize;
                room(a).cmp(&room(b)).then_with(|| b.cmp(&a))
            })
            .unwrap_or(0);
        for &index in group {
            assigned[index] = sides[choice];
        }
        filled[choice] += group.len();
    }

    Split::from_sides(&assigned)
}


/// Keeps [`hashed_side`]'s hash space apart from the fingerprints'.
const SPLIT_SEED: u64 = 0x5350_4C49_5400_0001;

/// A stable side for a key, for the split that has no dataset to count.
///
/// Hashing the scaffold key and cutting the hash space by the fractions gives
/// a molecule the same side whatever else is in the dataset -- so a model
/// trained today and one trained on next month's larger dataset have test sets
/// that agree rather than overlap. The sizes only approach the fractions as
/// the dataset grows, which is the price.
pub fn hashed_side(key: &str, fractions: Fractions) -> Side {
    let mut hasher = StableHasher::new(SPLIT_SEED);
    let bytes = key.as_bytes();
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        hasher.write_u64(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])));
    }
    for &byte in chunks.remainder() {
        hasher.write_u8(byte);
    }
    // The top 53 bits, which is every bit an `f64` in `[0, 1)` can hold.
    let position = (hasher.finish_u64() >> 11) as f64 / (1u64 << 53) as f64;

    let sum = fractions.train + fractions.valid + fractions.test;
    if sum <= 0.0 {
        return Side::Train;
    }
    let train = fractions.train / sum;
    let valid = fractions.valid / sum;
    if position < train {
        Side::Train
    } else if position < train + valid {
        Side::Valid
    } else {
        Side::Test
    }
}
