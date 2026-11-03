//! Getting a molecule into a model, and a dataset into a training loop.
//!
//! The chemistry is elsewhere: [`crate::molecule::featurize`] turns one
//! molecule into matrices, [`crate::molecule::scaffold`] says which molecules
//! belong to the same series. What is here is the shape of the data around
//! them -- how many molecules become one batch, which of them a model is
//! allowed to see, and what the numbers are scaled by.
//!
//! Nothing in this module needs a tensor library, and none of it is behind a
//! feature -- [`tokenize`] included, which is the alphabet a sequence decoder
//! writes molecules in rather than any part of the network itself. A dependent that trains its models in Python can read
//! [`batch::GraphBatch`] and hand the flat vectors straight to `torch`; the
//! `ml` feature adds the layers that would otherwise be the Python half.

pub mod batch;
pub mod dataset;
#[cfg(feature = "ml")]
pub mod nn;
pub mod normalize;
pub mod split;
pub mod tokenize;

/// `shuffle`, for a caller outside this crate that wants the same order.
pub fn shuffled(order: &mut [usize], seed: u64) {
    shuffle(order, seed);
}

/// A Fisher--Yates shuffle over a seeded SplitMix64 stream.
///
/// Not [`rand`]: a split and an epoch order are both part of a
/// published result, and the stream a general-purpose generator produces is
/// free to change between versions. This one cannot.
pub(crate) fn shuffle(order: &mut [usize], seed: u64) {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for position in (1..order.len()).rev() {
        let other = (next() % (position as u64 + 1)) as usize;
        order.swap(position, other);
    }
}
