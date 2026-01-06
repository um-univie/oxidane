//! Losses that skip the measurements nobody made.
//!
//! A chemical dataset is mostly holes. Twelve toxicity assays and a few
//! thousand molecules give a target matrix where each molecule was measured in
//! two or three of the twelve, and a loss that reads the other nine as zeros
//! teaches the model that almost everything is inactive -- confidently, and in
//! a way that shows up as an excellent training curve.
//!
//! Every loss here therefore takes a mask, averages over the entries the mask
//! keeps, and returns zero rather than a NaN when the mask keeps nothing. A
//! NaN loss is worse than a useless one: one backward pass through it and
//! every weight in the model is NaN too.

use candle_core::{Result, Tensor};

/// Mean squared error over the measured entries.
pub fn masked_mse(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let error = prediction.sub(target)?;
    masked_mean(&error.sqr()?, mask)
}

/// Mean absolute error over the measured entries.
///
/// What to reach for when the targets have outliers that a squared error would
/// let dominate -- a solubility set with a few insoluble compounds recorded at
/// the limit of the assay, say.
pub fn masked_l1(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    masked_mean(&prediction.sub(target)?.abs()?, mask)
}

/// Binary cross entropy, taking logits rather than probabilities.
///
/// Taking logits is not a convenience. `log(sigmoid(x))` overflows for a
/// confident negative prediction and `log(1 - sigmoid(x))` for a confident
/// positive one, which is exactly where a trained classifier spends its time;
/// the identity used here -- `max(x, 0) - x*y + log(1 + exp(-|x|))` -- is the
/// same function computed in a way that cannot.
pub fn masked_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let zero = logits.zeros_like()?;
    let stable = logits
        .maximum(&zero)?
        .sub(&logits.mul(target)?)?
        .add(&logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
    masked_mean(&stable, mask)
}

/// Cross entropy against a one-hot target, taking logits.
///
/// `logits` and `target` are `[rows, classes]`; `mask` is `[rows, 1]`, since a
/// row is either labelled or it is not.
pub fn masked_cross_entropy(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let log_probabilities = candle_nn::ops::log_softmax(logits, 1)?;
    let per_row = log_probabilities.mul(target)?.sum(1)?.neg()?.unsqueeze(1)?;
    masked_mean(&per_row, mask)
}

/// The mean of `values` over the entries `mask` keeps, or zero if it keeps
/// none.
///
/// `mask` broadcasts against `values`, so a per-row mask can weight a per-class
/// tensor.
pub fn masked_mean(values: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask = mask.to_dtype(values.dtype())?;
    let kept = values.broadcast_mul(&mask)?.sum_all()?;
    let count = mask.sum_all()?.broadcast_as(kept.shape())?;
    // Dividing by a floored count rather than branching on it: an empty batch
    // is then a zero loss with a zero gradient, which a training loop can carry
    // on from, instead of a NaN that spreads to every weight on the first step.
    let count = count.maximum(&Tensor::new(1.0f32, values.device())?.to_dtype(values.dtype())?)?;
    kept.broadcast_div(&count)
}

/// A mask tensor from the flags a [`crate::ml::dataset::Batch`] carries.
pub fn mask_from_flags(
    flags: &[bool],
    shape: (usize, usize),
    device: &candle_core::Device,
) -> Result<Tensor> {
    let values: Vec<f32> = flags
        .iter()
        .map(|&measured| if measured { 1.0 } else { 0.0 })
        .collect();
    Tensor::from_slice(&values, shape, device)
}
