//! Learning a representation with no labels at all.
//!
//! A graph autoencoder asks the model to reconstruct the molecule's bonds from
//! its own atom vectors. Two atoms that are bonded should have vectors whose
//! inner product is large, and two that are not should not, so the decoder is
//! just `sigmoid(z_i . z_j)` and the loss is a binary cross entropy over pairs.
//! There is nothing to label, so this trains on every structure available --
//! which for chemistry is millions of molecules against the thousands that
//! carry a measurement.
//!
//! The variational form ([`GraphAutoencoder::variational`]) encodes to a
//! distribution rather than a point and pays a KL term for straying from a
//! standard normal. What that buys is a latent space where the region between
//! two molecules is itself molecule-shaped, which is what interpolation and
//! sampling need; the plain form gives better reconstruction and a space with
//! holes in it.
//!
//! # Which pairs
//!
//! Every bond is a positive example. The negatives cannot be every non-bond --
//! there are `n²` of them against `n` bonds, and a model that answers "not
//! bonded" to everything would score well. As many negatives as positives are
//! sampled instead, uniformly from pairs within the same molecule; sampling
//! across molecules would be free marks, since two atoms in different
//! molecules are trivially not bonded.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

use crate::ml::batch::GraphBatch;
use crate::ml::nn::mpnn::{Encoder, EncoderConfig, EncoderKind};
use crate::ml::nn::tensors::GraphTensors;

/// How much latent space, and whether it is a distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoencoderConfig {
    pub encoder: EncoderConfig,
    pub kind: EncoderKind,
    /// The width of an atom's latent vector.
    pub latent: usize,
    /// Encode to a mean and a variance rather than a point.
    pub variational: bool,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            // Atoms rather than molecules are what gets embedded here, and an
            // atom is a much smaller thing than a molecule.
            kind: EncoderKind::Gine,
            latent: 32,
            variational: false,
        }
    }
}

/// An encoder and the projection into latent space.
///
/// ```
/// # fn main() -> candle_core::Result<()> {
/// use candle_core::{DType, Device};
/// use candle_nn::{VarBuilder, VarMap};
/// use oxidane::prelude::*;
/// use oxidane::molecule::featurize::graph_features;
/// use oxidane::ml::batch::GraphBatch;
/// use oxidane::ml::nn::autoencoder::{AutoencoderConfig, GraphAutoencoder};
/// use oxidane::ml::nn::{EncoderConfig, GraphTensors};
///
/// let device = Device::Cpu;
/// let varmap = VarMap::new();
/// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
/// let config = AutoencoderConfig {
///     encoder: EncoderConfig { hidden: 32, depth: 2, ..Default::default() },
///     latent: 8,
///     ..Default::default()
/// };
/// let autoencoder = GraphAutoencoder::new(config, vb)?;
///
/// let batch = GraphBatch::collate(&[graph_features(
///     &MolecularGraph::from_smiles_single("c1ccccc1O").unwrap(),
/// )]);
/// let graph = GraphTensors::new(&batch, &device)?;
///
/// let latent = autoencoder.encode(&graph, false)?;
/// assert_eq!(latent.mean.dims(), &[batch.atom_count, 8]);
///
/// // The loss is finite before a single step of training.
/// let loss = autoencoder.reconstruction_loss(&latent.mean, &batch, &device, 1)?;
/// assert!(loss.to_scalar::<f32>()?.is_finite());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GraphAutoencoder {
    encoder: Encoder,
    mean: Linear,
    log_variance: Option<Linear>,
    config: AutoencoderConfig,
}

impl GraphAutoencoder {
    pub fn new(config: AutoencoderConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(config.kind, config.encoder, vb.pp("encoder"))?;
        let width = encoder.width();
        Ok(Self {
            mean: linear(width, config.latent, vb.pp("mean"))?,
            log_variance: if config.variational {
                Some(linear(width, config.latent, vb.pp("log_variance"))?)
            } else {
                None
            },
            encoder,
            config,
        })
    }

    /// The variational form of [`GraphAutoencoder::new`].
    pub fn variational(mut config: AutoencoderConfig, vb: VarBuilder) -> Result<Self> {
        config.variational = true;
        Self::new(config, vb)
    }

    /// One latent vector per atom, and its spread if this is variational.
    pub fn encode(&self, graph: &GraphTensors, train: bool) -> Result<Latent> {
        let atoms = self.encoder.forward(graph, train)?;
        Ok(Latent {
            mean: self.mean.forward(&atoms)?,
            log_variance: match &self.log_variance {
                Some(layer) => Some(layer.forward(&atoms)?),
                None => None,
            },
        })
    }

    /// [`sample`], for a caller that has the model to hand.
    pub fn sample(&self, latent: &Latent, train: bool) -> Result<Tensor> {
        sample(latent, train)
    }

    /// Binary cross entropy over bonded and non-bonded pairs.
    ///
    /// `negatives_per_positive` says how many unbonded pairs are drawn for
    /// each bond; one is the usual choice and what keeps the two classes
    /// balanced.
    pub fn reconstruction_loss(
        &self,
        latent: &Tensor,
        batch: &GraphBatch,
        device: &candle_core::Device,
        negatives_per_positive: usize,
    ) -> Result<Tensor> {
        let (positives, negatives) = pair_samples(batch, negatives_per_positive);
        if positives.is_empty() && negatives.is_empty() {
            return Tensor::new(0.0f32, device)?.to_dtype(latent.dtype());
        }

        let mut left = Vec::with_capacity(positives.len() + negatives.len());
        let mut right = Vec::with_capacity(left.capacity());
        let mut labels = Vec::with_capacity(left.capacity());
        for &(a, b) in &positives {
            left.push(a);
            right.push(b);
            labels.push(1.0f32);
        }
        for &(a, b) in &negatives {
            left.push(a);
            right.push(b);
            labels.push(0.0f32);
        }

        let count = labels.len();
        let left = Tensor::from_slice(&left, count, device)?;
        let right = Tensor::from_slice(&right, count, device)?;
        let logits = latent
            .index_select(&left, 0)?
            .mul(&latent.index_select(&right, 0)?)?
            .sum(1)?
            .unsqueeze(1)?;
        let labels = Tensor::from_slice(&labels, (count, 1), device)?.to_dtype(logits.dtype())?;
        let ones = Tensor::ones((count, 1), logits.dtype(), device)?;

        crate::ml::nn::loss::masked_bce_with_logits(&logits, &labels, &ones)
    }

    /// [`kl_divergence`], for a caller that has the model to hand.
    pub fn kl_divergence(&self, latent: &Latent) -> Result<Tensor> {
        kl_divergence(latent)
    }

    /// How likely the model thinks each of `pairs` is to be a bond.
    ///
    /// The decoder, run forwards -- for looking at what the latent space
    /// learned rather than for training on it.
    pub fn bond_probabilities(
        &self,
        latent: &Tensor,
        pairs: &[(usize, usize)],
    ) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let device = latent.device();
        let left: Vec<u32> = pairs.iter().map(|&(a, _)| a as u32).collect();
        let right: Vec<u32> = pairs.iter().map(|&(_, b)| b as u32).collect();
        let left = Tensor::from_slice(&left, pairs.len(), device)?;
        let right = Tensor::from_slice(&right, pairs.len(), device)?;
        let logits = latent
            .index_select(&left, 0)?
            .mul(&latent.index_select(&right, 0)?)?
            .sum(1)?;
        candle_nn::ops::sigmoid(&logits)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()
    }

    pub fn config(&self) -> &AutoencoderConfig {
        &self.config
    }
}

/// What the encoder produced.
#[derive(Debug, Clone)]
pub struct Latent {
    /// `[atom_count, latent]`.
    pub mean: Tensor,
    /// The same shape, or `None` for a plain autoencoder.
    pub log_variance: Option<Tensor>,
}

/// A sample from an encoded distribution, or the mean if there is none.
///
/// The noise is added to the mean rather than drawn from the distribution
/// directly -- the reparameterisation trick -- so that the gradient reaches the
/// encoder. Sampling properly would put a random number between the loss and
/// the weights, and nothing would learn.
///
/// Evaluation takes the mean: a reconstruction is supposed to be a reading of
/// what went in, not a draw from the neighbourhood of it.
pub fn sample(latent: &Latent, train: bool) -> Result<Tensor> {
    let (Some(log_variance), true) = (&latent.log_variance, train) else {
        return Ok(latent.mean.clone());
    };
    let deviation = log_variance.affine(0.5, 0.0)?.exp()?;
    let noise = latent.mean.randn_like(0.0, 1.0)?;
    latent.mean.add(&deviation.mul(&noise)?)
}

/// The KL divergence of an encoded distribution from a standard normal,
/// averaged over rows.
///
/// Zero when the latent carries no spread, so a training loop can add it
/// unconditionally.
pub fn kl_divergence(latent: &Latent) -> Result<Tensor> {
    let Some(log_variance) = &latent.log_variance else {
        return Tensor::new(0.0f32, latent.mean.device())?.to_dtype(latent.mean.dtype());
    };
    // -0.5 * sum(1 + log s^2 - mu^2 - s^2), averaged over rows.
    let term = log_variance
        .affine(1.0, 1.0)?
        .sub(&latent.mean.sqr()?)?
        .sub(&log_variance.exp()?)?;
    term.sum_all()?
        .affine(-0.5 / latent.mean.dim(0)?.max(1) as f64, 0.0)
}

/// A pair of atom rows, as index-tensor material.
type Pair = (u32, u32);

/// Bonded pairs, and as many unbonded ones drawn from inside the same
/// molecules.
///
/// Deterministic given the batch and the count, because the sampler is seeded
/// from the batch's own shape rather than from a clock: two runs of the same
/// training script then see the same negatives, and a difference between them
/// is a difference in the model.
fn pair_samples(batch: &GraphBatch, negatives_per_positive: usize) -> (Vec<Pair>, Vec<Pair>) {
    let mut positives = Vec::with_capacity(batch.edge_count / 2);
    let mut bonded = std::collections::HashSet::with_capacity(batch.edge_count);
    for edge in 0..batch.edge_count {
        let (from, to) = batch.edge(edge);
        let key = (from.min(to) as u32, from.max(to) as u32);
        if bonded.insert(key) {
            positives.push(key);
        }
    }

    let wanted = positives.len() * negatives_per_positive;
    let mut negatives = Vec::with_capacity(wanted);
    if wanted == 0 {
        return (positives, negatives);
    }

    let mut state = (batch.atom_count as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(batch.edge_count as u64);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    // Molecules with at least two atoms, so that a pair exists to draw.
    let candidates: Vec<usize> = (0..batch.graph_count)
        .filter(|&graph| batch.graph_atom_count(graph) >= 2)
        .collect();
    if candidates.is_empty() {
        return (positives, negatives);
    }

    // Rejection sampling, with a cap so that a fully connected molecule -- one
    // where every pair is a bond and no negative exists -- cannot spin here.
    let mut attempts = 0;
    while negatives.len() < wanted && attempts < wanted * 8 {
        attempts += 1;
        let graph = candidates[(next() % candidates.len() as u64) as usize];
        let rows = batch.graph_rows(graph);
        let width = rows.len() as u64;
        let a = rows.start as u32 + (next() % width) as u32;
        let b = rows.start as u32 + (next() % width) as u32;
        if a == b {
            continue;
        }
        let key = (a.min(b), a.max(b));
        if bonded.contains(&key) {
            continue;
        }
        negatives.push(key);
    }

    (positives, negatives)
}
