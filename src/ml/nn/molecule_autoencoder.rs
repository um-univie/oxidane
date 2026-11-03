//! A molecule in, one vector out, a molecule back.
//!
//! [`crate::ml::nn::autoencoder`] embeds *atoms* and reconstructs the bonds
//! between them, which needs the molecule in hand to say which atoms are being
//! asked about. This is the other thing the word autoencoder means: one vector
//! stands for the whole molecule, and the molecule can be rebuilt from that
//! vector alone. That is what interpolation, sampling, and optimising a
//! property in latent space all need.
//!
//! ```text
//! MolecularGraph -> graph_features -> GraphBatch -> GraphTensors
//!                                                        |
//!                                                     Encoder
//!                                                        |
//!                                              pool over the atoms
//!                                                        |
//!                                             mean (and log variance)
//!                                                        |
//!                                                  z  [1, latent]
//!                                                        |
//!                                                 SmilesDecoder
//!                                                        |
//!                                          "c1ccccc1O" -> MolecularGraph
//! ```
//!
//! The encoder reads the graph and the decoder writes a string (see
//! [`crate::ml::nn::decoder`] for why the return trip goes through SMILES), so
//! **the round trip is not guaranteed to close.** A decoder can emit something that
//! does not parse, or that parses into a different molecule. The measure of
//! this model is therefore not its loss but the share of molecules that come
//! back identical, which [`MoleculeAutoencoder::decode`] is there to count.
//!
//! # Variational, and why it matters here
//!
//! [`MoleculeAutoencoder::variational`] encodes to a distribution and pays a KL
//! term for straying from a standard normal. Without it the latent space is a
//! scatter of points with nothing in between: reconstruction is better, and
//! every vector that is not the encoding of a real molecule decodes to
//! nonsense. With it, the space between two molecules is itself
//! molecule-shaped, which is the property that makes a latent space worth
//! having.
//!
//! The price is posterior collapse -- the decoder learns to ignore the latent,
//! the KL term goes to zero, and the model becomes a language model that
//! writes plausible molecules unrelated to its input. Two things here work
//! against it: the latent is fed at every decoder step rather than only at the
//! start, and [`MoleculeLoss::total`] takes the KL weight as an argument so a
//! training loop can raise it from zero over the first epochs instead of
//! paying it in full while the reconstruction is still random.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

use crate::ml::nn::autoencoder::Latent;
use crate::ml::nn::decoder::{DecoderConfig, SmilesDecoder};
use crate::ml::nn::mpnn::{Encoder, EncoderConfig, EncoderKind};
use crate::ml::nn::tensors::{GraphTensors, Pooling};
use crate::ml::tokenize::{SmilesVocabulary, TokenBatch};
use crate::molecule::molecular_graph::MolecularGraph;

/// How much model, and whether the latent is a distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoleculeAutoencoderConfig {
    pub encoder: EncoderConfig,
    pub kind: EncoderKind,
    /// How a molecule's atoms become one vector for it.
    ///
    /// [`Pooling::Mean`] by default. A sum grows with the molecule, so the
    /// latent of a large molecule is far from the origin and far from the
    /// standard normal a variational model is pulled towards -- the size of
    /// the molecule ends up spending the KL budget that its structure needs.
    pub pooling: Pooling,
    /// The width of the molecule's vector.
    pub latent: usize,
    pub decoder: DecoderConfig,
    /// Encode to a mean and a variance rather than a point.
    pub variational: bool,
}

impl Default for MoleculeAutoencoderConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            // The whole molecule is what gets embedded, and directed message
            // passing is the stronger encoder for a whole-molecule readout.
            kind: EncoderKind::Directed,
            pooling: Pooling::Mean,
            latent: 128,
            decoder: DecoderConfig::default(),
            variational: false,
        }
    }
}

/// An encoder, a latent projection, and a decoder that spells molecules.
///
/// ```
/// # fn main() -> candle_core::Result<()> {
/// use candle_core::{DType, Device};
/// use candle_nn::{VarBuilder, VarMap};
/// use oxidane::prelude::*;
/// use oxidane::molecule::featurize::graph_features;
/// use oxidane::ml::batch::GraphBatch;
/// use oxidane::ml::nn::decoder::DecoderConfig;
/// use oxidane::ml::nn::molecule_autoencoder::{MoleculeAutoencoder, MoleculeAutoencoderConfig};
/// use oxidane::ml::nn::{EncoderConfig, GraphTensors};
/// use oxidane::ml::tokenize::{SmilesVocabulary, TokenBatch};
///
/// let device = Device::Cpu;
/// let varmap = VarMap::new();
/// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
///
/// let smiles = ["c1ccccc1O", "CCO"];
/// let vocabulary = SmilesVocabulary::from_corpus(smiles, 1);
/// let config = MoleculeAutoencoderConfig {
///     encoder: EncoderConfig { hidden: 32, depth: 2, ..Default::default() },
///     decoder: DecoderConfig { embedding: 8, hidden: 16, ..Default::default() },
///     latent: 8,
///     ..Default::default()
/// };
/// let model = MoleculeAutoencoder::new(config, vocabulary.len(), vb)?;
///
/// let molecules: Vec<_> = smiles
///     .iter()
///     .map(|s| MolecularGraph::from_smiles_single(s).unwrap())
///     .collect();
/// let batch = GraphBatch::collate(
///     &molecules.iter().map(graph_features).collect::<Vec<_>>(),
/// );
/// let graph = GraphTensors::new(&batch, &device)?;
/// let tokens = TokenBatch::collate(
///     &smiles.iter().map(|s| vocabulary.encode(s)).collect::<Vec<_>>(),
/// );
///
/// // One vector per molecule, not per atom.
/// let latent = model.encode(&graph, false)?;
/// assert_eq!(latent.mean.dims(), &[2, 8]);
///
/// let loss = model.loss(&graph, &tokens, false)?;
/// assert!(loss.total(1.0)?.to_scalar::<f32>()?.is_finite());
///
/// // Untrained, so what comes back is nonsense -- but it comes back.
/// let decoded = model.decode(&latent.mean, &vocabulary, 0.0, 1)?;
/// assert_eq!(decoded.len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MoleculeAutoencoder {
    encoder: Encoder,
    mean: Linear,
    log_variance: Option<Linear>,
    decoder: SmilesDecoder,
    config: MoleculeAutoencoderConfig,
}

impl MoleculeAutoencoder {
    /// `vocabulary` is [`SmilesVocabulary::len`] -- the width of the decoder's
    /// output layer. Weights are only meaningful with the vocabulary they were
    /// trained against, so the two belong in the same directory.
    pub fn new(
        config: MoleculeAutoencoderConfig,
        vocabulary: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let encoder = Encoder::new(config.kind, config.encoder, vb.pp("encoder"))?;
        let width = encoder.width();
        Ok(Self {
            mean: linear(width, config.latent, vb.pp("mean"))?,
            log_variance: if config.variational {
                Some(linear(width, config.latent, vb.pp("log_variance"))?)
            } else {
                None
            },
            decoder: SmilesDecoder::new(
                config.latent,
                vocabulary,
                config.decoder,
                vb.pp("decoder"),
            )?,
            encoder,
            config,
        })
    }

    /// The variational form of [`MoleculeAutoencoder::new`].
    pub fn variational(
        mut config: MoleculeAutoencoderConfig,
        vocabulary: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        config.variational = true;
        Self::new(config, vocabulary, vb)
    }

    pub fn config(&self) -> &MoleculeAutoencoderConfig {
        &self.config
    }

    pub fn decoder(&self) -> &SmilesDecoder {
        &self.decoder
    }

    /// One latent vector per molecule, `[graph_count, latent]`, and its spread
    /// if this is variational.
    pub fn encode(&self, graph: &GraphTensors, train: bool) -> Result<Latent> {
        let atoms = self.encoder.forward(graph, train)?;
        let pooled = graph.pool(&atoms, self.config.pooling)?;
        Ok(Latent {
            mean: self.mean.forward(&pooled)?,
            log_variance: match &self.log_variance {
                Some(layer) => Some(layer.forward(&pooled)?),
                None => None,
            },
        })
    }

    /// [`crate::ml::nn::autoencoder::sample`], for a caller that has the model
    /// to hand.
    pub fn sample(&self, latent: &Latent, train: bool) -> Result<Tensor> {
        crate::ml::nn::autoencoder::sample(latent, train)
    }

    /// The two halves of the objective, for `graph` and the canonical SMILES
    /// of the same molecules in the same order.
    ///
    /// Keeping the two apart rather than returning a sum is what lets a
    /// training loop anneal the KL weight, and what lets a report say which of
    /// the two a run is actually spending its capacity on.
    ///
    /// The tokens must be the *canonical* SMILES of the molecules in `graph`.
    /// Nothing here checks that -- there is no cheap way to -- but a decoder
    /// trained against whichever spelling a file happened to use is being
    /// asked to reproduce an arbitrary choice from a vector that cannot
    /// possibly encode it, and it will spend real capacity trying.
    pub fn loss(
        &self,
        graph: &GraphTensors,
        tokens: &TokenBatch,
        train: bool,
    ) -> Result<MoleculeLoss> {
        let latent = self.encode(graph, train)?;
        let sampled = self.sample(&latent, train)?;
        let predicted: f32 = tokens.mask.iter().sum();
        Ok(MoleculeLoss {
            reconstruction: self.decoder.loss(&sampled, tokens, train)?,
            kl: crate::ml::nn::autoencoder::kl_divergence(&latent)?,
            tokens_per_molecule: (predicted as f64 / tokens.rows.max(1) as f64).max(1.0),
            latent,
        })
    }

    /// One SMILES per row of `latent`. See [`SmilesDecoder::generate`] for what
    /// `temperature` does.
    pub fn generate(
        &self,
        latent: &Tensor,
        vocabulary: &SmilesVocabulary,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<String>> {
        Ok(self
            .decoder
            .generate(latent, temperature, seed)?
            .iter()
            .map(|ids| vocabulary.decode(ids))
            .collect())
    }

    /// [`MoleculeAutoencoder::generate`], parsed.
    ///
    /// `None` where what the decoder wrote is not a molecule: an unbalanced
    /// ring closure, a valence no element has, a string cut off at the length
    /// limit. The share of `Some` is the model's validity rate, and the share
    /// of those that equal what went in is its reconstruction rate -- the two
    /// numbers that say whether this model works.
    pub fn decode(
        &self,
        latent: &Tensor,
        vocabulary: &SmilesVocabulary,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<Option<MolecularGraph>>> {
        Ok(self
            .generate(latent, vocabulary, temperature, seed)?
            .into_iter()
            .map(|text| parse(&text))
            .collect())
    }

    /// `rows` draws from the prior a variational model was trained towards.
    ///
    /// Decoding these is how a trained model is asked for molecules it has
    /// never seen. On a model that is not variational the prior means nothing
    /// -- the encodings of real molecules are wherever the encoder put them --
    /// and what comes back will mostly be nonsense.
    pub fn sample_prior(&self, rows: usize, device: &Device) -> Result<Tensor> {
        Tensor::randn(0.0f32, 1.0f32, (rows, self.config.latent), device)
    }
}

/// What the decoder wrote, as a molecule, or `None` if it was not one.
///
/// An empty string parses to no molecules at all rather than to an error,
/// which is a shape the rest of the crate never produces and would otherwise
/// panic on the way out.
fn parse(text: &str) -> Option<MolecularGraph> {
    let mut parsed = MolecularGraph::from_smiles(text).ok()?;
    (!parsed.is_empty()).then(|| parsed.swap_remove(0))
}

/// The reconstruction and the KL, kept apart.
#[derive(Debug, Clone)]
pub struct MoleculeLoss {
    /// Cross entropy over the SMILES tokens: how well the molecule was spelled.
    pub reconstruction: Tensor,
    /// How far the encoded distribution strayed from a standard normal, per
    /// molecule. Zero on a model that is not variational.
    pub kl: Tensor,
    /// How many tokens a molecule of this batch was scored on, on average --
    /// what puts `reconstruction`, a mean per token, and `kl`, a sum per
    /// molecule, on the same footing in [`MoleculeLoss::total`].
    pub tokens_per_molecule: f64,
    /// What the encoder produced, so that a caller wanting the embedding as
    /// well does not have to run the encoder twice -- which under dropout
    /// would not give the same answer anyway.
    pub latent: Latent,
}

impl MoleculeLoss {
    /// The number to take a gradient of.
    ///
    /// The KL is divided by [`MoleculeLoss::tokens_per_molecule`] before it is
    /// weighted, so that a weight of one is the evidence lower bound itself,
    /// per token. Adding the per-molecule KL to the per-token reconstruction
    /// as it stood charged the latent forty to a hundred times what the bound
    /// does, and paying that much for a latent is a strong reason for the
    /// decoder to stop reading it.
    ///
    /// `kl_weight` is the annealing knob: zero for the first epochs, so the
    /// decoder learns to use the latent before it is charged for it, then
    /// raised to one. Held at one from the start, a variational model of this
    /// shape reliably collapses -- the KL is paid immediately, the
    /// reconstruction has not yet earned anything, and the cheapest way out is
    /// a latent the decoder ignores.
    pub fn total(&self, kl_weight: f64) -> Result<Tensor> {
        self.reconstruction
            .add(&self.kl.affine(kl_weight / self.tokens_per_molecule, 0.0)?)
    }
}
