//! Teaching the encoder chemistry before it is shown a single measurement.
//!
//! The evidence for doing this is specific rather than fashionable. On these
//! datasets the model can already fit its training set almost exactly, dropout
//! does nothing, and a scaffold split costs the graph-only model 13% on ESOL
//! and 24% on Lipophilicity while costing a model given bulk descriptors
//! nothing at all. So the encoder is not short of capacity and it is not
//! overfitting in the way regularisation fixes: what it does badly is carry
//! what it learned to a skeleton it has not seen. Meanwhile the labelled sets
//! are one to four thousand molecules and unlabelled structures are effectively
//! unlimited.
//!
//! # The objective
//!
//! Hide the element of a fraction of the atoms and ask the encoder to say what
//! they were. An atom's identity is heavily determined by what surrounds it --
//! a two-connected atom between two aromatic carbons in a five-ring with a
//! hydrogen on it is a pyrrole nitrogen and very little else -- so predicting
//! it forces the vectors to describe neighbourhoods rather than to memorise
//! molecules.
//!
//! Reconstructing the *bonds* instead, which is what
//! [`crate::ml::nn::autoencoder`] does, is a weaker signal for a molecule: most
//! of the adjacency follows from valence, so a model can score well on it
//! without having learned anything transferable.
//!
//! Masking clears the element block **and the mass**, because the mass names
//! the element as plainly as the one-hot does. Leaving it would let the model
//! score perfectly by reading the answer off the row.
//!
//! Only heavy atoms are hidden. Hydrogens stay in the graph as context, but
//! asking for one gives nothing away to learn: a hydrogen row is the only one
//! with an unspecified hybridisation, a single heavy neighbour and no attached
//! hydrogens, so it is named exactly from what masking leaves behind. Before
//! this the hydrogens were 47.5% of the atoms asked about on PubChem, and
//! answering "hydrogen" for those and "carbon" for the rest scored 87%
//! without looking at a neighbourhood at all.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::ml::batch::GraphBatch;
use crate::ml::nn::layers::{Mlp, MlpConfig};
use crate::ml::nn::mpnn::{Encoder, EncoderConfig, EncoderKind};
use crate::ml::nn::tensors::GraphTensors;
use crate::molecule::featurize::{
    ATOM_CONTINUOUS_COLUMNS, ATOM_ELEMENT_CLASSES, ATOM_ELEMENT_COLUMNS, ATOM_FEATURE_WIDTH,
};

/// How much of the molecule to hide, and how much encoder to hide it from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PretrainConfig {
    pub encoder: EncoderConfig,
    pub kind: EncoderKind,
    /// The share of heavy atoms whose element is hidden in each batch.
    ///
    /// Fifteen percent is the usual choice and the reasoning carries over: too
    /// few and most of the forward pass teaches nothing, too many and the
    /// context that is supposed to give the answer away has itself been erased.
    pub mask_fraction: f64,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            kind: EncoderKind::default(),
            mask_fraction: 0.15,
        }
    }
}

/// An encoder and the head that reads an element back out of an atom vector.
///
/// The head is thrown away afterwards; the encoder is what is kept. See
/// [`crate::ml::nn::seed::save_prefixed`] for moving it.
#[derive(Debug)]
pub struct MaskedAtomModel {
    encoder: Encoder,
    head: Mlp,
    config: PretrainConfig,
}

impl MaskedAtomModel {
    /// Builds the model. The encoder is created under the `encoder` prefix, the
    /// same one [`crate::ml::nn::MolecularNet`] uses, so the weights move
    /// across by name without any translation.
    pub fn new(config: PretrainConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(config.kind, config.encoder, vb.pp("encoder"))?;
        let width = encoder.width();
        Ok(Self {
            head: Mlp::new(
                &[width, width, ATOM_ELEMENT_CLASSES],
                MlpConfig::default(),
                vb.pp("mask_head"),
            )?,
            encoder,
            config,
        })
    }

    /// The cross entropy of the masked atoms' elements for one batch.
    ///
    /// `atom_features` is the batch's atom matrix, already scaled by whatever
    /// the caller uses at fine-tuning time, so that the encoder is pretrained
    /// on the numbers it will later be given.
    pub fn loss(
        &self,
        batch: &GraphBatch,
        atom_features: &[f32],
        device: &Device,
        seed: u64,
        train: bool,
    ) -> Result<Tensor> {
        let masking = Masking::choose(batch, atom_features, self.config.mask_fraction, seed);
        if masking.rows.is_empty() {
            return Tensor::new(0.0f32, device);
        }

        let graph = GraphTensors::with_atom_features(batch, &masking.features, device)?;
        let atoms = self.encoder.forward(&graph, train)?;

        let rows = Tensor::from_slice(&masking.rows, masking.rows.len(), device)?;
        let logits = self.head.forward(&atoms.index_select(&rows, 0)?, train)?;

        // A one-hot target, so the same masked cross entropy the supervised
        // heads use can be reused rather than written again.
        let mut targets = vec![0.0f32; masking.rows.len() * ATOM_ELEMENT_CLASSES];
        for (position, &class) in masking.classes.iter().enumerate() {
            targets[position * ATOM_ELEMENT_CLASSES + class as usize] = 1.0;
        }
        let targets =
            Tensor::from_slice(&targets, (masking.rows.len(), ATOM_ELEMENT_CLASSES), device)?
                .to_dtype(logits.dtype())?;
        let kept = Tensor::ones((masking.rows.len(), 1), logits.dtype(), device)?;

        crate::ml::nn::loss::masked_cross_entropy(&logits, &targets, &kept)
    }

    /// How often the model names the hidden element correctly.
    ///
    /// The number to watch: the loss falling says the head is learning, and
    /// this says what it learned. Only heavy atoms are asked about, and they
    /// are mostly carbon, so the figure to beat is the share of heavy atoms
    /// that are carbon -- about three quarters -- not zero.
    pub fn accuracy(
        &self,
        batch: &GraphBatch,
        atom_features: &[f32],
        device: &Device,
        seed: u64,
    ) -> Result<(usize, usize)> {
        let masking = Masking::choose(batch, atom_features, self.config.mask_fraction, seed);
        if masking.rows.is_empty() {
            return Ok((0, 0));
        }
        let graph = GraphTensors::with_atom_features(batch, &masking.features, device)?;
        let atoms = self.encoder.forward(&graph, false)?;
        let rows = Tensor::from_slice(&masking.rows, masking.rows.len(), device)?;
        let logits = self.head.forward(&atoms.index_select(&rows, 0)?, false)?;

        let flat = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut right = 0;
        for (position, &class) in masking.classes.iter().enumerate() {
            let row = &flat[position * ATOM_ELEMENT_CLASSES..(position + 1) * ATOM_ELEMENT_CLASSES];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);
            if best == class as usize {
                right += 1;
            }
        }
        Ok((right, masking.classes.len()))
    }

    pub fn width(&self) -> usize {
        self.encoder.width()
    }
}

/// Which atoms were hidden, what they were, and the matrix with them hidden.
/// The element column a hydrogen sets: the first of
/// [`crate::molecule::featurize::FEATURED_ELEMENTS`].
const HYDROGEN_CLASS: usize = 0;
const _: () = assert!(crate::molecule::featurize::FEATURED_ELEMENTS[HYDROGEN_CLASS] == 1);

struct Masking {
    features: Vec<f32>,
    rows: Vec<u32>,
    classes: Vec<u32>,
}

impl Masking {
    /// Chooses the atoms to hide from a seeded stream, so an epoch is
    /// repeatable and two runs of one script can be compared.
    fn choose(batch: &GraphBatch, atom_features: &[f32], fraction: f64, seed: u64) -> Self {
        let mut features = atom_features.to_vec();
        let mut rows = Vec::new();
        let mut classes = Vec::new();

        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            (z ^ (z >> 31)) >> 11
        };
        let ceiling = (1u64 << 53) as f64;

        for row in 0..batch.atom_count {
            let start = row * ATOM_FEATURE_WIDTH;
            if features[start + HYDROGEN_CLASS] > 0.5 {
                continue;
            }
            if next() as f64 / ceiling >= fraction {
                continue;
            }
            // The element is whichever column of the block is set. A row with
            // none set cannot happen -- the last column is the catch-all -- but
            // if it ever did, hiding it would ask an unanswerable question.
            let Some(class) = ATOM_ELEMENT_COLUMNS
                .clone()
                .find(|&column| features[start + column] > 0.5)
            else {
                continue;
            };
            for column in ATOM_ELEMENT_COLUMNS {
                features[start + column] = 0.0;
            }
            for &column in ATOM_CONTINUOUS_COLUMNS.iter() {
                features[start + column] = 0.0;
            }
            rows.push(row as u32);
            classes.push(class as u32);
        }

        Self {
            features,
            rows,
            classes,
        }
    }
}
