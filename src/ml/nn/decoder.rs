//! Writing a molecule out again, one token at a time.
//!
//! An encoder collapses a molecule to a vector. Getting a molecule back from
//! that vector is the hard direction: the answer is a graph, and a graph has
//! no order, so a network that emits one has to be told which of its outputs
//! corresponds to which atom before it can be scored at all.
//!
//! Emitting a *string* sidesteps the whole problem. A canonical SMILES is a
//! total order on the molecule, so the target is a sequence and the loss is
//! the cross entropy every language model uses; and the crate's own parser is
//! the arbiter of whether what came out was a molecule. What is given up is
//! that nothing constrains the decoder to write something parseable -- an
//! unbalanced ring closure or a pentavalent carbon is a perfectly reachable
//! output -- so [`SmilesDecoder::generate`] returns strings and the caller
//! decides what to do with the ones that do not parse.
//!
//! # How the vector reaches the decoder
//!
//! Twice, deliberately. It sets the recurrent network's initial state, and it
//! is also concatenated onto every step's input. The first alone is the
//! textbook arrangement and it fails in a specific way on this problem: the
//! recurrence can carry the sequence by itself, the latent's contribution
//! decays over the string, and a variational model then finds it cheaper to
//! ignore the latent entirely and pay nothing in KL -- the molecule comes out
//! fluent, valid, and unrelated to what went in. Feeding it at every step
//! leaves the decoder no such option.

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::rnn::{GRU, GRUConfig, GRUState, RNN};
use candle_nn::{Dropout, Embedding, Linear, Module, VarBuilder, embedding, linear};

use crate::ml::nn::loss::masked_mean;
use crate::ml::tokenize::{END, PADDING, START};

/// How much decoder there is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecoderConfig {
    /// The width of a token's vector.
    pub embedding: usize,
    /// The width of the recurrent state.
    pub hidden: usize,
    /// How many tokens [`SmilesDecoder::generate`] will emit before giving up
    /// on an end marker. A string that hits the cap is truncated and will not
    /// parse, which is the honest outcome -- the model did not finish.
    pub max_length: usize,
    /// Applied to the state before the output projection.
    pub dropout: f32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            embedding: 64,
            hidden: 512,
            // Comfortably past a drug-sized molecule's canonical SMILES, which
            // runs to about a hundred tokens.
            max_length: 200,
            dropout: 0.0,
        }
    }
}

/// A recurrent network that spells a molecule out of a latent vector.
///
/// ```
/// # fn main() -> candle_core::Result<()> {
/// use candle_core::{DType, Device, Tensor};
/// use candle_nn::{VarBuilder, VarMap};
/// use oxidane::ml::nn::decoder::{DecoderConfig, SmilesDecoder};
/// use oxidane::ml::tokenize::{SmilesVocabulary, TokenBatch};
///
/// let device = Device::Cpu;
/// let varmap = VarMap::new();
/// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
///
/// let vocabulary = SmilesVocabulary::from_corpus(["CCO", "CC(=O)O"], 1);
/// let config = DecoderConfig { embedding: 8, hidden: 16, ..Default::default() };
/// let decoder = SmilesDecoder::new(4, vocabulary.len(), config, vb)?;
///
/// let batch = TokenBatch::collate(&[vocabulary.encode("CCO")]);
/// let latent = Tensor::zeros((1, 4), DType::F32, &device)?;
///
/// // One prediction per step, over the whole vocabulary.
/// let logits = decoder.logits(&latent, &batch, false)?;
/// assert_eq!(logits.dims(), &[batch.width, vocabulary.len()]);
/// assert!(decoder.loss(&latent, &batch, false)?.to_scalar::<f32>()?.is_finite());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SmilesDecoder {
    embedding: Embedding,
    /// The latent vector, made into the recurrent network's first state.
    initial: Linear,
    recurrent: GRU,
    output: Linear,
    dropout: Dropout,
    latent: usize,
    vocabulary: usize,
    config: DecoderConfig,
}

impl SmilesDecoder {
    pub fn new(
        latent: usize,
        vocabulary: usize,
        config: DecoderConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            embedding: embedding(vocabulary, config.embedding, vb.pp("embedding"))?,
            initial: linear(latent, config.hidden, vb.pp("initial"))?,
            recurrent: GRU::new(
                config.embedding + latent,
                config.hidden,
                GRUConfig::default(),
                vb.pp("recurrent"),
            )?,
            output: linear(config.hidden, vocabulary, vb.pp("output"))?,
            dropout: Dropout::new(config.dropout),
            latent,
            vocabulary,
            config,
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// How wide the latent vector it reads is.
    pub fn latent_width(&self) -> usize {
        self.latent
    }

    /// How many tokens it can emit.
    pub fn vocabulary_width(&self) -> usize {
        self.vocabulary
    }

    /// The first recurrent state, `tanh`-squashed so that an unbounded latent
    /// cannot start the recurrence saturated.
    fn first_state(&self, latent: &Tensor) -> Result<GRUState> {
        Ok(GRUState {
            h: self.initial.forward(latent)?.tanh()?.contiguous()?,
        })
    }

    /// One step of the recurrence: a token in, the next state out.
    fn step(&self, tokens: &Tensor, latent: &Tensor, state: &GRUState) -> Result<GRUState> {
        let embedded = self.embedding.forward(tokens)?;
        self.recurrent
            .step(&Tensor::cat(&[&embedded, latent], 1)?.contiguous()?, state)
    }

    /// A prediction for every step of every row: `[rows * width, vocabulary]`.
    ///
    /// Teacher forced -- the decoder reads [`crate::ml::tokenize::TokenBatch`]'s
    /// `inputs`, which is the true molecule shifted by one, rather than what it
    /// said itself. That is what makes the whole batch one parallel forward
    /// pass instead of `width` sequential ones per molecule, and it is why
    /// training loss is not a measure of what [`SmilesDecoder::generate`] will
    /// do: use a round-trip count for that.
    pub fn logits(
        &self,
        latent: &Tensor,
        batch: &crate::ml::tokenize::TokenBatch,
        train: bool,
    ) -> Result<Tensor> {
        let device = latent.device();
        let inputs = Tensor::from_slice(&batch.inputs, (batch.rows, batch.width), device)?;
        let mut state = self.first_state(latent)?;
        let mut states = Vec::with_capacity(batch.width);
        for step in 0..batch.width {
            state = self.step(&inputs.i((.., step))?.contiguous()?, latent, &state)?;
            states.push(state.h.clone());
        }
        let stacked =
            Tensor::stack(&states, 1)?.reshape((batch.rows * batch.width, self.config.hidden))?;
        self.output.forward(&self.dropout.forward(&stacked, train)?)
    }

    /// Cross entropy over the steps that are a real prediction.
    ///
    /// The padding after a molecule has ended is masked away rather than
    /// learned: a batch is as wide as its longest member, and teaching the
    /// model to predict padding after a short one teaches it to end early.
    pub fn loss(
        &self,
        latent: &Tensor,
        batch: &crate::ml::tokenize::TokenBatch,
        train: bool,
    ) -> Result<Tensor> {
        let logits = self.logits(latent, batch, train)?;
        let device = latent.device();
        let count = batch.rows * batch.width;
        let targets = Tensor::from_slice(&batch.targets, (count, 1), device)?;
        let mask = Tensor::from_slice(&batch.mask, (count, 1), device)?.to_dtype(logits.dtype())?;
        let chosen = candle_nn::ops::log_softmax(&logits, 1)?.gather(&targets, 1)?;
        masked_mean(&chosen.neg()?, &mask)
    }

    /// Spells out one molecule per row of `latent`.
    ///
    /// `temperature` at or below zero takes the most likely token at every
    /// step, which is deterministic and is what a round-trip test wants. Above
    /// zero it samples, with the usual meaning: below one sharpens the
    /// distribution, above one flattens it. Sampling is what a *generative*
    /// model is for -- greedy decoding from a random latent returns the same
    /// handful of molecules over and over.
    ///
    /// The ids come back without the start and end markers, ready for
    /// [`crate::ml::tokenize::SmilesVocabulary::decode`].
    pub fn generate(&self, latent: &Tensor, temperature: f64, seed: u64) -> Result<Vec<Vec<u32>>> {
        let device = latent.device();
        let rows = latent.dim(0)?;
        let mut state = self.first_state(latent)?;
        let mut current = vec![START; rows];
        let mut finished = vec![false; rows];
        let mut out = vec![Vec::new(); rows];
        let mut random = SplitMix::new(seed);

        for _ in 0..self.config.max_length {
            let tokens = Tensor::from_slice(&current, rows, device)?;
            state = self.step(&tokens, latent, &state)?;
            let logits = self.output.forward(&state.h)?.to_dtype(DType::F32)?;
            let rows_of_logits = logits.to_vec2::<f32>()?;

            for (row, values) in rows_of_logits.iter().enumerate() {
                if finished[row] {
                    // A finished row keeps being stepped so that the batch
                    // stays rectangular, and keeps being fed padding so that
                    // what it emits cannot depend on the rows beside it.
                    current[row] = PADDING;
                    continue;
                }
                let chosen = choose(values, temperature, &mut random);
                if chosen == END {
                    finished[row] = true;
                    current[row] = PADDING;
                    continue;
                }
                out[row].push(chosen);
                current[row] = chosen;
            }
            if finished.iter().all(|&done| done) {
                break;
            }
        }
        Ok(out)
    }
}

/// The token a row emits: the largest logit, or a draw from the softmax.
///
/// The special tokens other than the end marker are never emitted -- a decoder
/// has no use for padding, a start marker in the middle of a molecule is
/// meaningless, and an unknown token is by definition unspellable, so allowing
/// any of them only turns a usable string into an unparseable one.
fn choose(logits: &[f32], temperature: f64, random: &mut SplitMix) -> u32 {
    let allowed = |id: usize| id as u32 == END || id as u32 > crate::ml::tokenize::UNKNOWN;

    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .filter(|&(id, _)| allowed(id))
            .max_by(|left, right| {
                left.1
                    .partial_cmp(right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id as u32)
            .unwrap_or(END);
    }

    let highest = logits
        .iter()
        .enumerate()
        .filter(|&(id, _)| allowed(id))
        .fold(f32::NEG_INFINITY, |best, (_, &value)| best.max(value));
    let mut weights: Vec<f64> = logits
        .iter()
        .enumerate()
        .map(|(id, &value)| {
            if allowed(id) {
                (((value - highest) as f64) / temperature).exp()
            } else {
                0.0
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if !(total > 0.0) {
        return END;
    }
    let mut target = random.uniform() * total;
    for (id, weight) in weights.drain(..).enumerate() {
        target -= weight;
        if target <= 0.0 {
            return id as u32;
        }
    }
    END
}

/// The same SplitMix64 stream the rest of the crate seeds itself from, so that
/// a sampled generation is reproducible from its seed alone.
struct SplitMix(u64);

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
