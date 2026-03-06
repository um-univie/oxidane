//! AdamW, an epoch loop, and stopping when the validation score stops moving.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::ml::dataset::{Batch, Dataset};
use crate::ml::nn::loss::{mask_from_flags, masked_l1, masked_mse};
use crate::ml::nn::model::MolecularNet;
use crate::ml::nn::tensors::GraphTensors;
use crate::ml::normalize::{FeatureStats, GlobalStats};
use crate::ml::split::{Side, Split};

/// Which loss a graph-level task is trained against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Objective {
    /// Squared error, for a continuous target.
    #[default]
    Regression,
    /// Absolute error. Steadier than squared error on a target with outliers
    /// at an assay's detection limit, which most measured properties have.
    RobustRegression,
    /// Binary cross entropy on logits. The head's output stays a logit, so
    /// anything reading a probability out of a prediction has to apply a
    /// sigmoid itself.
    Classification,
}

/// How long to train and how hard.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub epochs: usize,
    pub batch_size: usize,
    /// Stop when the validation loss has not improved for this many epochs.
    /// `None` runs every epoch asked for.
    pub patience: Option<usize>,
    /// Rescale the gradient when its global norm exceeds this. Message passing
    /// re-applies the same weights every round, so a gradient can compound
    /// through the depth the way a recurrent network's does; a clip is cheaper
    /// insurance than finding out.
    pub grad_clip: Option<f64>,
    pub objective: Objective,
    /// The seed the epoch shuffles start from.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            epochs: 30,
            batch_size: 50,
            patience: Some(10),
            grad_clip: Some(5.0),
            objective: Objective::default(),
            seed: 0,
        }
    }
}

/// What one training run did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    /// Mean training loss per epoch.
    pub train_loss: Vec<f32>,
    /// Mean validation loss per epoch, empty if there was no validation set.
    pub valid_loss: Vec<f32>,
    /// The epoch with the lowest validation loss, or the last one run if there
    /// was no validation set.
    pub best_epoch: usize,
    /// That epoch's validation loss.
    pub best_loss: f32,
    /// Whether patience ran out before the epochs did.
    pub stopped_early: bool,
}

/// A model, its weights, and the optimizer stepping them.
///
/// The [`VarMap`] is the caller's, so saving a model is
/// `varmap.save("model.safetensors")` and loading it is `varmap.load(..)` on a
/// [`VarMap`] built from the same [`crate::ml::nn::ModelConfig`]. Store the
/// config and the [`FeatureStats`] alongside it: weights alone do not say what
/// they were trained to read.
pub struct Trainer<'a> {
    model: &'a MolecularNet,
    variables: Vec<Var>,
    optimizer: AdamW,
    config: TrainConfig,
    stats: Option<FeatureStats>,
    global_stats: Option<GlobalStats>,
    device: Device,
}

impl<'a> Trainer<'a> {
    pub fn new(
        model: &'a MolecularNet,
        varmap: &VarMap,
        config: TrainConfig,
        device: &Device,
    ) -> Result<Self> {
        let variables = varmap.all_vars();
        let optimizer = AdamW::new(
            variables.clone(),
            ParamsAdamW {
                lr: config.learning_rate,
                weight_decay: config.weight_decay,
                ..Default::default()
            },
        )?;
        Ok(Self {
            model,
            variables,
            optimizer,
            config,
            stats: None,
            global_stats: None,
            device: device.clone(),
        })
    }

    /// Scales the atom features by these statistics before every forward pass.
    ///
    /// Fit them on the training set alone -- see [`FeatureStats`] -- and keep
    /// them with the weights, because a model trained on scaled features
    /// predicts nonsense from unscaled ones and nothing about the numbers says
    /// so.
    pub fn with_normalization(mut self, stats: FeatureStats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Scales the whole-molecule descriptors the same way.
    ///
    /// Every one of them is a quantity, so all of them are scaled -- see
    /// [`GlobalStats`]. Keep these with the weights too.
    pub fn with_global_normalization(mut self, stats: GlobalStats) -> Self {
        self.global_stats = Some(stats);
        self
    }

    /// Trains on the `Train` side and watches the `Valid` side.
    ///
    /// A split with no validation set trains for the full number of epochs and
    /// reports the last one as the best, since there is nothing to choose
    /// between them with.
    pub fn fit(&mut self, dataset: &Dataset, split: &Split) -> Result<History> {
        let mut history = History {
            best_loss: f32::INFINITY,
            ..Default::default()
        };
        let mut since_improvement = 0;

        for epoch in 0..self.config.epochs {
            let seed = self.config.seed.wrapping_add(epoch as u64 + 1);
            let mut total = 0.0;
            let mut batches = 0;
            for batch in
                dataset.epoch_side(split, Side::Train, self.config.batch_size, seed)
            {
                total += self.step(&batch)?;
                batches += 1;
            }
            history
                .train_loss
                .push(if batches > 0 { total / batches as f32 } else { 0.0 });

            if split.valid.is_empty() {
                history.best_epoch = epoch;
                continue;
            }

            let validation = self.evaluate(dataset, split, Side::Valid)?;
            history.valid_loss.push(validation);
            if validation < history.best_loss {
                history.best_loss = validation;
                history.best_epoch = epoch;
                since_improvement = 0;
            } else {
                since_improvement += 1;
                if self.config.patience.is_some_and(|limit| since_improvement >= limit) {
                    history.stopped_early = true;
                    break;
                }
            }
        }

        if history.valid_loss.is_empty() {
            history.best_loss = *history.train_loss.last().unwrap_or(&0.0);
        }
        Ok(history)
    }

    /// One forward, one backward, one optimizer step. Returns the batch loss.
    pub fn step(&mut self, batch: &Batch) -> Result<f32> {
        let loss = self.batch_loss(batch, true)?;
        let mut gradients = loss.backward()?;
        if let Some(limit) = self.config.grad_clip {
            clip_global_norm(&mut gradients, &self.variables, limit)?;
        }
        self.optimizer.step(&gradients)?;
        loss.to_dtype(DType::F32)?.to_scalar::<f32>()
    }

    /// Sum of the per-head mean losses over the observed targets in a split.
    /// Each head is normalized once over the whole split, so missing labels,
    /// different molecule sizes and a short final batch do not change its weight.
    pub fn evaluate(&self, dataset: &Dataset, split: &Split, side: Side) -> Result<f32> {
        let mut sums = [0.0f64; 3];
        let mut counts = [0usize; 3];
        for batch in dataset.epoch_side(split, side, self.config.batch_size, 0) {
            for (head, loss) in self.head_losses(&batch, false)?.into_iter().enumerate() {
                if let Some((loss, count)) = loss && count > 0 {
                    sums[head] += loss.to_dtype(DType::F64)?.to_scalar::<f64>()? * count as f64;
                    counts[head] += count;
                }
            }
        }
        Ok(sums.into_iter().zip(counts).filter(|&(_, count)| count > 0)
            .map(|(sum, count)| sum / count as f64).sum::<f64>() as f32)
    }

    /// One prediction per molecule for every molecule of a side, in the order
    /// the split lists them.
    pub fn predict(&self, dataset: &Dataset, indices: &[usize]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(indices.len());
        for batch in dataset.epoch_of(indices, self.config.batch_size, 0) {
            let graph = self.upload(&batch)?;
            let predictions = self.model.forward(&graph, false)?;
            let Some(values) = predictions.graph else {
                continue;
            };
            let tasks = values.dim(1)?;
            let flat = values.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for row in 0..batch.graph.graph_count {
                out.push(flat[row * tasks..(row + 1) * tasks].to_vec());
            }
        }
        Ok(out)
    }

    fn head_losses(&self, batch: &Batch, train: bool) -> Result<[Option<(Tensor, usize)>; 3]> {
        let count = batch.target_mask.iter().filter(|&&measured| measured).count();
        Ok([Some((self.batch_loss(batch, train)?, count)), None, None])
    }

    fn batch_loss(&self, batch: &Batch, train: bool) -> Result<Tensor> {
        let graph = self.upload(batch)?;
        let predictions = self.model.forward(&graph, train)?;
        let Some(values) = predictions.graph else {
            return Tensor::new(0.0f32, &self.device);
        };

        let shape = (batch.graph.graph_count, batch.task_count);
        let targets = Tensor::from_slice(batch.targets.as_slice(), shape, &self.device)?
            .to_dtype(values.dtype())?;
        let mask = mask_from_flags(&batch.target_mask, shape, &self.device)?;

        match self.config.objective {
            Objective::Regression => masked_mse(&values, &targets, &mask),
            Objective::RobustRegression => masked_l1(&values, &targets, &mask),
            Objective::Classification => {
                crate::ml::nn::loss::masked_bce_with_logits(&values, &targets, &mask)
            }
        }
    }

    fn upload(&self, batch: &Batch) -> Result<GraphTensors> {
        let atoms = match &self.stats {
            Some(stats) => {
                let mut features = batch.graph.atom_features.clone();
                stats.apply(&mut features);
                std::borrow::Cow::Owned(features)
            }
            None => std::borrow::Cow::Borrowed(batch.graph.atom_features.as_slice()),
        };
        let globals = match &self.global_stats {
            Some(stats) if !stats.is_empty() => {
                let mut features = batch.graph.global_features.clone();
                stats.apply(&mut features);
                std::borrow::Cow::Owned(features)
            }
            _ => std::borrow::Cow::Borrowed(batch.graph.global_features.as_slice()),
        };
        GraphTensors::with_features(&batch.graph, &atoms, &globals, &self.device)
    }
}

/// Rescales every gradient by the same factor when their combined norm is too
/// large.
///
/// Per-parameter clipping would change the gradient's direction as well as its
/// size; scaling all of them together is the same step, taken shorter.
pub fn clip_global_norm(gradients: &mut GradStore, variables: &[Var], limit: f64) -> Result<()> {
    let mut total = 0.0f64;
    for variable in variables {
        if let Some(gradient) = gradients.get(variable) {
            total += gradient
                .sqr()?
                .sum_all()?
                .to_dtype(DType::F64)?
                .to_scalar::<f64>()?;
        }
    }

    let norm = total.sqrt();
    if !norm.is_finite() || norm <= limit {
        return Ok(());
    }

    let scale = limit / norm;
    for variable in variables {
        if let Some(gradient) = gradients.get(variable) {
            let scaled = gradient.affine(scale, 0.0)?;
            gradients.insert(variable, scaled);
        }
    }
    Ok(())
}
