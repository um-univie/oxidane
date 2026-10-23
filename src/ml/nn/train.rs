//! AdamW, an epoch loop, and stopping when the validation score stops moving.

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::ml::batch::GraphBatch;
use crate::ml::dataset::{Batch, Dataset};
use crate::ml::nn::loss::{mask_from_flags, masked_l1, masked_mse};
use crate::ml::nn::model::{MolecularNet, Predictions};
use crate::ml::nn::tensors::GraphTensors;
use crate::ml::normalize::{BondFeatureStats, ClassMeans, FeatureStats, GlobalStats, TargetStats};
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

/// How the learning rate moves over a run.
///
/// A flat rate has to be small enough that the first steps -- taken from a
/// random initialisation, through a network that reapplies the same weights
/// once per message-passing round -- do not throw the model somewhere it
/// cannot come back from. That same rate is then too small to converge with.
/// Warming up removes the first constraint, so the useful part of the run can
/// be faster, and decaying afterwards stops the last epochs bouncing around a
/// minimum they keep overshooting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Schedule {
    /// One rate throughout.
    Constant,
    /// Rises linearly to the configured rate over `warmup` epochs, then decays
    /// exponentially to `final_fraction` of it by the last epoch.
    ///
    /// This is the shape Chemprop uses. It is not the default here -- see
    /// [`Schedule::default`] for what it measured on these datasets.
    WarmupDecay { warmup: usize, final_fraction: f64 },
}

impl Default for Schedule {
    /// Constant, on the evidence.
    ///
    /// Warming up and decaying is what the published numbers for this
    /// architecture were produced with, so it was the default here first. Over
    /// five scaffold splits of each of ESOL, FreeSolv, BBBP and BACE, with the
    /// weights seeded so the comparison is paired rather than two noisy
    /// estimates subtracted, it helped one of the four and hurt the other
    /// three:
    ///
    /// ```text
    ///                 constant   warmup + decay
    /// ESOL     RMSE     0.581        0.593
    /// FreeSolv RMSE     1.054        0.995
    /// BBBP     AUC      0.907        0.901
    /// BACE     AUC      0.855        0.843
    /// ```
    ///
    /// The likely reason is that the decay is spread over the epoch budget
    /// while early stopping ends the run well before it, so the rate is below
    /// the flat one for most of the epochs that actually happen and the low-rate
    /// finish never arrives. A schedule fitted to where the run really stops
    /// may well win; this one does not, and defaulting to it on the strength of
    /// what it does elsewhere would be taking someone else's measurement for
    /// this one.
    fn default() -> Self {
        Schedule::Constant
    }
}

impl Schedule {
    /// The rate for `epoch` of `epochs`, as a multiple of the configured one.
    pub fn factor_for(self, epoch: usize, epochs: usize) -> f64 {
        match self {
            Schedule::Constant => 1.0,
            Schedule::WarmupDecay {
                warmup,
                final_fraction,
            } => {
                if epoch < warmup {
                    // The first epoch is not given a rate of zero: it would be
                    // an epoch of work that moves nothing.
                    return (epoch + 1) as f64 / (warmup + 1) as f64;
                }
                let remaining = epochs.saturating_sub(warmup).max(1);
                let through = (epoch - warmup) as f64 / remaining as f64;
                final_fraction.max(1e-6).powf(through)
            }
        }
    }
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
    /// How the learning rate moves; see [`Schedule`].
    pub schedule: Schedule,
    /// The seed the epoch shuffles start from.
    pub seed: u64,
    /// How many threads one training step is spread over.
    ///
    /// candle's CPU backend runs a matrix product on every core but every
    /// other operation on one, so a step over one batch leaves most of a
    /// machine idle: sixteen cores train D-MPNN only about 1.4 times faster
    /// than one. With more than one shard, [`Trainer::fit`] splits each
    /// batch into that many groups of molecules, runs their forward and
    /// backward passes on separate threads, and adds the gradients up before
    /// the one optimizer step. Each shard's loss is weighted by its share of
    /// the batch's measurements, so the sum is the batch's own loss and the
    /// step is the one a single thread would take, up to the order the
    /// floating-point sums are taken in -- which is fixed, so a run is still
    /// a function of its seed.
    ///
    /// One by default. Leave it there when several runs already share the
    /// machine, as `train_mpnn`'s splits and ensemble members do.
    pub shards: usize,
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
            schedule: Schedule::default(),
            seed: 0,
            shards: 1,
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
    /// Training steps whose gradient was not finite, and which were therefore
    /// not taken. Anything but zero means a batch produced a NaN or an
    /// overflow; the weights are what they were before it.
    pub skipped_steps: usize,
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
    bond_feature_stats: Option<BondFeatureStats>,
    global_stats: Option<GlobalStats>,
    target_stats: Option<TargetStats>,
    atom_stats: Option<TargetStats>,
    bond_stats: Option<TargetStats>,
    atom_class_means: Option<ClassMeans>,
    bond_class_means: Option<ClassMeans>,
    device: Device,
    /// Steps [`Trainer::step`] declined to take since the last [`Trainer::fit`]
    /// began. See [`History::skipped_steps`].
    skipped_steps: usize,
}

impl<'a> Trainer<'a> {
    pub fn new(
        model: &'a MolecularNet,
        varmap: &VarMap,
        config: TrainConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::over(model, varmap.all_vars(), config, device)
    }

    /// A trainer that moves only the weights it is given.
    ///
    /// Everything else is still used in the forward pass and still has a
    /// gradient computed for it; it is simply never stepped. That is what
    /// freezing a pretrained encoder means, and it matters more than it
    /// sounds: a freshly initialised head produces large, meaningless
    /// gradients for its first epochs, and those flow straight back into an
    /// encoder that took hours to train. Letting the head find its feet
    /// against a fixed encoder first is the difference between transferring a
    /// representation and destroying one.
    pub fn over(
        model: &'a MolecularNet,
        variables: Vec<Var>,
        config: TrainConfig,
        device: &Device,
    ) -> Result<Self> {
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
            bond_feature_stats: None,
            global_stats: None,
            target_stats: None,
            atom_stats: None,
            bond_stats: None,
            atom_class_means: None,
            bond_class_means: None,
            device: device.clone(),
            skipped_steps: 0,
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

    /// Scales the bond features (the 3D distance column, when
    /// [`crate::molecule::featurize::graph_features_with_geometry`] was used
    /// to build the dataset) the same way -- see [`BondFeatureStats`].
    pub fn with_bond_feature_normalization(mut self, stats: BondFeatureStats) -> Self {
        self.bond_feature_stats = Some(stats);
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

    /// Standardises the targets before the loss sees them, and puts the
    /// predictions back afterwards.
    ///
    /// Only meaningful for a regression: a classification's targets are
    /// already zero and one, and scaling them would move the decision boundary
    /// off the logit's zero. [`Trainer::fit`] ignores these when the objective
    /// is [`Objective::Classification`].
    pub fn with_target_normalization(mut self, stats: TargetStats) -> Self {
        self.target_stats = Some(stats);
        self
    }

    /// [`Trainer::with_target_normalization`] for an atom head's targets.
    pub fn with_atom_target_normalization(mut self, stats: TargetStats) -> Self {
        self.atom_stats = Some(stats);
        self
    }

    /// [`Trainer::with_target_normalization`] for a bond head's targets.
    pub fn with_bond_target_normalization(mut self, stats: TargetStats) -> Self {
        self.bond_stats = Some(stats);
        self
    }

    /// The target statistics in use, if any.
    pub fn target_stats(&self) -> Option<&TargetStats> {
        self.target_stats.as_ref()
    }

    /// The atom head's target statistics in use, if any.
    pub fn atom_target_stats(&self) -> Option<&TargetStats> {
        self.atom_stats.as_ref()
    }

    /// The bond head's target statistics in use, if any.
    pub fn bond_target_stats(&self) -> Option<&TargetStats> {
        self.bond_stats.as_ref()
    }

    /// Subtracts a per-atom-class mean from the atom head's targets before
    /// [`Trainer::with_atom_target_normalization`]'s scaling runs, and adds
    /// it back to a prediction after. See [`ClassMeans`] for why.
    pub fn with_atom_class_means(mut self, means: ClassMeans) -> Self {
        self.atom_class_means = Some(means);
        self
    }

    /// [`Trainer::with_atom_class_means`] for a bond head.
    pub fn with_bond_class_means(mut self, means: ClassMeans) -> Self {
        self.bond_class_means = Some(means);
        self
    }

    /// Trains on the `Train` side and watches the `Valid` side.
    ///
    /// The weights left behind are the ones from the epoch with the lowest
    /// validation loss, not the ones from the last epoch run. Those are not
    /// the same thing: training continues for `patience` epochs after the best
    /// one to find out whether it was the best, and every one of those is a
    /// step away from it. Keeping the final weights instead reports a model
    /// that early stopping deliberately declined to keep.
    ///
    /// A split with no validation set trains for the full number of epochs and
    /// keeps the last, since there is nothing to choose between them with.
    pub fn fit(&mut self, dataset: &Dataset, split: &Split) -> Result<History> {
        self.fit_with(dataset, split, |_history| {})
    }

    /// [`Trainer::fit`], calling `on_epoch` with the [`History`] so far after
    /// every epoch.
    ///
    /// A run over a large dataset can take long enough that silence until it
    /// returns is indistinguishable from a hang; this is the hook for a
    /// caller that wants to show it's still moving -- a progress line, a bar,
    /// whatever the caller's own output looks like. `on_epoch` sees the same
    /// `History` the return value will have, just truncated to the epochs
    /// run so far.
    pub fn fit_with(
        &mut self,
        dataset: &Dataset,
        split: &Split,
        mut on_epoch: impl FnMut(&History),
    ) -> Result<History> {
        let mut history = History {
            best_loss: f32::INFINITY,
            ..Default::default()
        };
        let mut since_improvement = 0;
        let mut best_weights: Option<Vec<Tensor>> = None;
        self.skipped_steps = 0;

        for epoch in 0..self.config.epochs {
            let factor = self.config.schedule.factor_for(epoch, self.config.epochs);
            self.optimizer
                .set_learning_rate(self.config.learning_rate * factor);
            let seed = self.config.seed.wrapping_add(epoch as u64 + 1);
            let mut total = 0.0;
            let mut batches = 0;
            for batch in dataset.epoch_side(split, Side::Train, self.config.batch_size, seed) {
                total += if self.config.shards > 1 && batch.indices.len() > 1 {
                    self.step_sharded(dataset, &batch)?
                } else {
                    self.step(&batch)?
                };
                batches += 1;
            }
            history.train_loss.push(if batches > 0 {
                total / batches as f32
            } else {
                0.0
            });
            history.skipped_steps = self.skipped_steps;

            if split.valid.is_empty() {
                history.best_epoch = epoch;
                on_epoch(&history);
                continue;
            }

            let validation = self.evaluate(dataset, split, Side::Valid)?;
            history.valid_loss.push(validation);
            if validation < history.best_loss {
                history.best_loss = validation;
                history.best_epoch = epoch;
                since_improvement = 0;
                best_weights = Some(self.snapshot()?);
            } else {
                since_improvement += 1;
                if self
                    .config
                    .patience
                    .is_some_and(|limit| since_improvement >= limit)
                {
                    history.stopped_early = true;
                    on_epoch(&history);
                    break;
                }
            }
            on_epoch(&history);
        }

        if let Some(weights) = best_weights {
            self.restore(&weights)?;
        }
        if history.valid_loss.is_empty() {
            history.best_loss = *history.train_loss.last().unwrap_or(&0.0);
        }
        Ok(history)
    }

    /// A detached copy of every weight, for putting back later.
    fn snapshot(&self) -> Result<Vec<Tensor>> {
        self.variables
            .iter()
            .map(|variable| variable.as_detached_tensor().copy())
            .collect()
    }

    fn restore(&self, weights: &[Tensor]) -> Result<()> {
        for (variable, saved) in self.variables.iter().zip(weights) {
            variable.set(saved)?;
        }
        Ok(())
    }

    /// One forward, one backward, one optimizer step. Returns the batch loss.
    ///
    /// A gradient that is not finite is not stepped along: AdamW would write
    /// the NaN into every weight it touches, and its moment estimates would
    /// carry it into every step after. The step is skipped and counted in
    /// [`History::skipped_steps`] instead, and the loss is returned as it was,
    /// so a NaN still shows in the epoch's mean.
    pub fn step(&mut self, batch: &Batch) -> Result<f32> {
        let loss = self.batch_loss(batch, true)?;
        let mut gradients = loss.backward()?;
        let norm = global_norm(&gradients, &self.variables)?;
        if !norm.is_finite() {
            self.skipped_steps += 1;
        } else {
            if let Some(limit) = self.config.grad_clip {
                scale_gradients(&mut gradients, &self.variables, norm, limit)?;
            }
            self.optimizer.step(&gradients)?;
        }
        loss.to_dtype(DType::F32)?.to_scalar::<f32>()
    }

    /// [`Trainer::step`] with the batch spread over
    /// [`TrainConfig::shards`] threads. See there for why the step is the
    /// same one.
    fn step_sharded(&mut self, dataset: &Dataset, batch: &Batch) -> Result<f32> {
        let totals = measured_counts(batch);
        let size = batch
            .indices
            .len()
            .div_ceil(self.config.shards.min(batch.indices.len()));
        let shards: Vec<Batch> = batch
            .indices
            .chunks(size)
            .map(|chunk| dataset.batch(chunk))
            .collect();

        // Threads of their own rather than rayon's: a caller running this
        // inside a pool of one -- `train_mpnn` sizes its pool to the runs it
        // has -- would otherwise get its shards one after another.
        let this = &*self;
        let run = |shard: &Batch| -> Result<(GradStore, f32)> {
            let mut loss = Tensor::new(0.0f32, &this.device)?;
            for (head, part) in this.head_losses(shard, true)?.into_iter().enumerate() {
                if let Some((mean, count)) = part
                    && count > 0
                {
                    let share = count as f64 / totals[head] as f64;
                    loss = loss.add(&mean.affine(share, 0.0)?.to_dtype(DType::F32)?)?;
                }
            }
            let value = loss.to_scalar::<f32>()?;
            Ok((loss.backward()?, value))
        };
        let run = &run;
        let results: Vec<Result<(GradStore, f32)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = shards
                .iter()
                .map(|shard| scope.spawn(move || run(shard)))
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(candle_core::Error::Msg(
                            "a training shard panicked".to_string(),
                        ))
                    })
                })
                .collect()
        });

        let mut total = 0.0;
        let mut gradients: Option<GradStore> = None;
        for result in results {
            let (shard_gradients, value) = result?;
            total += value;
            match &mut gradients {
                None => gradients = Some(shard_gradients),
                Some(sum) => {
                    for variable in &self.variables {
                        if let Some(gradient) = shard_gradients.get(variable) {
                            let added = match sum.get(variable) {
                                Some(previous) => previous.add(gradient)?,
                                None => gradient.clone(),
                            };
                            sum.insert(variable, added);
                        }
                    }
                }
            }
        }
        let Some(mut gradients) = gradients else {
            return Ok(total);
        };
        let norm = global_norm(&gradients, &self.variables)?;
        if !norm.is_finite() {
            self.skipped_steps += 1;
        } else {
            if let Some(limit) = self.config.grad_clip {
                scale_gradients(&mut gradients, &self.variables, norm, limit)?;
            }
            self.optimizer.step(&gradients)?;
        }
        Ok(total)
    }

    /// Sum of the per-head mean losses over the observed targets in a split.
    /// Each head is normalized once over the whole split, so missing labels,
    /// different molecule sizes and a short final batch do not change its weight.
    pub fn evaluate(&self, dataset: &Dataset, split: &Split, side: Side) -> Result<f32> {
        let mut sums = [0.0f64; 3];
        let mut counts = [0usize; 3];
        for batch in dataset.epoch_side(split, side, self.config.batch_size, 0) {
            for (head, loss) in self.head_losses(&batch, false)?.into_iter().enumerate() {
                if let Some((loss, count)) = loss
                    && count > 0
                {
                    sums[head] += loss.to_dtype(DType::F64)?.to_scalar::<f64>()? * count as f64;
                    counts[head] += count;
                }
            }
        }
        Ok(sums
            .into_iter()
            .zip(counts)
            .filter(|&(_, count)| count > 0)
            .map(|(sum, count)| sum / count as f64)
            .sum::<f64>() as f32)
    }

    /// One prediction per molecule for every molecule of a side, in the order
    /// the split lists them.
    pub fn predict(&self, dataset: &Dataset, indices: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.predict_level(
            dataset,
            indices,
            self.target_stats.as_ref(),
            None,
            |predictions| predictions.graph.as_ref(),
            |graph| graph.graph_count,
        )
    }

    /// [`Trainer::predict`] for an atom head: one row per atom row, in the
    /// same order [`Dataset::atom_targets`] gives them for the same
    /// `indices`.
    pub fn predict_atoms(&self, dataset: &Dataset, indices: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.predict_level(
            dataset,
            indices,
            self.atom_stats.as_ref(),
            self.atom_class_means.as_ref().map(|means| {
                (
                    means,
                    (|batch: &Batch| batch.atom_class.as_slice()) as fn(&Batch) -> &[u32],
                )
            }),
            |predictions| predictions.atom.as_ref(),
            |graph| graph.atom_count,
        )
    }

    /// [`Trainer::predict`] for a bond head: one row per **directed edge**,
    /// matching [`Dataset::bond_targets`].
    pub fn predict_bonds(&self, dataset: &Dataset, indices: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.predict_level(
            dataset,
            indices,
            self.bond_stats.as_ref(),
            self.bond_class_means.as_ref().map(|means| {
                (
                    means,
                    (|batch: &Batch| batch.bond_class.as_slice()) as fn(&Batch) -> &[u32],
                )
            }),
            |predictions| predictions.bond.as_ref(),
            |graph| graph.edge_count,
        )
    }

    /// The body shared by [`Trainer::predict`] and its atom and bond
    /// siblings: run the model over every batch of `indices`, pull out
    /// whichever head `select` names, undo `stats`' scaling and (if
    /// `class_means` is given) add each row's class mean back, and slice it
    /// into `rows_in(batch.graph)`-sized chunks.
    #[allow(clippy::type_complexity)]
    fn predict_level(
        &self,
        dataset: &Dataset,
        indices: &[usize],
        stats: Option<&TargetStats>,
        class_means: Option<(&ClassMeans, fn(&Batch) -> &[u32])>,
        select: impl Fn(&Predictions) -> Option<&Tensor>,
        rows_in: impl Fn(&GraphBatch) -> usize,
    ) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(indices.len());
        for batch in dataset.epoch_of(indices, self.config.batch_size, 0) {
            let graph = self.upload(&batch)?;
            let predictions = self.model.forward(&graph, false)?;
            let Some(values) = select(&predictions) else {
                continue;
            };
            let tasks = values.dim(1)?;
            let mut flat = values
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if let Some(stats) = self.scaler_of(stats) {
                stats.invert(&mut flat);
            }
            if let Some((means, classes_of)) = class_means {
                means.add_back(classes_of(&batch), &mut flat);
            }
            for row in 0..rows_in(&batch.graph) {
                out.push(flat[row * tasks..(row + 1) * tasks].to_vec());
            }
        }
        Ok(out)
    }

    fn batch_loss(&self, batch: &Batch, train: bool) -> Result<Tensor> {
        let mut total = Tensor::new(0.0f32, &self.device)?;
        for (loss, _) in self.head_losses(batch, train)?.into_iter().flatten() {
            total = total.add(&loss)?;
        }
        Ok(total)
    }

    fn head_losses(&self, batch: &Batch, train: bool) -> Result<[Option<(Tensor, usize)>; 3]> {
        let graph = self.upload(batch)?;
        let predictions = self.model.forward(&graph, train)?;

        let mut losses = [None, None, None];
        if batch.task_count > 0
            && let Some(values) = &predictions.graph
        {
            losses[0] = Some((
                self.level_loss(
                    values,
                    &batch.targets,
                    &batch.target_mask,
                    batch.graph.graph_count,
                    batch.task_count,
                    self.target_stats.as_ref(),
                    None,
                )?,
                batch
                    .target_mask
                    .iter()
                    .filter(|&&measured| measured)
                    .count(),
            ));
        }
        if batch.atom_task_count > 0
            && let Some(values) = &predictions.atom
        {
            losses[1] = Some((
                self.level_loss(
                    values,
                    &batch.atom_targets,
                    &batch.atom_mask,
                    batch.graph.atom_count,
                    batch.atom_task_count,
                    self.atom_stats.as_ref(),
                    self.atom_class_means
                        .as_ref()
                        .map(|means| (batch.atom_class.as_slice(), means)),
                )?,
                batch.atom_mask.iter().filter(|&&measured| measured).count(),
            ));
        }
        if batch.bond_task_count > 0
            && let Some(values) = &predictions.bond
        {
            losses[2] = Some((
                self.level_loss(
                    values,
                    &batch.bond_targets,
                    &batch.bond_mask,
                    batch.graph.edge_count,
                    batch.bond_task_count,
                    self.bond_stats.as_ref(),
                    self.bond_class_means
                        .as_ref()
                        .map(|means| (batch.bond_class.as_slice(), means)),
                )?,
                batch.bond_mask.iter().filter(|&&measured| measured).count(),
            ));
        }
        Ok(losses)
    }

    /// The masked loss for one head: `values` against `targets`/`mask`
    /// reshaped to `rows * tasks`, with `classes`' per-class mean subtracted
    /// first where given (see [`ClassMeans`]) and `stats`' scaling applied
    /// after, where the objective allows it.
    ///
    /// Shared by every head [`Trainer::batch_loss`] finds active, so a run
    /// that trains more than one at once -- what
    /// [`crate::ml::nn::MolecularNet`] was built to allow -- sums their
    /// losses here rather than needing three copies of this method.
    #[allow(clippy::too_many_arguments)]
    fn level_loss(
        &self,
        values: &Tensor,
        targets: &[f32],
        mask: &[bool],
        rows: usize,
        tasks: usize,
        stats: Option<&TargetStats>,
        classes: Option<(&[u32], &ClassMeans)>,
    ) -> Result<Tensor> {
        let shape = (rows, tasks);
        let scaled;
        let targets = if classes.is_some() || self.scaler_of(stats).is_some() {
            let mut adjusted = targets.to_vec();
            if let Some((classes, means)) = classes {
                means.subtract(classes, &mut adjusted);
            }
            if let Some(stats) = self.scaler_of(stats) {
                stats.apply(&mut adjusted);
            }
            scaled = adjusted;
            scaled.as_slice()
        } else {
            targets
        };
        let targets = Tensor::from_slice(targets, shape, &self.device)?.to_dtype(values.dtype())?;
        let mask = mask_from_flags(mask, shape, &self.device)?;

        match self.config.objective {
            Objective::Regression => masked_mse(values, &targets, &mask),
            Objective::RobustRegression => masked_l1(values, &targets, &mask),
            Objective::Classification => {
                crate::ml::nn::loss::masked_bce_with_logits(values, &targets, &mask)
            }
        }
    }

    /// The target scaler, or none where scaling a target would be wrong.
    fn scaler_of<'s>(&self, stats: Option<&'s TargetStats>) -> Option<&'s TargetStats> {
        match self.config.objective {
            Objective::Classification => None,
            _ => stats.filter(|stats| !stats.is_empty()),
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
        let bonds = match &self.bond_feature_stats {
            Some(stats) => {
                let mut features = batch.graph.bond_features.clone();
                stats.apply(&mut features);
                std::borrow::Cow::Owned(features)
            }
            None => std::borrow::Cow::Borrowed(batch.graph.bond_features.as_slice()),
        };
        let globals = match &self.global_stats {
            Some(stats) if !stats.is_empty() => {
                let mut features = batch.graph.global_features.clone();
                stats.apply(&mut features);
                std::borrow::Cow::Owned(features)
            }
            _ => std::borrow::Cow::Borrowed(batch.graph.global_features.as_slice()),
        };
        GraphTensors::with_bond_features(&batch.graph, &atoms, &bonds, &globals, &self.device)
    }
}

/// How many targets each head -- molecule, atom, bond -- has measured in a
/// batch: what a shard's share of the loss is reckoned against.
fn measured_counts(batch: &Batch) -> [usize; 3] {
    let measured = |flags: &[bool]| flags.iter().filter(|&&measured| measured).count();
    [
        measured(&batch.target_mask),
        measured(&batch.atom_mask),
        measured(&batch.bond_mask),
    ]
}

/// Rescales every gradient by the same factor when their combined norm is too
/// large.
///
/// Per-parameter clipping would change the gradient's direction as well as its
/// size; scaling all of them together is the same step, taken shorter.
pub fn clip_global_norm(gradients: &mut GradStore, variables: &[Var], limit: f64) -> Result<()> {
    let norm = global_norm(gradients, variables)?;
    scale_gradients(gradients, variables, norm, limit)
}

/// The norm of every gradient taken together, as if they were one vector.
pub fn global_norm(gradients: &GradStore, variables: &[Var]) -> Result<f64> {
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
    Ok(total.sqrt())
}

/// [`clip_global_norm`] with the norm already known.
fn scale_gradients(
    gradients: &mut GradStore,
    variables: &[Var],
    norm: f64,
    limit: f64,
) -> Result<()> {
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
