//! Putting the feature columns on the same scale, without flattening the ones
//! that are already flat.
//!
//! Almost every column [`crate::molecule::featurize`] produces is one hot: it
//! is zero or one, it means something on its own, and standardising it would
//! turn "this atom is not a nitrogen" from an exact zero into a small negative
//! number that a bias term then has to undo. What actually needs scaling is
//! the handful of columns holding a quantity -- at schema version 2, the
//! atomic mass, which arrives two orders of magnitude larger than everything
//! next to it and dominates the first layer until it is scaled.
//!
//! So [`FeatureStats`] standardises those columns and leaves the rest exactly
//! as they were. Which columns those are comes from the schema, not from
//! looking at the data, because a column that happens to be constant in one
//! batch is not thereby an indicator.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::featurize::graph_features;
//! use oxidane::ml::batch::GraphBatch;
//! use oxidane::ml::normalize::FeatureStats;
//!
//! let batch = GraphBatch::collate(&[
//!     graph_features(&MolecularGraph::from_smiles_single("CCO").unwrap()),
//!     graph_features(&MolecularGraph::from_smiles_single("c1ccccc1Br").unwrap()),
//! ]);
//!
//! let stats = FeatureStats::fit(&batch);
//! let mut scaled = batch.atom_features.clone();
//! stats.apply(&mut scaled);
//! // Bromine's mass no longer swamps the one-hot columns beside it.
//! assert!(scaled.iter().all(|value| value.abs() < 10.0));
//! ```

use std::collections::HashMap;

use crate::ml::batch::GraphBatch;
use crate::molecule::featurize::{
    ATOM_CONTINUOUS_COLUMNS, ATOM_FEATURE_WIDTH, BOND_CONTINUOUS_COLUMNS, BOND_FEATURE_WIDTH,
    FEATURE_SCHEMA_VERSION,
};

/// The mean and standard deviation of each continuous atom column.
///
/// Fit on the training set and applied to all three -- fitting on the whole
/// dataset leaks the test set's distribution into training, which is a small
/// leak but a real one and free to avoid.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureStats {
    /// The schema these statistics were measured against. A model stores this
    /// with its weights; loading them against a different schema is what the
    /// check in [`FeatureStats::apply_checked`] is for.
    pub schema_version: u32,
    /// One entry per column of [`ATOM_CONTINUOUS_COLUMNS`], in that order.
    pub mean: Vec<f32>,
    /// The same, and never zero: a column that does not vary is left alone
    /// rather than divided by nothing.
    pub deviation: Vec<f32>,
}

impl FeatureStats {
    /// Measures the continuous columns of one batch.
    pub fn fit(batch: &GraphBatch) -> Self {
        Self::fit_rows(&batch.atom_features)
    }

    /// Measures the continuous columns of several batches at once, which is
    /// how a training set larger than memory is fitted.
    pub fn fit_batches<'a, I>(batches: I) -> Self
    where
        I: IntoIterator<Item = &'a GraphBatch>,
    {
        let mut accumulator = StatsAccumulator::new(ATOM_FEATURE_WIDTH, &ATOM_CONTINUOUS_COLUMNS);
        for batch in batches {
            accumulator.add(&batch.atom_features);
        }
        let (mean, deviation) = accumulator.finish();
        Self {
            schema_version: FEATURE_SCHEMA_VERSION,
            mean,
            deviation,
        }
    }

    /// [`FeatureStats::fit`] over a raw atom feature matrix.
    pub fn fit_rows(rows: &[f32]) -> Self {
        let mut accumulator = StatsAccumulator::new(ATOM_FEATURE_WIDTH, &ATOM_CONTINUOUS_COLUMNS);
        accumulator.add(rows);
        let (mean, deviation) = accumulator.finish();
        Self {
            schema_version: FEATURE_SCHEMA_VERSION,
            mean,
            deviation,
        }
    }

    /// Standardises the continuous columns of an atom feature matrix in place.
    pub fn apply(&self, rows: &mut [f32]) {
        apply_continuous(
            rows,
            ATOM_FEATURE_WIDTH,
            &ATOM_CONTINUOUS_COLUMNS,
            &self.mean,
            &self.deviation,
        );
    }

    /// [`FeatureStats::apply`], refusing statistics measured against a
    /// different schema.
    ///
    /// The columns are identified by position, so applying an older schema's
    /// statistics to a newer matrix would scale whatever now sits where an
    /// old column used to -- quietly, and with no way to tell from the
    /// numbers.
    pub fn apply_checked(&self, rows: &mut [f32]) -> Result<(), SchemaMismatch> {
        if self.schema_version != FEATURE_SCHEMA_VERSION {
            return Err(SchemaMismatch {
                expected: FEATURE_SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        self.apply(rows);
        Ok(())
    }

    /// Undoes [`FeatureStats::apply`], for looking at what a model was given.
    pub fn invert(&self, rows: &mut [f32]) {
        invert_continuous(
            rows,
            ATOM_FEATURE_WIDTH,
            &ATOM_CONTINUOUS_COLUMNS,
            &self.mean,
            &self.deviation,
        );
    }
}

/// The mean and standard deviation of the one continuous bond column (the 3D
/// distance -- see [`crate::molecule::featurize::graph_features_with_geometry`]).
/// Structurally [`FeatureStats`]'s twin, over bond rows instead of atom rows;
/// every bond feature besides the distance is a 0/1 indicator, the same
/// reason `FeatureStats` leaves everything but its continuous columns alone.
#[derive(Debug, Clone, PartialEq)]
pub struct BondFeatureStats {
    pub schema_version: u32,
    /// One entry per column of [`BOND_CONTINUOUS_COLUMNS`], in that order.
    pub mean: Vec<f32>,
    pub deviation: Vec<f32>,
}

impl BondFeatureStats {
    /// Measures the continuous columns of one batch.
    pub fn fit(batch: &GraphBatch) -> Self {
        Self::fit_rows(&batch.bond_features)
    }

    /// [`BondFeatureStats::fit`] over several batches at once.
    pub fn fit_batches<'a, I>(batches: I) -> Self
    where
        I: IntoIterator<Item = &'a GraphBatch>,
    {
        let mut accumulator = StatsAccumulator::new(BOND_FEATURE_WIDTH, &BOND_CONTINUOUS_COLUMNS);
        for batch in batches {
            accumulator.add(&batch.bond_features);
        }
        let (mean, deviation) = accumulator.finish();
        Self {
            schema_version: FEATURE_SCHEMA_VERSION,
            mean,
            deviation,
        }
    }

    /// [`BondFeatureStats::fit`] over a raw bond feature matrix.
    pub fn fit_rows(rows: &[f32]) -> Self {
        let mut accumulator = StatsAccumulator::new(BOND_FEATURE_WIDTH, &BOND_CONTINUOUS_COLUMNS);
        accumulator.add(rows);
        let (mean, deviation) = accumulator.finish();
        Self {
            schema_version: FEATURE_SCHEMA_VERSION,
            mean,
            deviation,
        }
    }

    /// Standardises the continuous columns of a bond feature matrix in place.
    pub fn apply(&self, rows: &mut [f32]) {
        apply_continuous(
            rows,
            BOND_FEATURE_WIDTH,
            &BOND_CONTINUOUS_COLUMNS,
            &self.mean,
            &self.deviation,
        );
    }

    /// [`FeatureStats::apply_checked`]'s counterpart for bond features.
    pub fn apply_checked(&self, rows: &mut [f32]) -> Result<(), SchemaMismatch> {
        if self.schema_version != FEATURE_SCHEMA_VERSION {
            return Err(SchemaMismatch {
                expected: FEATURE_SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        self.apply(rows);
        Ok(())
    }

    /// Undoes [`BondFeatureStats::apply`].
    pub fn invert(&self, rows: &mut [f32]) {
        invert_continuous(
            rows,
            BOND_FEATURE_WIDTH,
            &BOND_CONTINUOUS_COLUMNS,
            &self.mean,
            &self.deviation,
        );
    }
}

/// Statistics from one schema were applied to a matrix from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaMismatch {
    pub expected: u32,
    pub found: u32,
}

impl std::fmt::Display for SchemaMismatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "feature statistics were measured against schema version {}, but this build produces version {}",
            self.found, self.expected
        )
    }
}

impl std::error::Error for SchemaMismatch {}

/// Scales `rows` (row-major, `width` wide) in place, standardising the
/// columns named by `columns` against `mean`/`deviation` (each the same
/// length as `columns`, in that order). Shared by [`FeatureStats::apply`] and
/// [`BondFeatureStats::apply`].
fn apply_continuous(
    rows: &mut [f32],
    width: usize,
    columns: &[usize],
    mean: &[f32],
    deviation: &[f32],
) {
    for (slot, &column) in columns.iter().enumerate() {
        let mean = mean[slot];
        let scale = 1.0 / deviation[slot];
        let mut index = column;
        while index < rows.len() {
            rows[index] = (rows[index] - mean) * scale;
            index += width;
        }
    }
}

/// Undoes [`apply_continuous`].
fn invert_continuous(
    rows: &mut [f32],
    width: usize,
    columns: &[usize],
    mean: &[f32],
    deviation: &[f32],
) {
    for (slot, &column) in columns.iter().enumerate() {
        let mean = mean[slot];
        let scale = deviation[slot];
        let mut index = column;
        while index < rows.len() {
            rows[index] = rows[index] * scale + mean;
            index += width;
        }
    }
}

/// Welford's running mean and variance, so a training set can be fitted in one
/// pass without holding it or losing precision to a sum of squares. Shared by
/// [`FeatureStats`] and [`BondFeatureStats`]'s fits, parameterised by which
/// columns of how wide a row are the continuous ones.
#[derive(Debug)]
struct StatsAccumulator {
    width: usize,
    columns: Vec<usize>,
    count: u64,
    mean: Vec<f64>,
    sum_squares: Vec<f64>,
}

impl StatsAccumulator {
    fn new(width: usize, columns: &[usize]) -> Self {
        Self {
            width,
            columns: columns.to_vec(),
            count: 0,
            mean: vec![0.0; columns.len()],
            sum_squares: vec![0.0; columns.len()],
        }
    }

    fn add(&mut self, rows: &[f32]) {
        for row in rows.chunks_exact(self.width) {
            self.count += 1;
            let weight = 1.0 / self.count as f64;
            for (slot, &column) in self.columns.iter().enumerate() {
                let value = row[column] as f64;
                let previous = self.mean[slot];
                self.mean[slot] += (value - previous) * weight;
                self.sum_squares[slot] += (value - previous) * (value - self.mean[slot]);
            }
        }
    }

    /// `(mean, deviation)`, one entry per column, in `columns`' order.
    fn finish(self) -> (Vec<f32>, Vec<f32>) {
        let width = self.columns.len();
        if self.count == 0 {
            return (vec![0.0; width], vec![1.0; width]);
        }
        let deviation = (0..width)
            .map(|slot| {
                let variance = self.sum_squares[slot] / self.count as f64;
                let value = variance.sqrt() as f32;
                // A column that does not vary is passed through rather than
                // divided by zero; centring it is still worth doing.
                if value > f32::EPSILON { value } else { 1.0 }
            })
            .collect();
        (
            self.mean.into_iter().map(|value| value as f32).collect(),
            deviation,
        )
    }
}

/// The mean and deviation of every whole-molecule descriptor column.
///
/// Unlike the atom features, *all* of these are quantities -- a molecular
/// weight, a polar surface area, a count of rings -- so all of them are
/// scaled. A model handed molecular weight beside a fraction between zero and
/// one spends its first layer undoing the difference in size.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GlobalStats {
    pub schema_version: u32,
    pub mean: Vec<f32>,
    pub deviation: Vec<f32>,
}

impl GlobalStats {
    /// Measures the descriptor rows of a batch. Empty if it carried none.
    pub fn fit(batch: &GraphBatch) -> Self {
        Self::fit_rows(&batch.global_features, batch.global_width)
    }

    /// [`GlobalStats::fit`] over a raw matrix and its width.
    pub fn fit_rows(rows: &[f32], width: usize) -> Self {
        if width == 0 || rows.is_empty() {
            return Self {
                schema_version: FEATURE_SCHEMA_VERSION,
                ..Default::default()
            };
        }
        let count = rows.len() / width;
        let mut mean = vec![0.0f64; width];
        let mut sum_squares = vec![0.0f64; width];
        for (seen, row) in rows.chunks_exact(width).enumerate() {
            let weight = 1.0 / (seen + 1) as f64;
            for column in 0..width {
                let value = row[column] as f64;
                let previous = mean[column];
                mean[column] += (value - previous) * weight;
                sum_squares[column] += (value - previous) * (value - mean[column]);
            }
        }
        Self {
            schema_version: FEATURE_SCHEMA_VERSION,
            mean: mean.iter().map(|&value| value as f32).collect(),
            deviation: (0..width)
                .map(|column| {
                    let value = (sum_squares[column] / count.max(1) as f64).sqrt() as f32;
                    if value > f32::EPSILON { value } else { 1.0 }
                })
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Standardises a descriptor matrix in place.
    pub fn apply(&self, rows: &mut [f32]) {
        if self.mean.is_empty() {
            return;
        }
        let width = self.mean.len();
        for row in rows.chunks_exact_mut(width) {
            for ((value, mean), deviation) in row.iter_mut().zip(&self.mean).zip(&self.deviation) {
                *value = (*value - mean) / deviation;
            }
        }
    }
}

/// The mean and deviation of each regression target.
///
/// A model's last layer starts near zero, so on raw targets it begins by
/// predicting nothing and the first epochs go on finding the mean. FreeSolv's
/// hydration free energies average -3.8 kcal/mol, so a squared error starts
/// near 29 and the gradient that reduces it is the same one that would fit the
/// chemistry -- the two are not separable, and the fitting starts late.
/// Standardising the targets puts the useful gradient first.
///
/// It also makes a multi-task loss mean anything. Two tasks measured in
/// different units contribute in proportion to their spread, so the one
/// recorded in larger numbers takes the model whether or not it is the one
/// that matters.
///
/// Fit on the training set only, like the feature statistics, and applied to
/// the predictions in reverse before anything is reported -- a root mean
/// squared error in units of standard deviations is not the number anyone
/// wants.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TargetStats {
    pub mean: Vec<f32>,
    pub deviation: Vec<f32>,
}

impl TargetStats {
    /// Measures the targets a dataset holds, over the rows named by `indices`
    /// and over the entries their masks keep.
    ///
    /// Missing measurements are skipped rather than counted as zero, which
    /// would drag both statistics toward the origin in proportion to how
    /// sparse the task is.
    pub fn fit(dataset: &crate::ml::dataset::Dataset, indices: &[usize]) -> Self {
        Self::fit_rows(
            indices.iter().flat_map(|&index| {
                let (targets, mask) = dataset.targets(index);
                targets.iter().copied().zip(mask.iter().copied())
            }),
            dataset.task_count(),
        )
    }

    /// [`TargetStats::fit`] over a dataset's per-atom targets instead.
    pub fn fit_atom(dataset: &crate::ml::dataset::Dataset, indices: &[usize]) -> Self {
        Self::fit_rows(
            indices.iter().flat_map(|&index| {
                let (targets, mask) = dataset.atom_targets(index);
                targets.iter().copied().zip(mask.iter().copied())
            }),
            dataset.atom_task_count(),
        )
    }

    /// [`TargetStats::fit`] over a dataset's per-bond targets instead.
    pub fn fit_bond(dataset: &crate::ml::dataset::Dataset, indices: &[usize]) -> Self {
        Self::fit_rows(
            indices.iter().flat_map(|&index| {
                let (targets, mask) = dataset.bond_targets(index);
                targets.iter().copied().zip(mask.iter().copied())
            }),
            dataset.bond_task_count(),
        )
    }

    /// [`TargetStats::fit_atom`], but fit over the residual after `means`'
    /// per-class mean is subtracted from each value first.
    ///
    /// What `--residual`-style training actually asks the model to predict
    /// is that residual, not the raw value, so these are the statistics that
    /// scaling needs to match -- fitting on the raw value instead would
    /// centre and scale for a distribution the model never sees.
    pub fn fit_atom_residual(
        dataset: &crate::ml::dataset::Dataset,
        indices: &[usize],
        means: &ClassMeans,
    ) -> Self {
        Self::fit_rows(
            indices.iter().flat_map(|&index| {
                let classes = dataset.atom_classes(index);
                let (targets, mask) = dataset.atom_targets(index);
                classes
                    .iter()
                    .zip(targets.iter().zip(mask.iter()))
                    .map(|(&class, (&value, &measured))| (value - means.mean_of(class), measured))
            }),
            dataset.atom_task_count(),
        )
    }

    /// [`TargetStats::fit_atom_residual`] over a dataset's per-bond targets
    /// instead.
    pub fn fit_bond_residual(
        dataset: &crate::ml::dataset::Dataset,
        indices: &[usize],
        means: &ClassMeans,
    ) -> Self {
        Self::fit_rows(
            indices.iter().flat_map(|&index| {
                let classes = dataset.bond_classes(index);
                let (targets, mask) = dataset.bond_targets(index);
                classes
                    .iter()
                    .zip(targets.iter().zip(mask.iter()))
                    .map(|(&class, (&value, &measured))| (value - means.mean_of(class), measured))
            }),
            dataset.bond_task_count(),
        )
    }

    /// The mean and deviation of `tasks` columns, read off a flat stream of
    /// `(value, measured)` pairs in row-major order -- a task's own column is
    /// every `tasks`-th entry.
    ///
    /// Shared by [`TargetStats::fit`] and its atom and bond siblings: none of
    /// them need to know how many rows one molecule contributes, only that
    /// the rows are `tasks` wide.
    fn fit_rows(rows: impl Iterator<Item = (f32, bool)>, tasks: usize) -> Self {
        let mut count = vec![0u64; tasks];
        let mut mean = vec![0.0f64; tasks];
        let mut sum_squares = vec![0.0f64; tasks];
        for (position, (value, measured)) in rows.enumerate() {
            if !measured {
                continue;
            }
            let task = position % tasks;
            let value = value as f64;
            count[task] += 1;
            let previous = mean[task];
            mean[task] += (value - previous) / count[task] as f64;
            sum_squares[task] += (value - previous) * (value - mean[task]);
        }
        Self {
            mean: mean.iter().map(|&value| value as f32).collect(),
            deviation: (0..tasks)
                .map(|task| {
                    if count[task] == 0 {
                        return 1.0;
                    }
                    let value = (sum_squares[task] / count[task] as f64).sqrt() as f32;
                    if value > f32::EPSILON { value } else { 1.0 }
                })
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Standardises a row-major target matrix in place.
    pub fn apply(&self, targets: &mut [f32]) {
        if self.mean.is_empty() {
            return;
        }
        for row in targets.chunks_exact_mut(self.mean.len()) {
            for ((value, mean), deviation) in row.iter_mut().zip(&self.mean).zip(&self.deviation) {
                *value = (*value - mean) / deviation;
            }
        }
    }

    /// Puts predictions back into the units the targets were given in.
    pub fn invert(&self, predictions: &mut [f32]) {
        if self.mean.is_empty() {
            return;
        }
        for row in predictions.chunks_exact_mut(self.mean.len()) {
            for ((value, mean), deviation) in row.iter_mut().zip(&self.mean).zip(&self.deviation) {
                *value = *value * deviation + mean;
            }
        }
    }
}

/// A per-class mean, subtracted from a target before the ordinary
/// standardisation in [`TargetStats`] runs and added back to a prediction
/// after it.
///
/// The class is whatever [`crate::molecule::featurize::atom_class`] or
/// [`crate::molecule::featurize::bond_class`] says it is -- a coarse grouping
/// ("this is an aromatic C-H", "this is a C-C single bond") that accounts for
/// most of a raw bond dissociation energy's spread before any
/// environment-specific chemistry comes into it. Predicting the residual on
/// top of this, rather than the raw value, is the training trick behind
/// ALFABET's near-chemical-accuracy bond dissociation energy model (St John,
/// Guan, Kim, Kim & Paton 2020, *Nat. Commun.* 11, 2328): "these predictions
/// were added to a single mean BDE value for each bond class."
///
/// Fit on the training set alone, like every other statistic in this module.
#[derive(Debug, Clone, Default)]
pub struct ClassMeans {
    means: HashMap<u32, f32>,
    /// The plain mean over every measured entry, used for a class the fit
    /// never saw -- a validation or test row is not thereby unscoreable.
    fallback: f32,
}

impl ClassMeans {
    /// Fits over a dataset's per-atom targets and classes.
    ///
    /// # Panics
    ///
    /// If the dataset carries more than one atom task -- a class is a
    /// property of the atom, not of one of several tasks it might be given,
    /// and there is nothing sound to average it against a second task with.
    pub fn fit_atom(dataset: &crate::ml::dataset::Dataset, indices: &[usize]) -> Self {
        assert!(
            dataset.atom_task_count() <= 1,
            "ClassMeans::fit_atom assumes a single atom task, found {}",
            dataset.atom_task_count()
        );
        Self::fit(indices.iter().flat_map(|&index| {
            let classes = dataset.atom_classes(index);
            let (targets, mask) = dataset.atom_targets(index);
            classes
                .iter()
                .copied()
                .zip(targets.iter().copied().zip(mask.iter().copied()))
        }))
    }

    /// [`ClassMeans::fit_atom`] over a dataset's per-bond targets and classes
    /// instead.
    ///
    /// # Panics
    ///
    /// If the dataset carries more than one bond task -- see
    /// [`ClassMeans::fit_atom`].
    pub fn fit_bond(dataset: &crate::ml::dataset::Dataset, indices: &[usize]) -> Self {
        assert!(
            dataset.bond_task_count() <= 1,
            "ClassMeans::fit_bond assumes a single bond task, found {}",
            dataset.bond_task_count()
        );
        Self::fit(indices.iter().flat_map(|&index| {
            let classes = dataset.bond_classes(index);
            let (targets, mask) = dataset.bond_targets(index);
            classes
                .iter()
                .copied()
                .zip(targets.iter().copied().zip(mask.iter().copied()))
        }))
    }

    fn fit(rows: impl Iterator<Item = (u32, (f32, bool))>) -> Self {
        let mut sum: HashMap<u32, f64> = HashMap::new();
        let mut count: HashMap<u32, u64> = HashMap::new();
        let mut total = 0.0f64;
        let mut total_count = 0u64;
        for (class, (value, measured)) in rows {
            if !measured {
                continue;
            }
            *sum.entry(class).or_insert(0.0) += value as f64;
            *count.entry(class).or_insert(0) += 1;
            total += value as f64;
            total_count += 1;
        }
        let fallback = if total_count > 0 {
            (total / total_count as f64) as f32
        } else {
            0.0
        };
        let means = sum
            .into_iter()
            .map(|(class, class_total)| (class, (class_total / count[&class] as f64) as f32))
            .collect();
        Self { means, fallback }
    }

    /// The mean for a class, or the fallback if the fit never saw it.
    fn mean_of(&self, class: u32) -> f32 {
        self.means.get(&class).copied().unwrap_or(self.fallback)
    }

    /// Subtracts each row's class mean from it, in place.
    pub fn subtract(&self, classes: &[u32], values: &mut [f32]) {
        for (value, &class) in values.iter_mut().zip(classes) {
            *value -= self.mean_of(class);
        }
    }

    /// Undoes [`ClassMeans::subtract`].
    pub fn add_back(&self, classes: &[u32], values: &mut [f32]) {
        for (value, &class) in values.iter_mut().zip(classes) {
            *value += self.mean_of(class);
        }
    }
}
