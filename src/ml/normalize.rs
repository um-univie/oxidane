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

use crate::ml::batch::GraphBatch;
use crate::molecule::featurize::{
    ATOM_CONTINUOUS_COLUMNS, ATOM_FEATURE_WIDTH, FEATURE_SCHEMA_VERSION,
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
        let mut accumulator = StatsAccumulator::default();
        for batch in batches {
            accumulator.add(&batch.atom_features);
        }
        accumulator.finish()
    }

    /// [`FeatureStats::fit`] over a raw atom feature matrix.
    pub fn fit_rows(rows: &[f32]) -> Self {
        let mut accumulator = StatsAccumulator::default();
        accumulator.add(rows);
        accumulator.finish()
    }

    /// Standardises the continuous columns of an atom feature matrix in place.
    pub fn apply(&self, rows: &mut [f32]) {
        for (slot, &column) in ATOM_CONTINUOUS_COLUMNS.iter().enumerate() {
            let mean = self.mean[slot];
            let scale = 1.0 / self.deviation[slot];
            let mut index = column;
            while index < rows.len() {
                rows[index] = (rows[index] - mean) * scale;
                index += ATOM_FEATURE_WIDTH;
            }
        }
    }

    /// [`FeatureStats::apply`], refusing statistics measured against a
    /// different schema.
    ///
    /// The columns are identified by position, so applying version 2's
    /// statistics to version 3's matrix would scale whatever now sits where
    /// the mass used to -- quietly, and with no way to tell from the numbers.
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
        for (slot, &column) in ATOM_CONTINUOUS_COLUMNS.iter().enumerate() {
            let mean = self.mean[slot];
            let scale = self.deviation[slot];
            let mut index = column;
            while index < rows.len() {
                rows[index] = rows[index] * scale + mean;
                index += ATOM_FEATURE_WIDTH;
            }
        }
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

/// Welford's running mean and variance, so a training set can be fitted in one
/// pass without holding it or losing precision to a sum of squares.
#[derive(Debug, Default)]
struct StatsAccumulator {
    count: u64,
    mean: Vec<f64>,
    sum_squares: Vec<f64>,
}

impl StatsAccumulator {
    fn add(&mut self, rows: &[f32]) {
        if self.mean.is_empty() {
            self.mean = vec![0.0; ATOM_CONTINUOUS_COLUMNS.len()];
            self.sum_squares = vec![0.0; ATOM_CONTINUOUS_COLUMNS.len()];
        }
        for row in rows.chunks_exact(ATOM_FEATURE_WIDTH) {
            self.count += 1;
            let weight = 1.0 / self.count as f64;
            for (slot, &column) in ATOM_CONTINUOUS_COLUMNS.iter().enumerate() {
                let value = row[column] as f64;
                let previous = self.mean[slot];
                self.mean[slot] += (value - previous) * weight;
                self.sum_squares[slot] += (value - previous) * (value - self.mean[slot]);
            }
        }
    }

    fn finish(self) -> FeatureStats {
        let width = ATOM_CONTINUOUS_COLUMNS.len();
        if self.count == 0 {
            return FeatureStats {
                schema_version: FEATURE_SCHEMA_VERSION,
                mean: vec![0.0; width],
                deviation: vec![1.0; width],
            };
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
        FeatureStats {
            schema_version: FEATURE_SCHEMA_VERSION,
            mean: self.mean.into_iter().map(|value| value as f32).collect(),
            deviation,
        }
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
            for ((value, mean), deviation) in
                row.iter_mut().zip(&self.mean).zip(&self.deviation)
            {
                *value = (*value - mean) / deviation;
            }
        }
    }
}
