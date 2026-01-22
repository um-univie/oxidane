//! Molecules with labels on them, handed out a minibatch at a time.
//!
//! Two things make a chemical dataset awkward to iterate. The first is that
//! most of the labels are missing: a multi-task set like Tox12 has twelve
//! assays and hardly a molecule measured in all of them, so a target matrix is
//! mostly holes and a loss that reads them as zeros trains the model to
//! predict "inactive" everywhere. Every target here therefore comes with a
//! mask, and [`Batch::target_mask`] says which entries are real.
//!
//! The second is that a molecule is not a row. Shuffling is over molecules,
//! but what a model receives is one concatenated graph, so the shuffle has to
//! happen before the collation rather than over a matrix afterwards -- which
//! is what [`Dataset::epoch`] does.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::ml::dataset::Dataset;
//!
//! let mut dataset = Dataset::new(1);
//! for (smiles, solubility) in [("CCO", -0.24), ("c1ccccc1", -1.64), ("CCCCCC", -3.84)] {
//!     let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
//!     dataset.push(&molecule, &[Some(solubility)]);
//! }
//!
//! let batches: Vec<_> = dataset.epoch(2, 0).collect();
//! assert_eq!(batches.len(), 2);
//! assert_eq!(batches[0].graph.graph_count, 2);
//! assert_eq!(batches[1].graph.graph_count, 1);
//! ```

use crate::ml::batch::GraphBatch;
use crate::ml::shuffle;
use crate::ml::split::{Side, Split};
use crate::molecule::base::MoleculeCore;
use crate::molecule::descriptors::MolecularDescriptors;
use crate::molecule::featurize::{GraphFeatures, graph_features, molecule_features};

/// Featurised molecules and their targets.
///
/// Targets are stored dense with a parallel mask rather than sparse, because a
/// model reads them dense and the mask is a bit per target either way.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    features: Vec<GraphFeatures>,
    globals: Vec<f32>,
    global_width: usize,
    targets: Vec<f32>,
    mask: Vec<bool>,
    task_count: usize,
}

impl Dataset {
    /// An empty dataset for molecules with `task_count` targets each.
    pub fn new(task_count: usize) -> Self {
        Self {
            task_count,
            ..Default::default()
        }
    }

    /// Featurises a molecule and stores it with its targets.
    ///
    /// `targets` must have one entry per task; `None` is a measurement that
    /// was not made, which the loss then skips rather than treats as zero.
    ///
    /// # Panics
    ///
    /// If `targets` is not `task_count` long. A ragged target row is a bug in
    /// the caller's reader, and one that would otherwise show up as a model
    /// that trains but does not learn.
    pub fn push<M: MoleculeCore + ?Sized>(&mut self, molecule: &M, targets: &[Option<f64>]) {
        self.push_features(graph_features(molecule), targets);
    }

    /// [`Dataset::push`] with a row of whole-molecule descriptors as well.
    ///
    /// Message passing only sees a neighbourhood at a time, so a property that
    /// depends on the size or the polarity of the whole molecule has to travel
    /// far to reach the readout. Handing those to the readout directly is what
    /// [`crate::molecule::featurize::molecule_features`] is for.
    ///
    /// # Panics
    ///
    /// If the row is a different width from the ones already stored.
    pub fn push_with_globals<M: MolecularDescriptors + ?Sized>(
        &mut self,
        molecule: &M,
        targets: &[Option<f64>],
    ) {
        let globals = molecule_features(molecule);
        self.push_features_with_globals(graph_features(molecule), &globals, targets);
    }

    /// [`Dataset::push_with_globals`] for a molecule already featurised.
    pub fn push_features_with_globals(
        &mut self,
        features: GraphFeatures,
        globals: &[f32],
        targets: &[Option<f64>],
    ) {
        if self.features.is_empty() {
            self.global_width = globals.len();
        }
        assert_eq!(
            globals.len(),
            self.global_width,
            "a molecule was given {} descriptors in a dataset of {}",
            globals.len(),
            self.global_width
        );
        self.globals.extend_from_slice(globals);
        self.push_features(features, targets);
    }

    /// How wide a row of whole-molecule descriptors is; zero if there are none.
    pub fn global_width(&self) -> usize {
        self.global_width
    }

    /// One molecule's whole-molecule descriptors.
    pub fn globals(&self, index: usize) -> &[f32] {
        if self.global_width == 0 {
            return &[];
        }
        &self.globals[index * self.global_width..(index + 1) * self.global_width]
    }

    /// [`Dataset::push`] for a molecule already featurised.
    pub fn push_features(&mut self, features: GraphFeatures, targets: &[Option<f64>]) {
        assert_eq!(
            targets.len(),
            self.task_count,
            "a molecule was given {} targets in a dataset of {} tasks",
            targets.len(),
            self.task_count
        );
        self.features.push(features);
        for target in targets {
            self.targets.push(target.unwrap_or(0.0) as f32);
            self.mask.push(target.is_some());
        }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn task_count(&self) -> usize {
        self.task_count
    }

    /// One molecule's feature matrices.
    pub fn features(&self, index: usize) -> &GraphFeatures {
        &self.features[index]
    }

    /// One molecule's targets, and which of them were measured.
    pub fn targets(&self, index: usize) -> (&[f32], &[bool]) {
        let span = index * self.task_count..(index + 1) * self.task_count;
        (&self.targets[span.clone()], &self.mask[span])
    }

    /// The molecules named by `indices`, collated into one batch.
    pub fn batch(&self, indices: &[usize]) -> Batch {
        let mut graph = GraphBatch::collate_iter(indices.iter().map(|&index| &self.features[index]));
        if self.global_width > 0 {
            graph.global_width = self.global_width;
            graph.global_features = Vec::with_capacity(indices.len() * self.global_width);
            for &index in indices {
                graph.global_features.extend_from_slice(self.globals(index));
            }
        }
        let mut targets = Vec::with_capacity(indices.len() * self.task_count);
        let mut mask = Vec::with_capacity(indices.len() * self.task_count);
        for &index in indices {
            let (row, row_mask) = self.targets(index);
            targets.extend_from_slice(row);
            mask.extend_from_slice(row_mask);
        }
        Batch {
            graph,
            targets,
            target_mask: mask,
            task_count: self.task_count,
            indices: indices.to_vec(),
        }
    }

    /// Minibatches covering every molecule once, in a seeded shuffled order.
    ///
    /// The last batch is short rather than dropped -- a validation pass that
    /// silently ignored the tail would report a score for a set nobody chose.
    /// Vary `seed` by epoch to get a different order each time.
    pub fn epoch(&self, batch_size: usize, seed: u64) -> Batches<'_> {
        self.epoch_of(&(0..self.len()).collect::<Vec<_>>(), batch_size, seed)
    }

    /// [`Dataset::epoch`] restricted to one side of a [`Split`].
    pub fn epoch_side(
        &self,
        split: &Split,
        side: Side,
        batch_size: usize,
        seed: u64,
    ) -> Batches<'_> {
        let indices = match side {
            Side::Train => &split.train,
            Side::Valid => &split.valid,
            Side::Test => &split.test,
        };
        self.epoch_of(indices, batch_size, seed)
    }

    /// [`Dataset::epoch`] over a chosen subset.
    ///
    /// A seed of zero leaves the order alone, which is what an evaluation pass
    /// wants: shuffling a set nothing is learned from only makes its per-batch
    /// numbers harder to compare between runs.
    pub fn epoch_of(&self, indices: &[usize], batch_size: usize, seed: u64) -> Batches<'_> {
        let mut order = indices.to_vec();
        if seed != 0 {
            shuffle(&mut order, seed);
        }
        Batches {
            dataset: self,
            order,
            position: 0,
            batch_size: batch_size.max(1),
        }
    }
}

/// One minibatch: the concatenated graph, and the targets that go with it.
#[derive(Debug, Clone)]
pub struct Batch {
    /// The molecules laid out as one graph.
    pub graph: GraphBatch,
    /// `graph.graph_count * task_count` targets, row-major by molecule.
    pub targets: Vec<f32>,
    /// Which of `targets` were measured. Same shape.
    pub target_mask: Vec<bool>,
    /// How many targets each molecule has.
    pub task_count: usize,
    /// Which molecules of the dataset these are, in batch order -- for
    /// reporting a prediction against the molecule it came from.
    pub indices: Vec<usize>,
}

impl Batch {
    /// One molecule's targets and mask.
    pub fn targets_of(&self, graph: usize) -> (&[f32], &[bool]) {
        let span = graph * self.task_count..(graph + 1) * self.task_count;
        (&self.targets[span.clone()], &self.target_mask[span])
    }

    /// How many targets in this batch were actually measured.
    ///
    /// The denominator of a masked mean loss. Zero is possible, and a loss
    /// that divides by it should return zero rather than a NaN that then
    /// spreads through every weight in the model.
    pub fn measured_count(&self) -> usize {
        self.target_mask.iter().filter(|&&measured| measured).count()
    }
}

/// The minibatches of one epoch. See [`Dataset::epoch`].
pub struct Batches<'a> {
    dataset: &'a Dataset,
    order: Vec<usize>,
    position: usize,
    batch_size: usize,
}

impl Iterator for Batches<'_> {
    type Item = Batch;

    fn next(&mut self) -> Option<Batch> {
        if self.position >= self.order.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.order.len());
        let batch = self.dataset.batch(&self.order[self.position..end]);
        self.position = end;
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.order.len() - self.position;
        let batches = left.div_ceil(self.batch_size);
        (batches, Some(batches))
    }
}

impl ExactSizeIterator for Batches<'_> {}
