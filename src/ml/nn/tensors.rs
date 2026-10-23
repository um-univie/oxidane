//! A [`GraphBatch`] as tensors, with the two scatters message passing is made of.
//!
//! Nothing here is a molecule any more. What a message-passing layer needs from
//! a batch is four index arrays and two matrices, and the only operations it
//! performs on them are "gather a row per edge" and "add every edge's row into
//! the atom it points at". Both are `index_select` and `index_add`, both are
//! differentiable, and having them named once means a layer never touches an
//! index array itself.

use candle_core::{Device, IndexOp, Result, Tensor};

use crate::ml::batch::GraphBatch;
use crate::molecule::featurize::{ATOM_FEATURE_WIDTH, BOND_FEATURE_WIDTH};

/// One batch of molecules on a device.
#[derive(Debug, Clone)]
pub struct GraphTensors {
    /// `[atom_count, ATOM_FEATURE_WIDTH]`.
    pub atom_features: Tensor,
    /// `[edge_count, BOND_FEATURE_WIDTH]`.
    pub bond_features: Tensor,
    /// `[edge_count]`, the atom each edge leaves.
    pub source: Tensor,
    /// `[edge_count]`, the atom each edge arrives at.
    pub target: Tensor,
    /// `[edge_count]`, the edge running the other way. See
    /// [`GraphBatch::reverse_edges`].
    pub reverse: Tensor,
    /// `[edge_count, 1]`, one where the reverse edge exists and zero where it
    /// does not, so that a subtraction can be multiplied away instead of
    /// branched on.
    pub reverse_mask: Tensor,
    /// Whether every edge has its reverse, so that `reverse_mask` is all ones
    /// and a layer can skip multiplying by it.
    pub reverse_complete: bool,
    /// `[atom_count]`, the molecule each atom belongs to.
    pub batch: Tensor,
    /// `[graph_count, 1]`, how many atoms each molecule has, floored at one so
    /// that a mean never divides by zero.
    pub atom_counts: Tensor,
    /// `[graph_count, global_width]`, or `None` when the batch carried no
    /// whole-molecule descriptors.
    pub global_features: Option<Tensor>,
    pub atom_count: usize,
    pub edge_count: usize,
    pub graph_count: usize,
    pub device: Device,
}

impl GraphTensors {
    /// Uploads a batch, using its features as they are.
    pub fn new(batch: &GraphBatch, device: &Device) -> Result<Self> {
        Self::with_atom_features(batch, &batch.atom_features, device)
    }

    /// Uploads a batch with atom features the caller has already scaled.
    ///
    /// This is the entry point a normalised pipeline uses: apply
    /// [`crate::ml::normalize::FeatureStats`] to a copy of
    /// [`GraphBatch::atom_features`] and hand the result in, rather than
    /// mutating the batch.
    pub fn with_atom_features(
        batch: &GraphBatch,
        atom_features: &[f32],
        device: &Device,
    ) -> Result<Self> {
        Self::with_features(batch, atom_features, &batch.global_features, device)
    }

    /// [`GraphTensors::with_atom_features`] with the whole-molecule
    /// descriptors scaled by the caller as well.
    pub fn with_features(
        batch: &GraphBatch,
        atom_features: &[f32],
        global_features: &[f32],
        device: &Device,
    ) -> Result<Self> {
        Self::with_bond_features(
            batch,
            atom_features,
            &batch.bond_features,
            global_features,
            device,
        )
    }

    /// [`GraphTensors::with_features`] with the bond features scaled by the
    /// caller as well -- see [`crate::ml::normalize::BondFeatureStats`].
    pub fn with_bond_features(
        batch: &GraphBatch,
        atom_features: &[f32],
        bond_features: &[f32],
        global_features: &[f32],
        device: &Device,
    ) -> Result<Self> {
        let (reverse, present) = batch.reverse_edges();
        let mask: Vec<f32> = present
            .iter()
            .map(|&found| if found { 1.0 } else { 0.0 })
            .collect();
        let counts: Vec<f32> = (0..batch.graph_count)
            .map(|graph| batch.graph_atom_count(graph).max(1) as f32)
            .collect();

        Ok(Self {
            atom_features: Tensor::from_slice(
                atom_features,
                (batch.atom_count, ATOM_FEATURE_WIDTH),
                device,
            )?,
            bond_features: Tensor::from_slice(
                bond_features,
                (batch.edge_count, BOND_FEATURE_WIDTH),
                device,
            )?,
            source: Tensor::from_slice(&batch.edge_sources(), batch.edge_count, device)?,
            target: Tensor::from_slice(&batch.edge_targets(), batch.edge_count, device)?,
            reverse: Tensor::from_slice(&reverse, batch.edge_count, device)?,
            reverse_mask: Tensor::from_slice(&mask, (batch.edge_count, 1), device)?,
            reverse_complete: present.iter().all(|&found| found),
            batch: Tensor::from_slice(batch.batch.as_slice(), batch.atom_count, device)?,
            atom_counts: Tensor::from_slice(&counts, (batch.graph_count, 1), device)?,
            global_features: if batch.global_width == 0 || global_features.is_empty() {
                None
            } else {
                Some(Tensor::from_slice(
                    global_features,
                    (batch.graph_count, batch.global_width),
                    device,
                )?)
            },
            atom_count: batch.atom_count,
            edge_count: batch.edge_count,
            graph_count: batch.graph_count,
            device: device.clone(),
        })
    }

    /// The rows of `atoms` belonging to the atom each edge leaves.
    ///
    /// `atoms` is `[atom_count, width]`; the result is `[edge_count, width]`.
    pub fn gather_sources(&self, atoms: &Tensor) -> Result<Tensor> {
        atoms.index_select(&self.source, 0)
    }

    /// The rows of `atoms` belonging to the atom each edge arrives at.
    pub fn gather_targets(&self, atoms: &Tensor) -> Result<Tensor> {
        atoms.index_select(&self.target, 0)
    }

    /// Each edge's row, taken from the edge running the other way.
    pub fn gather_reverse(&self, edges: &Tensor) -> Result<Tensor> {
        edges.index_select(&self.reverse, 0)
    }

    /// Adds every edge's row into the atom it arrives at.
    ///
    /// `edges` is `[edge_count, width]`; the result is `[atom_count, width]`.
    /// This is the aggregation half of message passing, and the reason
    /// featurization gives every bond twice.
    pub fn scatter_to_atoms(&self, edges: &Tensor) -> Result<Tensor> {
        let width = edges.dim(1)?;
        Tensor::zeros((self.atom_count, width), edges.dtype(), &self.device)?.index_add(
            &self.target,
            edges,
            0,
        )
    }

    /// Adds every atom's row into the molecule it belongs to.
    ///
    /// `atoms` is `[atom_count, width]`; the result is `[graph_count, width]`.
    pub fn pool_sum(&self, atoms: &Tensor) -> Result<Tensor> {
        let width = atoms.dim(1)?;
        Tensor::zeros((self.graph_count, width), atoms.dtype(), &self.device)?.index_add(
            &self.batch,
            atoms,
            0,
        )
    }

    /// [`GraphTensors::pool_sum`] divided by each molecule's atom count.
    ///
    /// A molecule with no atom rows -- a hydrogen-only fragment has none --
    /// divides by one rather than by zero, and so pools to the zeros its
    /// empty sum already is.
    pub fn pool_mean(&self, atoms: &Tensor) -> Result<Tensor> {
        self.pool_sum(atoms)?
            .broadcast_div(&self.atom_counts.to_dtype(atoms.dtype())?)
    }

    /// The readout named by `how`.
    ///
    /// There is no segment maximum here. Candle has no differentiable one, and
    /// building it out of padding costs more than the pooling it replaces;
    /// [`GraphBatch::segment_max`] is available for the places -- reporting,
    /// analysis -- where no gradient has to flow back through it.
    pub fn pool(&self, atoms: &Tensor, how: Pooling) -> Result<Tensor> {
        match how {
            Pooling::Sum => self.pool_sum(atoms),
            Pooling::Mean => self.pool_mean(atoms),
        }
    }

    /// The rows of a per-molecule tensor, one per atom.
    ///
    /// The opposite of pooling: gives every atom the vector of the molecule it
    /// is in, so that a per-atom head can see what the whole molecule looks
    /// like.
    pub fn broadcast_to_atoms(&self, graphs: &Tensor) -> Result<Tensor> {
        graphs.index_select(&self.batch, 0)
    }

    /// The atom rows of one molecule of the batch, for reading a prediction
    /// back out against the atoms it was made for.
    pub fn graph_rows(&self, atoms: &Tensor, batch: &GraphBatch, graph: usize) -> Result<Tensor> {
        let rows = batch.graph_rows(graph);
        atoms.i(rows)
    }
}

/// How a molecule's atom vectors become one vector for the molecule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// Adds them. Sensitive to size, which is right for an extensive property
    /// -- a molecule twice the size has twice the atoms contributing to its
    /// molar refractivity.
    #[default]
    Sum,
    /// Averages them. Right for an intensive property, where size is a
    /// confound rather than a signal.
    Mean,
}
