//! An encoder with heads on it: what actually gets trained.
//!
//! The three heads answer the three questions a molecular graph can be asked.
//! A *graph* head predicts one number for the molecule -- solubility, a
//! toxicity call. An *atom* head predicts one per atom: a partial charge, a
//! chemical shift, whether a position is a site of metabolism. A *bond* head
//! predicts one per bond: a rotational barrier, whether a bond is the one a
//! retrosynthesis should cut.
//!
//! All three read the same atom vectors, so a model can carry any combination
//! of them and the encoder is trained by all of them at once.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::ml::batch::GraphBatch;
use crate::ml::nn::layers::{Mlp, MlpConfig};
use crate::ml::nn::mpnn::{Encoder, EncoderConfig, EncoderKind};
use crate::ml::nn::tensors::{GraphTensors, Pooling};
use crate::molecule::featurize::BOND_FEATURE_WIDTH;

/// What a model is asked to predict, and how much network to do it with.
///
/// The default predicts nothing: every head is empty, which is a model that
/// only encodes. Give it a head, or start from one of the constructors below.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub kind: EncoderKind,
    /// How a molecule's atoms become one vector for it.
    pub pooling: Pooling,
    /// The widths of the head that predicts per molecule, output last. Empty
    /// for a model that does not predict per molecule.
    pub graph_head: Vec<usize>,
    /// The same, per atom.
    pub atom_head: Vec<usize>,
    /// The same, per bond.
    pub bond_head: Vec<usize>,
    /// How wide a row of whole-molecule descriptors is, or zero for none.
    ///
    /// When it is not zero the graph head reads the pooled atom vectors with
    /// that row concatenated on. Message passing sees a neighbourhood at a
    /// time, so a property that turns on the size or the polarity of the whole
    /// molecule reaches the readout the long way round or not at all;
    /// [`crate::molecule::featurize::molecule_features`] hands it over
    /// directly. It changes only the graph head -- an atom's prediction is
    /// still about that atom.
    pub global_width: usize,
    pub head: MlpConfig,
}

impl ModelConfig {
    /// A model predicting `tasks` numbers per molecule, through one hidden
    /// layer.
    pub fn graph_tasks(tasks: usize) -> Self {
        let hidden = EncoderConfig::default().hidden;
        Self {
            graph_head: vec![hidden, tasks],
            ..Default::default()
        }
    }

    /// A model predicting `tasks` numbers per atom.
    pub fn atom_tasks(tasks: usize) -> Self {
        let hidden = EncoderConfig::default().hidden;
        Self {
            atom_head: vec![hidden, tasks],
            ..Default::default()
        }
    }

    /// A model predicting `tasks` numbers per bond.
    pub fn bond_tasks(tasks: usize) -> Self {
        let hidden = EncoderConfig::default().hidden;
        Self {
            bond_head: vec![hidden, tasks],
            ..Default::default()
        }
    }
}

/// An encoder and whichever heads were asked for.
///
/// ```
/// # fn main() -> candle_core::Result<()> {
/// use candle_core::{DType, Device};
/// use candle_nn::{VarBuilder, VarMap};
/// use oxidane::prelude::*;
/// use oxidane::molecule::featurize::graph_features;
/// use oxidane::ml::batch::GraphBatch;
/// use oxidane::ml::nn::{EncoderConfig, GraphTensors, ModelConfig, MolecularNet};
///
/// let device = Device::Cpu;
/// let varmap = VarMap::new();
/// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
///
/// let config = ModelConfig {
///     encoder: EncoderConfig { hidden: 32, depth: 2, ..Default::default() },
///     graph_head: vec![32, 1],
///     atom_head: vec![32, 1],
///     ..ModelConfig::graph_tasks(1)
/// };
/// let model = MolecularNet::new(config, vb)?;
///
/// let batch = GraphBatch::collate(&[
///     graph_features(&MolecularGraph::from_smiles_single("CCO").unwrap()),
///     graph_features(&MolecularGraph::from_smiles_single("c1ccccc1").unwrap()),
/// ]);
/// let graph = GraphTensors::new(&batch, &device)?;
/// let predictions = model.forward(&graph, false)?;
///
/// assert_eq!(predictions.graph.unwrap().dims(), &[2, 1]);
/// // CCO: 3 heavy + 6 H = 9. Benzene: 6 heavy + 6 H = 12. 21 atom rows total.
/// assert_eq!(predictions.atom.unwrap().dims(), &[21, 1]);
/// assert!(predictions.bond.is_none());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MolecularNet {
    encoder: Encoder,
    graph_head: Option<Mlp>,
    atom_head: Option<Mlp>,
    bond_head: Option<Mlp>,
    config: ModelConfig,
}

impl MolecularNet {
    pub fn new(config: ModelConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(config.kind, config.encoder, vb.pp("encoder"))?;
        let width = encoder.width();

        let head = |sizes: &[usize], name: &str, input: usize| -> Result<Option<Mlp>> {
            if sizes.is_empty() {
                return Ok(None);
            }
            let mut widths = Vec::with_capacity(sizes.len() + 1);
            widths.push(input);
            widths.extend_from_slice(sizes);
            Ok(Some(Mlp::new(&widths, config.head, vb.pp(name))?))
        };

        Ok(Self {
            graph_head: head(
                &config.graph_head,
                "graph_head",
                width * config.pooling.width_factor() + config.global_width,
            )?,
            atom_head: head(&config.atom_head, "atom_head", width)?,
            // See `bond_representation` for what a bond head reads.
            bond_head: head(
                &config.bond_head,
                "bond_head",
                2 * width + BOND_FEATURE_WIDTH + if encoder.has_edge_states() { width } else { 0 },
            )?,
            encoder,
            config,
        })
    }

    /// The atom vectors, before any head sees them.
    ///
    /// This is the molecule's learned representation, and what to take from a
    /// trained model when the point was the embedding rather than the label.
    pub fn encode(&self, graph: &GraphTensors, train: bool) -> Result<Tensor> {
        self.encoder.forward(graph, train)
    }

    /// [`MolecularNet::encode`], and the encoder's edge messages where it
    /// keeps them -- what [`MolecularNet::predict_with_edges`] takes.
    pub fn encode_with_edges(
        &self,
        graph: &GraphTensors,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        self.encoder.forward_with_edges(graph, train)
    }

    /// One vector per molecule, `[graph_count, width]` -- twice the width for
    /// [`Pooling::SumMean`].
    pub fn encode_molecules(&self, graph: &GraphTensors, train: bool) -> Result<Tensor> {
        let atoms = self.encode(graph, train)?;
        graph.pool(&atoms, self.config.pooling)
    }

    pub fn forward(&self, graph: &GraphTensors, train: bool) -> Result<Predictions> {
        if self.bond_head.is_none() {
            return self.predict(&self.encode(graph, train)?, graph, train);
        }
        let (atoms, edges) = self.encoder.forward_with_edges(graph, train)?;
        self.predict_with_edges(&atoms, edges.as_ref(), graph, train)
    }

    /// The heads, applied to atom vectors already computed.
    ///
    /// Worth having apart from [`MolecularNet::forward`] when the same
    /// encoding feeds something else as well -- a contrastive term, an
    /// autoencoder's decoder -- and running the encoder twice would be both
    /// slower and, under dropout, not the same encoding.
    ///
    /// A bond head on a D-MPNN also reads the encoder's edge messages, which
    /// atom vectors alone do not carry; this passes it zeros in their place.
    /// [`MolecularNet::predict_with_edges`] takes them as well.
    pub fn predict(
        &self,
        atoms: &Tensor,
        graph: &GraphTensors,
        train: bool,
    ) -> Result<Predictions> {
        self.predict_with_edges(atoms, None, graph, train)
    }

    /// [`MolecularNet::predict`] with the edge messages
    /// [`crate::ml::nn::Encoder::forward_with_edges`] gave beside the atoms.
    pub fn predict_with_edges(
        &self,
        atoms: &Tensor,
        edges: Option<&Tensor>,
        graph: &GraphTensors,
        train: bool,
    ) -> Result<Predictions> {
        let graph_predictions = match &self.graph_head {
            Some(head) => {
                let pooled = graph.pool(atoms, self.config.pooling)?;
                let read = match (&graph.global_features, self.config.global_width) {
                    (Some(globals), width) if width > 0 => {
                        Tensor::cat(&[&pooled, &globals.to_dtype(pooled.dtype())?], 1)?
                    }
                    _ => pooled,
                };
                Some(head.forward(&read, train)?)
            }
            None => None,
        };
        let atom_predictions = match &self.atom_head {
            Some(head) => Some(head.forward(atoms, train)?),
            None => None,
        };
        let bond_predictions = match &self.bond_head {
            Some(head) if graph.edge_count > 0 => {
                let edges = match edges {
                    Some(edges) => Some(edges.clone()),
                    None if self.encoder.has_edge_states() => Some(Tensor::zeros(
                        (graph.edge_count, self.encoder.width()),
                        atoms.dtype(),
                        atoms.device(),
                    )?),
                    None => None,
                };
                let read = bond_representation(atoms, edges.as_ref(), graph)?;
                Some(head.forward(&read, train)?)
            }
            Some(_) => None,
            None => None,
        };

        Ok(Predictions {
            graph: graph_predictions,
            atom: atom_predictions,
            bond: bond_predictions,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Whatever the model was asked for.
#[derive(Debug, Clone)]
pub struct Predictions {
    /// `[graph_count, tasks]`.
    pub graph: Option<Tensor>,
    /// `[atom_count, tasks]`.
    pub atom: Option<Tensor>,
    /// `[edge_count, tasks]` -- one row per *directed* edge, so a bond appears
    /// twice with the same value. [`bond_predictions`] halves it.
    pub bond: Option<Tensor>,
}

/// What a bond head reads, the same from either end of the bond: the two
/// atoms' sum and the size of their difference, the bond's own feature row,
/// and -- from an encoder that keeps them -- the messages along both of its
/// directions, summed.
///
/// Concatenating the two atom vectors in edge order would make a prediction
/// that depends on which way round the bond happened to be written. And the
/// atom vectors alone are not enough: each holds every bond its atom has,
/// summed, so what is particular to this one bond -- its order, whether it
/// closes a ring, its geometry -- has to be inferred back out of two sums.
/// The bond row says it directly, and a D-MPNN's messages are the network's
/// own view of the bond, which is what ALFABET's bond model reads too.
fn bond_representation(
    atoms: &Tensor,
    edges: Option<&Tensor>,
    graph: &GraphTensors,
) -> Result<Tensor> {
    let sources = graph.gather_sources(atoms)?;
    let targets = graph.gather_targets(atoms)?;
    let bond = graph.bond_features.to_dtype(atoms.dtype())?;
    let mut parts = vec![sources.add(&targets)?, sources.sub(&targets)?.abs()?, bond];
    if let Some(edges) = edges {
        parts.push(edges.add(&graph.gather_reverse(edges)?)?);
    }
    Tensor::cat(&parts, 1)
}

/// A bond, named by the two atom rows it joins, and what was predicted for it.
pub type BondPrediction = ((usize, usize), Vec<f32>);

/// One prediction per bond rather than per directed edge.
///
/// A bond head runs on edges, and every bond is two of them. The two rows are
/// equal by construction -- `bond_representation` is symmetric -- so this
/// keeps whichever comes first and reports which atoms it joins.
pub fn bond_predictions(predictions: &Tensor, batch: &GraphBatch) -> Result<Vec<BondPrediction>> {
    let tasks = predictions.dim(1)?;
    let values = predictions
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for edge in 0..batch.edge_count {
        let (from, to) = batch.edge(edge);
        let key = (from.min(to), from.max(to));
        if !seen.insert(key) {
            continue;
        }
        out.push((key, values[edge * tasks..(edge + 1) * tasks].to_vec()));
    }
    Ok(out)
}
