//! The encoders: what turns a molecule into one vector per atom.
//!
//! Two are here because they answer to different shapes of problem.
//!
//! [`DMpnn`] passes messages along *directed edges* rather than atoms, which
//! is what Chemprop does and what the bond features are already laid out for.
//! Its advantage is that a message can be stopped from bouncing back the way
//! it came, so a two-step walk cannot revisit its own start and the
//! representation of a bond is built from what the rest of the molecule looks
//! like from it. On molecular property prediction it is the stronger of the
//! two more often than not, and it is the one to reach for first.
//!
//! [`Gine`] passes messages between atoms, conditioned on the bond in between.
//! It is cheaper, it is what most of the graph-network literature outside
//! chemistry is written against, and its atom vectors are the natural input to
//! an autoencoder because every layer is a function of atoms rather than of
//! edges.

use candle_core::{Result, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder, linear};

use crate::ml::nn::layers::{Activation, Mlp, MlpConfig};
use crate::ml::nn::tensors::GraphTensors;
use crate::molecule::featurize::{ATOM_FEATURE_WIDTH, BOND_FEATURE_WIDTH};

/// How wide and how deep an encoder is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderConfig {
    /// The width of an atom's vector, and of a message.
    pub hidden: usize,
    /// How many rounds of message passing follow the first embedding.
    ///
    /// Three is the usual choice. Each round widens the neighbourhood an atom
    /// can see by one bond, and past four or so the vectors of atoms in the
    /// same molecule start to converge on each other.
    pub depth: usize,
    pub activation: Activation,
    /// Applied after every round, and after the final atom projection.
    pub dropout: f32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            hidden: 300,
            depth: 3,
            activation: Activation::Relu,
            dropout: 0.0,
        }
    }
}

/// Chemprop's directed message-passing network.
///
/// ```
/// # fn main() -> candle_core::Result<()> {
/// use candle_core::{DType, Device};
/// use candle_nn::{VarBuilder, VarMap};
/// use oxidane::prelude::*;
/// use oxidane::molecule::featurize::graph_features;
/// use oxidane::ml::batch::GraphBatch;
/// use oxidane::ml::nn::{DMpnn, EncoderConfig, GraphTensors};
///
/// let device = Device::Cpu;
/// let varmap = VarMap::new();
/// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
/// let config = EncoderConfig { hidden: 32, depth: 2, ..Default::default() };
/// let encoder = DMpnn::new(config, vb)?;
///
/// let batch = GraphBatch::collate(&[graph_features(
///     &MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap(),
/// )]);
/// let graph = GraphTensors::new(&batch, &device)?;
/// let atoms = encoder.forward(&graph, false)?;
///
/// // One vector per heavy atom.
/// assert_eq!(atoms.dims(), &[batch.atom_count, 32]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct DMpnn {
    /// The first embedding, from an edge's source atom and the bond itself.
    input: Linear,
    /// The recurrent step, shared across every round.
    hidden: Linear,
    /// The atom projection, from an atom and everything that arrived at it.
    output: Linear,
    dropout: Dropout,
    config: EncoderConfig,
}

impl DMpnn {
    pub fn new(config: EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input: linear(
                ATOM_FEATURE_WIDTH + BOND_FEATURE_WIDTH,
                config.hidden,
                vb.pp("input"),
            )?,
            hidden: linear(config.hidden, config.hidden, vb.pp("hidden"))?,
            output: linear(
                ATOM_FEATURE_WIDTH + config.hidden,
                config.hidden,
                vb.pp("output"),
            )?,
            dropout: Dropout::new(config.dropout),
            config,
        })
    }

    /// How wide each atom's vector comes out.
    pub fn width(&self) -> usize {
        self.config.hidden
    }

    /// One vector per atom row, `[atom_count, hidden]`.
    pub fn forward(&self, graph: &GraphTensors, train: bool) -> Result<Tensor> {
        let arrived = if graph.edge_count == 0 {
            // A molecule of one heavy atom has no bonds, and there is nothing
            // for message passing to do. Its atom vector is its own features
            // put through the output projection, which is what the zeros here
            // arrange.
            Tensor::zeros(
                (graph.atom_count, self.config.hidden),
                graph.atom_features.dtype(),
                &graph.device,
            )?
        } else {
            self.pass_messages(graph, train)?
        };

        let atoms = Tensor::cat(&[&graph.atom_features, &arrived], 1)?;
        let atoms = self.config.activation.apply(&self.output.forward(&atoms)?)?;
        self.maybe_drop(atoms, train)
    }

    fn pass_messages(&self, graph: &GraphTensors, train: bool) -> Result<Tensor> {
        // What an edge starts out knowing: the atom it leaves and the bond it
        // is.
        let sources = graph.gather_sources(&graph.atom_features)?;
        let initial = Tensor::cat(&[&sources, &graph.bond_features], 1)?;
        let initial = self
            .config
            .activation
            .apply(&self.input.forward(&initial)?)?;

        let mut messages = initial.clone();
        for _ in 0..self.config.depth {
            // Everything that arrived at each atom...
            let arrived = graph.scatter_to_atoms(&messages)?;
            // ...as seen from the far end of each edge, less what that edge
            // itself sent back the other way. Without the subtraction a
            // message returns to its own source in two rounds and the network
            // spends its depth listening to itself.
            let incoming = graph.gather_sources(&arrived)?;
            let returning = graph
                .gather_reverse(&messages)?
                .broadcast_mul(&graph.reverse_mask)?;
            let carried = incoming.sub(&returning)?;

            let updated = initial.add(&self.hidden.forward(&carried)?)?;
            messages = self.config.activation.apply(&updated)?;
            messages = self.maybe_drop(messages, train)?;
        }

        graph.scatter_to_atoms(&messages)
    }

    fn maybe_drop(&self, xs: Tensor, train: bool) -> Result<Tensor> {
        if self.config.dropout > 0.0 {
            self.dropout.forward(&xs, train)
        } else {
            Ok(xs)
        }
    }
}

/// Graph isomorphism network with edge features.
///
/// Messages travel between atoms, each carrying the bond it crossed. The
/// `eps` that scales an atom's own vector against the sum of its neighbours'
/// is learned rather than fixed, which is what keeps the sum injective -- the
/// property the architecture is named for and the reason it can tell apart
/// neighbourhoods that a mean or a maximum cannot.
#[derive(Debug)]
pub struct Gine {
    embed: Linear,
    bonds: Vec<Linear>,
    updates: Vec<Mlp>,
    epsilons: Vec<Tensor>,
    dropout: Dropout,
    config: EncoderConfig,
}

impl Gine {
    pub fn new(config: EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut bonds = Vec::new();
        let mut updates = Vec::new();
        let mut epsilons = Vec::new();
        let update_config = MlpConfig {
            activation: config.activation,
            layer_norm: true,
            activate_output: false,
            ..Default::default()
        };
        for round in 0..config.depth {
            let vb = vb.pp(format!("round{round}"));
            bonds.push(linear(BOND_FEATURE_WIDTH, config.hidden, vb.pp("bond"))?);
            updates.push(Mlp::new(
                &[config.hidden, config.hidden, config.hidden],
                update_config,
                vb.pp("update"),
            )?);
            epsilons.push(vb.get(1, "eps")?);
        }
        Ok(Self {
            embed: linear(ATOM_FEATURE_WIDTH, config.hidden, vb.pp("embed"))?,
            bonds,
            updates,
            epsilons,
            dropout: Dropout::new(config.dropout),
            config,
        })
    }

    pub fn width(&self) -> usize {
        self.config.hidden
    }

    pub fn forward(&self, graph: &GraphTensors, train: bool) -> Result<Tensor> {
        let mut atoms = self.embed.forward(&graph.atom_features)?;

        for round in 0..self.config.depth {
            // A molecule of one heavy atom has no bonds and so receives
            // nothing. Returning its embedding here instead of running the
            // rounds over an empty sum would be a different function -- and
            // one that changed depending on whether the molecule was batched
            // with something that does have bonds.
            let arrived = if graph.edge_count == 0 {
                Tensor::zeros(atoms.shape(), atoms.dtype(), &graph.device)?
            } else {
                let bonds = self.bonds[round].forward(&graph.bond_features)?;
                let neighbours = graph.gather_sources(&atoms)?.add(&bonds)?;
                let messages = self.config.activation.apply(&neighbours)?;
                graph.scatter_to_atoms(&messages)?
            };

            // `(1 + eps) * h + arrived`, with eps a scalar this round learned.
            let scale = self.epsilons[round]
                .to_dtype(atoms.dtype())?
                .broadcast_add(&Tensor::ones(1, atoms.dtype(), &graph.device)?)?;
            let combined = atoms.broadcast_mul(&scale)?.add(&arrived)?;

            atoms = self.updates[round].forward(&combined, train)?;
            if self.config.dropout > 0.0 {
                atoms = self.dropout.forward(&atoms, train)?;
            }
        }
        Ok(atoms)
    }
}

/// Either encoder, so that a model can be configured rather than rewritten.
#[derive(Debug)]
pub enum Encoder {
    Directed(DMpnn),
    Gine(Gine),
}

impl Encoder {
    /// The encoder named by `kind`.
    pub fn new(kind: EncoderKind, config: EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(match kind {
            EncoderKind::Directed => Encoder::Directed(DMpnn::new(config, vb)?),
            EncoderKind::Gine => Encoder::Gine(Gine::new(config, vb)?),
        })
    }

    pub fn width(&self) -> usize {
        match self {
            Encoder::Directed(encoder) => encoder.width(),
            Encoder::Gine(encoder) => encoder.width(),
        }
    }

    pub fn forward(&self, graph: &GraphTensors, train: bool) -> Result<Tensor> {
        match self {
            Encoder::Directed(encoder) => encoder.forward(graph, train),
            Encoder::Gine(encoder) => encoder.forward(graph, train),
        }
    }
}

/// Which encoder a model is built around.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderKind {
    /// [`DMpnn`].
    #[default]
    Directed,
    /// [`Gine`].
    Gine,
}
