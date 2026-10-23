//! The pieces every part of the model is built out of.

use candle_core::{Result, Tensor};
use candle_nn::{Dropout, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};

/// The non-linearity between two layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// Cheap, and what the message-passing literature is written against.
    #[default]
    Relu,
    /// Smooth. Usually a little better in the head, rarely worth the cost in
    /// the message passing.
    Gelu,
    /// Smooth and self-gating.
    Silu,
    Tanh,
}

impl Activation {
    pub fn apply(self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Activation::Relu => xs.relu(),
            Activation::Gelu => xs.gelu_erf(),
            Activation::Silu => candle_nn::ops::silu(xs),
            Activation::Tanh => xs.tanh(),
        }
    }
}

/// How a stack of linear layers is put together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlpConfig {
    pub activation: Activation,
    /// Dropped between layers while training, and ignored while evaluating.
    pub dropout: f32,
    /// A layer norm before each non-linearity. Worth having in a deep stack
    /// and noise in a shallow one.
    pub layer_norm: bool,
    /// Whether the last layer gets an activation. Off for a head, whose output
    /// is a prediction rather than a representation.
    pub activate_output: bool,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            activation: Activation::Relu,
            dropout: 0.0,
            layer_norm: false,
            activate_output: false,
        }
    }
}

/// A stack of linear layers.
#[derive(Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    norms: Vec<LayerNorm>,
    dropout: Dropout,
    config: MlpConfig,
}

impl Mlp {
    /// A stack whose layer widths are `sizes`, input first and output last.
    ///
    /// `sizes` of `[64, 128, 1]` is two layers: 64 in to 128, then 128 to 1.
    /// Two entries is one layer, and one entry is the identity -- which is
    /// useful rather than degenerate, since it lets a head be configured away
    /// without a branch at the call site.
    pub fn new(sizes: &[usize], config: MlpConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut norms = Vec::new();
        for (index, pair) in sizes.windows(2).enumerate() {
            layers.push(linear(pair[0], pair[1], vb.pp(format!("layer{index}")))?);
            let last = index + 2 == sizes.len();
            if config.layer_norm && (!last || config.activate_output) {
                norms.push(layer_norm(
                    pair[1],
                    candle_nn::LayerNormConfig::default(),
                    vb.pp(format!("norm{index}")),
                )?);
            }
        }
        Ok(Self {
            layers,
            norms,
            dropout: Dropout::new(config.dropout),
            config,
        })
    }

    /// How wide the input has to be, or `None` for an empty stack.
    pub fn input_width(&self) -> Option<usize> {
        self.layers
            .first()
            .map(|layer| layer.weight().dim(1).unwrap_or(0))
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut current = xs.clone();
        let last = self.layers.len().saturating_sub(1);
        for (index, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if index == last && !self.config.activate_output {
                break;
            }
            if let Some(norm) = self.norms.get(index) {
                current = norm.forward(&current)?;
            }
            current = self.config.activation.apply(&current)?;
            if self.config.dropout > 0.0 {
                current = self.dropout.forward(&current, train)?;
            }
        }
        Ok(current)
    }
}
