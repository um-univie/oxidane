//! The neural half: tensors, message passing, heads, losses and a training loop.
//!
//! Behind the `ml` feature, because this is the part that needs a tensor
//! library. Everything else under [`crate::ml`] is dependency-free and always
//! compiled.
//!
//! The pipeline reads in one direction:
//!
//! ```text
//! MolecularGraph -> graph_features -> GraphBatch -> GraphTensors
//!                                                       |
//!                                             Encoder (DMpnn / Gine)
//!                                                       |
//!                                          one vector per atom
//!                                          /            |            \
//!                                    pool + head    atom head    bond head
//! ```
//!
//! [`crate::molecule::featurize`] does the first arrow, [`crate::ml::batch`]
//! the second, and [`tensors`] the third. From there it is
//! [`mpnn::Encoder`] and [`model::MolecularNet`], with [`loss`] and [`train`]
//! around them.
//!
//! The device is the caller's choice throughout. candle's default build is
//! pure Rust on the CPU; a dependent that wants a GPU turns on candle's own
//! `cuda` or `metal` feature and passes a different [`candle_core::Device`],
//! and nothing here changes.

pub mod layers;
pub mod loss;
pub mod model;
pub mod mpnn;
pub mod tensors;
pub mod train;

pub use layers::{Activation, Mlp, MlpConfig};
pub use loss::{masked_bce_with_logits, masked_l1, masked_mse, mask_from_flags};
pub use model::{ModelConfig, MolecularNet, Predictions, bond_predictions};
pub use mpnn::{DMpnn, Encoder, EncoderConfig, EncoderKind, Gine};
pub use tensors::{GraphTensors, Pooling};
pub use train::{History, Objective, TrainConfig, Trainer};
