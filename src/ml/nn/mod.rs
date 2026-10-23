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
//! around them, and [`autoencoder`] for the unsupervised case.
//!
//! Two of those go the other way as well. [`autoencoder`] reconstructs the
//! bonds between atoms it has been given; [`molecule_autoencoder`] reconstructs
//! the whole molecule from one vector, spelling it out through
//! [`decoder::SmilesDecoder`], which is the round trip a latent space has to
//! close for interpolation or generation to mean anything.
//!
//! The device is the caller's choice throughout. candle's default build is
//! pure Rust on the CPU; a dependent that wants a GPU turns on candle's own
//! `cuda` or `metal` feature and passes a different [`candle_core::Device`],
//! and nothing here changes.

pub mod autoencoder;
pub mod decoder;
pub mod layers;
pub mod loss;
pub mod model;
pub mod molecule_autoencoder;
pub mod mpnn;
pub mod pretrain;
pub mod seed;
pub mod tensors;
pub mod train;

pub use decoder::{DecoderConfig, SmilesDecoder};
pub use layers::{Activation, Mlp, MlpConfig};
pub use loss::{mask_from_flags, masked_bce_with_logits, masked_l1, masked_mse};
pub use model::{ModelConfig, MolecularNet, Predictions, bond_predictions};
pub use molecule_autoencoder::{MoleculeAutoencoder, MoleculeAutoencoderConfig, MoleculeLoss};
pub use mpnn::{DMpnn, Encoder, EncoderConfig, EncoderKind, Gine};
pub use pretrain::{MaskedAtomModel, PretrainConfig};
pub use seed::{load_prefixed, save_prefixed, seed_variables, variables_outside};
pub use tensors::{GraphTensors, Pooling};
pub use train::{History, Objective, Schedule, TrainConfig, Trainer};
