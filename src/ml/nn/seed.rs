//! Making a training run repeatable.
//!
//! Candle draws its initial weights from a thread-local generator seeded from
//! the operating system, and its CPU backend refuses `Device::set_seed`
//! outright. So two runs of one training script start from different weights
//! and finish at different numbers, and there is no argument that changes
//! that.
//!
//! On a benchmark that is not a nuisance, it is the whole problem. Two
//! configurations measured a percent apart cannot be told from one
//! configuration measured twice, so no change can be shown to have helped and
//! no published number can be checked by anyone.
//!
//! [`seed_variables`] draws every weight again from a stream this crate
//! controls. Call it after building a model and before training it; from that
//! point the run is a function of the seed, the data and the code.
//!
//! ```
//! # fn main() -> candle_core::Result<()> {
//! use candle_core::{DType, Device};
//! use candle_nn::{VarBuilder, VarMap};
//! use oxidane::ml::nn::{EncoderConfig, ModelConfig, MolecularNet, seed_variables};
//!
//! let device = Device::Cpu;
//! let weights = |seed| -> candle_core::Result<Vec<f32>> {
//!     let varmap = VarMap::new();
//!     let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
//!     let _ = MolecularNet::new(
//!         ModelConfig {
//!             encoder: EncoderConfig { hidden: 8, depth: 1, ..Default::default() },
//!             graph_head: vec![8, 1],
//!             ..Default::default()
//!         },
//!         vb,
//!     )?;
//!     seed_variables(&varmap, seed)?;
//!     let mut all = Vec::new();
//!     for variable in varmap.all_vars() {
//!         all.extend(variable.flatten_all()?.to_vec1::<f32>()?);
//!     }
//!     all.sort_by(|a, b| a.partial_cmp(b).unwrap());
//!     Ok(all)
//! };
//!
//! assert_eq!(weights(7)?, weights(7)?);
//! assert_ne!(weights(7)?, weights(8)?);
//! # Ok(())
//! # }
//! ```

use candle_core::{Result, Tensor};
use candle_nn::VarMap;

/// Draws every weight in `varmap` again from a stream seeded by `seed`.
///
/// The distribution is the one PyTorch and candle use for a linear layer:
/// uniform over plus and minus the reciprocal square root of the fan-in, which
/// is the last dimension of the weight. A bias takes the fan-in of the weight
/// beside it -- `layer0.bias` that of `layer0.weight`, a GRU's `bias_ih_l0`
/// that of its `weight_ih_l0` -- as PyTorch's does, rather than its own
/// length, which for a one-output head made the bias uniform over plus and
/// minus one.
///
/// Not every variable is drawn. A one-dimensional `weight` is a layer norm's
/// gain, and is set to one with its bias at zero; a learned `eps` is set to
/// zero. Those are the values every library starts them at, and drawing them
/// like a linear layer's weights instead started each GINE layer norm with
/// gains of a few hundredths, half of them negative, which shrank every
/// round's output to noise -- seeded GINE runs scored about a third worse for
/// it.
///
/// # Why the names are sorted
///
/// [`VarMap`] stores its variables in a hash map, so the order they come out
/// in is not a property of the model. Walking them in that order and drawing
/// from one stream would give a different assignment of numbers to weights on
/// every run, which is the problem this function exists to remove rather than
/// a detail of how it removes it.
pub fn seed_variables(varmap: &VarMap, seed: u64) -> Result<()> {
    let mut named: Vec<(String, candle_core::Var)> = {
        let data = varmap
            .data()
            .lock()
            .map_err(|_| candle_core::Error::Msg("the variable map was poisoned".to_string()))?;
        data.iter()
            .map(|(name, variable)| (name.clone(), variable.clone()))
            .collect()
    };
    named.sort_by(|a, b| a.0.cmp(&b.0));
    let shapes: std::collections::HashMap<String, Vec<usize>> = named
        .iter()
        .map(|(name, variable)| (name.clone(), variable.dims().to_vec()))
        .collect();

    let mut stream = SplitMix64::new(seed);
    for (name, variable) in named {
        let shape = variable.dims().to_vec();
        let (path, leaf) = name.rsplit_once('.').unwrap_or(("", name.as_str()));
        let sibling = |leaf: &str| {
            let key = if path.is_empty() {
                leaf.to_string()
            } else {
                format!("{path}.{leaf}")
            };
            shapes.get(&key)
        };
        let constant = if leaf == "eps" {
            Some(0.0)
        } else if leaf == "weight" && shape.len() == 1 {
            Some(1.0)
        } else if leaf == "bias" && sibling("weight").is_some_and(|weight| weight.len() == 1) {
            Some(0.0)
        } else {
            None
        };
        if let Some(value) = constant {
            let filled = Tensor::full(value as f32, shape.as_slice(), variable.device())?
                .to_dtype(variable.dtype())?;
            variable.set(&filled)?;
            continue;
        }

        let fan_in = if leaf.starts_with("bias") {
            sibling(&leaf.replacen("bias", "weight", 1))
                .and_then(|weight| weight.last().copied())
                .unwrap_or(*shape.last().unwrap_or(&1))
        } else {
            *shape.last().unwrap_or(&1)
        };
        let count: usize = shape.iter().product();
        let bound = 1.0 / (fan_in as f64).sqrt();

        let values: Vec<f32> = (0..count)
            .map(|_| (stream.uniform() * 2.0 - 1.0) as f32 * bound as f32)
            .collect();
        let drawn =
            Tensor::from_vec(values, shape, variable.device())?.to_dtype(variable.dtype())?;
        variable.set(&drawn)?;
    }
    Ok(())
}

/// The same generator the splits and the epoch orders use, for the same
/// reason: a stream a dependency upgrade can change is one that makes a
/// published number unrepeatable.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, 1)`, from the 53 bits an `f64` can hold exactly.
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Whether the device this crate is running on can be seeded by candle itself.
///
/// False on the CPU, which is the default build, and the reason
/// [`seed_variables`] exists.
pub fn device_is_seedable(device: &candle_core::Device) -> bool {
    device.set_seed(0).is_ok()
}

/// Writes the variables whose names begin with `prefix` to a safetensors file.
///
/// What pretraining produces that is worth keeping is the encoder; the head
/// that read elements out of atom vectors has done its job and is thrown away.
/// Saving by prefix rather than saving everything means the file can be loaded
/// into a model that has different heads on it, which is the entire point.
pub fn save_prefixed<P: AsRef<std::path::Path>>(
    varmap: &VarMap,
    prefix: &str,
    path: P,
) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("the variable map was poisoned".to_string()))?;
    let kept: std::collections::HashMap<String, Tensor> = data
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, variable)| (name.clone(), variable.as_detached_tensor()))
        .collect();
    if kept.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "no variable is named with the prefix {prefix:?}"
        )));
    }
    candle_core::safetensors::save(&kept, path)
}

/// Loads variables saved by [`save_prefixed`] into a model that already has
/// them, and says how many it set.
///
/// A name in the file that the model does not have is an error rather than
/// something to skip: it means the two were built from different
/// configurations, and a partly loaded encoder is worse than none -- it trains,
/// it converges, and the transfer it was supposed to provide silently did not
/// happen. A shape that disagrees is refused for the same reason.
pub fn load_prefixed<P: AsRef<std::path::Path>>(
    varmap: &VarMap,
    prefix: &str,
    path: P,
    device: &candle_core::Device,
) -> Result<usize> {
    let saved = candle_core::safetensors::load(path, device)?;
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("the variable map was poisoned".to_string()))?;

    let mut set = 0;
    for (name, tensor) in saved.iter() {
        if !name.starts_with(prefix) {
            continue;
        }
        let Some(variable) = data.get(name) else {
            return Err(candle_core::Error::Msg(format!(
                "the saved encoder has {name:?} and this model does not; \
                 they were built from different configurations"
            )));
        };
        if variable.dims() != tensor.dims() {
            return Err(candle_core::Error::Msg(format!(
                "{name:?} is {:?} here and {:?} in the file",
                variable.dims(),
                tensor.dims()
            )));
        }
        variable.set(&tensor.to_dtype(variable.dtype())?)?;
        set += 1;
    }

    // The other direction matters just as much: a model with encoder weights
    // the file did not carry is one whose transfer is partial.
    let wanted = data.keys().filter(|name| name.starts_with(prefix)).count();
    if set != wanted {
        return Err(candle_core::Error::Msg(format!(
            "this model has {wanted} variables under {prefix:?} and the file supplied {set}"
        )));
    }
    Ok(set)
}

/// The variables whose names do *not* begin with `prefix`, in a stable order.
///
/// What a caller hands [`crate::ml::nn::Trainer::over`] to freeze an encoder:
/// pass the complement of `"encoder"` and only the heads move.
pub fn variables_outside(varmap: &VarMap, prefix: &str) -> Result<Vec<candle_core::Var>> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| candle_core::Error::Msg("the variable map was poisoned".to_string()))?;
    let mut named: Vec<(String, candle_core::Var)> = data
        .iter()
        .filter(|(name, _)| !name.starts_with(prefix))
        .map(|(name, variable)| (name.clone(), variable.clone()))
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(named.into_iter().map(|(_, variable)| variable).collect())
}
