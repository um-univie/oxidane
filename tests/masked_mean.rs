#![cfg(feature = "ml")]
use candle_core::{Device, Tensor, Var};
use oxidane::ml::nn::loss::masked_mean;

#[test]
fn broadcast_masks_count_every_retained_value() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let values = Tensor::new(&[[1f32, 3.], [5., 7.]], &device)?;
    for (mask, expected) in [
        (Tensor::new(&[[1f32], [1.]], &device)?, 4.),
        (Tensor::new(&[[1f32], [0.]], &device)?, 2.),
        (Tensor::new(&[[0f32, 1.]], &device)?, 5.),
        (Tensor::new(1f32, &device)?, 4.),
        (Tensor::new(&[[0f32], [0.]], &device)?, 0.),
        (Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?, 4.),
    ] {
        assert_eq!(masked_mean(&values, &mask)?.to_scalar::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn broadcast_mask_normalizes_gradients_and_empty_masks_zero_them() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let values = Var::new(&[[1f32, 3.], [5., 7.]], &device)?;
    for (flags, expected) in [
        ([[1f32], [0.]], vec![vec![0.5, 0.5], vec![0., 0.]]),
        ([[0f32], [0.]], vec![vec![0., 0.], vec![0., 0.]]),
    ] {
        let mask = Tensor::new(&flags, &device)?;
        let gradients = masked_mean(&values, &mask)?.backward()?;
        assert_eq!(gradients.get(&values).unwrap().to_vec2::<f32>()?, expected);
    }
    Ok(())
}
