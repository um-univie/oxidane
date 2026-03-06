#![cfg(feature = "ml")]

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use oxidane::ml::dataset::Dataset;
use oxidane::ml::nn::{ModelConfig, MolecularNet, TrainConfig, Trainer};
use oxidane::ml::split::{Side, Split};
use oxidane::prelude::*;

fn scores(data: &Dataset, config: ModelConfig) -> candle_core::Result<Vec<f32>> {
    let device = Device::Cpu;
    let variables = VarMap::new();
    let model = MolecularNet::new(
        config,
        VarBuilder::from_varmap(&variables, DType::F32, &device),
    )?;
    for variable in variables.all_vars() {
        variable.set(&variable.zeros_like()?)?;
    }
    let mut scores = Vec::new();
    for indices in [
        (0..data.len()).collect::<Vec<_>>(),
        (0..data.len()).rev().collect(),
    ] {
        let split = Split {
            valid: indices,
            ..Default::default()
        };
        for batch_size in [1, 2, 3, 10] {
            let trainer = Trainer::new(
                &model,
                &variables,
                TrainConfig {
                    batch_size,
                    ..Default::default()
                },
                &device,
            )?;
            scores.push(trainer.evaluate(data, &split, Side::Valid)?);
        }
    }
    Ok(scores)
}

#[test]
fn validation_mse_does_not_depend_on_batch_size_or_order() -> candle_core::Result<()> {
    let molecule = MolecularGraph::from_smiles_single("C").unwrap();
    let mut data = Dataset::new(1);
    for label in [Some(0.), Some(0.), Some(3.)] {
        data.push(&molecule, &[label]);
    }
    for score in scores(&data, ModelConfig::graph_tasks(1))? {
        assert!((score - 3.).abs() < 1e-5, "{score}");
    }
    Ok(())
}

#[test]
fn unmeasured_rows_do_not_dilute_the_validation_loss() -> candle_core::Result<()> {
    let molecule = MolecularGraph::from_smiles_single("C").unwrap();
    let mut data = Dataset::new(2);
    for labels in [[Some(2.), None], [None, None], [None, Some(4.)]] {
        data.push(&molecule, &labels);
    }
    for score in scores(&data, ModelConfig::graph_tasks(2))? {
        assert!((score - 10.).abs() < 1e-5, "{score}");
    }
    let empty = Dataset::new(1);
    assert!(
        scores(&empty, ModelConfig::graph_tasks(1))?
            .iter()
            .all(|&v| v == 0.)
    );
    let mut masked = Dataset::new(1);
    masked.push(&molecule, &[None]);
    assert!(
        scores(&masked, ModelConfig::graph_tasks(1))?
            .iter()
            .all(|&v| v == 0.)
    );
    Ok(())
}
