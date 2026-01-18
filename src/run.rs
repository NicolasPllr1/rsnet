use crate::model::{Module, NN};
use ndarray::prelude::*;
use ndarray::Array2;
use std::fs;

/// Run inference using a loaded neural network
pub fn run(checkpoint_path: &str, example_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Load neural network from checkpoint
    let mut nn = NN::from_checkpoint(checkpoint_path)?;
    println!("Loaded checkpoint from: {}", checkpoint_path);

    // Read example file (expecting 784 bytes for 28x28 MNIST image)
    let image_bytes = fs::read(example_path)?;
    if image_bytes.len() != 784 {
        return Err(format!(
            "Expected 784 bytes (28x28 image), got {} bytes",
            image_bytes.len()
        )
        .into());
    }

    // Convert image from Vec<u8> to Array2<f32> and normalize
    let img_f32: Vec<f32> = image_bytes.iter().map(|&x| x as f32 / 255.0).collect();
    let input = Array2::from_shape_vec((1, 784), img_f32)
        .map_err(|e| format!("Failed to create input array: {}", e))?;

    // Run forward pass
    let output = nn
        .forward(input.into_dyn())
        .into_dimensionality::<Ix2>()
        .expect("Output should be castable to 2D");

    // Get prediction (argmax - index with highest value)
    let predicted_label = output
        .row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // Print prediction
    println!("Prediction: {}", predicted_label);
    println!("Output probabilities: {:?}", output.row(0).to_vec());

    Ok(())
}
