// use indicatif::ProgressIterator;
use ndarray::Array2;

use crate::{
    load_mnist, Conv2Dlayer, FcLayer, FlattenLayer, Layer, Module, ReluLayer, SoftMaxLayer, NN,
};
use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use std::io::Write;
use std::path::Path;

type CostFunction = fn(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>);

fn cross_entropy(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let batch_size = labels.len();
    let num_classes = actual_y.ncols();

    // Convert labels to one-hot encoding Array2
    let mut expected_y = Array2::zeros((batch_size, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        expected_y[(i, label as usize)] = 1.0;
    }

    // Calculate cross-entropy for each sample in batch: -log(p)
    let loss = -(expected_y.clone() * actual_y.ln()).fold_axis(Axis(1), 0.0, |&a, &b| a + b);

    (loss, expected_y)
}

trait Optimizer {
    fn step(
        &self,
        nn: &mut NN,
        cost_function: CostFunction,
        labels: &[u8],
        output: &Array2<f32>,
    ) -> Array1<f32>;
}

struct SGD {
    learning_rate: f32,
}

impl Optimizer for SGD {
    fn step(
        &self,
        nn: &mut NN,
        cost_function: CostFunction,
        labels: &[u8],
        output: &Array2<f32>,
    ) -> Array1<f32> {
        // Calculate loss and gradient for the batch
        let (loss, grad) = cost_function(labels, output);
        // Backpropagate through layers in reverse order
        nn.backward(grad.into_dyn());

        nn.step(self.learning_rate);

        loss
    }
}

/// Train a neural network (stub)
pub fn train(
    train_steps: usize,
    learning_rate: f32,
    checkpoint_folder: &str,
    checkpoint_stride: usize,
    loss_csv_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create checkpoint folder if it doesn't exist
    fs::create_dir_all(checkpoint_folder)?;

    let mut nn = NN {
        layers: vec![
            Layer::Conv(Conv2Dlayer::new(1, 10, (5, 5))), // feature maps: (1, 28x28) --> (10, 24,24)
            Layer::ReLU(ReluLayer::new()),
            Layer::Conv(Conv2Dlayer::new(10, 20, (2, 2))), // feature maps: (10, 24x24) --> (20, 23,23)
            Layer::ReLU(ReluLayer::new()),
            Layer::Flatten(FlattenLayer::new()), // flatten feature maps into a single 1D vector
            Layer::FC(FcLayer::new(20 * 23 * 23, 10)),
            Layer::Softmax(SoftMaxLayer::new()),
        ],
    };

    // Load MNIST dataset
    let (train_images, train_labels, test_images, test_labels) = load_mnist();
    let num_images = train_images.len() / 784;
    let num_test_images = test_images.len() / 784;
    println!("Loaded {} training images", num_images);
    println!("Loaded {} test images", num_test_images);

    // Prepare training data for shuffling per epoch
    // Collect data into vectors for efficient access
    let train_data_vec: Vec<(&[u8], &u8)> =
        train_images.chunks(784).zip(train_labels.iter()).collect();

    // Create indices that will be shuffled for each epoch
    let mut indices: Vec<usize> = (0..num_images).collect();

    // Original main() code from before rebase:
    /*
    fn main() {
        let mut nn = NN {
            layers: vec![
                Layer::FC(FcLayer::new(784, 10)),
            ],
        };

        let (train_images, train_labels, test_images, test_labels) = load_mnist();
        println!("Train images: {:?}", train_images.len());
        println!("Train labels: {:?}", train_labels.len());
        println!("Test images: {:?}", test_images.len());
        println!("Test labels: {:?}", test_labels.len());

        // train loop
        // TODO: calculate loss and do backpropagation
        println!("Training...");
        for (image, _label) in train_images.chunks(784).zip(train_labels.iter()).progress_count(60_000) {
            let img_f32: Vec<f32> = image.iter().map(|&x| x as f32).collect();
            let input = Array2::from_shape_vec((1, 784), img_f32).unwrap();
            let _output = nn.forward(input);
        }

        // test loop
        // TODO: calculate accuracy
        println!("Testing...");
        let mut total_correct = 0;
        let mut total_samples = 0;
        for (image, label) in test_images.chunks(784).zip(test_labels.iter()).progress_count(10_000) {
            let img_f32: Vec<f32> = image.iter().map(|&x| x as f32).collect();
            let input = Array2::from_shape_vec((1, 784), img_f32).unwrap();
            let output = nn.forward(input);
            // Find index of max value (argmax)
            let predicted_label = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap() as u8;
            if predicted_label == *label {
                total_correct += 1;
            }
            total_samples += 1;
        }
        let accuracy = total_correct as f32 / total_samples as f32;
        println!("Accuracy: {:?}", accuracy);
    }
    */

    // Create or truncate CSV file and write header
    let mut csv_file = fs::File::create(loss_csv_path)?;
    writeln!(csv_file, "epoch,loss,accuracy")?;

    let optimizer = SGD { learning_rate };

    const BATCH_SIZE: usize = 128;
    // Iterate through epochs
    for epoch in 0..train_steps {
        // Shuffle indices for this epoch
        indices.shuffle(&mut thread_rng());

        // Track loss for this epoch
        let mut epoch_loss_sum = 0.0;
        let mut epoch_loss_count = 0;

        // Iterate through shuffled data in batches
        for batch_indices in indices.chunks(BATCH_SIZE) {
            let batch_size = batch_indices.len(); // used to catch the remainder

            // Collect images and labels for this batch efficiently
            let mut batch_images = Vec::with_capacity(batch_size * 784);
            let mut batch_labels = Vec::with_capacity(batch_size);

            for &idx in batch_indices {
                let (image, label) = train_data_vec[idx];
                // Normalize and extend image to batch using iterator
                batch_images.extend(image.iter().map(|&pixel| pixel as f32 / 255.0));
                batch_labels.push(*label);
            }

            // Create batch input Array4 with shape (batch_size, channels=1, 28, 28)
            let input = Array4::from_shape_vec((batch_size, 1, 28, 28), batch_images)
                .map_err(|e| format!("Failed to create input array: {}", e))?;
            println!("[train] input to network: {:?}", input.shape());

            // Run forward pass (output shape: (batch_size, num_classes))
            let output = nn.forward(input.into_dyn());

            // Calculate loss and do backpropagation on the batch
            // The cost function will convert labels to one-hot encoding internally
            let loss = optimizer.step(
                &mut nn,
                cross_entropy,
                &batch_labels,
                &output
                    .into_dimensionality::<Ix2>()
                    .expect("Output should be castable to 2D"),
            );

            // Accumulate loss for epoch average
            epoch_loss_sum += loss.mean().unwrap();
            epoch_loss_count += 1;
        }

        // Save checkpoint every checkpoint_stride epochs
        if (epoch + 1) % checkpoint_stride == 0 {
            let checkpoint_num = (epoch + 1) / checkpoint_stride;
            let checkpoint_path =
                Path::new(checkpoint_folder).join(format!("checkpoint_{}.json", checkpoint_num));
            nn.to_checkpoint(checkpoint_path.to_str().unwrap())?;

            // Get average loss for this epoch
            let epoch_avg_loss = epoch_loss_sum / epoch_loss_count as f32;

            // Run test loop to calculate accuracy
            println!("Running test loop...");
            let mut total_correct = 0;
            let mut total_samples = 0;
            for (image, label) in test_images.chunks(784).zip(test_labels.iter()) {
                // Normalize image
                let img_f32: Vec<f32> = image.iter().map(|&x| x as f32 / 255.0).collect();
                let input = Array2::from_shape_vec((1, 784), img_f32)
                    .map_err(|e| format!("Failed to create test input array: {}", e))?;

                let output = nn.forward(input.into_dyn());

                // Find index of max value (argmax) - predicted label
                let predicted_label = output
                    .into_dimensionality::<Ix2>()
                    .expect("Output shoud be castable to 2D")
                    .row(0)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap() as u8;

                if predicted_label == *label {
                    total_correct += 1;
                }
                total_samples += 1;
            }
            let accuracy = total_correct as f32 / total_samples as f32;

            // Write to CSV
            writeln!(csv_file, "{},{},{}", epoch + 1, epoch_avg_loss, accuracy)?;
            csv_file.flush()?;

            println!(
                "Saved checkpoint {} at epoch {} (loss: {:.4}, accuracy: {:.2}%)",
                checkpoint_num,
                epoch + 1,
                epoch_avg_loss,
                accuracy * 100.0
            );
        }
    }

    println!("Training completed!");
    println!("Checkpoint folder: {}", checkpoint_folder);
    Ok(())
}
