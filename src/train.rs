// use indicatif::ProgressIterator;
use ndarray::Array2;

use crate::{load_mnist, FcLayer, Layer, Module, ReluLayer, SoftMaxLayer, NN};
use ndarray::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;

type CostFunction = fn(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>);

fn least_squares(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let batch_size = labels.len();
    let num_classes = actual_y.ncols();

    // Convert labels to one-hot encoding Array2
    let mut expected_y = Array2::zeros((batch_size, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        expected_y[(i, label as usize)] = 1.0;
    }

    // Calculate least squares for each sample in batch: (expected - actual)^2
    let loss = (expected_y.clone() - actual_y)
        .mapv(|x| x * x)
        .fold_axis(Axis(1), 0.0, |&a, &b| a + b);
    let grad = 2.0 * (expected_y.clone() - actual_y);
    (loss, grad)
}

fn cross_entropy(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let batch_size = labels.len();
    let num_classes = actual_y.ncols();

    // Convert labels to one-hot encoding Array2
    let mut expected_y = Array2::zeros((batch_size, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        expected_y[(i, label as usize)] = 1.0;
    }

    // Calculate least squares for each sample in batch: (expected - actual)^2
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
        nn.backward(grad);

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

    // TODO: Implement training logic
    // For now, use a hardcoded neural network
    let mut nn = NN {
        layers: vec![
            Layer::FC(FcLayer::new(28 * 28, 3)),
            Layer::ReLU(ReluLayer::new()),
            Layer::FC(FcLayer::new(3, 10)),
            Layer::Softmax(SoftMaxLayer::new()),
        ],
    };

    // Load MNIST dataset
    let (train_images, train_labels, _test_images, _test_labels) = load_mnist();
    println!("Loaded {} training images", train_images.len() / 784);

    // Create iterator over training data
    let train_data = train_images
        .chunks(784)
        .zip(train_labels.iter())
        .cycle() // Cycle through the dataset for multiple epochs
        .take(train_steps);

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
    writeln!(csv_file, "step,loss")?;

    let optimizer = SGD { learning_rate };

    for (step, (image, label)) in train_data.enumerate() {
        // Convert image from Vec<u8> to Array2<f32>
        let img_f32: Vec<f32> = image.iter().map(|&x| x as f32 / 255.0).collect(); // Normalize to [0, 1]
        let input = Array2::from_shape_vec((1, 784), img_f32)
            .map_err(|e| format!("Failed to create input array: {}", e))?;

        // Run forward pass (output shape: (batch_size, num_classes))
        let output = nn.forward(input);

        // Calculate loss and do backpropagation
        // The cost function will convert labels to one-hot encoding internally
        let loss = optimizer.step(&mut nn, cross_entropy, &[*label], &output);

        // Save checkpoint every checkpoint_stride steps
        if (step + 1) % checkpoint_stride == 0 {
            let checkpoint_num = (step + 1) / checkpoint_stride;
            let checkpoint_path =
                Path::new(checkpoint_folder).join(format!("checkpoint_{}.json", checkpoint_num));
            nn.to_checkpoint(checkpoint_path.to_str().unwrap())?;

            // Get loss at this checkpoint (average over batch)
            let checkpoint_loss = loss.mean().unwrap();

            // Write to CSV
            writeln!(csv_file, "{},{}", step + 1, checkpoint_loss)?;
            csv_file.flush()?;

            println!(
                "Saved checkpoint {} at step {} (loss: {:.4})",
                checkpoint_num,
                step + 1,
                checkpoint_loss
            );
        }
    }

    println!("Training completed!");
    println!("Checkpoint folder: {}", checkpoint_folder);
    Ok(())
}
