// use crate::mnist_dataset::load_mnist;
use crate::custom_dataset::{load_metadata, process_image, Dataset};
use crate::model::{Module, NN};
use crate::optim::SGDMomentum;
pub(crate) use crate::optim::{CostFunction, Optimizer};
use crate::DEBUG;
use rayon::prelude::*;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

/// Train a neural network
pub fn train(
    mut nn: NN,
    data_dir: &str,
    batch_size: usize,
    nb_epochs: usize,
    cost_function: CostFunction,
    learning_rate: f32,
    checkpoint_folder: &str,
    checkpoint_stride: usize,
    loss_csv_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create checkpoint folder if it doesn't exist
    fs::create_dir_all(checkpoint_folder)?;

    let (train_dataset, test_dataset) = load_metadata(&data_dir, None);
    println!("[TRAIN] len: {}", train_dataset.samples.len());
    println!("[TEST] len: {}\n", test_dataset.samples.len());

    let mut indices: Vec<usize> = (0..train_dataset.samples.len()).collect();

    let (in_channels, h, w) = (1, 128, 128); // Downscaled size

    // Create or truncate CSV file and write header
    let mut csv_file = fs::File::create(loss_csv_path)?;
    writeln!(csv_file, "step,loss,duration,accuracy")?;

    let viscosity = 0.9;
    let mut optimizer = SGDMomentum::new(&nn, learning_rate, viscosity);

    let pb = ProgressBar::new(nb_epochs as u64 * (train_dataset.samples.len() / batch_size) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap(),
    );
    let mut optim_step = 1;
    // Iterate through epochs;
    for _epoch in 0..nb_epochs {
        indices.shuffle(&mut thread_rng());

        // Track loss running loss
        let mut running_loss = 0.0;
        let mut stride_duration = std::time::Instant::now();

        // Iterate through shuffled data in batches
        for (_batch_idx, batch_indices) in indices.chunks_exact(batch_size).enumerate() {
            // for batch_idx in 0..200 {
            let step_start = std::time::Instant::now();

            let mut batch_images = Vec::with_capacity(batch_size * 3 * h * w);
            let mut batch_labels = Vec::new();

            for &idx in batch_indices {
                let (path, label) = &train_dataset.samples[idx];
                // Actual loading of the image
                let processed_pixels = process_image(path, h as u32, w as u32);

                batch_images.extend(processed_pixels);
                batch_labels.push(*label);
            }

            // Input batch: (batch_size, in_channels, h, w)
            let input = Array4::from_shape_vec((batch_size, in_channels, h, w), batch_images)
                .map_err(|e| format!("Failed to create input array: {}", e))?;

            // 0. Zero gradients
            nn.zero_grad();
            // 1. Forward pass
            let output = nn.forward(input.into_dyn());
            let output = output // (batch_size, num_classes)
                .into_dimensionality::<Ix2>()
                .expect("Network output should be 2D: (batch_size, num_classes)");
            // 2. Compute loss
            let (loss, init_grad) = cost_function(&batch_labels, &output);
            // 3. Backward pass
            nn.backward(init_grad.into_dyn());
            // 4. Optim. step
            optimizer.step(&mut nn);

            if optim_step % checkpoint_stride == 0 {
                let avg_loss = loss.sum() / batch_size as f32;
                let duration = step_start.elapsed();
                pb.println(format!(
                    "[BATCH] step {} loss: {:.4} duration: {:?}",
                    optim_step, avg_loss, duration
                ));
            }

            // Accumulate loss for epoch average
            running_loss += loss.mean().unwrap();

            // Save checkpoint every checkpoint_stride optimization steps
            if optim_step % checkpoint_stride == 0 {
                validation(
                    &mut nn,
                    running_loss,
                    optim_step,
                    stride_duration.elapsed(),
                    checkpoint_stride,
                    &test_dataset,
                    checkpoint_folder,
                    &mut csv_file,
                    in_channels,
                    h,
                    w,
                    optim_step % 5 * checkpoint_stride == 0,
                    &pb,
                )?;

                // Reset
                running_loss = 0.0;
                stride_duration = std::time::Instant::now();
            }

            optim_step += 1;
            pb.inc(1);
        }
    }

    let ckpt_path = save_model(&nn, checkpoint_folder, optim_step)?;
    println!("Model saved at {}", ckpt_path);

    println!("Training completed!");
    Ok(())
}

fn validation(
    nn: &mut NN,
    running_loss: f32,
    optim_step: usize,
    duration: Duration,
    checkpoint_stride: usize,
    test_dataset: &Dataset,
    checkpoint_folder: &str,
    csv_file: &mut File,
    //
    in_channels: usize,
    h: usize,
    w: usize,
    save_model: bool,
    pb: &ProgressBar,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get average loss for this epoch
    let running_avg_loss = running_loss / checkpoint_stride as f32;
    if DEBUG {
        println!("[STRIDE DURATION] {:?}", duration);
        println!("[STRIDE AVG LOSS] {:.4}", running_avg_loss);
    }

    // Run test loop to calculate accuracy
    if DEBUG {
        println!("Running test loop...");
    }

    let (total_correct, total_samples) = test_dataset
        .samples
        .par_iter()
        .map_init(
            || nn.clone(),
            |local_nn, (image_path, label)| {
                // Load the processed image
                let img_f32 = process_image(&image_path, h as u32, w as u32);
                let input = Array4::from_shape_vec((1, in_channels, h, w), img_f32)
                    .expect("Failed to create test input array");
                // .map_err(|e| format!("Failed to create test input array: {}", e))?;

                let output = local_nn.forward(input.into_dyn());

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
                    (1, 1)
                } else {
                    (0, 1)
                }
            },
        )
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    let accuracy = total_correct as f32 / total_samples as f32;
    pb.println(format!("[TEST ACC] {:.3}", accuracy));

    // Write to CSV
    writeln!(
        csv_file,
        "{},{:.4},{:?},{:.4}",
        optim_step, running_avg_loss, duration, accuracy
    )?;
    csv_file.flush()?;

    if DEBUG {
        println!(
            "[VALIDATION] step {} (loss: {:.4}, accuracy: {:.2}%)",
            optim_step,
            running_avg_loss,
            accuracy * 100.0
        );
    }
    Ok(())
}

fn save_model(
    nn: &NN,
    checkpoint_folder: &str,
    optim_step: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let ckpt_path = Path::new(checkpoint_folder)
        .join(format!("checkpoint_{optim_step}.json"))
        .to_str()
        .unwrap();
    println!("Saving model to: {}", ckpt_path.to_str());
    nn.to_checkpoint(ckpt_path)?;

    Ok(ckpt_path.to_string())
}
