use crate::custom_dataset::{load_and_preprocess_image, load_dataset, Dataset};
use crate::model::{Module, NN};
use crate::optim::{Adam, CostFunction, Optimizer, SGDMomentum};
use crate::DEBUG;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::{self, File};
use std::io::Write;

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
    fs::create_dir_all(checkpoint_folder)?; // in case the folder does not exist

    let (train_dataset, test_dataset) = load_dataset(&data_dir, None);
    println!("[TRAIN] len: {}", train_dataset.samples.len());
    println!("[TEST] len: {}\n", test_dataset.samples.len());

    let mut indices: Vec<usize> = (0..train_dataset.samples.len()).collect();

    let (in_channels, h, w) = (1, 64, 64); // Downscaled size

    // Create or truncate CSV file and write header
    let mut csv_file = fs::File::create(loss_csv_path)?;
    writeln!(csv_file, "step,loss,duration,accuracy,stats")?;

    let mut optimizer = SGDMomentum::new(&nn, learning_rate);
    // let mut optimizer = Adam::new(&nn, learning_rate);

    let pb = ProgressBar::new(nb_epochs as u64 * (train_dataset.samples.len() / batch_size) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap(),
    );
    let mut optim_step = 1;

    for _epoch in 0..nb_epochs {
        indices.shuffle(&mut thread_rng());

        for (_batch_idx, batch_indices) in indices.chunks_exact(batch_size).enumerate() {
            let mut batch_images = Vec::with_capacity(batch_size * in_channels * h * w);
            let mut batch_labels = Vec::new();

            for &idx in batch_indices {
                let (path, label) = &train_dataset.samples[idx];
                let processed_pixels = load_and_preprocess_image(path, h as u32, w as u32);

                batch_images.extend(processed_pixels);
                batch_labels.push(*label);
            }

            let input = Array4::from_shape_vec((batch_size, in_channels, h, w), batch_images)
                .map_err(|e| format!("Failed to create input array: {}", e))?;

            // ----------
            nn.zero_grad();
            let output = nn.forward(input.into_dyn());
            let output = output // (batch_size, num_classes)
                .into_dimensionality::<Ix2>()
                .expect("Network output should be 2D: (batch_size, num_classes)");
            let (loss, init_grad) = cost_function(&batch_labels, &output);
            nn.backward(init_grad.into_dyn());
            optimizer.step(&mut nn);
            // ----------

            if optim_step % checkpoint_stride == 0 {
                let avg_loss = loss.sum() / loss.len() as f32;
                pb.println(format!("[BATCH] step {} loss: {:.4}", optim_step, avg_loss));
                let (acc, pred_stats) = validation(&mut nn, &test_dataset, in_channels, h, w, &pb)?;
                save_metrics(&mut csv_file, optim_step, avg_loss, acc, pred_stats)?;
                save_model(&nn, checkpoint_folder, optim_step)?;
            }

            optim_step += 1;
            pb.inc(1);
        }
    }

    println!("Final validation");
    validation(&mut nn, &test_dataset, in_channels, h, w, &pb)?;
    nn.to_checkpoint(&format!(
        "{checkpoint_folder}/final_imgsize{h}_batch{batch_size}_lr{learning_rate:0.4}_adam.csv"
    ))?;
    println!("Training completed!");
    Ok(())
}

fn save_metrics(
    csv_file: &mut File,
    //
    optim_step: usize,
    avg_loss: f32,
    val_acc: f32,
    pred_stats: HashMap<u8, u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(
        csv_file,
        "{},{:.4},{:.4},'{:?}'",
        optim_step, avg_loss, val_acc, pred_stats
    )?;
    csv_file.flush()?;
    Ok(())
}

fn validation(
    nn: &mut NN,
    test_dataset: &Dataset,
    //
    in_channels: usize,
    h: usize,
    w: usize,
    pb: &ProgressBar,
) -> Result<(f32, HashMap<u8, u64>), Box<dyn std::error::Error>> {
    // Run test loop to calculate accuracy
    if DEBUG {
        println!("Running test loop...");
    }

    let (total_correct, total_samples, pred_stats) = test_dataset
        .samples
        .par_iter()
        .fold(
            || (0, 0, HashMap::<u8, u64>::new()),
            |(mut correct, mut total, mut stats), (image_path, label)| {
                let img_f32 = load_and_preprocess_image(&image_path, h as u32, w as u32);
                let input = Array4::from_shape_vec((1, in_channels, h, w), img_f32)
                    .expect("Failed to create test input array");

                let output = nn.clone().forward(input.into_dyn());

                let predicted_label = output
                    .into_dimensionality::<Ix2>()
                    .expect("Output should be castable to 2D")
                    .row(0)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in output"))
                    .map(|(idx, _)| idx)
                    .unwrap() as u8;

                // Update local thread stats
                *stats.entry(predicted_label).or_insert(0) += 1; // NOTE: how is there not a data
                                                                 // race here ?!

                if predicted_label == *label {
                    correct += 1;
                }
                total += 1;

                (correct, total, stats)
            },
        )
        .reduce(
            || (0, 0, HashMap::new()),
            |mut a, b| {
                // Merge the maps from different threads
                for (label, count) in b.2 {
                    *a.2.entry(label).or_insert(0) += count;
                }
                (a.0 + b.0, a.1 + b.1, a.2)
            },
        );

    let accuracy = total_correct as f32 / total_samples as f32;
    pb.println(format!("[TEST ACC] {:.3}", accuracy));
    pb.println(format!("[TEST STATS] {:?}\n", pred_stats));

    Ok((accuracy, pred_stats))
}

fn save_model(
    nn: &NN,
    checkpoint_folder: &str,
    optim_step: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let ckpt_path = Path::new(checkpoint_folder).join(format!("checkpoint_{optim_step}.json"));
    let ckpt_path = ckpt_path.to_str().unwrap();
    nn.to_checkpoint(ckpt_path)?;

    Ok(ckpt_path.to_string())
}
