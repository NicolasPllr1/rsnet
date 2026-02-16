use crate::custom_dataset::{load_and_preprocess_image, load_dataset, Dataset};
use crate::model::{Module, NN};
use crate::optim::{CostFunction, OptiName};
use crate::DEBUG;

use indicatif::{ProgressBar, ProgressStyle};
use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub struct TrainConfig {
    pub data_dir: String,
    //
    pub batch_size: usize,
    pub nb_epochs: usize,
    //
    pub cost_function: CostFunction,
    pub optimizer_name: OptiName,
    pub learning_rate: f32,
}

pub struct CheckpointConfig {
    pub checkpoint_folder: Option<String>,
    pub checkpoint_stride: usize,
    pub loss_csv_path: String,
}

/// Train a neural network
pub fn train(
    mut nn: NN,
    train_cfg: TrainConfig,
    ckpt_cfg: CheckpointConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ref ckpt_path) = ckpt_cfg.checkpoint_folder {
        fs::create_dir_all(ckpt_path)?; // in case the folder does not exist
    }

    let (train_dataset, test_dataset) = load_dataset(&train_cfg.data_dir, None);
    println!("[TRAIN] len: {}", train_dataset.samples.len());
    println!("[TEST] len: {}\n", test_dataset.samples.len());

    let mut indices: Vec<usize> = (0..train_dataset.samples.len()).collect();

    let (in_channels, h, w) = (1, 64, 64); // Downscaled size

    // Create or truncate CSV file and write header
    let mut csv_file = fs::File::create(ckpt_cfg.loss_csv_path)?;
    writeln!(csv_file, "step,loss,duration,accuracy,stats")?;

    let mut optimizer = train_cfg
        .optimizer_name
        .build_optimizer(&nn, train_cfg.learning_rate);
    println!("[OPTIMIZER] {:?}", train_cfg.optimizer_name);

    let nb_epochs = train_cfg.nb_epochs;
    let batch_size = train_cfg.batch_size;

    let pb = ProgressBar::new(
        nb_epochs as u64 * (train_dataset.samples.len() / train_cfg.batch_size) as u64,
    );
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .expect("progress bar init"),
    );
    let mut optim_step = 1;

    for _epoch in 0..train_cfg.nb_epochs {
        indices.shuffle(&mut thread_rng());
        for batch_indices in indices.chunks_exact(batch_size) {
            let mut batch_images = Vec::with_capacity(batch_size * in_channels * h * w);
            let mut batch_labels = Vec::new();

            // TODO: measure this loop perf normal/release and try a // version
            for &idx in batch_indices {
                let (path, label) = &train_dataset.samples[idx];
                let processed_pixels = load_and_preprocess_image(path, h as u32, w as u32);

                batch_images.extend(processed_pixels);
                batch_labels.push(*label);
            }

            let input = if nn.is_cnn() {
                Array4::from_shape_vec((batch_size, in_channels, h, w), batch_images.clone())
                    .map_err(|e| format!("Failed to create input array: {}", e))?
                    .into_dyn()
            } else {
                Array2::from_shape_vec((batch_size, h * w), batch_images)
                    .map_err(|e| format!("Failed to create input array: {}", e))?
                    .into_dyn()
            };

            // ----------
            nn.zero_grad();
            let output = nn.forward(input);
            let output = output
                .into_dimensionality::<Ix2>() // (batch_size, num_classes)
                .expect("Network output should be 2D: (batch_size, num_classes)");
            let (loss, init_grad) = (train_cfg.cost_function)(&batch_labels, &output);
            nn.backward(init_grad.into_dyn());
            optimizer.step(&mut nn);
            // ----------

            if optim_step % ckpt_cfg.checkpoint_stride == 0 {
                let avg_loss = loss.sum() / loss.len() as f32; // batch loss
                pb.println(format!(
                    "[BATCH] step {} loss: {:.4}\n",
                    optim_step, avg_loss
                ));
                let (acc, pred_stats) = validation(&mut nn, &test_dataset, in_channels, h, w, &pb)?;

                save_metrics(&mut csv_file, optim_step, avg_loss, acc, pred_stats)?;

                if let Some(ref ckpt_path) = ckpt_cfg.checkpoint_folder {
                    save_model(&nn, ckpt_path, optim_step)?;
                }
            }

            optim_step += 1;
            pb.inc(1);
        }
    }

    pb.println("Final validation");
    validation(&mut nn, &test_dataset, in_channels, h, w, &pb)?;

    if let Some(ref ckpt_path) = ckpt_cfg.checkpoint_folder {
        nn.to_checkpoint(&format!(
            "{ckpt_path}/final_imgsize{h}_batch{batch_size}_lr{:0.4}_adam.csv",
            train_cfg.learning_rate
        ))?;
    }
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
                let img_f32 = load_and_preprocess_image(image_path, h as u32, w as u32);

                let input = if nn.is_cnn() {
                    Array4::from_shape_vec((1, in_channels, h, w), img_f32)
                        .expect("[VAL] Failed to create 4D input array for CNN")
                        .into_dyn()
                } else {
                    Array2::from_shape_vec((1, h * w), img_f32)
                        .expect("[VAL] Failed to create 2D input array")
                        .into_dyn()
                };

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
