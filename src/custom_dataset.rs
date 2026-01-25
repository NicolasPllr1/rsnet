use image::imageops::FilterType;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct MetadataSchema {
    train: HashMap<String, Vec<String>>,
    test: HashMap<String, Vec<String>>,
}

pub struct Dataset {
    pub samples: Vec<(PathBuf, u8)>, // path, label
}

fn load_metadata(root_data_dir: &Path) -> MetadataSchema {
    let json_str =
        fs::read_to_string(root_data_dir.join("metadata.json")).expect("Metadata missing");
    let metadata: MetadataSchema = serde_json::from_str(&json_str).expect("JSON parse error");
    metadata
}

pub fn load_dataset(data_dir: &str, test_data_percentage: Option<f32>) -> (Dataset, Dataset) {
    if let Some(p) = test_data_percentage {
        assert!(0.0 < p && p < 1.0);
    }

    let root = Path::new(data_dir);
    let metadata = load_metadata(root);

    let mut train_samples = Vec::new();
    let mut test_samples = Vec::new();

    for (label_str, paths) in metadata.train {
        let label = label_str.parse::<u8>().unwrap() - 1; // NOTE: careful here!
        for p in paths {
            train_samples.push((root.join(p), label));
        }
    }
    for (label_str, paths) in metadata.test {
        let label = label_str.parse::<u8>().unwrap() - 1; // NOTE: careful here!
        for p in paths {
            test_samples.push((root.join(p), label));
        }
    }

    // Shuffling
    let mut rng = thread_rng();
    train_samples.shuffle(&mut rng);
    test_samples.shuffle(&mut rng);

    if let Some(p) = test_data_percentage {
        // let train_len = (train_samples.len() as f32 * r) as usize;
        // train_samples.truncate(train_len);

        let test_len = (test_samples.len() as f32 * p) as usize;
        test_samples.truncate(test_len);
    }

    (
        Dataset {
            samples: train_samples,
        },
        Dataset {
            samples: test_samples,
        },
    )
}

pub fn process_image(path: &PathBuf, target_h: u32, target_w: u32) -> Vec<f32> {
    let img = image::open(path).expect("Failed to open image");
    let greyscale = img.grayscale();
    let resized = greyscale.resize_exact(target_w, target_h, FilterType::Triangle);
    let rgb = resized.to_luma32f();
    let pixels = rgb.into_raw();
    let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
    pixels.into_iter().map(|p| p - mean).collect()
}
