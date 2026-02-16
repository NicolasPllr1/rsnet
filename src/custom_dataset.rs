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

pub struct KFoldDataset {
    pub k: usize,
    pub samples: Vec<(PathBuf, u8)>, // path, label
}

impl KFoldDataset {
    pub fn new(data_dir: &str, k: usize) -> KFoldDataset {
        let root = Path::new(data_dir);
        let metadata = load_metadata(root);

        let mut samples = Vec::new();

        for (label_str, paths) in metadata.train {
            let label = label_str.parse::<u8>().unwrap() - 1; // NOTE: careful here!
            for p in paths {
                samples.push((root.join(p), label));
            }
        }
        for (label_str, paths) in metadata.test {
            let label = label_str.parse::<u8>().unwrap() - 1; // NOTE: careful here!
            for p in paths {
                samples.push((root.join(p), label));
            }
        }

        // Shuffling
        let mut rng = thread_rng();
        samples.shuffle(&mut rng);

        KFoldDataset { k, samples }
    }

    /// Returns a train / test datasets split, where the test dataset
    /// is the `test_fold_idx` fold, and all other folds are put in the
    /// training dataset.
    pub fn get_fold(&self, test_fold_idx: usize) -> (Dataset, Dataset) {
        assert!(test_fold_idx < self.k, "Fold index out of bounds");

        let total_len = self.samples.len();
        let fold_size = total_len / self.k;

        let mut train_samples = Vec::new();
        let mut test_samples = Vec::new();

        // Calculate the start and end indices for the test fold
        let start = test_fold_idx * fold_size;
        let end = if test_fold_idx == self.k - 1 {
            total_len
        } else {
            start + fold_size
        };

        for (i, sample) in self.samples.iter().enumerate() {
            if start <= i && i < end {
                test_samples.push(sample.clone());
            } else {
                train_samples.push(sample.clone());
            }
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
}

pub fn load_and_preprocess_image(path: &PathBuf, target_h: u32, target_w: u32) -> Vec<f32> {
    let img = image::open(path).expect("Failed to open image");
    let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
    let pixels = resized.to_luma32f(); // luma32f is greyscale in [0,1] ; rgb32f for rgb
    let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
    pixels.iter().map(|p| p - mean).collect()
}
