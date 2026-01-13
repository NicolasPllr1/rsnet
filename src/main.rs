mod run;
mod train;

use ndarray::prelude::*;

use indicatif::ProgressIterator; // Adds .progress() to iterators (like tqdm)
use mnist::MnistBuilder;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

/// Loads the MNIST dataset - downloads automatically if not cached
/// This is just some black magic which uses the ubyte files in data. Do not touch those.
/// Returns: (train_images, train_labels, test_images, test_labels)
/// - Images are Vec<u8> with pixel values 0-255
/// - Labels are Vec<u8> with digit values 0-9
pub fn load_mnist() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mnist = MnistBuilder::new()
        .base_path("data/")
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    (
        mnist.trn_img, // 60,000 * 784 bytes (28x28 images flattened)
        mnist.trn_lbl, // 60,000 labels
        mnist.tst_img, // 10,000 * 784 bytes
        mnist.tst_lbl, // 10,000 labels
    )
}

trait Module {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>; // Input is (batch_size, features)
    fn backward(&mut self, next_layer_err: Array2<f32>) -> Array2<f32>;
    fn step(&mut self, learning_rate: f32);
}

#[derive(Serialize, Deserialize, Debug)]
///  z = W.a_prev + b
struct FcLayer {
    input_size: usize,
    output_size: usize,
    //
    weights: Array2<f32>, // (input_size, output_size)
    bias: Array1<f32>,    //  (output_size)
    // for backprop
    last_input: Option<Array2<f32>>, // (batch_size, input_size), this is the prev layer activation
    //
    w_grad: Option<Array2<f32>>, // (input_size, output_size)
    b_grad: Option<Array1<f32>>, // (output_size)
}

impl FcLayer {
    fn new(input_size: usize, output_size: usize) -> FcLayer {
        FcLayer {
            input_size,
            output_size,
            weights: FcLayer::init_2d_mat(input_size, output_size),
            bias: FcLayer::init_bias(output_size),
            //
            last_input: None,
            //
            w_grad: None,
            b_grad: None,
        }
    }
    fn init_bias(output_size: usize) -> Array1<f32> {
        return Array1::random(output_size, Uniform::new(0., 1.0).unwrap());
    }
    fn init_2d_mat(input_size: usize, output_size: usize) -> Array2<f32> {
        return Array2::random((input_size, output_size), Uniform::new(0., 1.0).unwrap());
    }
}

impl Module for FcLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        // store input for backprop computations
        self.last_input = Some(input.clone()); // cloning ...

        // (batch_size, input_size) X (input_size, output_size) = (batch_size, output_size)
        input.dot(&self.weights) + self.bias.clone()
    }

    fn backward(&mut self, next_layer_err: Array2<f32>) -> Array2<f32> {
        let last_input = self
            .last_input
            .take()
            .expect("Need to do a forward pass before the backward");

        // Gradients for this layer weights
        // w: (batch_size, input_size)^T X (batch_size, output_size) = (input_size, output_size)
        self.w_grad = Some(last_input.t().dot(&next_layer_err));
        // b: (batch_size, output_size) summed over batch-axis = (output_size)
        self.b_grad = Some(next_layer_err.sum_axis(Axis(0)));

        //  What needs to be passed on to the 'previous' layer in the network
        //  (batch_size, output_size) X (input_size, output_size)^T
        next_layer_err.dot(&self.weights.t()) // (input_size, batch_size)
    }
    fn step(&mut self, learning_rate: f32) {
        self.weights -= self.w_grad.unwrap() * learning_rate;
        self.bias -= self.b_grad.unwrap() * learning_rate;
        // reset gradients
        self.w_grad = None;
        self.b_grad = None;
        // reset input
        self.last_input = None;
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct ReluLayer {
    last_input: Array2<f32>,
}

impl Module for ReluLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.0))
    }

    fn backward(&mut self, next_layer_err: Array2<f32>) -> Array2<f32> {
        self.last_input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * next_layer_err
    }
    fn step(&mut self, learning_rate: f32) {
        self.last_input = None;
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SoftMaxLayer {
    last_output: Array2<f32>,
}

impl Module for SoftMaxLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        let max = input.fold_axis(Axis(1), f32::NEG_INFINITY, |&a, &b| a.max(b));
        // exp(x - max)
        let mut out = input.clone() - max.insert_axis(Axis(1));
        out.mapv_inplace(|x| x.exp());

        let sum = out.sum_axis(Axis(1));
        let out = input.mapv(|x| x.exp()) / sum;

        // for backprop
        self.last_output = out.clone();

        out
    }

    fn backward(&mut self, labels: Array2<f32>) -> Array2<f32> {
        // NOTE: input to the softmax backward is the labels
        // labels: (batch_size, K), and there are K=9 classes for MNIST

        // NOTE: ASSUMING cross-entropy loss, which simplifies nicely with softmax during backprop
        self.last_output.clone() - labels // TODO: maybe broadcast?
    }
}

// #[derive(Debug)]
// struct ConvLayer {
//     nb_channels: usize,
//     height: usize,
//     width: usize,
//     kernels: Vec<Array4<f32>>, // (output_channels, input_channels, height, width)
//     bias: Array1<f32>,         // (features)
// }

// impl ConvLayer {
//     fn new(
//         nb_channels: usize,
//         in_channels: usize,
//         out_channels: usize,
//         height: usize,
//         width: usize,
//     ) -> ConvLayer {
//         let mut kernels = Vec::new();
//         for _ in 0..nb_channels {
//             kernels.push(ConvLayer::init_conv_kernel(
//                 in_channels,
//                 out_channels,
//                 height,
//                 width,
//             ));
//         }

//         ConvLayer {
//             nb_channels,
//             height,
//             width,
//             kernels,
//             bias: ConvLayer::init_bias(out_channels), // TODO: check if this is really one
//                                                       // bias per output channels
//         }
//     }

//     fn init_conv_kernel(
//         in_channels: usize,
//         out_channels: usize,
//         height: usize,
//         width: usize,
//     ) -> Array4<f32> {
//         return Array4::random(
//             (in_channels, out_channels, height, width),
//             Uniform::new(0., 1.0).unwrap(),
//         );
//     }

//     fn init_bias(output_size: usize) -> Array1<f32> {
//         return Array1::random(output_size, Uniform::new(0., 1.0).unwrap());
//     }
// }

// impl Module for ConvLayer {
//     fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
//         todo!()
//     }

//     fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
//         todo!()
//     }
// }

#[derive(Serialize, Deserialize, Debug)]
enum Layer {
    FC(FcLayer),
    ReLU(ReluLayer),
    // Conv(ConvLayer),
    // MaxPooling,
    Softmax(SoftMaxLayer),
}

impl Module for Layer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::FC(l) => l.forward(input),
            Layer::ReLU(l) => l.forward(input),
            // Layer::Conv(_) => todo!(),
            Layer::Softmax(l) => l.forward(input),
        }
    }

    fn backward(&mut self, next_layer_err: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::FC(l) => l.backward(next_layer_err),
            Layer::ReLU(l) => l.backward(next_layer_err),
            // Layer::Conv(_) => todo!(),
            Layer::Softmax(l) => l.backward(next_layer_err),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct NN {
    // layers: Vec<Box<dyn Module>>,
    layers: Vec<Layer>,
}

impl Module for NN {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        let mut x = input;
        for layer in &mut self.layers {
            x = layer.forward(x);
        }
        x
    }

    fn backward(&mut self, next_layer_err: Array2<f32>) -> Array2<f32> {
        let mut x = next_layer_err;
        // Iterate layers in reverse order, mutate each as we go
        for layer in self.layers.iter_mut().rev() {
            x = layer.backward(x);
        }
        x
    }
    fn step(&mut self, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.step(learning_rate);
        }
    }
}

impl NN {
    /// Save the neural network to a checkpoint file
    pub fn to_checkpoint(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(filepath)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load a neural network from a checkpoint file
    pub fn from_checkpoint(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filepath)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let nn: NN = serde_json::from_str(&contents)?;
        Ok(nn)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <train|run> [arguments...]", args[0]);
        eprintln!("  train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride>  - Train a neural network");
        eprintln!("  run <checkpoint> <example_file>                                          - Run inference using a checkpoint file");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            if args.len() < 6 {
                eprintln!(
                    "Error: 'train' requires gradient steps, learning rate, checkpoint folder, and checkpoint stride"
                );
                eprintln!(
                    "Usage: {} train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride>",
                    args[0]
                );
                std::process::exit(1);
            }
            let train_steps: usize = match args[2].parse() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("Error: Invalid number of steps '{}': {}", args[2], e);
                    std::process::exit(1);
                }
            };
            let learning_rate: f32 = match args[3].parse() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("Error: Invalid learning rate '{}': {}", args[3], e);
                    std::process::exit(1);
                }
            };
            let checkpoint_folder = &args[4];
            let checkpoint_stride: usize = match args[5].parse() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("Error: Invalid checkpoint stride '{}': {}", args[5], e);
                    std::process::exit(1);
                }
            };

            // Validate that train_steps is a multiple of checkpoint_stride
            if train_steps % checkpoint_stride != 0 {
                eprintln!(
                    "Error: train_steps ({}) must be a multiple of checkpoint_stride ({})",
                    train_steps, checkpoint_stride
                );
                std::process::exit(1);
            }

            if let Err(e) = train::train(
                train_steps,
                learning_rate,
                checkpoint_folder,
                checkpoint_stride,
            ) {
                eprintln!("Error during training: {}", e);
                std::process::exit(1);
            }
        }
        "run" => {
            if args.len() < 4 {
                eprintln!("Error: 'run' requires a checkpoint file path and example file");
                eprintln!("Usage: {} run <checkpoint_path> <example_file>", args[0]);
                eprintln!(
                    "  example_file should be a raw binary file with 784 bytes (28x28 MNIST image)"
                );
                std::process::exit(1);
            }
            if let Err(e) = run::run(&args[2], &args[3]) {
                eprintln!("Error running inference: {}", e);
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Error: Unknown command '{}'", args[1]);
            eprintln!("Usage: {} <train|run> [arguments...]", args[0]);
            eprintln!(
                "  train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride>  - Train a neural network"
            );
            eprintln!("  run <checkpoint> <example_file>                     - Run inference using a checkpoint file");
            std::process::exit(1);
        }
    }
}
