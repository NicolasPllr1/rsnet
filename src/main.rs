use ndarray::prelude::*;

use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use mnist::MnistBuilder;
use indicatif::ProgressIterator;  // Adds .progress() to iterators (like tqdm)

/// Loads the MNIST dataset - downloads automatically if not cached
/// This is just some black magic which uses the ubyte files in data. Do not touch those.
/// Returns: (train_images, train_labels, test_images, test_labels)
/// - Images are Vec<u8> with pixel values 0-255
/// - Labels are Vec<u8> with digit values 0-9
fn load_mnist() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mnist = MnistBuilder::new()
        .base_path("data/")
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    (
        mnist.trn_img,  // 60,000 * 784 bytes (28x28 images flattened)
        mnist.trn_lbl,  // 60,000 labels
        mnist.tst_img,  // 10,000 * 784 bytes
        mnist.tst_lbl,  // 10,000 labels
    )
}

trait Module {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>; // Input is (batch_size, features)
    fn backward(&mut self, next_layer_grad: Array2<f32>) -> Array2<f32>;
}

#[derive(Debug)]
struct FcLayer {
    input_size: usize,
    output_size: usize,
    weights: Array2<f32>, // (input_size, output_size)
    bias: Array1<f32>,    //  (output_size)
    // for backprop
    last_input: Option<Array2<f32>>, // (batch_size, input_size)
    //
    w_grad: Option<Array2<f32>>, // (batch_size, input_size, output_size)
    b_grad: Option<Array2<f32>>, // (batch_size, output_size)
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
        todo!()
    }
}

#[derive(Debug)]
struct ConvLayer {
    nb_channels: usize,
    height: usize,
    width: usize,
    kernels: Vec<Array4<f32>>, // (output_channels, input_channels, height, width)
    bias: Array1<f32>,         // (features)
}

impl ConvLayer {
    fn new(
        nb_channels: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> ConvLayer {
        let mut kernels = Vec::new();
        for _ in 0..nb_channels {
            kernels.push(ConvLayer::init_conv_kernel(
                in_channels,
                out_channels,
                height,
                width,
            ));
        }

        ConvLayer {
            nb_channels,
            height,
            width,
            kernels,
            bias: ConvLayer::init_bias(out_channels), // TODO: check if this is really one
                                                      // bias per output channels
        }
    }

    fn init_conv_kernel(
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> Array4<f32> {
        return Array4::random(
            (in_channels, out_channels, height, width),
            Uniform::new(0., 1.0).unwrap(),
        );
    }

    fn init_bias(output_size: usize) -> Array1<f32> {
        return Array1::random(output_size, Uniform::new(0., 1.0).unwrap());
    }
}

impl Module for ConvLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        todo!()
    }

    fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

#[derive(Debug)]
enum Layer {
    FC(FcLayer),
    Conv(ConvLayer),
    ReLU,
    Softmax,
}

impl Module for Layer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        match self {
            Layer::FC(fc_layer) => fc_layer.forward(input),
            Layer::Conv(_) => todo!(),
            Layer::ReLU => todo!(),
            Layer::Softmax => todo!(),
        }
    }

    fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

#[derive(Debug)]
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

    fn backward(&mut self, _post_grad: Array2<f32>) -> Array2<f32> {
        todo!()
    }
}

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
