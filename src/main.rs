mod run;
mod train;

use ndarray::prelude::*;

// use indicatif::ProgressIterator; // Adds .progress() to iterators (like tqdm)
use mnist::MnistBuilder;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::f32;
use std::fs::File;
use std::io::{Read, Write};

trait Module {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32>; // Input is (batch_size, features)
    /// Backward pass
    ///
    /// The `backward` function receives a gradient `dz` which corresponds to dLoss/dz,
    /// where z is the current layer output.
    /// The job of the backward pass is to combine its 'local gradients' - namely dz/dx, dz/dw -
    /// with this incoming gradient - dLoss/dz - using the chain rule.
    /// The backward should:
    /// - return dLoss/dx = dz/dx * dLoss/dz
    /// - fill in the gradients for the layer's own weights dLoss/dw = dz/dw * dLoss/dz
    ///
    /// Notice how both dLoss/dx (which will get returned) and dLoss/dw (which will get saved for later
    /// optimization) are computed combining the incoming dLoss/dz and local gradient information.
    /// The combination itself being specified by the chain rule.
    ///
    /// Note:
    /// - the dz/dx Jacobian matrix is not materialized (too wasteful). Instead, each layer
    /// directly computes the matrix-vector product of interest.
    /// - the shape of the function output - which corresponds to dLoss/dx - is the same shape
    /// as the layer inputs.
    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32>;
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
            bias: FcLayer::init_bias(input_size, output_size),
            //
            last_input: None,
            //
            w_grad: None,
            b_grad: None,
        }
    }
    fn get_scale(input_size: usize) -> f32 {
        (2.0 / input_size as f32).sqrt()
    }
    fn init_bias(input_size: usize, output_size: usize) -> Array1<f32> {
        return Array1::random(output_size, Uniform::new(-1.0, 1.0).unwrap())
            * FcLayer::get_scale(input_size);
    }
    fn init_2d_mat(input_size: usize, output_size: usize) -> Array2<f32> {
        return Array2::random((input_size, output_size), Uniform::new(-1.0, 1.0).unwrap())
            * FcLayer::get_scale(input_size);
    }
}

impl Module for FcLayer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        // store input for backprop computations
        let input = input
            .into_dimensionality::<Ix2>()
            .expect("FC layer input should be 2D");
        self.last_input = Some(input.clone()); // cloning ...

        // (batch_size, input_size) X (input_size, output_size) = (batch_size, output_size)
        let out = input.dot(&self.weights) + self.bias.clone();
        let out = out.into_dyn(); // dynamic array type
        out
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        let dz = dz
            .into_dimensionality::<Ix2>()
            .expect("FC layer backward input should be 2D");

        let last_input = self
            .last_input
            .take()
            .expect("Need to do a forward pass before the backward");

        // Gradients for this layer weights
        // w: (batch_size, input_size)^T X (batch_size, output_size) = (input_size, output_size)
        let batch_size = dz.shape()[0] as f32;
        self.w_grad = Some(last_input.t().dot(&dz) / batch_size);
        // b: (batch_size, output_size) summed over batch-axis = (output_size)
        self.b_grad = Some(dz.sum_axis(Axis(0)) / batch_size);

        //  What needs to be passed on to the 'previous' layer in the network
        //  (batch_size, output_size) X (input_size, output_size)^T
        let new_dz = dz.dot(&self.weights.t()); // (input_size, batch_size)
        let new_dz = new_dz.into_dyn(); // dynamic array type
        new_dz
    }
    fn step(&mut self, lr: f32) {
        self.weights -= &(self.w_grad.take().unwrap() * lr);

        self.bias -= &(self.b_grad.take().unwrap() * lr);
        // reset gradients
        self.w_grad = None;
        self.b_grad = None;
        // reset input
        self.last_input = None;
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct ReluLayer {
    last_input: Option<ArrayD<f32>>,
}

impl ReluLayer {
    fn new() -> ReluLayer {
        ReluLayer { last_input: None }
    }
}

impl Module for ReluLayer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        self.last_input = Some(input.clone());
        input.mapv(|x| x.max(0.0))
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        self.last_input
            .clone()
            .expect("run forward before backward")
            .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
            * dz
    }
    fn step(&mut self, _learning_rate: f32) {
        self.last_input = None;
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SoftMaxLayer {
    last_output: Option<Array2<f32>>,
}

impl SoftMaxLayer {
    fn new() -> SoftMaxLayer {
        SoftMaxLayer { last_output: None }
    }
}

impl Module for SoftMaxLayer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        let input = input
            .into_dimensionality::<Ix2>()
            .expect("Input to sofmax should be 2D");

        let max = input.fold_axis(Axis(1), f32::NEG_INFINITY, |&a, &b| a.max(b));
        // exp(x - max)
        let mut out = input.clone() - max.insert_axis(Axis(1));
        out.mapv_inplace(|x| x.exp());

        let sum = out.sum_axis(Axis(1));
        let out = out / sum.insert_axis(Axis(1));
        let out = out.into_dyn();

        // for backprop
        self.last_output = Some(
            out.clone()
                .into_dimensionality::<Ix2>()
                .expect("Softmax output should be 2D"),
        );

        out
    }

    fn backward(&mut self, labels: ArrayD<f32>) -> ArrayD<f32> {
        // NOTE: input to the softmax backward is the labels
        // labels: (batch_size, K), and there are K=9 classes for MNIST

        // NOTE: ASSUMING cross-entropy loss, which simplifies nicely with softmax during backprop
        self.last_output.clone().unwrap() - labels // TODO: maybe broadcast?
    }

    fn step(&mut self, _learning_rate: f32) {
        self.last_output = None;
    }
}

/// 2D convolution layer (without padding and with stride=1).
/// pytorch doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
#[derive(Serialize, Deserialize, Debug)]
struct Conv2Dlayer {
    in_channels: usize,          // Number of channels in the input image
    out_channels: usize,         // Number of channels produced by the convolution
    kernel_size: (usize, usize), // Size of all the 2d convolving kernels used in this layer.
    stride: usize,               // Stride of the convolution. Will be hardocoded to 1 for now.
    // weights
    kernels_mat: Array2<f32>, // Layout for img2col: (out_channels, in_channels*k^2)
    b: Array1<f32>,           // One bias per output channel: (output_channels)
    // for backprop
    last_input: Option<Array3<f32>>, // The 'patches' matrix in img2col: (batch_size, locations, in_channels * k^2)
    //
    k_grad: Option<Array2<f32>>, // (in_channels, out_channels, kernel_size)
    b_grad: Option<Array1<f32>>, // (out_channels)
}

impl Conv2Dlayer {
    fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Conv2Dlayer {
        Conv2Dlayer {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            //
            kernels_mat: Conv2Dlayer::init_kernel(in_channels, out_channels, kernel_size),
            b: Conv2Dlayer::init_bias(in_channels, out_channels, kernel_size),
            //
            last_input: None,
            //
            k_grad: None,
            b_grad: None,
        }
    }
    fn get_scale(in_channels: usize, kernel_size: (usize, usize)) -> f32 {
        (2.0 / (in_channels * kernel_size.0 * kernel_size.1) as f32).sqrt()
    }

    fn init_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Array1<f32> {
        return Array1::random(out_channels, Uniform::new(-1.0, 1.0).unwrap())
            * Conv2Dlayer::get_scale(in_channels, kernel_size);
    }
    fn init_kernel(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Array2<f32> {
        assert!(kernel_size.0 == kernel_size.1);
        let k = kernel_size.0;
        // Kernel weights layed-out for the 'img2col' method to compute the convolution.
        // Dimensions: (out_channels, in_channels*k^2)
        return Array2::random(
            (out_channels, in_channels * k * k),
            Uniform::new(-1.0, 1.0).unwrap(),
        ) * Conv2Dlayer::get_scale(in_channels, kernel_size);
    }
}

impl Module for Conv2Dlayer {
    /// Forward for the convolution layer using the 'img2col' method.
    ///
    /// The 'img2col' idea is to map the convolution operation to a single matmul.
    /// The goal is to compute OUT = kernels_mat x patches_mat, where patches_mat
    /// is a matrix where columns correspond to entire input patches to the convolution kernel.
    /// In terme of size (ommiting about the batch dim to simplify):
    /// - kernels_mat: (out_channels, channels_in * k^2)
    /// - patches_mat: (channels_in * k ^2, locations)
    /// So their multiplication yiels: (out_channels, locations)
    /// By locations, we means every valid coordinate in the input feature map volume
    /// where the kernel can be used to compute a value through the convolution.
    ///
    /// Input: (batch_size, in_channels, height, width)
    /// Output: (batch_size, out_channels, height-k+1, width-k+1), where kernel_size=(k,k)
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        println!("[forward] [conv] input: {:?}", input.shape());
        let input = input
            .into_dimensionality::<Ix4>()
            .expect("Conv layer input should be 4D");

        // Using input dimensions to initialize the output 4D tensor
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        assert!(in_channels == self.in_channels);
        assert!(self.kernel_size.0 == self.kernel_size.1);
        let k = self.kernel_size.0;
        let out_height = in_height - k + 1;
        let out_width = in_height - k + 1;
        let mut out = Array4::zeros((batch_size, self.out_channels, out_height, out_width));

        // NOTE: maybe switch batch and channels dim for easier operations?

        // The 'kernels' 2D tensor already has the correct shape for the img2col method.
        // kernels_mat: (out_channels, channels_in * k ^2)
        // - Each row is associated with a single output channel
        // - Each row is the vector of kernel weights associated to every input channels

        // Preparing the 'patches' matrix for the img2col method.
        // patches_mat: (in_channels * k^2, locations)^T
        // - Each column is associated to a single location where the kernels will be applied
        // - Each column is the flattened vector of input values for the given location accross all input channels
        //
        // We will build this 'patches' matrix by iterating over the locations in the input volume.
        // Each input location (in_channels, k, k) we will get flattened to (in_channels*k*k) and
        // inserted in the patches_matrix as a column.

        // The number of locations for a 'valid' convolution with stride=1
        let nb_locations = (in_height - k + 1) * (in_width - k + 1);

        // Init the 'last input' which corresponds to the entire patches matrix.
        // Will be needed during backprop.
        let mut last_input = Array3::<f32>::zeros((batch_size, nb_locations, in_channels * k * k));

        // Computing the 'img2col' matmul on a per-batch basis.
        // So we are computing the convolution one item from the batch at a time, and progressively filling the output matrix.
        for (batch_idx, input_feature_maps) in input.outer_iter().enumerate() {
            // Get input patches: (in_channels, in_height, in_width) -> (L, in_channels*k^2)
            // where (k,k) is the kernel size and L the number of patch locations.

            // There are L patches of size (in_channels, k, k)
            let patches = input_feature_maps.windows((in_channels, k, k));
            // We just want to lay them in a matrix where each row is a flattened patch
            let mut patches_mat = Array2::zeros((nb_locations, in_channels * k * k));

            // For each patch, we flatten it and put it as a row in the patches matrix
            for (mut patches_mat_row, patch) in patches_mat.rows_mut().into_iter().zip(patches) {
                patches_mat_row.assign(&patch.flatten());
            }

            // Cache this batch 'patches' matrix
            last_input
                .slice_mut(s![batch_idx, .., ..])
                .assign(&patches_mat);

            // The "img2col" matmul which implement the convolution as a single GEMM.
            // (out_channels, L) = (out_channels, in_channels*k^2) dot (L, in_channels*k^2)^T
            let mut flattened_output_feature_map = self.kernels_mat.dot(&patches_mat.t());

            // Add bias: same bias per output_channel
            // --> (out_channels, 1) broadcasted to (out_channels, L)
            flattened_output_feature_map += &self.b.view().insert_axis(Axis(1));

            // Re-shaping the output feature map
            // (out_channels, L) -> (out_channels, out_height, out_width)

            assert!(out_width * out_height == nb_locations);
            let output_feature_map = flattened_output_feature_map.into_shape_with_order((
                self.out_channels,
                out_height,
                out_width,
            )).expect("Flattened output feature map dimensions are compatible with the 3d shape (out_channels, out_width, out_height)");

            // Add this batch output feature maps to the 4D output tensor
            // (we are computing the convolution one item from the batch at a time)
            out.index_axis_mut(Axis(0), batch_idx)
                .assign(&output_feature_map);
        }

        // Cache the input patches matrix
        self.last_input = Some(last_input);

        let out = out.into_dyn();
        out
    }

    /// Backward for the convolution layer using the 'img2col' method.
    ///
    /// With the 'img2col' method, the convolution forward becomes: output = kernels_mat dot patches_mat,
    /// With dims: (out_channels, locations) = (out_channels, channels_in * k ^2) dot (channels_in * k ^2, locations)
    /// This matmul is done on a per-batch basis, i.e. we for-loop over the batch dim
    /// and process one batch item at a time.
    /// Then the full output is reshaped to (batch_size, out_channels, height-k+1, width-k+1)
    ///
    /// For the backward, we reshape the incoming dz:
    /// from (out_channels, height-k+1, width-k+1) to (out_channels, locations),
    /// where locations = (height-k+1)*(width-k+1) = out_h*out_w.
    ///
    /// And we make use of the last input saved by the layer, to compute local gradient
    /// and use the chain rule to get kernels/bias gradient and dL/dx to propagate to the prev
    /// layer.
    ///
    /// Gradients:
    /// - dL/dkernels_mat = dL/dconv_output * (dconv_output/dkernels_mat)^T = dz dot patches_mat^T.
    /// In sizes: (out_channels, in_channels * k^2) = (out_channels, locations) dot (in_channels * k^2, locations)^T.
    /// Note that dz here refered to the reshaped incoming dz.
    ///
    /// - DL/dbias : (out_channels) = dz.sum(-1).sum(0)
    /// For the bias, we want to reduce for every output channels all the gradient values at every
    /// location, since the same bias was added to all these location (for a given output channel), and also sum over the batch dimension.
    ///
    /// - dL/dinput ? We compute dL/dpatches_mat and then reshape it for the input matrix.
    /// - DL/dpatches_mat = doutput/dinput dot dL/doutput = kernels_mat^T dot dz
    /// in sizes: (in_channels * k^2, locations) = (out_channels, in_channels * k^2)^T dot (out_channels, locations)
    /// Which we then map carefully to: (channels_in, height, width).
    /// Method to re-shape from (channels_in * k ^2, locations) to (channels_in, height, width):
    /// 0. Init empty input grad tensor with zeros and shape (channels_in, height, width)
    /// 1. transpose&reshape the matmul output (dL/dpatches_mat) from (channels_in * k^2, locations) to (locations, channels_in, k, k)
    /// - Let's call this reshaped tensor grad_patches
    /// 2. Iterate over the locations. For each location:
    ///     - we have a (channels_in, k, k) row (the patch which influenced the output at that particular location)
    ///     - we then populate the input grad tensor, += (accumulation!) the kxk values which are all around this particular location
    /// Mapping between a patch location and the corresponding coordinate in the input window:
    /// - The method to do this mapping is to find the top left corner coordinate and go from there
    /// - For location index l: top_y = l // out_width, top_x = l % out_width
    /// - And then we can get all the input coordinate using ranges
    /// - We want to fill this volume (ignoring the batch dim): input_grad[:, top_y..top_y+k,top_x..top_x+k] '=' grad_patches[l]
    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        // dz: (batch_size, out_channels, out_height, out_width)
        println!("[backward] [conv] incoming dz: {:?}", dz.shape());
        let dz = dz
            .into_dimensionality::<Ix4>()
            .expect("[backward] [conv] incoming dz is 4D");
        let (batch_size, out_channels, out_height, out_width) = dz.dim();

        // Reshape incoming dz
        let nb_locations = out_height * out_width;
        // new dz shape: (batch_size, out_channels, locations)
        let dz = dz
            .to_shape((batch_size, out_channels, nb_locations))
            .expect("[backward] [conv] incoming dz is compatible with img2col shape");

        // last input ~ 'patches' matrix: (batch_size, in_channels * k ^2, locations)
        let last_patches_mat = self
            .last_input
            .as_ref()
            .expect("Run forward before the backward");

        // transpose the last two dim
        let k = self.kernel_size.0;
        let last_patches_mat = last_patches_mat
            .to_shape((batch_size, nb_locations, self.in_channels * k * k))
            .expect("Last patch matrix shape is known and we can transpose the last two dim");

        // dL/dkernels_mat: (batch_size, out_channels, in_channels * k ^2)
        // = (batch_size, out_channels, locations) dot (batch_size, locations, in_channels * k^2)
        // Iterating over the batch dimension to accumulate the gradient, only computing matmuls (2D)

        let mut dkernels_mat: Array2<f32> = Array2::zeros((out_channels, self.in_channels * k * k));
        for batch_idx in 0..batch_size {
            dkernels_mat += &dz
                .slice(s![batch_idx, .., ..])
                .dot(&last_patches_mat.slice(s![batch_idx, .., ..]));
        }

        // Average over the batch
        let dkernels_mat = dkernels_mat / batch_size as f32;

        self.k_grad = Some(dkernels_mat);

        // dL/dbias
        self.b_grad = Some(
            dz.fold_axis(Axis(2), 0.0, |&a, &b| a + b) // sum over locations
                .fold_axis(Axis(0), 0.0, |&a, &b| a + b)
                / batch_size as f32, // sum over batch
        );

        // dL/dinput, to be returned for the prev. layer to use for its own backprop
        let height = out_height + k - 1;
        let width = out_width + k - 1;
        // First compute dL/dpatches.
        // Will accumulate gradients over the batch dim
        let mut acc_dpatches = Array2::<f32>::zeros((self.in_channels * k * k, nb_locations));
        // Accumulate the gradient accross the batch
        for sample_dz in dz.outer_iter() {
            // (in_channels*k^2, locations) = (out_channels, in_channels*k^2)^T dot (out_channels, locations)
            acc_dpatches += &self.kernels_mat.t().dot(&sample_dz);
        }

        // Building the dL/dinput from the dL/dpatches
        let mut dinput = Array3::zeros((self.in_channels, height, width));

        // Transpose / reshape dpatches from (in_channels * k^2, locations)
        // to (locations, in_channels, k, k)
        let binding = acc_dpatches.t();
        let grad_patches = binding.to_shape((nb_locations, self.in_channels, k, k)).expect("[backward] [conv] img2col patches gradient is compatible with the shape (locations, in_channels, k, k)");
        // Iterating over the patches
        for (patch_idx, patch_grad) in grad_patches.outer_iter().enumerate() {
            // patch top corner position
            let top_y = patch_idx / out_width;
            let top_x = patch_idx - top_y * out_width; // = patch_idx modulo out_width
            dinput
                .slice_mut(s![.., top_y..top_y + k, top_x..top_x + k])
                .assign(&patch_grad);
            // accumulating the gradient within dinput
        }

        dinput.into_dyn()
    }

    fn step(&mut self, lr: f32) {
        self.kernels_mat -= &(self.k_grad.take().unwrap() * lr);

        self.b -= &(self.b_grad.take().unwrap() * lr);
        // reset gradients
        self.k_grad = None;
        self.b_grad = None;
        // reset input
        self.last_input = None;
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct FlattenLayer {
    last_input: Option<ArrayD<f32>>,
}

impl FlattenLayer {
    fn new() -> FlattenLayer {
        FlattenLayer { last_input: None }
    }
}

impl Module for FlattenLayer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        self.last_input = Some(input.clone());
        // Assuming the input is a batch of feature maps
        let input = input
            .into_dimensionality::<Ix4>()
            .expect("Flatten layer input should be 4D");
        let (batch_size, in_channels, height, width) = input.dim();
        let output = input
            .to_shape((batch_size, in_channels * height * width))
            .expect("flatten input to 2D array should not fail")
            .to_owned()
            .into_dyn();
        output
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        let last_input = self
            .last_input
            .clone()
            .expect("Need to do a forward pass before the backward");
        let new_dz = dz
            .to_shape(last_input.shape())
            .expect("should be able to reshape the incoming gradient")
            .to_owned();

        new_dz
    }
    fn step(&mut self, _learning_rate: f32) {
        self.last_input = None;
    }
}

#[derive(Serialize, Deserialize, Debug)]
enum Layer {
    FC(FcLayer),
    Conv(Conv2Dlayer),
    ReLU(ReluLayer),
    Softmax(SoftMaxLayer),
    Flatten(FlattenLayer),
}

impl Module for Layer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Layer::FC(l) => l.forward(input),
            Layer::Conv(l) => l.forward(input),
            Layer::ReLU(l) => l.forward(input),
            Layer::Softmax(l) => l.forward(input),
            Layer::Flatten(l) => l.forward(input),
        }
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Layer::FC(l) => l.backward(dz),
            Layer::Conv(l) => l.backward(dz),
            Layer::ReLU(l) => l.backward(dz),
            Layer::Softmax(l) => l.backward(dz),
            Layer::Flatten(l) => l.backward(dz),
        }
    }

    fn step(&mut self, lr: f32) {
        match self {
            Layer::FC(l) => l.step(lr),
            Layer::Conv(l) => l.step(lr),
            Layer::ReLU(l) => l.step(lr),
            Layer::Softmax(l) => l.step(lr),
            Layer::Flatten(l) => l.step(lr),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct NN {
    // layers: Vec<Box<dyn Module>>,
    layers: Vec<Layer>,
}

impl Module for NN {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        let mut x = input;
        for layer in &mut self.layers {
            x = layer.forward(x);
        }
        x
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        let mut x = dz;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <train|run> [arguments...]", args[0]);
        eprintln!("  train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride> <loss_csv>  - Train a neural network");
        eprintln!("  run <checkpoint> <example_file>                                          - Run inference using a checkpoint file");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            if args.len() < 7 {
                eprintln!(
                    "Error: 'train' requires gradient steps, learning rate, checkpoint folder, checkpoint stride, and loss CSV path"
                );
                eprintln!(
                    "Usage: {} train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride> <loss_csv>",
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

            let loss_csv_path = &args[6];

            if let Err(e) = train::train(
                train_steps,
                learning_rate,
                checkpoint_folder,
                checkpoint_stride,
                loss_csv_path,
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
                "  train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride> <loss_csv>  - Train a neural network"
            );
            eprintln!("  run <checkpoint> <example_file>                     - Run inference using a checkpoint file");
            std::process::exit(1);
        }
    }
}
