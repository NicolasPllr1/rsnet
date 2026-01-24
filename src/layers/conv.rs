use crate::model::Module;
use crate::DEBUG;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::f32;

/// 2D convolution layer (without padding and with stride=1).
/// pytorch doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Conv2Dlayer {
    in_channels: usize,          // Number of channels in the input image
    out_channels: usize,         // Number of channels produced by the convolution
    kernel_size: (usize, usize), // Size of all the 2d convolving kernels used in this layer.
    stride: usize,               // Stride of the convolution. Will be hardocoded to 1 for now.
    // weights
    pub kernels_mat: Array2<f32>, // Layout for img2col: (out_channels, in_channels*k^2)
    pub b: Array1<f32>,           // One bias per output channel: (output_channels)
    // for backprop
    last_input: Option<Array3<f32>>, // The 'patches' matrix in img2col: (batch_size, locations, in_channels * k^2)
    //
    pub k_grad: Option<Array2<f32>>, // (in_channels, out_channels, kernel_size)
    pub b_grad: Option<Array1<f32>>, // (out_channels)
}

impl Conv2Dlayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Conv2Dlayer {
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
        // (2.0 / (in_channels * kernel_size.0 * kernel_size.1) as f32).sqrt()
        (6.0 / (in_channels * kernel_size.0 * kernel_size.1) as f32).sqrt() // 2 -> 6, uniform vs normal
    }

    fn init_bias(
        _in_channels: usize,
        out_channels: usize,
        _kernel_size: (usize, usize),
    ) -> Array1<f32> {
        // return Array1::random(out_channels, Uniform::new(-1.0, 1.0).unwrap())
        //     * Conv2Dlayer::get_scale(in_channels, kernel_size);
        Array1::zeros(out_channels)
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
        if DEBUG {
            println!("[forward] [conv] input: {:?}", input.shape());
        }
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
        if DEBUG {
            println!("[backward] [conv] incoming dz: {:?}", dz.shape());
        }
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
        let mut dpatches =
            Array3::<f32>::zeros((batch_size, self.in_channels * k * k, nb_locations));
        // Compute gradient for each sample in the batch
        for batch_idx in 0..batch_size {
            let dz_sample = dz.slice(s![batch_idx, .., ..]);
            // (in_channels*k^2, locations) = (out_channels, in_channels*k^2)^T dot (out_channels, locations)
            let sample_patches_grad = self.kernels_mat.t().dot(&dz_sample);
            dpatches
                .slice_mut(s![batch_idx, .., ..])
                .assign(&sample_patches_grad);
        }

        // Building the dL/dinput from the dL/dpatches
        let mut dinput = Array4::zeros((batch_size, self.in_channels, height, width));

        // Transpose / reshape dpatches from (batch_size, in_channels * k^2, locations)
        // to (batch_size, locations, in_channels, k, k)
        let binding = dpatches.permuted_axes([0, 2, 1]); // (batch_size, locations, in_channels * k^2)
        let all_grad_patches = binding.to_shape((batch_size, nb_locations, self.in_channels, k, k)).expect("[backward] [conv] img2col patches gradient is compatible with the shape (batch_size, locations, in_channels, k, k)");
        // Iterating over the patches
        for batch_idx in 0..batch_size {
            let grad_patches = all_grad_patches.slice(s![batch_idx, .., .., .., ..]);
            for (patch_idx, patch_grad) in grad_patches.outer_iter().enumerate() {
                // patch top corner position
                let top_y = patch_idx / out_width;
                let top_x = patch_idx - top_y * out_width; // = patch_idx modulo out_width
                let mut dinput_slice =
                    dinput.slice_mut(s![batch_idx, .., top_y..top_y + k, top_x..top_x + k]);
                dinput_slice += &patch_grad;
                // accumulating the gradient within dinput
            }
        }

        dinput.into_dyn()
    }

    fn zero_grad(&mut self) {
        self.k_grad = None;
        self.b_grad = None;
    }

    /*
    fn get_weight_grads(&mut self) -> Option<Vec<(ArrayD<f32>, Option<ArrayD<f32>>)>> {
        let k_grad = self
            .k_grad
            .take()
            .expect("Gradient should be filled")
            .into_dyn();

        let b_grad = self
            .b_grad
            .take()
            .expect("Gradient should be filled")
            .into_dyn();

        Some(vec![(k_grad, Some(b_grad))])
    }
    */
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MaxPoolLayer {
    pool_size: (usize, usize),
    // for backprop
    last_input_max_mask: Option<Array6<f32>>, // (batch_size, in_channels, height/k, k, width/k, k)
}

impl MaxPoolLayer {
    pub fn new(pool_size: (usize, usize)) -> MaxPoolLayer {
        assert!(pool_size.0 == pool_size.1);
        MaxPoolLayer {
            pool_size,
            last_input_max_mask: None,
        }
    }
}

impl Module for MaxPoolLayer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        let input = input
            .into_dimensionality::<Ix4>()
            .expect("[forward] [maxPool] input is a 4D tensor");

        let (batch_size, in_channels, height, width) = input.dim();
        let k = self.pool_size.0;

        // Reshape to (batch_size, in_channels, height / k, k, width / k, k)
        let input_6d = input
            .to_shape((batch_size, in_channels, height / k, k, width / k, k))
            .expect("[forward] [maxPool] input is compatible with 6D tensor for the pooling");

        // Fold the dims with size k, i.e axis 3 and 5 of input
        let pooled: Array4<f32> = input_6d
            .fold_axis(Axis(3), f32::NEG_INFINITY, |&a, &b| a.max(b))
            .fold_axis(Axis(5 - 1), f32::NEG_INFINITY, |&a, &b| a.max(b));

        // Creating a mask for the input to know where the max values are
        // (for backprop)
        let pooled_6d = pooled
            .to_shape((batch_size, in_channels, height / k, 1, width / k, 1))
            .unwrap();

        let mut input_mask_6d =
            Array6::zeros((batch_size, in_channels, height / k, k, width / k, k));

        // expected: [128, 10, 12, 2, 12, 2], got: [128, 10, 12, 1, 12, 1]

        Zip::from(&mut input_mask_6d)
            .and(&input_6d)
            .and_broadcast(&pooled_6d)
            .for_each(|w, &in_val, &max_val| {
                if in_val == max_val {
                    // NOTE: multiple value could be == max and thus be marked
                    // as 1.0 in the mask, which will lead to the gradient being duplicated durin
                    // gthe backward. Leaving as is for now.
                    *w += 1.0
                }
            });

        self.last_input_max_mask = Some(input_mask_6d);

        pooled.into_dyn()
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        // dz: (batch_size, channels, height/k, width/k)
        let dz = dz
            .into_dimensionality::<Ix4>()
            .expect("[backward] [maxPool] dz is 4D");

        let (batch_size, channels, out_height, out_width) = dz.dim();
        let k = self.pool_size.0;
        let (height, width) = (out_height * k, out_width * k);

        let dz_6d = dz
            .to_shape((batch_size, channels, out_height, 1, out_width, 1))
            .unwrap();

        let input_mask = self
            .last_input_max_mask
            .as_ref()
            .expect("[backward] [maxPool] Run forward before backward");

        let mut dinput = Array6::zeros((batch_size, channels, out_height, k, out_width, k));

        // input mask: (batch_size, in_channels, height/k, k, width/k, k)
        Zip::from(&mut dinput)
            .and(input_mask)
            .and_broadcast(&dz_6d)
            .for_each(|din, &mask_val, &dz_val| {
                if mask_val == 1.0 {
                    *din += dz_val;
                }
            });

        let dinput = dinput
            .into_shape_with_order((batch_size, channels, height, width))
            .expect("[backward] [maxPool] dinput is compatible with expected 4D tensor");

        dinput.to_owned().into_dyn()
    }

    fn zero_grad(&mut self) {
        self.last_input_max_mask = None;
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlattenLayer {
    last_input: Option<ArrayD<f32>>,
}

impl FlattenLayer {
    pub fn new() -> FlattenLayer {
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

    fn zero_grad(&mut self) {
        self.last_input = None;
    }
}
