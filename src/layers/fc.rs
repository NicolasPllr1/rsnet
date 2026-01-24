pub use crate::model::Module;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::f32;

#[derive(Serialize, Deserialize, Debug)]
///  z = W.a_prev + b
pub struct FcLayer {
    input_size: usize,
    output_size: usize,
    //
    pub weights: Array2<f32>, // (input_size, output_size)
    pub bias: Array1<f32>,    //  (output_size)
    // for backprop
    last_input: Option<Array2<f32>>, // (batch_size, input_size), this is the prev layer activation
    //
    pub w_grad: Option<Array2<f32>>, // (input_size, output_size)
    pub b_grad: Option<Array1<f32>>, // (output_size)
}

impl FcLayer {
    pub fn new(input_size: usize, output_size: usize) -> FcLayer {
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
    fn init_bias(_input_size: usize, output_size: usize) -> Array1<f32> {
        // return Array1::random(output_size, Uniform::new(-1.0, 1.0).unwrap())
        //     * FcLayer::get_scale(input_size);
        return Array1::zeros(output_size);
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

    // fn get_weight_grads(&mut self) -> Option<Vec<(ArrayD<f32>, Option<ArrayD<f32>>)>> {
    //     let w_grad = self
    //         .w_grad
    //         .take()
    //         .expect("Gradient should be filled")
    //         .into_dyn();
    //     let b_grad = self
    //         .b_grad
    //         .take()
    //         .expect("Gradient should be filled")
    //         .into_dyn();
    //
    //     Some(vec![(w_grad, Some(b_grad))])
    // }

    fn zero_grad(&mut self) {
        self.w_grad = None;
        self.b_grad = None;
    }
}
