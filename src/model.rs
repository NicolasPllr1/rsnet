use crate::layers::Layer;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32;

use std::fs::File;
use std::io::{Read, Write};

pub trait Module {
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
    // fn get_weight_grads(&mut self) -> Option<(ArrayD<f32>, ArrayD<f32>)>;
    fn zero_grad(&mut self);
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NN {
    // layers: Vec<Box<dyn Module>>,
    pub layers: Vec<Layer>,
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

    // fn get_weight_grads(&mut self) -> Option<(ArrayD<f32>, ArrayD<f32>)> {
    // let mut w = Vec::new();
    // let mut count = 0;
    // for layer in &mut self.layers {
    //     if let Some(l_w) = layer.get_weight_grads() {
    //         w.extend(l_w);
    //         count += 1;
    //     }
    // }
    //
    // if count == 0 {
    //     return None;
    // }
    // Some(w)
    //     todo!()
    // }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
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
