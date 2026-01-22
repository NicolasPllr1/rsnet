pub use crate::layers::activations::{ReluLayer, SoftMaxLayer};
pub use crate::layers::conv::{Conv2Dlayer, FlattenLayer, MaxPoolLayer};
pub use crate::layers::fc::FcLayer;
pub use crate::model::Module;

use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32;

pub mod activations;
pub mod conv;
pub mod fc;

#[derive(Serialize, Deserialize, Debug)]
pub enum Layer {
    FC(FcLayer),
    Conv(Conv2Dlayer),
    Pool(MaxPoolLayer),
    ReLU(ReluLayer),
    Softmax(SoftMaxLayer),
    Flatten(FlattenLayer),
}

impl Module for Layer {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Layer::FC(l) => l.forward(input),
            Layer::Conv(l) => l.forward(input),
            Layer::Pool(l) => l.forward(input),
            Layer::ReLU(l) => l.forward(input),
            Layer::Softmax(l) => l.forward(input),
            Layer::Flatten(l) => l.forward(input),
        }
    }

    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Layer::FC(l) => l.backward(dz),
            Layer::Conv(l) => l.backward(dz),
            Layer::Pool(l) => l.backward(dz),
            Layer::ReLU(l) => l.backward(dz),
            Layer::Softmax(l) => l.backward(dz),
            Layer::Flatten(l) => l.backward(dz),
        }
    }

    fn zero_grad(&mut self) {
        match self {
            Layer::FC(l) => l.zero_grad(),
            Layer::Conv(l) => l.zero_grad(),
            Layer::Pool(l) => l.zero_grad(),
            Layer::ReLU(l) => l.zero_grad(),
            Layer::Softmax(l) => l.zero_grad(),
            Layer::Flatten(l) => l.zero_grad(),
        }
    }

    /*
    fn step(&mut self, lr: f32) {
        match self {
            Layer::FC(l) => l.step(lr),
            Layer::Conv(l) => l.step(lr),
            Layer::Pool(l) => l.step(lr),
            Layer::ReLU(l) => l.step(lr),
            Layer::Softmax(l) => l.step(lr),
            Layer::Flatten(l) => l.step(lr),
        }
    }

    fn get_weight_grads(&mut self) -> Option<Vec<(ArrayD<f32>, Option<ArrayD<f32>>)>> {
        match self {
            Layer::FC(l) => l.get_weight_grads(),
            Layer::Conv(l) => l.get_weight_grads(),
            Layer::Pool(l) => l.get_weight_grads(),
            Layer::ReLU(l) => l.get_weight_grads(),
            Layer::Softmax(l) => l.get_weight_grads(),
            Layer::Flatten(l) => l.get_weight_grads(),
        }
    }
    */
}
