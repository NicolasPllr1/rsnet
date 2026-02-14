pub use crate::layers::activations::{ReluLayer, SoftMaxLayer};
pub use crate::layers::conv::{Conv2Dlayer, FlattenLayer, MaxPoolLayer};
pub use crate::layers::fc::FcLayer;
pub use crate::model::Module;

use ndarray::prelude::*;
use onnx_protobuf::GraphProto;
use serde::{Deserialize, Serialize};
use std::f32;

pub mod activations;
pub mod conv;
pub mod fc;

#[derive(Serialize, Deserialize, Debug, Clone)]
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

    fn to_onnx(&self, input_name: String, layer_idx: usize, graph: &mut GraphProto) -> String {
        match self {
            Layer::FC(fc_layer) => fc_layer.to_onnx(input_name, layer_idx, graph),
            Layer::Conv(conv2_dlayer) => conv2_dlayer.to_onnx(input_name, layer_idx, graph),
            Layer::Pool(max_pool_layer) => max_pool_layer.to_onnx(input_name, layer_idx, graph),
            Layer::ReLU(relu_layer) => relu_layer.to_onnx(input_name, layer_idx, graph),
            Layer::Softmax(soft_max_layer) => soft_max_layer.to_onnx(input_name, layer_idx, graph),
            Layer::Flatten(flatten_layer) => flatten_layer.to_onnx(input_name, layer_idx, graph),
        }
    }
}
