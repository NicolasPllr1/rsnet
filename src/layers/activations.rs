pub use crate::model::Module;

use ndarray::prelude::*;
use onnx_protobuf::NodeProto;
use serde::{Deserialize, Serialize};

use std::f32;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ReluLayer {
    last_input: Option<ArrayD<f32>>,
}

impl ReluLayer {
    pub fn new() -> ReluLayer {
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

    fn zero_grad(&mut self) {
        self.last_input = None;
    }

    fn to_onnx(
        &self,
        input_name: String,
        layer_idx: usize,
        graph: &mut onnx_protobuf::GraphProto,
    ) -> String {
        let layer_name = format!("relu_{layer_idx}");
        let output_name = format!("{layer_name}_out");

        let relu_node = NodeProto {
            name: layer_name,
            input: vec![input_name],
            output: vec![output_name.clone()],
            op_type: "Relu".to_string(),
            ..Default::default()
        };
        graph.node.push(relu_node);
        output_name
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct SoftMaxLayer {
    last_output: Option<Array2<f32>>,
}

impl SoftMaxLayer {
    pub fn new() -> SoftMaxLayer {
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

        let batch_size = labels.shape()[0];

        // NOTE: ASSUMING cross-entropy loss, which simplifies nicely with softmax during backprop
        let unormalized_dz = self.last_output.clone().unwrap() - labels; // TODO: maybe broadcast?
        unormalized_dz / batch_size as f32
    }

    fn zero_grad(&mut self) {
        self.last_output = None;
    }

    fn to_onnx(
        &self,
        input_name: String,
        layer_idx: usize,
        graph: &mut onnx_protobuf::GraphProto,
    ) -> String {
        let layer_name = format!("softmax_{layer_idx}");
        let output_name = "probs".to_string(); // NOTE: assume its the last layer and is unique in the
                                               // network

        // default: softmax over the last axis
        // https://onnx.ai/onnx/operators/onnx__Softmax.html
        let softmax_node = NodeProto {
            name: layer_name,
            input: vec![input_name],
            output: vec![output_name.clone()],
            op_type: "Softmax".to_string(),
            ..Default::default()
        };
        graph.node.push(softmax_node);
        output_name
    }
}
