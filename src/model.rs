use crate::layers::Layer;
use ndarray::prelude::*;
use onnx_protobuf::tensor_proto::DataType;
use onnx_protobuf::tensor_shape_proto::{dimension::Value, Dimension};
use onnx_protobuf::type_proto;
use onnx_protobuf::{
    GraphProto, Message, ModelProto, OperatorSetIdProto, TensorShapeProto, TypeProto,
    ValueInfoProto,
};
use protobuf::MessageField;
use serde::{Deserialize, Serialize};
use std::f32;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug)]
pub struct ImageShape {
    pub channels: i64,
    pub height: i64,
    pub width: i64,
}

impl ImageShape {
    pub fn new(channels: i64, height: i64, width: i64) -> ImageShape {
        ImageShape {
            channels,
            height,
            width,
        }
    }
}

pub trait Module {
    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32>; // Input is (batch_size, features)
    /// Backward pass
    ///
    /// The `backward` function receives a gradient `dz` which corresponds to dLoss/dz,
    /// where z is the current layer output.
    /// The job of the backward pass is to combine its 'local gradients' - namely dz/dx, dz/dw -
    /// with this incoming gradient - dLoss/dz - using the chain rule.
    ///
    /// The backward should:
    ///
    /// - return dLoss/dx = dz/dx * dLoss/dz
    /// - fill in the gradients for the layer's own weights dLoss/dw = dz/dw * dLoss/dz
    ///
    /// Notice how both dLoss/dx (which will get returned) and dLoss/dw (which will get saved for later
    /// optimization) are computed combining the incoming dLoss/dz and local gradient information.
    /// The combination itself being specified by the chain rule.
    ///
    /// Note:
    /// - the dz/dx Jacobian matrix is not materialized (too wasteful). Instead, each layer
    ///   directly computes the matrix-vector product of interest.
    /// - the shape of `backward` output - which corresponds to dLoss/dx - is the same shape as the layer inputs.
    fn backward(&mut self, dz: ArrayD<f32>) -> ArrayD<f32>;
    fn zero_grad(&mut self);
    fn to_onnx(&self, input_name: String, layer_idx: usize, graph: &mut GraphProto) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NN {
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
        // Iterate layers in *reverse* order, mutate each as we go
        for layer in self.layers.iter_mut().rev() {
            x = layer.backward(x);
        }
        x
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    fn to_onnx(&self, input_name: String, layer_idx: usize, graph: &mut GraphProto) -> String {
        graph.name = "model".to_string();

        let mut input_name = input_name;
        let mut layer_idx = layer_idx;
        for layer in &self.layers {
            input_name = layer.to_onnx(input_name, layer_idx, graph);
            layer_idx += 1;
        }

        let final_output_name = input_name; // This is returned from the last layer
        graph.output = vec![ValueInfoProto {
            name: final_output_name.clone(),
            type_: MessageField::some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DataType::FLOAT as i32,
                    shape: MessageField::some(TensorShapeProto {
                        dim: vec![
                            Dimension {
                                value: Some(Value::DimParam("batch".to_string())),
                                ..Default::default()
                            },
                            Dimension {
                                value: Some(Value::DimValue(5)),
                                ..Default::default()
                            }, // probs over output classes
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }];

        final_output_name
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

    pub fn save_as_onnx_model(
        &self,
        file_path: &str,
        image_shape: ImageShape,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(file_path)?;
        let mut graph = GraphProto {
            input: NN::_onnx_input_info(image_shape),
            ..Default::default()
        };

        self.to_onnx("input".to_string(), 0, &mut graph);

        let model = ModelProto {
            graph: protobuf::MessageField(Some(Box::new(graph))),
            ir_version: 11,
            opset_import: vec![OperatorSetIdProto {
                version: 15,
                ..Default::default()
            }],
            ..Default::default()
        };
        file.write_all(&model.write_to_bytes()?)?;
        Ok(())
    }

    pub fn is_cnn(&self) -> bool {
        matches!(self.layers.first(), Some(Layer::Conv(_)))
    }

    /// Specifies the input information for the ONNX format.
    /// Note: assumes input is an image, i.e. has a channels and height/width.
    fn _onnx_input_info(image_shape: ImageShape) -> Vec<ValueInfoProto> {
        let input_dims = NN::_onnx_input_dims(image_shape);
        vec![ValueInfoProto {
            name: "input".to_string(),
            type_: MessageField::some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: DataType::FLOAT as i32,
                    shape: MessageField::some(TensorShapeProto {
                        dim: input_dims,
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }]
    }

    /// Specifies the input dimensions (dynamic batch size and fixed spatial dims)
    /// of the model in the ONNX format.
    /// Note: assumes input is an image, i.e. has a channels and height/width.
    fn _onnx_input_dims(image_shape: ImageShape) -> Vec<Dimension> {
        vec![
            Dimension {
                value: Some(Value::DimParam("batch".to_string())),
                ..Default::default()
            },
            Dimension {
                value: Some(Value::DimValue(image_shape.channels)),
                ..Default::default()
            }, // channels
            Dimension {
                value: Some(Value::DimValue(image_shape.height)),
                ..Default::default()
            }, // height
            Dimension {
                value: Some(Value::DimValue(image_shape.width)),
                ..Default::default()
            }, // width
        ]
    }
}
