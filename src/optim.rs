use crate::layers::Layer;
use crate::model::NN;

use ndarray::Array2;

use ndarray::prelude::*;

pub type CostFunction = fn(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>);

pub fn cross_entropy(labels: &[u8], actual_y: &Array2<f32>) -> (Array1<f32>, Array2<f32>) {
    let batch_size = labels.len();
    let num_classes = actual_y.ncols();

    // Convert labels to one-hot encoding Array2
    let mut expected_y = Array2::zeros((batch_size, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        let label_idx = label - 1;
        expected_y[(i, label_idx as usize)] = 1.0;
    }

    // Calculate cross-entropy for each sample in batch: -log(p)
    let log_probs = (actual_y + 1e-10).ln();
    let loss = -(expected_y.clone() * log_probs).fold_axis(Axis(1), 0.0, |&a, &b| a + b);

    (loss, expected_y)
}

pub trait Optimizer {
    /// Update the neural network weights in place using the deltas (w := w + delta).
    /// TODO: write a default impl. for all optimizer
    // fn update_weights(nn: &mut NN, deltas: Vec<Option<(ArrayD<f32>, ArrayD<f32>)>>);

    /// Run one optimization step.
    /// Note:
    /// - Assumes forward+backward pass have been done.
    /// - May mutate the optimizer internal state (e.g. momentum).
    fn step(&mut self, nn: &mut NN);
}

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    /// NOTE: Ideally, optimizers should have to known about all layers.
    /// Layers should be the one exposing their weights and gradients in an mutably iterable
    /// fashion, so that you can get all tuples (&mut w, &mut w_grad) for any weights the layer may hold and
    /// want the optimizer to optimize
    fn step(&mut self, nn: &mut NN) {
        for layer in &mut nn.layers {
            match layer {
                Layer::FC(fc_layer) => {
                    fc_layer.weights -= &(fc_layer.w_grad.as_ref().unwrap() * self.learning_rate);
                    fc_layer.bias -= &(fc_layer.b_grad.as_ref().unwrap() * self.learning_rate);
                }
                Layer::Conv(conv2_dlayer) => {
                    conv2_dlayer.kernels_mat -=
                        &(conv2_dlayer.k_grad.as_ref().unwrap() * self.learning_rate);
                    conv2_dlayer.b -= &(conv2_dlayer.b_grad.as_ref().unwrap() * self.learning_rate);
                }
                _ => (), // no weights to update in other layers
            }
        }
    }
}

pub struct SGDMomentum {
    pub learning_rate: f32,
    pub viscosity: f32,
    pub velocity: Vec<Option<(ArrayD<f32>, ArrayD<f32>)>>, // velocity per grad
}

impl SGDMomentum {
    pub fn new(nn: &NN, learning_rate: f32, viscosity: f32) -> SGDMomentum {
        let mut velocity = Vec::new();
        for layer in &nn.layers {
            match layer {
                Layer::FC(_) => {
                    velocity.push(None);
                }
                Layer::Conv(_) => {
                    velocity.push(None);
                }
                _ => (), // no weights to update in other layers
            }
        }
        SGDMomentum {
            learning_rate,
            viscosity,
            velocity,
        }
    }
}

impl Optimizer for SGDMomentum {
    fn step(&mut self, nn: &mut NN) {
        let mut next_velocity = Vec::new();
        for (layer, prev_velocities) in nn.layers.iter_mut().zip(self.velocity.iter_mut()) {
            match layer {
                Layer::FC(fc_layer) => {
                    if prev_velocities.is_none() {
                        // First step ever. No velocities yet. Update == vanilla SGD update.
                        let w_delta = -fc_layer.w_grad.as_ref().unwrap() * self.learning_rate;
                        fc_layer.weights += &(w_delta);

                        let b_delta = -fc_layer.b_grad.as_ref().unwrap() * self.learning_rate;
                        fc_layer.bias += &(b_delta);

                        // Record velocities
                        next_velocity.push(Some((w_delta, b_delta)));
                    }

                    let (prev_w_delta, prev_b_delta) = prev_velocities.clone().unwrap(); // NOTE: Q: clone necessary??

                    // dynamic --> fixed dimensionality.  NOTE: necessary?
                    let prev_w_delta = prev_w_delta.into_dimensionality::<Ix2>().unwrap();
                    let prev_b_delta = prev_b_delta.into_dimensionality::<Ix1>().unwrap();

                    let w_delta = prev_w_delta * self.viscosity
                        - fc_layer.w_grad.as_ref().unwrap() * self.learning_rate;
                    fc_layer.weights += &(w_delta);

                    let b_delta = prev_b_delta * self.viscosity
                        - fc_layer.b_grad.as_ref().unwrap() * self.learning_rate;
                    fc_layer.bias += &(b_delta);

                    // Record velocities
                    next_velocity.push(Some((w_delta, b_delta)));

                    fc_layer.weights -= &(fc_layer.w_grad.as_ref().unwrap() * self.learning_rate);
                    fc_layer.bias -= &(fc_layer.b_grad.as_ref().unwrap() * self.learning_rate);
                }
                Layer::Conv(conv2_dlayer) => {
                    conv2_dlayer.kernels_mat -=
                        &(conv2_dlayer.k_grad.as_ref().unwrap() * self.learning_rate);
                    conv2_dlayer.b -= &(conv2_dlayer.b_grad.as_ref().unwrap() * self.learning_rate);
                }
                _ => (), // no weights to update in other layers
            }
        }
    }
}
