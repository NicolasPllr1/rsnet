use crate::model::{Module, NN};

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
    fn step(
        &self,
        nn: &mut NN,
        cost_function: CostFunction,
        labels: &[u8],
        output: &Array2<f32>,
    ) -> Array1<f32>;
}

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    fn step(
        &self,
        nn: &mut NN,
        cost_function: CostFunction,
        labels: &[u8],
        output: &Array2<f32>,
    ) -> Array1<f32> {
        // Calculate loss and gradient for the batch
        let (loss, grad) = cost_function(labels, output);
        // Backpropagate through layers in reverse order
        nn.backward(grad.into_dyn());

        nn.step(self.learning_rate);

        loss
    }
}
