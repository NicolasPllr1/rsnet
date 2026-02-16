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
        expected_y[(i, label as usize)] = 1.0;
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
    pub fn new(nn: &NN, learning_rate: f32) -> SGDMomentum {
        let mut velocity = Vec::new();
        for layer in &nn.layers {
            match layer {
                Layer::FC(_) => {
                    velocity.push(None);
                }
                Layer::Conv(_) => {
                    velocity.push(None);
                }
                _ => velocity.push(None), // no weights to update in other layers
            }
        }
        SGDMomentum {
            learning_rate,
            viscosity: 0.9,
            velocity,
        }
    }
}

impl Optimizer for SGDMomentum {
    ///  SDG with momentum does two updates:
    ///
    ///  Pytorch does:
    ///  1) v_{t+1} = viscosity*v_t + g_t
    ///  2) w_{t+1} = w_t - lr*v_{t+1}
    ///
    ///  In the original paper (~Sutskever), the updates are:
    ///  1) v_{t+1} = viscosity*v_t - lr*g_t
    ///  2) w_{t+1} = w_t + v_{t+1}
    ///
    /// Implementing the Sutskever version here.
    fn step(&mut self, nn: &mut NN) {
        let mut next_velocity: Vec<Option<(ArrayD<f32>, ArrayD<f32>)>> = Vec::new();

        let mut total_grad_norm_sq = 0.0;
        let mut zero_grad_count = 0;
        let mut total_param_count = 0;

        for (layer, prev_velocities) in nn.layers.iter_mut().zip(self.velocity.iter_mut()) {
            match layer {
                Layer::FC(fc_layer) => {
                    let w_grad = fc_layer.w_grad.as_ref().expect("[FC] w grad");
                    let b_grad = fc_layer.b_grad.as_ref().expect("[FC] bias grad");

                    // Metrics Collection
                    total_grad_norm_sq += w_grad.iter().map(|x| x * x).sum::<f32>();
                    total_grad_norm_sq += b_grad.iter().map(|x| x * x).sum::<f32>();
                    zero_grad_count += w_grad.iter().filter(|&&x| x == 0.0).count();
                    total_param_count += w_grad.len() + b_grad.len();

                    let max_w = fc_layer.weights.iter().map(|x| x.abs()).fold(0.0, f32::max);
                    println!("[OPTIM] Max FC Weight: {}", max_w);

                    let lr = self.learning_rate;
                    let mu = self.viscosity;

                    if prev_velocities.is_none() {
                        // First step ever. No velocities yet. Update == vanilla SGD update.
                        let w_velocity = -lr * w_grad;
                        fc_layer.weights += &(w_velocity);

                        let b_velocity = -lr * b_grad;
                        fc_layer.bias += &(b_velocity);

                        // Record velocities
                        next_velocity.push(Some((w_velocity.into_dyn(), b_velocity.into_dyn())));
                    } else {
                        let (prev_w_velocity, prev_b_velocity) =
                            prev_velocities.clone().expect("[FC] prev velocities"); // NOTE: Q: clone necessary??

                        // dynamic --> fixed dimensionality.  NOTE: necessary?
                        let prev_w_velocity = prev_w_velocity.into_dimensionality::<Ix2>().unwrap();
                        let prev_b_velocity = prev_b_velocity.into_dimensionality::<Ix1>().unwrap();

                        let w_velocity = mu * prev_w_velocity - lr * w_grad;
                        fc_layer.weights += &(w_velocity);

                        let b_velocity = mu * prev_b_velocity - lr * b_grad;
                        fc_layer.bias += &(b_velocity);

                        // Record velocities
                        next_velocity.push(Some((w_velocity.into_dyn(), b_velocity.into_dyn())));
                    }
                }
                Layer::Conv(conv2_dlayer) => {
                    let k_grad = conv2_dlayer.k_grad.as_ref().expect("[CONV] kernel grad");
                    let b_grad = conv2_dlayer.b_grad.as_ref().expect("[CONV] bias grad");

                    total_grad_norm_sq += k_grad.iter().map(|x| x * x).sum::<f32>();
                    total_grad_norm_sq += b_grad.iter().map(|x| x * x).sum::<f32>();
                    zero_grad_count += k_grad.iter().filter(|&&x| x == 0.0).count();
                    total_param_count += k_grad.len() + b_grad.len();

                    let lr = self.learning_rate;
                    let mu = self.viscosity;

                    if prev_velocities.is_none() {
                        // First step ever. No velocities yet. Update == vanilla SGD update.
                        let k_velocity = -lr * k_grad;
                        conv2_dlayer.kernels_mat += &(k_velocity);

                        let b_velocity = -lr * b_grad;
                        conv2_dlayer.b += &(b_velocity);

                        next_velocity.push(Some((k_velocity.into_dyn(), b_velocity.into_dyn())));
                    } else {
                        let (prev_k_velocity, prev_b_velocity) =
                            prev_velocities.clone().expect("[CONV] prev velocities"); // NOTE: Q: clone necessary??

                        // dynamic --> fixed dimensionality.  NOTE: necessary?
                        let prev_k_velocity = prev_k_velocity.into_dimensionality::<Ix2>().unwrap();
                        let prev_b_velocity = prev_b_velocity.into_dimensionality::<Ix1>().unwrap();

                        let k_velocity = mu * prev_k_velocity - lr * k_grad;
                        conv2_dlayer.kernels_mat += &(k_velocity);

                        let b_velocity = mu * prev_b_velocity - lr * b_grad;
                        conv2_dlayer.b += &(b_velocity);

                        next_velocity.push(Some((k_velocity.into_dyn(), b_velocity.into_dyn())));
                    }
                }
                _ => next_velocity.push(None), // no weights to update in other layers
            }
        }
        let g_norm = total_grad_norm_sq.sqrt();
        let dead_ratio = zero_grad_count as f32 / total_param_count as f32;

        println!(
            "[OPTIMIZER] G_Norm: {:.6}, Dead_Params: {:.2}%",
            g_norm,
            dead_ratio * 100.0
        );

        self.velocity = next_velocity;
    }
}

#[derive(Clone)]
pub struct AdamState {
    // Each element is a Vec of (m, v) for each parameter in that layer
    // e.g., for FC: [(m_w, v_w), (m_b, v_b)]
    pub params: Vec<(ArrayD<f32>, ArrayD<f32>)>,
}

pub struct Adam {
    pub t: i32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub states: Vec<Option<AdamState>>,
}

impl Adam {
    pub fn new(nn: &NN, learning_rate: f32) -> Self {
        let states = vec![None; nn.layers.len()];

        Adam {
            t: 0,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            states,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, nn: &mut NN) {
        self.t += 1;
        let t_f32 = self.t as f32;

        for (i, layer) in nn.layers.iter_mut().enumerate() {
            match layer {
                Layer::FC(l) => {
                    let w_grad = l.w_grad.as_ref().unwrap().clone().into_dyn();
                    let b_grad = l.b_grad.as_ref().unwrap().clone().into_dyn();

                    // Initialize state if first time
                    if self.states[i].is_none() {
                        self.states[i] = Some(AdamState {
                            params: vec![
                                (ArrayD::zeros(w_grad.dim()), ArrayD::zeros(w_grad.dim())),
                                (ArrayD::zeros(b_grad.dim()), ArrayD::zeros(b_grad.dim())),
                            ],
                        });
                    }

                    let state = self.states[i].as_mut().unwrap();

                    // Update Weights
                    update_param(
                        &mut l.weights.view_mut().into_dyn(),
                        &w_grad,
                        &mut state.params[0],
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        t_f32,
                    );

                    // Update Bias
                    update_param(
                        &mut l.bias.view_mut().into_dyn(),
                        &b_grad,
                        &mut state.params[1],
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        t_f32,
                    );
                }
                Layer::Conv(l) => {
                    let k_grad = l.k_grad.as_ref().unwrap().clone().into_dyn();
                    let b_grad = l.b_grad.as_ref().unwrap().clone().into_dyn();

                    if self.states[i].is_none() {
                        self.states[i] = Some(AdamState {
                            params: vec![
                                (ArrayD::zeros(k_grad.dim()), ArrayD::zeros(k_grad.dim())),
                                (ArrayD::zeros(b_grad.dim()), ArrayD::zeros(b_grad.dim())),
                            ],
                        });
                    }

                    let state = self.states[i].as_mut().unwrap();

                    update_param(
                        &mut l.kernels_mat.view_mut().into_dyn(),
                        &k_grad,
                        &mut state.params[0],
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        t_f32,
                    );

                    update_param(
                        &mut l.b.view_mut().into_dyn(),
                        &b_grad,
                        &mut state.params[1],
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        t_f32,
                    );
                }
                _ => {}
            }
        }
    }
}

fn update_param(
    param: &mut ArrayViewMutD<f32>,
    grad: &ArrayD<f32>,
    state: &mut (ArrayD<f32>, ArrayD<f32>),
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: f32,
) {
    let (m, v) = state;

    // m = beta1 * m + (1 - beta1) * grad
    m.zip_mut_with(grad, |m_val, g_val| {
        *m_val = beta1 * *m_val + (1.0 - beta1) * g_val;
    });

    // v = beta2 * v + (1 - beta2) * grad^2
    v.zip_mut_with(grad, |v_val, g_val| {
        *v_val = beta2 * *v_val + (1.0 - beta2) * g_val.powi(2);
    });

    // Bias correction
    let m_corr = 1.0 - beta1.powf(t);
    let v_corr = 1.0 - beta2.powf(t);

    // Update weight: w = w - lr * (m / m_corr) / (sqrt(v / v_corr) + eps)
    azip!((p in param, mv in &*m, vv in &*v) {
        let m_hat = mv / m_corr;
        let v_hat = vv / v_corr;
        *p -= lr * m_hat / (v_hat.sqrt() + eps);
    });
}
