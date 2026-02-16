use mnist_rust::layers::{
    Conv2Dlayer, FcLayer, FlattenLayer, Layer, MaxPoolLayer, ReluLayer, SoftMaxLayer,
};
use mnist_rust::model::{Module, NN};
use mnist_rust::optim::{cross_entropy, Optimizer, SGDMomentum};
use ndarray::prelude::*;
use rand::{distributions::Uniform, thread_rng, Rng};

#[derive(Debug)]
struct TestDataset {
    pub samples: Vec<(Vec<f32>, u8)>, // input data, label
}

/// Generate a random batch of data.
/// This data has the same shape as greyscale images.
fn gen_test_batch(batch_size: usize, height: usize, width: usize, nb_classes: u8) -> TestDataset {
    let nb_pixels = height * width;
    let mut rng = thread_rng();
    let pixels_distrib = Uniform::new(0.0, 1.0);
    let labels_distrib = Uniform::new(0, nb_classes);
    let mut samples = Vec::new();
    for _ in 0..batch_size {
        let input_data: Vec<f32> = (0..nb_pixels).map(|_| rng.sample(pixels_distrib)).collect();
        let label = rng.sample(labels_distrib);
        samples.push((input_data, label));
    }
    TestDataset { samples }
}

#[test]
/// Test that the SGD with momentum optimizer can train a CNN to overfit
/// a batch of 64 random 64x64 greyscale images associated to random
/// integer labels ranging from 0 to 5.
fn test_sgd_momentum_can_overfit_single_batch() -> Result<(), Box<dyn std::error::Error>> {
    const MAX_RETRIES: usize = 3;
    let mut last_error = String::new();

    for attempt in 1..MAX_RETRIES {
        println!("Attempt {attempt}");
        let mut cnn = NN {
            layers: vec![
                Layer::Conv(Conv2Dlayer::new(1, 4, (3, 3))), // (1, 64, 64) --> (4, 62, 62)
                Layer::ReLU(ReluLayer::new()),
                Layer::Pool(MaxPoolLayer::new((2, 2))), // (4, 62, 62) --> (4, 31, 31)
                //
                Layer::Conv(Conv2Dlayer::new(4, 4, (2, 2))), // (4, 31, 31) --> (4, 30, 30)
                Layer::ReLU(ReluLayer::new()),
                Layer::Pool(MaxPoolLayer::new((5, 5))), // (4, 30, 30) --> (4, 6, 6)
                //
                Layer::Flatten(FlattenLayer::new()),
                //
                Layer::FC(FcLayer::new(4 * 6 * 6, 5)), // 5 classes
                Layer::Softmax(SoftMaxLayer::new()),
            ],
        };

        const OVERFITTED_LOSS_VALUE: f32 = 0.1;
        const MAX_OPTIM_STEPS: usize = 300; // Generous. Worse cases seem to need ~15 epochs to converge.
        let batch_size: usize = 64;
        let lr = 1.0;
        let mut optimizer = SGDMomentum::new(&cnn, lr);

        let (in_channels, h, w) = (1, 64, 64);
        let nb_classes = 5;
        let test_ds = gen_test_batch(batch_size, h, w, nb_classes);

        let batch_images: Vec<f32> = test_ds
            .samples
            .iter()
            .flat_map(|(pixels, _)| pixels.clone())
            .collect();
        let batch_images =
            Array4::from_shape_vec((batch_size, in_channels, h, w), batch_images)?.into_dyn();
        let batch_labels: Vec<u8> = test_ds.samples.iter().map(|(_, label)| *label).collect();

        let avg_loss: f32 = f32::MAX;
        for optim_step in 1..(MAX_OPTIM_STEPS + 1) {
            cnn.zero_grad();

            let output = cnn.forward(batch_images.clone());
            let output2d = output // (batch_size, num_classes)
                .into_dimensionality::<Ix2>()
                .expect("Network output should be 2D: (batch_size, num_classes)");
            let (loss, init_grad) = cross_entropy(&batch_labels, &output2d);

            let avg_loss = loss.sum() / loss.len() as f32; // batch loss
            if avg_loss < OVERFITTED_LOSS_VALUE {
                println!("Reached loss<{OVERFITTED_LOSS_VALUE} in attempt {attempt} with {optim_step} optimisation steps",);
                return Ok(());
            }

            if optim_step % 50 == 0 {
                println!("step {optim_step}, loss {avg_loss}");
            }

            cnn.backward(init_grad.into_dyn());
            optimizer.step(&mut cnn);
        }
        last_error = format!("Attempt {attempt} failed: Final loss was {avg_loss}");
        println!("{}", last_error);
    }
    Err(format!(
        "Failed to overfit after {MAX_RETRIES} attempts. Last error: {last_error}"
    ))?
}
