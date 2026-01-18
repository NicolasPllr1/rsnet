use mnist::MnistBuilder;

/// Loads the MNIST dataset - downloads automatically if not cached
/// This is just some black magic which uses the ubyte files in data. Do not touch those.
/// Returns: (train_images, train_labels, test_images, test_labels)
/// - Images are Vec<u8> with pixel values 0-255
/// - Labels are Vec<u8> with digit values 0-9
pub fn load_mnist() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mnist = MnistBuilder::new()
        .base_path("data/")
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    (
        mnist.trn_img, // 60,000 * 784 bytes (28x28 images flattened)
        mnist.trn_lbl, // 60,000 labels
        mnist.tst_img, // 10,000 * 784 bytes
        mnist.tst_lbl, // 10,000 labels
    )
}
