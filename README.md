<p align="center">
    <img src="./MNIST_dataset_illustration.png" alt="MNIST dataset illustration" width="200">
</p>

# Neural Network from Scratch

A lightweight implementation of neural networks from 'scratch' in Rust. It
supports both training and inference. For now, it focuses on the MNIST dataset
where the task is digit recognition on 28x28 black-and-white images.

## Features

Dependencies:

- Core dependency is
  [ndarray](https://docs.rs/ndarray/latest/ndarray/index.html) for efficient
  operations on ( n )-dimensional arrays.

Layers available:

- Fully-connected layer
- Convolution, max pooling, flatten layers
- ReLU and softmax activations

Training

- Stochastic gradient descent (SGD) with cross-entropy loss.
- Persistence: save and load models via JSON checkpoints (courtesy of serde).

### Note on convolutions

To implement the convolution operation efficiently, the _img2col_ method is
used. The idea is to turn this operation into one single matrix multipication
which benefits from optimized kernels (see
[GEMM](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)).

Ressource:

- [Anatomy of a High-Speed Convolution](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/)

## Usage

### Training

To train the model on MNIST, specify the number of steps, learning rate, and
output paths:

```bash
cargo run --release -- train 1000 0.01 ./checkpoints 100 loss.csv
```

### Inference

To run a single image through a saved checkpoint:

```bash
cargo run --release -- run ./checkpoints/checkpoint_10.json ./example_digit.bin
```
