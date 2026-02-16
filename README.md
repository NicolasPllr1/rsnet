<p align="center">
    <img src="./assets/MNIST_dataset_illustration.png" alt="MNIST dataset illustration" width="400">
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

To train the model on MNIST, use the `train`. You can pass a bunch of standard hyper-parameters as well as specify paths for check-pointing:

```bash
cargo run --release -- train \
  --learning-rate 0.001 \
  --batch-size 64 \
  --nb-epochs 30 \
  --train-data-dir data/ \
  --checkpoint-folder ckpt/ \
  --checkpoint-stride 5 \
  --loss-csv-path loss.csv \
```


Note: use the `--help` flag to get more info (`cargo run -- train --help`)

### Inference

To run a single image through a saved checkpoint:

```bash
cargo run --release -- run ckpt/my_latest_ckpt.json image_sample.bin
```

Note: use the `--help` flag to get more info (`cargo run -- run --help`)

### ONNX

Models defined with this Rust engine can be exported to the ONNX format. Use the
`export` command:

```bash
cargo run -- export \
  --checkpoint-path <path-to-checkpoint> \
  --onnx-path <output-onnx-file-path>
```

Resources:

- MMAP blog post: [ONNX introduction](https://mmapped.blog/posts/37-onnx-intro)
- [Doc](https://onnx.ai/onnx/index.html):
  - [Concepts](https://onnx.ai/onnx/intro/index.html)
  - [Operators](https://onnx.ai/onnx/operators/index.html)
