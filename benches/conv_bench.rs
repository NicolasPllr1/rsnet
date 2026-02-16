use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use ndarray::{Array4, ArrayD};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rsnet::layers::Conv2Dlayer;
use rsnet::model::Module;
use std::f32;
use std::hint::black_box;

fn gen_input(b: usize, c: usize, h: usize, w: usize) -> ArrayD<f32> {
    Array4::<f32>::random((b, c, h, w), Uniform::new(-1.0, 1.0).unwrap()).into_dyn()
}

fn forward_pass_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Convolution_Forward");

    // (batch_size, in_channels, height, width)
    let params = vec![
        (16, 10, 28, 28),   // Standard MNIST size
        (64, 10, 28, 28),   // Larger batch
        (16, 10, 128, 128), // Larger resolution
    ];

    let out_channels = 20;
    let kernel_size = (5, 5);
    let k = kernel_size.0;

    for (batch_size, in_channels, height, width) in params {
        let mut conv_layer = Conv2Dlayer::new(in_channels, out_channels, kernel_size);
        let test_input = gen_input(batch_size, in_channels, height, width);

        let flops =
            batch_size * in_channels * out_channels * k * k * (height - k + 1) * (width - k + 1);
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::new(
                "forward_pass",
                format!("{}x{}x{}x{}", batch_size, in_channels, height, width),
            ),
            &test_input,
            |b, input| {
                b.iter_batched(
                    || input.clone(),
                    |data| conv_layer.forward(black_box(data)),
                    BatchSize::LargeInput,
                );
            },
        );

        // (Future) Add your Rayon bench here:
        // group.bench_with_input(BenchmarkId::new("im2col_rayon", ...), ...)
    }
    group.finish();
}

fn backward_pass_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Convolution_Backward");

    let (batch_size, in_c, out_c, h, w) = (16, 10, 20, 28, 28);
    let kernel = 5;
    let mut conv_layer = Conv2Dlayer::new(in_c, out_c, (kernel, kernel));
    let test_input = gen_input(batch_size, in_c, h, w);

    // We MUST run forward once to populate the layer's internal state
    let _ = conv_layer.forward(test_input.clone());

    // Prepare "grad_output" (the gradient coming from the next layer)
    let out_h = h - kernel + 1;
    let out_w = w - kernel + 1;
    let grad_output = Array4::<f32>::random(
        (batch_size, out_c, out_h, out_w),
        Uniform::new(-1.0, 1.0).unwrap(),
    )
    .into_dyn();

    group.throughput(Throughput::Elements(
        (batch_size * out_c * out_h * out_w) as u64,
    ));

    group.bench_function("backward_pass", |b| {
        b.iter_batched(
            || grad_output.clone(),
            |grad| conv_layer.backward(black_box(grad)),
            BatchSize::LargeInput,
        );
    });

    group.finish();
}
fn forward_then_backward_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Convolution_Full_Step");
    let (b, in_c, out_c, h, w) = (16, 10, 20, 28, 28);
    let k = 5;
    let mut conv_layer = Conv2Dlayer::new(in_c, out_c, (k, k));

    let input = gen_input(b, in_c, h, w);
    let grad_output = gen_input(b, out_c, h - k + 1, w - k + 1);

    group.bench_function("forward_then_backward", |bench| {
        bench.iter_batched(
            || (input.clone(), grad_output.clone()),
            |(in_data, g_out)| {
                conv_layer.forward(black_box(in_data));
                conv_layer.backward(black_box(g_out))
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    forward_pass_benchmark,
    backward_pass_benchmark,
    forward_then_backward_benchmark
);
criterion_main!(benches);
