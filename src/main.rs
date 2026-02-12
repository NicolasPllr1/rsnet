use mnist_rust::layers::{
    Conv2Dlayer, FcLayer, FlattenLayer, Layer, MaxPoolLayer, ReluLayer, SoftMaxLayer,
};
use mnist_rust::model::NN;
use mnist_rust::optim::cross_entropy;
use mnist_rust::run;
use mnist_rust::train;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new model
    Train {
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, default_value_t = 10)]
        nb_epochs: usize,
        #[arg(long, short = 'd')]
        train_data_dir: String,
        #[arg(long, default_value = "checkpoints/")]
        checkpoint_folder: String,
        #[arg(long, default_value_t = 1)]
        checkpoint_stride: usize, // Every how many epochs do we checkpoint?
        #[arg(long, default_value = "loss.csv")]
        loss_csv_path: String,
    },
    /// Run inference on a file
    Run {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        image_path: String,
    },
    Export {
        #[arg(long)]
        checkpoint_path: String, // input
        #[arg(long)]
        onnx_path: String, // output
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            learning_rate,
            batch_size,
            nb_epochs,
            train_data_dir,
            checkpoint_folder,
            checkpoint_stride,
            loss_csv_path,
        } => {
            // The neural network to train
            let nn = NN {
                layers: vec![
                    Layer::Conv(Conv2Dlayer::new(1, 4, (3, 3))), // Input image (1, 64, 64) --> (4, 62, 62)
                    Layer::ReLU(ReluLayer::new()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (4, 62, 62) --> (4, 31, 31)
                    //
                    Layer::Conv(Conv2Dlayer::new(4, 8, (2, 2))), // (4, 31, 31) --> (8, 30, 30)
                    Layer::ReLU(ReluLayer::new()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (8, 30, 30) --> (8, 15, 15)
                    //
                    Layer::Conv(Conv2Dlayer::new(8, 16, (2, 2))), // (8, 15, 15) --> (16, 14, 14)
                    Layer::ReLU(ReluLayer::new()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (16, 14, 14) --> (16, 7, 7)
                    //
                    Layer::Flatten(FlattenLayer::new()), // Flatten feature maps into a single 1D vector
                    //
                    Layer::FC(FcLayer::new(16 * 7 * 7, 128)), // Compress down to 128 features
                    Layer::ReLU(ReluLayer::new()),
                    //
                    Layer::FC(FcLayer::new(128, 5)), // Classes: {1, 2, 3, 5}
                    Layer::Softmax(SoftMaxLayer::new()),
                ],
            };
            if let Err(e) = train::train(
                nn,
                &train_data_dir,
                batch_size,
                nb_epochs,
                cross_entropy,
                learning_rate,
                &checkpoint_folder,
                checkpoint_stride,
                &loss_csv_path,
            ) {
                eprintln!("Error during training: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Run {
            checkpoint,
            image_path,
        } => {
            if let Err(e) = run::run(&checkpoint, &image_path) {
                eprintln!("Error running inference: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Export {
            checkpoint_path,
            onnx_path,
        } => {
            let nn =
                NN::from_checkpoint(&checkpoint_path).expect("Could not load model checkpoint");
            if let Err(e) = nn.save_as_onnx_model(&onnx_path) {
                eprintln!("Error saving model as ONNX: {}", e);
                std::process::exit(1);
            }
        }
    }
}
