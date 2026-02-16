use rsnet::layers::{
    Conv2Dlayer, FcLayer, FlattenLayer, Layer, MaxPoolLayer, ReluLayer, SoftMaxLayer,
};
use rsnet::model::{ImageShape, NN};
use rsnet::optim::cross_entropy;
use rsnet::optim::OptiName;
use rsnet::run;
use rsnet::train::{self, CheckpointConfig, TrainConfig};

use std::str::FromStr;

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
        #[arg(long, default_value = "adam")]
        optimizer_name: String,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, default_value_t = 10)]
        nb_epochs: usize,
        #[arg(long, short = 'd')]
        train_data_dir: String,
        #[arg(long, default_value = "checkpoints/")]
        checkpoint_folder: Option<String>,
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
        #[arg(long, default_value_t = 1)]
        input_channels: i64,
        #[arg(long, default_value_t = 64)]
        input_height: i64,
        #[arg(long, default_value_t = 64)]
        input_width: i64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            optimizer_name,
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
                    Layer::Conv(Conv2Dlayer::new(4, 4, (2, 2))), // Input image (4, 31, 31) --> (4, 30, 30)
                    Layer::ReLU(ReluLayer::default()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (4, 30, 30) --> (4, 15, 15)
                    //
                    Layer::Flatten(FlattenLayer::default()), // Flatten feature maps into a single 1D vector
                    //
                    Layer::FC(FcLayer::new(4 * 15 * 15, 5)), // Classes: {1, 2, 3, 5}
                    Layer::Softmax(SoftMaxLayer::default()),
                ],
            };

            let optimizer_name = OptiName::from_str(&optimizer_name)?;

            if let Err(e) = train::train(
                nn,
                TrainConfig {
                    data_dir: train_data_dir,
                    batch_size,
                    nb_epochs,
                    cost_function: cross_entropy,
                    optimizer_name,
                    learning_rate,
                },
                CheckpointConfig {
                    checkpoint_folder,
                    checkpoint_stride,
                    loss_csv_path,
                },
            ) {
                eprintln!("Error during training: {}", e);
                std::process::exit(1);
            }
            Ok(())
        }
        Commands::Run {
            checkpoint,
            image_path,
        } => {
            if let Err(e) = run::run(&checkpoint, &image_path) {
                eprintln!("Error running inference: {}", e);
                std::process::exit(1);
            }
            Ok(())
        }
        Commands::Export {
            checkpoint_path,
            onnx_path,
            input_channels,
            input_height,
            input_width,
        } => {
            let nn =
                NN::from_checkpoint(&checkpoint_path).expect("Could not load model checkpoint");

            let input_shape = ImageShape::new(input_channels, input_height, input_width);
            if let Err(e) = nn.save_as_onnx_model(&onnx_path, input_shape) {
                eprintln!("Error saving model as ONNX: {}", e);
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
