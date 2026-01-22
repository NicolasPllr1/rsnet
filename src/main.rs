use mnist_rust::run;
use mnist_rust::train;

use mnist_rust::layers::{
    Conv2Dlayer, FcLayer, FlattenLayer, Layer, MaxPoolLayer, ReluLayer, SoftMaxLayer,
};
use mnist_rust::model::NN;

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
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, short = 'd')]
        train_data_dir: String,
        #[arg(long, default_value_t = 10)]
        nb_epochs: usize,
        #[arg(long, default_value_t = 0.1)]
        learning_rate: f32,
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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            batch_size,
            train_data_dir,
            nb_epochs,
            learning_rate,
            checkpoint_folder,
            checkpoint_stride,
            loss_csv_path,
        } => {
            // The neural network to train
            let nn = NN {
                layers: vec![
                    Layer::Conv(Conv2Dlayer::new(3, 16, (3, 3))), // Input image (3, 128x128) --> (16, 126, 126)
                    Layer::ReLU(ReluLayer::new()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (16, 126, 126) --> (16, 63, 63)
                    //
                    Layer::Conv(Conv2Dlayer::new(16, 32, (4, 4))), // (16, 63, 63) --> (32, 60, 60)
                    Layer::ReLU(ReluLayer::new()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (32, 60, 60) --> (32, 30, 30)
                    //
                    Layer::Conv(Conv2Dlayer::new(32, 64, (3, 3))), // (32, 30, 30) --> (64, 28, 28)
                    Layer::ReLU(ReluLayer::new()),
                    Layer::Pool(MaxPoolLayer::new((2, 2))), // (64, 28, 28) --> (64, 14, 14)
                    //
                    Layer::Flatten(FlattenLayer::new()), // Flatten feature maps into a single 1D vector
                    //
                    Layer::FC(FcLayer::new(64 * 14 * 14, 128)), // Compress down to 128 features
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
    }
}
