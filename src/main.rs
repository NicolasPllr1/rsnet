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
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, default_value_t = 10)]
        nb_epochs: usize,
        #[arg(long, default_value_t = 0.003)]
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
            nb_epochs,
            learning_rate,
            checkpoint_folder,
            checkpoint_stride,
            loss_csv_path,
        } => {
            if let Err(e) = train::train(
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
