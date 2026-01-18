use mnist_rust::run;
use mnist_rust::train;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <train|run> [arguments...]", args[0]);
        eprintln!("  train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride> <loss_csv>  - Train a neural network");
        eprintln!("  run <checkpoint> <example_file>                                          - Run inference using a checkpoint file");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            if args.len() < 7 {
                eprintln!(
                    "Error: 'train' requires gradient steps, learning rate, checkpoint folder, checkpoint stride, and loss CSV path"
                );
                eprintln!(
                    "Usage: {} train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride> <loss_csv>",
                    args[0]
                );
                std::process::exit(1);
            }
            let train_steps: usize = match args[2].parse() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("Error: Invalid number of steps '{}': {}", args[2], e);
                    std::process::exit(1);
                }
            };
            let learning_rate: f32 = match args[3].parse() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("Error: Invalid learning rate '{}': {}", args[3], e);
                    std::process::exit(1);
                }
            };
            let checkpoint_folder = &args[4];
            let checkpoint_stride: usize = match args[5].parse() {
                Ok(val) => val,
                Err(e) => {
                    eprintln!("Error: Invalid checkpoint stride '{}': {}", args[5], e);
                    std::process::exit(1);
                }
            };

            // Validate that train_steps is a multiple of checkpoint_stride
            if train_steps % checkpoint_stride != 0 {
                eprintln!(
                    "Error: train_steps ({}) must be a multiple of checkpoint_stride ({})",
                    train_steps, checkpoint_stride
                );
                std::process::exit(1);
            }

            let loss_csv_path = &args[6];

            if let Err(e) = train::train(
                train_steps,
                learning_rate,
                checkpoint_folder,
                checkpoint_stride,
                loss_csv_path,
            ) {
                eprintln!("Error during training: {}", e);
                std::process::exit(1);
            }
        }
        "run" => {
            if args.len() < 4 {
                eprintln!("Error: 'run' requires a checkpoint file path and example file");
                eprintln!("Usage: {} run <checkpoint_path> <example_file>", args[0]);
                eprintln!(
                    "  example_file should be a raw binary file with 784 bytes (28x28 MNIST image)"
                );
                std::process::exit(1);
            }
            if let Err(e) = run::run(&args[2], &args[3]) {
                eprintln!("Error running inference: {}", e);
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Error: Unknown command '{}'", args[1]);
            eprintln!("Usage: {} <train|run> [arguments...]", args[0]);
            eprintln!(
                "  train <steps> <learning_rate> <checkpoint_folder> <checkpoint_stride> <loss_csv>  - Train a neural network"
            );
            eprintln!("  run <checkpoint> <example_file>                     - Run inference using a checkpoint file");
            std::process::exit(1);
        }
    }
}
