#!/bin/bash

set -e

DATASET_DIR=$1 # where all images are stored
TRAIN_P=$2 # percentage of images assigned to the training split. The rest is for validation (test).

if [ -z "$DATASET_DIR" ] || [ -z "$TRAIN_P" ]; then
    echo "Usage: $0 <dataset_dir> <train_percentage>"
    exit 1
fi
echo "Starting augmentation..."
python augment.py -d "$DATASET_DIR"

echo "Building metadata..."
python metadata.py -d "$DATASET_DIR/augment" -p "$TRAIN_P"

echo "Done!"
