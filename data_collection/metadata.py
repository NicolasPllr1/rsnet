import argparse
import json
import random
from pathlib import Path
from typing import TypedDict


class MetadataJson(TypedDict):
    train: dict[int, list[str]]
    test: dict[int, list[str]]


def build_metadata_json(data_dir: Path, train_p: float) -> Path:
    # Initialize the metadata structure
    # MNIST has classes 0-9
    metadata: MetadataJson = {
        "train": {i: [] for i in range(10)},
        "test": {i: [] for i in range(10)},
    }

    # rglob .png to find all images
    image_paths = list(data_dir.rglob("*.png"))

    for img_path in image_paths:
        try:
            # Get the label from the parent directory name
            label = int(img_path.parent.name)
        except ValueError as e:
            # Skip directories that are not integer labels
            raise e

        # Convert to relative path string for the JSON
        rel_path = str(img_path.relative_to(data_dir))

        # Split based on train_p probability
        if random.random() < train_p:
            metadata["train"][label].append(rel_path)
        else:
            metadata["test"][label].append(rel_path)

    metadata_json_path = data_dir / "metadata.json"

    # Write the dictionary to a JSON file
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return metadata_json_path


def main():
    parser = argparse.ArgumentParser(description="Build dataset metadata.json")
    parser.add_argument(
        "-d", "--data_dir", type=str, help="Directory containing all images"
    )
    parser.add_argument(
        "-p",
        "--train_percentage",
        type=float,
        help="Percentage of images used for training, as opposed to testing",
    )
    args = parser.parse_args()

    if not args.data_dir or args.train_percentage is None:
        parser.print_help()
        return

    input_path = Path(args.data_dir)
    if not input_path.is_dir():
        print(f"Error: {args.data_dir} is not a valid directory.")
        return

    train_percentage = float(args.train_percentage)  # percentage
    if not (0 < train_percentage and train_percentage < 1):
        print(f"Error: {args.train_percentage} should be in ]0, 1[")
        return

    result_path = build_metadata_json(input_path, train_percentage)
    print(f"Metadata successfully saved to {result_path}")

    return


if __name__ == "__main__":
    main()
