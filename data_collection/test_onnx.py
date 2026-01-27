import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import onnx


def load_model(model_path: Path):
    return onnx.load(model_path)


def load_input(image_path: Path, target_size: tuple[int, int]):
    with Image.open(image_path) as img:
        img = img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
        # Convert to float32 and scale to [0, 1]
        pixels = np.array(img).astype(np.float32) / 255.0
    # Normalization (zero-mean)
    pixels -= pixels.mean()
    # Add batch and channel dims: (1, 1, H, W)
    return np.expand_dims(pixels, axis=(0, 1))


def get_test_images_path(data_dir: Path) -> list[Path]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return [
        data_dir / p for label_paths in metadata["test"].values() for p in label_paths
    ]


def inference_loop(
    session: ort.InferenceSession, image_paths: list[Path], target_size: tuple[int, int]
) -> tuple[int, float]:
    correct = 0
    for img_path in image_paths:
        try:
            input_tensor = load_input(img_path, target_size)

            # Run inference
            outputs = session.run(["probs"], {"input": input_tensor})
            probs = outputs[0]

            label = int(img_path.parent.name)
            pred = int(probs[0].argmax() + 1)  # type: ignore

            is_correct = pred == label
            correct += int(is_correct)
            print(
                f"{img_path.relative_to(data_dir)} | {'OK' if is_correct else 'X'} | {label=} | {pred=}:\n{probs=}\n"
            )

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    accuracy = correct / len(image_paths)
    return correct, accuracy


def main():
    parser = argparse.ArgumentParser(description="ONNX Model Inference Script")
    parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to ONNX model"
    )
    parser.add_argument(
        "-d", "--data", type=Path, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "-t", "--test-only", action="store_true", help="Only use test set from metadata"
    )
    args = parser.parse_args()

    TARGET_SIZE = (64, 64)

    model_path = Path(args.onnx_model_path)
    data_dir = Path(args.data_dir)
    test_set_only = bool(args.test_set_only)

    # Validation
    if not args.model.is_file():
        sys.exit(f"Error: Model not found at {args.model}")
    if not args.data.is_dir():
        sys.exit(f"Error: Data directory not found at {args.data}")

    try:
        onnx_model = onnx.load(args.model)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(args.model)
    except Exception as e:
        sys.exit(f"Error loading ONNX model: {e}")

    try:
        if test_set_only:
            image_paths = get_test_images_path(data_dir)
        else:
            image_paths = list(data_dir.rglob("*.png"))
    except Exception as e:
        sys.exit(f"Error loading test images: {e}")

    if not image_paths:
        sys.exit("No images found to process.")

    correct, accuracy = inference_loop(session, image_paths, TARGET_SIZE)
    print(f"[ACC] {100 * accuracy:.3f}% ({correct}/{len(image_paths)})")


if __name__ == "__main__":
    main()
