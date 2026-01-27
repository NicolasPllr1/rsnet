import argparse
import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image


def load_model(model_path: Path):
    return onnx.load(model_path)


def load_input(image_path: Path, target_size: tuple[int, int]):
    img = Image.open(image_path).convert("L")
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to tensor (scales to [0, 1] automatically)
    pixels = transforms.ToTensor()(img)

    # Normalization
    mean = pixels.mean()
    pixels = pixels - mean

    return pixels


def run_model(
    session: ort.InferenceSession, image_path: Path, target_size: tuple[int, int]
) -> npt.NDArray:
    img = load_input(image_path, target_size)
    # add batch dim
    img = np.expand_dims(img, axis=0)
    print("img shape:", img.shape)
    probs = session.run(["probs"], {"input": img})
    return probs


def get_test_images_path(data_dir: Path) -> list[Path]:
    """
    Assumes there is a 'metadata.json' file in the `data_dir` dir.
    Uses it to only load _test_ images.
    """
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    paths = []
    for _, label_paths in metadata["test"].items():
        paths.extend([data_dir / p for p in label_paths])
    return paths


if __name__ == "__main__":
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

    no_metadata_json = args.data / "metadata.json" not in args.data.glob("*.json")
    if args.test_only and no_metadata_json:
        sys.exit(
            f"Error: Flag --test-only set but 'metadata.json' file not found in {args.data}"
        )

    try:
        onnx_model = onnx.load(args.model)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(args.model)
    except Exception as e:
        sys.exit(f"Error loading ONNX model: {e}")

    if test_set_only:
        image_paths = get_test_images_path(data_dir)
    else:
        image_paths = list(data_dir.rglob("*.png"))

    if not image_paths:
        sys.exit("No images found to process.")

    correct = 0
    for img_path in image_paths:
        try:
            probs = run_model(session, img_path, TARGET_SIZE)

            label = int(img_path.parent.name)
            pred = int(probs[0].argmax() + 1)

            is_correct = pred == label
            correct += int(is_correct)
            print(
                f"{img_path.relative_to(data_dir)} | {'OK' if is_correct else 'X'} | {label=} | {pred=}:\n{probs=}\n"
            )

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    accuracy = correct / len(image_paths)
    print(f"[ACC] {100 * accuracy:.3f}% ({correct}/{len(image_paths)})")
