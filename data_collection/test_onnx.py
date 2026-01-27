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
    rt: ort.InferenceSession, image_path: Path, target_size: tuple[int, int]
) -> npt.NDArray:
    img = load_input(image_path, target_size)
    # add batch dim
    img = np.expand_dims(img, axis=0)
    print("img shape:", img.shape)
    probs = rt.run(["probs"], {"input": img})
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--onnx_model_path", type=str)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-t", "--test_set_only", type=bool)

    TARGET_SIZE = (64, 64)

    args = parser.parse_args()

    model_path = Path(args.onnx_model_path)
    data_dir = Path(args.data_dir)
    test_set_only = bool(args.test_set_only)

    if not model_path.is_file():
        print("Error: invalid model path (not a file)")
        sys.exit(1)

    if not data_dir.is_dir():
        print("Error: invalid test data dir (not a dir)")
        sys.exit(1)

    no_metadata_json = data_dir / "metadata.json" not in data_dir.glob("*.json")
    if test_set_only and no_metadata_json:
        print(
            "Error: test_set_only flag is set but no 'metadata.json' file in the data dir"
        )
        sys.exit(1)

    onnx_model = load_model(model_path)
    onnx.checker.check_model(onnx_model)

    rt = ort.InferenceSession(model_path)

    if test_set_only:
        image_paths = get_test_images_path(data_dir)
    else:
        image_paths = list(data_dir.rglob("*.png"))

    acc = 0
    samples = 0
    for img_path in image_paths:
        label = int(img_path.parent.name)
        probs = run_model(rt, img_path, TARGET_SIZE)

        img_path_display = img_path.relative_to(data_dir)
        pred = int(probs[0].argmax() + 1)
        print(f"{img_path_display} | {label=} | {pred=}| {probs=}\n")

        if pred == label:
            acc += 1
        samples += 1
    acc /= samples
    print(f"[ACC] {100 * acc:.3f}%")
