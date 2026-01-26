import onnx
import numpy as np
import numpy.typing as npt
import sys
from pathlib import Path
import argparse
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--onnx_model_path", type=str)
    parser.add_argument("-t", "--test_data_dir", type=str)

    TARGET_SIZE = (64, 64)

    args = parser.parse_args()

    model_path = Path(args.onnx_model_path)
    test_data_dir = Path(args.test_data_dir)

    if not model_path.is_file():
        print("Error: invalid model path (not a file)")
        sys.exit(1)

    if not test_data_dir.is_dir():
        print("Error: invalid test data dir (not a dir)")
        sys.exit(1)

    onnx_model = load_model(model_path)
    onnx.checker.check_model(onnx_model)

    rt = ort.InferenceSession(model_path)

    for img_path in test_data_dir.rglob("*.png"):
        label = int(img_path.parent.name)
        probs = run_model(rt, img_path, TARGET_SIZE)

        img_path_display = img_path.relative_to(test_data_dir)
        pred = int(probs[0].argmax() + 1)
        print(f"{img_path_display} | {label=} | {pred=}| {probs=}\n")
