import argparse
import itertools
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

import cv2
from tqdm import tqdm

IMAGE_SIZE = (1080, 1920)
img_height, img_width = IMAGE_SIZE

# Augmentation hyper-parameters
ROTATIONS = [-15, 0, 15]  # Degrees
SCALE_FACTOR = 0.3
SCALES = [1 - SCALE_FACTOR, 1.0, 1 + SCALE_FACTOR]  # Factor

TRANSLATIONS_X = [-img_width / 10, 0, img_width / 10]  # Pixels
TRANSLATIONS_Y = [-img_height / 10, 0, img_height / 10]  # Pixels
IMAGE_EXTENSIONS = {".png"}


def process_single_image(img_p: Path, grid: list, out_dir: Path):
    img = cv2.imread(str(img_p))
    if img is None:
        return

    mean_color = cv2.mean(img)[:3]

    rows, cols = img.shape[:2]
    center = (cols / 2, rows / 2)

    for rot, scale, tx, ty in grid:
        # Skip the 'Identity' transformation (no change)
        if rot == 0 and scale == 1.0 and tx == 0 and ty == 0:
            continue

        # 1. Generate Rotation and Scale Matrix
        # (Center, Angle, Scale)
        rot_map = cv2.getRotationMatrix2D(center, rot, scale)

        # 2. Add Translation to the same Matrix
        # The 3rd column of the affine matrix handles translation
        rot_map[0, 2] += tx
        rot_map[1, 2] += ty

        # 3. Apply the combined transformation
        augmented = cv2.warpAffine(
            img,
            rot_map,
            (cols, rows),
            flags=cv2.INTER_LINEAR,
            borderValue=mean_color,
        )

        # # Optional: Add a slight Gaussian Blur to some to simulate webcam motion
        # # augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

        # 4. Save with descriptive suffix
        suffix = f"_R{rot}_S{scale}_TX{tx}_TY{ty}"
        out_name = f"{img_p.stem}{suffix}{img_p.suffix}"
        out_full_path = out_dir / out_name

        success = cv2.imwrite(str(out_full_path), augmented)
        if not success:
            raise ValueError(f"Could not save augmented image to {out_full_path}")


def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Pipeline")
    parser.add_argument(
        "-d", "--data_dir", type=str, help="Directory containing source images"
    )
    args = parser.parse_args()

    input_path = Path(args.data_dir)
    if not input_path.is_dir():
        print(f"Error: {args.data_dir} is not a valid directory.")
        return

    # Create the output directory
    augment_dir = input_path / "augment"
    augment_dir.mkdir(exist_ok=True)

    # Prepare the grid of transformations
    # This creates a list of all possible combinations
    grid = list(itertools.product(ROTATIONS, SCALES, TRANSLATIONS_X, TRANSLATIONS_Y))

    image_files = [f for ext in IMAGE_EXTENSIONS for f in input_path.rglob(f"*{ext}")]

    print(f"Source images: {len(image_files)}")
    print(
        f"Grid size: {len(grid) - 1} transformations per image"
    )  # -1 as we skip the identity transformation
    print(f"Target total: {len(image_files) * len(grid)} images")

    with ProcessPoolExecutor() as executor:
        futures = []
        for img_p in image_files:
            out_dir = augment_dir / (img_p.parent.relative_to(input_path))
            out_dir.mkdir(exist_ok=True, parents=True)
            futures.append(executor.submit(process_single_image, img_p, grid, out_dir))

        # Wrap in tqdm to see progress
        for _ in tqdm(futures, total=len(image_files), desc="Augmenting"):
            _.result()

    print(f"Augmentation done! Check the results in: {augment_dir}")


if __name__ == "__main__":
    main()
