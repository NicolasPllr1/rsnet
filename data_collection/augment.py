import argparse
import itertools
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_SIZE = (512, 512)
img_size_x = IMAGE_SIZE[0]
img_size_y = IMAGE_SIZE[1]

# Augmentation hyper-parameters
ROTATIONS = [-15, 0, 15]  # Degrees
SCALES = [0.5, 1.0, 1.5]  # Factor
TRANSLATIONS_X = [-img_size_x / 10, 0, img_size_x / 10]  # Pixels
TRANSLATIONS_Y = [-img_size_y / 10, 0, img_size_y / 10]  # Pixels
IMAGE_EXTENSIONS = {".png"}


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

    for img_p in tqdm(image_files):
        img = cv2.imread(str(img_p))
        if img is None:
            continue

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
                # borderMode=cv2.BORDER_CONSTANT,
                # borderMode=cv2.BORDER_REPLICATE,
                borderMode=cv2.BORDER_REFLECT_101,
                # borderValue=mean_color,
            )

            # # Optional: Add a slight Gaussian Blur to some to simulate webcam motion
            # # augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

            # 4. Save with descriptive suffix
            suffix = f"_R{rot}_S{scale}_TX{tx}_TY{ty}"
            out_name = f"{img_p.stem}{suffix}{img_p.suffix}"
            cv2.imwrite(str(augment_dir / out_name), augmented)

    print(f"Augmentation done! Check the results in: {augment_dir}")


if __name__ == "__main__":
    main()
