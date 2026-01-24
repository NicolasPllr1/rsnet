import time
import sys
from pathlib import Path
import argparse

import cv2

# Config
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def collect_data(label: str, *, nb_samples_to_collect: int, dir: Path):
    """
    Collect full-resolution RGB images from the webcam for the `label` class.
    """

    save_path = dir / label
    if not save_path.exists():
        print(f"Creating class ({label=}) directory: {save_path}")
        save_path.mkdir(parents=True)

    cap = cv2.VideoCapture(0)

    # Check if webcam opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Collecting for class: {label}")
    print("Commands: 's' to save image, 'q' to finish class")

    samples_collected = 0
    while samples_collected < nb_samples_to_collect:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for "mirror" effect (easier to position hand)
        frame = cv2.flip(frame, 1)

        cv2.imshow("Webcam feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            timestamp = int(time.time() * 1000)
            filename = save_path / f"{samples_collected + 1}_{timestamp}.png"
            success = cv2.imwrite(filename, frame)
            if success:
                print(
                    f"[Sample {samples_collected + 1}] Saved ({frame.shape=}): {filename}",
                    flush=True,
                )
                samples_collected += 1
            else:
                print(f"Failed to save: {filename}", flush=True)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", type=str)
    parser.add_argument("-l", "--label", type=str)  # [0, 9] (as str)
    parser.add_argument("-c", "--count", type=int)

    args = parser.parse_args()

    data_dir = Path(str(args.dataset_dir))
    if not data_dir.exists():
        print(f"Creating {data_dir=}")
        data_dir.mkdir()

    nb_samples_to_collect = int(args.count)

    current_digit = str(args.label)
    if current_digit not in CLASSES:
        print("Invalid digit.")
        sys.exit(0)

    collect_data(
        current_digit, nb_samples_to_collect=nb_samples_to_collect, dir=data_dir
    )
