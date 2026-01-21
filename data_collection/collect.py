import time
from pathlib import Path

import cv2

# Config
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
DATA_DIR = Path("hand_dataset")

if not DATA_DIR.exists():
    print(f"Creating {DATA_DIR=}")
    DATA_DIR.mkdir()


def collect_data(label: str, *, nb_samples_to_collect: int, sub_dir: str = ""):
    """
    Collect full-resolution RGB images from the webcam for the `label` class.
    """

    if d := sub_dir.strip():
        # like: "data/at_home/1" if capturing images for label 1 at home
        save_path = DATA_DIR / d / label
    else:
        save_path = DATA_DIR / label

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
    current_digit = input("Enter the digit you are recording (0-9): ").strip()
    sub_dir = input(f"Sub-dir from {DATA_DIR} to store data (empty if no sub dir): ")
    nb_samples_to_collect = int(
        input("How many samples do you want to collect?").strip()
    )
    if current_digit in CLASSES:
        collect_data(
            current_digit, nb_samples_to_collect=nb_samples_to_collect, sub_dir=sub_dir
        )
    else:
        print("Invalid digit.")
