import time
from pathlib import Path

import cv2

# Config
TARGET_SIZE = (512, 512)
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
DATA_DIR = Path("hand_dataset")

if not DATA_DIR.exists():
    print(f"Creating {DATA_DIR=}")
    DATA_DIR.mkdir()


def collect_data(label: str):
    save_path = DATA_DIR / label
    if not save_path.exists():
        print(f"Creating class ({label=}) directory: {save_path}")
        save_path.mkdir()

    cap = cv2.VideoCapture(0)

    # Check if webcam opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Collecting for class: {label}")
    print("Commands: 's' to save image, 'q' to finish class")

    count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for "mirror" effect (easier to position hand)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Square ROI in the center
        size = min(h, w, 720)  # ensure we don't go out of frame bounds
        x1, y1 = (w // 2 - size // 2), (h // 2 - size // 2)
        x2, y2 = x1 + size, y1 + size

        # Extract ROI before drawing rectangle so the box isn't in the data
        roi = frame[y1:y2, x1:x2]
        resized = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Draw ROI box for display only
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Collector", frame)
        cv2.imshow("Preview (Input to CNN)", resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            timestamp = int(time.time() * 1000)
            filename = save_path / f"{count}_{timestamp}.png"
            success = cv2.imwrite(filename, resized)
            if success:
                print(f"Saved: {filename}", flush=True)
                count += 1
            else:
                print(f"Failed to save: {filename}", flush=True)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    current_digit = input("Enter the digit you are recording (0-9): ").strip()
    if current_digit in CLASSES:
        collect_data(current_digit)
    else:
        print("Invalid digit.")
