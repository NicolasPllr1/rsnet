import json
import time
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, metadata, split="train", target_size=(64, 64)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.samples = []

        # Load paths and labels from the specific split (train or test)
        for label_str, paths in metadata[split].items():
            label = int(label_str)
            for p in paths:
                self.samples.append((os.path.join(data_dir, p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Open and resize (Image.BILINEAR is roughly equivalent to Triangle)
        img = Image.open(path).convert("RGB")
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert to tensor (scales to [0, 1] automatically)
        pixels = transforms.ToTensor()(img)

        mean = pixels.mean()
        pixels = pixels - mean

        raw_labels = torch.tensor(label, dtype=torch.long)
        labels = raw_labels - 1

        return pixels, labels


class OptimizerName(Enum):
    SGD = "SGD"
    SGD_WITH_MOMENTUM = "SGD_WITH_MOMENTUM"
    ADAM = "ADAM"


@dataclass
class Cfg:
    data_dir: str
    batch_size: int
    target_size: tuple[int, int]
    input_dim: int
    num_classes: int
    epochs: int
    optimizer_name: OptimizerName
    loss_csv: str


def train(cfg: Cfg, model: nn.Sequential):
    with open(os.path.join(cfg.data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    train_ds = CustomImageDataset(cfg.data_dir, metadata, "train", cfg.target_size)
    test_ds = CustomImageDataset(cfg.data_dir, metadata, "test", cfg.target_size)

    print(f"[TRAIN] {len(train_ds)=}")
    print(f"[TEST] {len(test_ds)=}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    criterion = nn.CrossEntropyLoss()

    match cfg.optimizer_name:
        case OptimizerName.SGD:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        case OptimizerName.SGD_WITH_MOMENTUM:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        case OptimizerName.ADAM:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

    with open(cfg.loss_csv, "w") as f:
        f.write("batch,step,duration,loss\n")

    # Training Loop
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        time_start = time.time()
        with open(cfg.loss_csv, "a") as f:
            for i, (inputs, labels) in tqdm(enumerate(train_loader), desc="batch"):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Print/log every 10 steps
                    time_end = time.time()
                    duration = time_end - time_start
                    batch_nb = i + 1
                    step_nb = (epoch * len(train_loader)) + batch_nb
                    avg_loss = running_loss / 10

                    print(
                        f"[{epoch + 1}, {batch_nb}, {duration:.2f}s] loss: {avg_loss:.3f}",
                        flush=True,
                    )
                    f.write(
                        f"{batch_nb // 10},{step_nb},{duration:.2f},{avg_loss:.3f}\n"
                    )
                    f.flush()

                    running_loss = 0.0
                    time_start = time.time()

    # Simple Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="eval"):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total}%")


if __name__ == "__main__":
    parser = ArgumentParser("Train a CNN")
    parser.add_argument("-l", "--loss_csv", type=str)
    parser.add_argument("-n", "--optimizer_name", type=str)
    args = parser.parse_args()

    loss_csv = str(args.loss_csv).strip()
    optimizer_name = OptimizerName(str(args.optimizer_name).strip().upper())

    target_size = (64, 64)
    cfg = Cfg(
        data_dir="./hand_dataset/augment",
        batch_size=32,
        target_size=target_size,
        input_dim=target_size[0] * target_size[1] * 3,
        num_classes=5,
        epochs=10,
        optimizer_name=optimizer_name,
        loss_csv=loss_csv,
    )
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * (cfg.target_size[0] // 4) * (cfg.target_size[1] // 4), 128),
        nn.ReLU(),
        nn.Linear(128, cfg.num_classes),
    )

    train(cfg, model)
