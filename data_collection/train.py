import json
import time
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim

# from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class OptimizerName(Enum):
    SGD = "SGD"
    SGD_WITH_MOMENTUM = "SGD_WITH_MOMENTUM"
    ADAM = "ADAM"


@dataclass
class Cfg:
    data_dir: str
    grey_scale: bool
    batch_size: int
    target_size: tuple[int, int]
    input_dim: int
    num_classes: int
    epochs: int
    optimizer_name: OptimizerName
    loss_csv: str


class CustomImageDataset(Dataset):
    def __init__(
        self,
        cfg: Cfg,
        metadata,
        split="train",
    ):
        self.data_dir = cfg.data_dir
        self.target_size = cfg.target_size
        self.grey_scale = cfg.grey_scale
        self.samples = []

        # Load paths and labels from the specific split (train or test)
        for label_str, paths in metadata[split].items():
            label = int(label_str)
            for p in paths:
                self.samples.append((os.path.join(cfg.data_dir, p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # type: ignore
        path, label = self.samples[idx]
        # Open and resize (Image.BILINEAR is roughly equivalent to Triangle)
        img = Image.open(path).convert("L" if self.grey_scale else "RGB")
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        # Convert to tensor (scales to [0, 1] automatically)
        pixels = transforms.ToTensor()(img)

        mean = pixels.mean()
        pixels = pixels - mean

        raw_labels = torch.tensor(label, dtype=torch.long)
        labels = raw_labels - 1

        return pixels, labels


def eval(model: nn.Sequential, test_loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train(cfg: Cfg, model: nn.Sequential):
    with open(os.path.join(cfg.data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    train_ds = CustomImageDataset(cfg, metadata, "train")
    test_ds = CustomImageDataset(cfg, metadata, "test")

    print(f"[TRAIN] {len(train_ds)=}")
    print(f"[TEST] {len(test_ds)=}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    criterion = nn.CrossEntropyLoss()

    match cfg.optimizer_name:
        case OptimizerName.SGD:
            optimizer = optim.SGD(model.parameters(), lr=0.001)
        case OptimizerName.SGD_WITH_MOMENTUM:
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        case OptimizerName.ADAM:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    with open(cfg.loss_csv, "w") as f:
        f.write("batch,step,duration,loss\n")

    # Training Loop
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        time_start = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        # print(f"Epoch {epoch + 1}/{cfg.epochs} - LR: {current_lr:.6f}")

        with open(cfg.loss_csv, "a") as f:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}",
            )
            for i, (inputs, labels) in pbar:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if True:
                    time_end = time.time()
                    duration = time_end - time_start
                    batch_nb = i + 1
                    step_nb = (epoch * len(train_loader)) + batch_nb
                    avg_loss = running_loss / 10
                    # pbar.set_postfix(loss=f"{avg_loss:.3f}")
                    pbar.write(f"loss: {avg_loss:.3f}")
                    f.write(f"{step_nb},{duration:.2f},{avg_loss:.3f}\n")
                    f.flush()

                    running_loss = 0.0
                    time_start = time.time()

        # scheduler.step()
        epoch_acc = eval(model, test_loader)
        pbar.write(f"epoch acc: {epoch_acc}%")


if __name__ == "__main__":
    parser = ArgumentParser("Train a CNN")
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-g", "--grey", type=bool)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-l", "--loss_csv", type=str)
    parser.add_argument("-n", "--optimizer_name", type=str)
    args = parser.parse_args()

    loss_csv = str(args.loss_csv).strip()
    optimizer_name = OptimizerName(str(args.optimizer_name).strip().upper())

    TARGET_SIZE = (64, 64)
    cfg = Cfg(
        data_dir=args.data,
        grey_scale=args.grey,
        batch_size=args.batch_size,
        target_size=TARGET_SIZE,
        input_dim=TARGET_SIZE[0] * TARGET_SIZE[1] * 3,
        num_classes=5,
        epochs=args.epochs,
        optimizer_name=optimizer_name,
        loss_csv=loss_csv,
    )

    print("Grey?", args.grey)
    in_channels = 1 if cfg.grey_scale else 3
    model = nn.Sequential(
        nn.Conv2d(in_channels, 8, kernel_size=3),  # 64x64 --> 62x62
        nn.ReLU(),
        nn.MaxPool2d(2),  # --> 31x31
        #
        nn.Conv2d(8, 16, kernel_size=2),  # --> 30x30
        nn.ReLU(),
        nn.MaxPool2d(2),  # --> 15x15
        #
        nn.Flatten(),
        nn.Linear(16 * 15 * 15, 128),  # 3600 --> 128
        nn.ReLU(),
        nn.Linear(128, cfg.num_classes),
    )

    train(cfg, model)
