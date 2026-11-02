"""Train a site-aware ResNet18 baseline for the DrivenData Conser-vision task."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

CLASS_NAMES = [
    "antelope_duiker", "bird", "blank", "civet_genet",
    "hog", "leopard", "monkey_prosimian", "rodent",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CameraTrapDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_dir: Path, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.data_dir / str(row["filepath"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


def build_transforms(image_size: int):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    valid_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tfms, valid_tfms


def load_training_frame(data_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(data_dir / "train_features.csv")
    labels = pd.read_csv(data_dir / "train_labels.csv")
    required_feature_cols = {"id", "filepath", "site"}
    missing = required_feature_cols - set(features.columns)
    if missing:
        raise ValueError(f"train_features.csv is missing columns: {sorted(missing)}")
    missing_labels = set(CLASS_NAMES) - set(labels.columns)
    if missing_labels:
        raise ValueError(f"train_labels.csv is missing class columns: {sorted(missing_labels)}")

    frame = features.merge(labels[["id"] + CLASS_NAMES], on="id", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No matching image IDs found between train_features.csv and train_labels.csv.")
    if frame[CLASS_NAMES].isna().any().any():
        raise ValueError("Label file contains missing class values.")
    frame["label"] = frame[CLASS_NAMES].to_numpy().argmax(axis=1)
    return frame


def make_model(num_classes: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(labels.numpy())
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=np.arange(len(CLASS_NAMES)))),
        "accuracy": float(accuracy_score(y_true, y_prob.argmax(axis=1))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")

    frame = load_training_frame(args.data_dir)
    if frame["site"].nunique() < 2:
        raise ValueError("At least two distinct sites are required for a site-grouped validation split.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(frame, groups=frame["site"]))
    train_frame = frame.iloc[train_idx].copy()
    val_frame = frame.iloc[val_idx].copy()

    train_tfms, valid_tfms = build_transforms(args.image_size)
    train_ds = CameraTrapDataset(train_frame, args.data_dir, train_tfms)
    val_ds = CameraTrapDataset(val_frame, args.data_dir, valid_tfms)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = make_model(len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []

    print(f"Device: {device}")
    print(f"Train images: {len(train_ds)} | validation images: {len(val_ds)}")
    print(f"Train sites: {train_frame['site'].nunique()} | validation sites: {val_frame['site'].nunique()}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["log_loss"])
        record = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(record)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_log_loss={metrics['log_loss']:.4f}, val_accuracy={metrics['accuracy']:.4f}"
        )

        if metrics["log_loss"] < best_loss:
            best_loss = metrics["log_loss"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": args.image_size,
                "validation_metrics": metrics,
                "epoch": epoch,
            }
            torch.save(checkpoint, args.output_dir / "best_model.pt")

    with open(args.output_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"Best validation log loss: {best_loss:.5f}")
    print(f"Saved checkpoint: {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
