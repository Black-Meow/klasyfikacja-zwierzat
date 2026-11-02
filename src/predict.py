"""Create a DrivenData-format probability submission from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch import nn
from tqdm import tqdm


class TestImageDataset(Dataset):
    def __init__(self, frame, data_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.data_dir / str(row["filepath"])) as image:
            image = image.convert("RGB")
        return self.transform(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    test_features = pd.read_csv(args.data_dir / "test_features.csv")
    if not {"id", "filepath"}.issubset(test_features.columns):
        raise ValueError("test_features.csv must contain id and filepath columns.")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = TestImageDataset(test_features, args.data_dir, transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    predictions = []
    with torch.no_grad():
        for images in tqdm(loader, desc="Predicting"):
            logits = model(images.to(device))
            predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(predictions, axis=0)

    output = pd.DataFrame(probabilities, columns=class_names)
    output.insert(0, "id", test_features["id"].values)

    # Preserve the official submission column order when the template is available.
    template_path = args.data_dir / "submission_format.csv"
    if template_path.exists():
        template = pd.read_csv(template_path)
        expected_columns = list(template.columns)
        if len(template) != len(output) or not np.array_equal(template["id"].values, output["id"].values):
            raise ValueError("Prediction IDs/order do not match submission_format.csv.")
        output = output[expected_columns]

    if not np.isfinite(output[class_names].to_numpy()).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    row_sums = output[class_names].sum(axis=1).to_numpy()
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Predicted class probabilities do not sum to 1.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
