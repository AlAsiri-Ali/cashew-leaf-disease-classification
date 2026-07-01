from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_ds = datasets.ImageFolder(args.splits / "test", transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = timm.create_model(args.arch, pretrained=False, num_classes=len(test_ds.classes)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.tolist())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(targets, preds, target_names=test_ds.classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(args.output_dir / "classification_report.csv")

    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
    }
    pd.DataFrame([metrics]).to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics)

    cm = confusion_matrix(targets, preds, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=test_ds.classes, yticklabels=test_ds.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(args.output_dir / "normalized_confusion_matrix.png", dpi=200)


if __name__ == "__main__":
    main()
