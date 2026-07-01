from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_transforms(image_size: int = 224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train_tfms, eval_tfms


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(y.tolist())
    macro_f1 = f1_score(targets, preds, average="macro")
    accuracy = sum(int(p == t) for p, t in zip(preds, targets)) / len(targets)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-weights", type=Path, default=Path("results/class_weights.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tfms, eval_tfms = get_transforms()

    train_ds = datasets.ImageFolder(args.splits / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(args.splits / "val", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(train_ds.classes)).to(device)

    if args.class_weights.exists():
        weights_by_name = json.loads(args.class_weights.read_text(encoding="utf-8"))
        weights = [weights_by_name.get(c, 1.0) for c in train_ds.classes]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), args.output / f"best_{args.arch}_lr{args.lr}.pt")


if __name__ == "__main__":
    main()
