from __future__ import annotations

import timm
import torch
from torch import nn


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a timm image classification model with a task-specific classification head."""
    return timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)


def make_weighted_loss(class_weights: list[float], device: str | torch.device) -> nn.Module:
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights)
