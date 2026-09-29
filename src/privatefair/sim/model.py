"""Model construction, weight transfer and preprocessing (task A2).

Pretrained ImageNet ResNet-18 with a new classifier head. Small MedMNIST images are
upsampled (default 64x64) rather than to 224: on CPU that is ~12x cheaper, and the
pretrained stem still gives useful features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

Weights = dict[str, torch.Tensor]


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 (ImageNet weights if `pretrained`) with an `num_classes`-way head."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_weights(model: nn.Module) -> Weights:
    """Detached CPU copy of the full state (incl. BatchNorm stats). Goes into ClientUpdate.weights."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: Weights) -> None:
    model.load_state_dict(weights, strict=True)


def preprocess(x: np.ndarray, input_size: int) -> torch.Tensor:
    """uint8 (N, H, W, C) -> normalized float (N, 3, input_size, input_size)."""
    t = torch.from_numpy(np.ascontiguousarray(x)).float().div_(255.0).permute(0, 3, 1, 2)
    if t.shape[1] == 1:  # grayscale MedMNIST variants
        t = t.expand(-1, 3, -1, -1)
    if t.shape[-1] != input_size or t.shape[-2] != input_size:
        t = F.interpolate(t, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD
