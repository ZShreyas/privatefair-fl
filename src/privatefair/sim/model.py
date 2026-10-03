"""Model construction, weight transfer and preprocessing (task A2).

Pretrained ImageNet ResNet-18 with a new classifier head. Small MedMNIST images are
upsampled (default 64x64) rather than to 224: on CPU that is ~12x cheaper, and the
pretrained stem still gives useful features.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

Weights = dict[str, torch.Tensor]


DEVICE_CHOICES = ("auto", "cuda", "cpu")


def resolve_device(name: str = "cpu") -> torch.device:
    """ "auto" = CUDA if available else CPU; "cuda" fails loudly if there is no GPU."""
    if name not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but torch.cuda.is_available() is False")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-18 (ImageNet weights if `pretrained`) with an `num_classes`-way head.

    `freeze_backbone` leaves only the last residual block (`layer4[-1]`) and the head trainable.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
        for part in (model.layer4[-1], model.fc):
            for p in part.parameters():
                p.requires_grad_(True)
    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), total


def trainable_keys(model: nn.Module) -> list[str] | None:
    """State-dict keys that local training can change, or None when nothing is frozen.

    Includes the BatchNorm running stats of trainable BN layers (buffers, but they do change).
    """
    if all(p.requires_grad for p in model.parameters()):
        return None
    keys: set[str] = set()
    for name, mod in model.named_modules():
        if any(p.requires_grad for p in mod.parameters(recurse=False)):
            prefix = f"{name}." if name else ""
            keys |= {prefix + n for n, _ in mod.named_parameters(recurse=False)}
            keys |= {prefix + n for n, _ in mod.named_buffers(recurse=False)}
    return [k for k in model.state_dict() if k in keys]


def train_mode(model: nn.Module) -> None:
    """model.train(), except fully frozen BatchNorm layers stay in eval so their running stats do not drift."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def get_weights(model: nn.Module, keys: Sequence[str] | None = None) -> Weights:
    """Detached CPU copy of the state (incl. BatchNorm stats), optionally only `keys`. For ClientUpdate.weights."""
    sd = model.state_dict()
    return {k: sd[k].detach().cpu().clone() for k in (sd if keys is None else keys)}


def set_weights(model: nn.Module, weights: Weights) -> None:
    model.load_state_dict(weights, strict=True)  # copies into the model's own device


def preprocess(x: np.ndarray, input_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    """uint8 (N, H, W, C) -> normalized float (N, 3, input_size, input_size), on `device` (default CPU)."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if device is not None:
        t = t.to(device)  # move uint8 first: 4x fewer bytes over PCIe than float32
    t = t.float().div_(255.0).permute(0, 3, 1, 2)
    if t.shape[1] == 1:  # grayscale MedMNIST variants
        t = t.expand(-1, 3, -1, -1)
    if t.shape[-1] != input_size or t.shape[-2] != input_size:
        t = F.interpolate(t, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return (t - _IMAGENET_MEAN.to(t.device)) / _IMAGENET_STD.to(t.device)
