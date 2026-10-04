"""Local training and evaluation for one site (task A2).

Local work is a fixed number of SGD steps, not epochs: A1 sites have very unequal sizes,
and epochs would give large sites far more compute per round. It also makes A6's
"compressed" mode a smaller `local_steps`.

Batch order comes from the caller's numpy Generator (no DataLoader), so the same rng and
starting weights give identical results on CPU (Gate G1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from privatefair.sim.metrics import balanced_accuracy
from privatefair.sim.model import (
    DEVICE_CHOICES,
    Weights,
    get_weights,
    preprocess,
    resolve_device,
    train_mode,
    trainable_keys,
)


@dataclass(frozen=True)
class TrainConfig:
    local_steps: int = 20
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    input_size: int = 64
    eval_batch_size: int = 256
    device: str = "cpu"  # "auto" = CUDA if available else CPU; "cpu" keeps results bit-reproducible
    amp: bool = False  # fp16 autocast + GradScaler; only active on CUDA (intended for T4-class GPUs)

    def __post_init__(self) -> None:
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"train.device must be one of {DEVICE_CHOICES}, got {self.device!r}")

    @property
    def torch_device(self) -> torch.device:
        return resolve_device(self.device)

    @property
    def amp_active(self) -> bool:
        return self.amp and self.torch_device.type == "cuda"


@dataclass
class TrainResult:
    weights: Weights
    steps: int
    mean_loss: float  # mean training loss over the local steps (sim-side only, never telemetry)


def _batches(n: int, batch_size: int, steps: int, rng: np.random.Generator):
    """Yield `steps` index batches, cycling through fresh permutations of range(n)."""
    order, pos = rng.permutation(n), 0
    for _ in range(steps):
        if pos + batch_size > n:
            order, pos = rng.permutation(n), 0
        yield order[pos : pos + batch_size]
        pos += batch_size


def train_local(
    model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: TrainConfig, rng: np.random.Generator
) -> TrainResult:
    """Train `model` in place from the weights it currently holds (caller loads the global model first).

    A fresh optimizer per call: standard FedAvg keeps no optimizer state across rounds.
    """
    if len(y) == 0:
        raise ValueError("cannot train on an empty site")
    device = cfg.torch_device
    model.to(device)
    train_mode(model)
    params = [p for p in model.parameters() if p.requires_grad]  # frozen backbone: only the trainable part
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if cfg.amp_active else None  # never built on the non-AMP path
    y_t = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.long, device=device)
    losses = []
    for idx in _batches(len(y_t), min(cfg.batch_size, len(y_t)), cfg.local_steps, rng):
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp_active):
            loss = F.cross_entropy(model(preprocess(x[idx], cfg.input_size, device)), y_t[idx])
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        losses.append(loss.item())
    return TrainResult(
        get_weights(model, trainable_keys(model)), cfg.local_steps, float(np.mean(losses)) if losses else float("nan")
    )


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: TrainConfig, num_classes: int) -> dict[str, float]:
    """Metrics for one site's split. Keys match RoundLog.per_site_metrics."""
    if len(y) == 0:
        raise ValueError("cannot evaluate on an empty split")
    device = cfg.torch_device
    model.to(device)
    model.eval()
    y = np.asarray(y).reshape(-1)
    total_loss, preds = 0.0, []
    for start in range(0, len(y), cfg.eval_batch_size):
        sl = slice(start, start + cfg.eval_batch_size)
        logits = model(preprocess(x[sl], cfg.input_size, device))  # fp32 even with amp: telemetry loss stays clean
        y_b = torch.as_tensor(y[sl], dtype=torch.long, device=device)
        total_loss += F.cross_entropy(logits, y_b, reduction="sum").item()
        preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return {
        "balanced_acc": balanced_accuracy(y, y_pred, num_classes),
        "acc": float((y_pred == y).mean()),
        "loss": total_loss / len(y),
    }
