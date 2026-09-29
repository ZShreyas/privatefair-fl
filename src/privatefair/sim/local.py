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
from privatefair.sim.model import Weights, get_weights, preprocess


@dataclass(frozen=True)
class TrainConfig:
    local_steps: int = 20
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    input_size: int = 64
    eval_batch_size: int = 256


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
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    y_t = torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.long)
    losses = []
    for idx in _batches(len(y_t), min(cfg.batch_size, len(y_t)), cfg.local_steps, rng):
        loss = F.cross_entropy(model(preprocess(x[idx], cfg.input_size)), y_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return TrainResult(get_weights(model), cfg.local_steps, float(np.mean(losses)) if losses else float("nan"))


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, cfg: TrainConfig, num_classes: int) -> dict[str, float]:
    """Metrics for one site's split. Keys match RoundLog.per_site_metrics."""
    if len(y) == 0:
        raise ValueError("cannot evaluate on an empty split")
    model.eval()
    y = np.asarray(y).reshape(-1)
    total_loss, preds = 0.0, []
    for start in range(0, len(y), cfg.eval_batch_size):
        sl = slice(start, start + cfg.eval_batch_size)
        logits = model(preprocess(x[sl], cfg.input_size))
        total_loss += F.cross_entropy(logits, torch.as_tensor(y[sl], dtype=torch.long), reduction="sum").item()
        preds.append(logits.argmax(dim=1).numpy())
    y_pred = np.concatenate(preds)
    return {
        "balanced_acc": balanced_accuracy(y, y_pred, num_classes),
        "acc": float((y_pred == y).mean()),
        "loss": total_loss / len(y),
    }
