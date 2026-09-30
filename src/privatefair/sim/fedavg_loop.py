"""Plain FedAvg training loop: every site trains every round (task A3, Gate G1).

Determinism (same seed -> byte-identical rounds.jsonl):
  * each site gets its own numpy Generator from SeedSequence([seed, site_id]), so a site's
    batches do not depend on how many other sites exist or their order;
  * torch.manual_seed(seed) fixes the new classifier head's initialization;
  * wall-clock fields (round_seconds, controller_ms) are logged as 0.0. A4 fills
    round_seconds from *simulated* time; there is no controller in this baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from privatefair.data.pathmnist import SiteData, load_sites
from privatefair.fl.fedavg import FedAvg
from privatefair.interfaces import ClientUpdate, RoundLog, ScheduleMode
from privatefair.runlog import RunLogger
from privatefair.sim.local import TrainConfig, evaluate, train_local
from privatefair.sim.model import Weights, build_model, get_weights, set_weights

POLICY_NAME = "fedavg_all"


@dataclass(frozen=True)
class FLConfig:
    rounds: int = 50
    eval_every: int = 5  # evaluate every site at rounds eval_every, 2*eval_every, ... and the last round


def model_bytes(weights: Weights) -> int:
    return sum(v.numel() * v.element_size() for v in weights.values())


def site_rng(seed: int, site_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, site_id]))


def run_fedavg(
    sites: Sequence[SiteData],
    *,
    num_classes: int,
    train_cfg: TrainConfig,
    fl_cfg: FLConfig,
    seed: int,
    logger: RunLogger,
    pretrained: bool = True,
) -> Weights:
    """Run FedAvg over all sites, logging one RoundLog per round. Returns the final global weights."""
    torch.manual_seed(seed)
    model = build_model(num_classes, pretrained=pretrained)
    global_w = get_weights(model)
    nbytes = model_bytes(global_w)
    aggregator = FedAvg()
    ids = [s.site_id for s in sites]
    rngs = {s.site_id: site_rng(seed, s.site_id) for s in sites}
    shifted = sorted(s.site_id for s in sites if s.shifted)

    for epoch in range(fl_cfg.rounds):
        updates, train_losses = [], []
        for s in sites:
            set_weights(model, global_w)
            res = train_local(model, s.x_train, s.y_train, train_cfg, rngs[s.site_id])
            updates.append(ClientUpdate(s.site_id, epoch, ScheduleMode.FULL, res.weights, 0.0, nbytes))
            train_losses.append(res.mean_loss)
        global_w = aggregator.aggregate(global_w, updates)

        global_metrics: dict[str, float] = {"mean_train_loss": float(np.mean(train_losses))}
        per_site: dict[int, dict[str, float]] = {}
        if (epoch + 1) % fl_cfg.eval_every == 0 or epoch == fl_cfg.rounds - 1:
            set_weights(model, global_w)
            per_site = {s.site_id: evaluate(model, s.x_test, s.y_test, train_cfg, num_classes) for s in sites}
            bal = [m["balanced_acc"] for m in per_site.values()]
            global_metrics |= {"mean_balanced_acc": float(np.mean(bal)), "worst_balanced_acc": float(min(bal))}

        logger.log(
            RoundLog(
                run_id=logger.run_id,
                seed=seed,
                policy=POLICY_NAME,
                epoch=epoch,
                selected=ids,
                modes={i: ScheduleMode.FULL.value for i in ids},
                mandatory_sites=[],
                deferral_reasons={},
                telemetry=[],
                posteriors=[],
                participation_age={i: 0 for i in ids},  # everyone participates every round
                epsilon_spent={i: 0.0 for i in ids},  # no telemetry released
                round_seconds=0.0,
                bytes_up=nbytes * len(ids),
                bytes_down=nbytes * len(ids),
                controller_ms=0.0,
                per_site_metrics=per_site,
                global_metrics=global_metrics,
                sim_only={"shifted_sites": shifted},
            )
        )
    return global_w


def run_from_config(
    cfg: Mapping[str, Any],
    seed: int,
    out_dir: str | Path,
    run_id: str,
    sites: Sequence[SiteData] | None = None,
) -> Path:
    """Build everything from a parsed YAML config, run, and return the rounds.jsonl path."""
    model_cfg = cfg.get("model", {})
    if model_cfg.get("arch", "resnet18") != "resnet18":
        raise ValueError(f"unsupported arch {model_cfg['arch']!r}")
    if sites is None:
        sites = load_sites(cfg["data"], seed)
    logger = RunLogger(out_dir, run_id, config={**cfg, "seed": seed})
    run_fedavg(
        sites,
        num_classes=model_cfg.get("num_classes", 9),
        train_cfg=TrainConfig(**cfg.get("train", {})),
        fl_cfg=FLConfig(**cfg.get("fl", {})),
        seed=seed,
        logger=logger,
        pretrained=model_cfg.get("pretrained", True),
    )
    return logger.path
