# PrivateFair-FL — instructions for Claude Code

Two people build this repo with two separate Claude accounts, relaying work when one hits a usage limit.
You have NO memory of the other person's sessions. The repo, the GitHub issues and their handoff comments
are the only shared memory. Keep them accurate.

## The project in one paragraph
Simulated cross-silo federated learning for medical images. Each round, a coordinator picks which simulated
hospitals train. Sites send only three categorical signals (utility/tail-risk 5 levels, readiness 4, shift 3),
each privatized with K-ary randomized response (local DP). The coordinator decodes them into posteriors with
Bayes' rule, scores sites, enforces a hard participation-age/coverage rule plus a shifted-site slot, and
assigns full/compressed/deferred. Models are aggregated with equal-weight FedAvg. The research output is the
trade-off between telemetry privacy (epsilon), worst-site accuracy, coverage and cost vs. baselines
(random, coverage-random, raw-telemetry oracle, quantized non-private). Full spec: `docs/PROJECT_REPORT.md`
(sections 4, 7, 8, 10). Task list: `docs/TEAM_PLAN.md`. Relay rules: `docs/RELAY.md`.

## Priorities (in order)
1. Speed to a working end-to-end result. Smallest thing that works, then improve.
2. Correctness of the privacy math and the privacy boundary (tests prove it).
3. Reproducibility: seeds, config files, JSONL logs.
Reuse libraries and pretrained models; never write from scratch what torchvision/medmnist/flwr-datasets provide.

## Hard rules
- `src/privatefair/interfaces.py` is the contract. Do not change it inside a feature PR. If a change is needed,
  stop, tell the user, and make a separate tiny PR that both leads approve.
- Coordinator code (`src/privatefair/coordinator/`) never imports `sim`/`data` or uses `TrueBins`/`sim_only`.
  `tests/test_boundaries.py` enforces this. Never weaken that test.
- Nothing the coordinator receives may contain raw losses, accuracies, histograms, sample counts or hardware info.
- Aggregation is equal-weight FedAvg unless a task explicitly says otherwise. Noisy telemetry is never an aggregation weight.
- Core logic (privacy, coordinator, policies, logging) must import without torch, so CI stays fast.
  Tests needing torch get `@pytest.mark.ml`; long ones get `@pytest.mark.slow`.
- Never commit datasets, checkpoints or run outputs (see `.gitignore`).
- Never push to `main`. Never force-push a shared branch.

## Layout (create folders as tasks need them)
```
src/privatefair/interfaces.py   shared contract (frozen)
src/privatefair/data/           Track A: dataset loading, site partitioning, synthetic shift
src/privatefair/sim/            Track A: client simulator, local train/eval, timing, availability, FL loop
src/privatefair/fl/             Track A: aggregation (FedAvg)
src/privatefair/privacy/        Track B: randomized response, privacy ledger
src/privatefair/coordinator/    Track B: posterior decoding, policies, participation age
src/privatefair/analysis/       Member C: plots/tables from RoundLog JSONL
configs/                        YAML experiment configs
scripts/                        entry points (run_experiment.py, make_plots.py)
tests/                          pytest; tests/fixtures/ holds test vectors
```

## Conventions
- Python 3.11, type hints, dataclasses, `numpy.random.Generator` passed in explicitly (no global RNG).
- `ruff check . && ruff format .` and `pytest -q` must pass before every commit you call "done".
- One branch per task: `task/<ID>-<slug>` (e.g. `task/B2-posterior`). Small PRs; body uses the PR template.
- Commit early and often. Work-in-progress commits are fine on task branches; prefix them `WIP:`.
- Datasets: MedMNIST PathMNIST (dev, CPU-friendly, auto-download) → Fed-ISIC2019 via
  `flwr-datasets` (6 real sites, ~9GB, GPU). Model: torchvision pretrained ResNet-18 / EfficientNet-B0.

## How to work with this team
- The users are beginners with this stack and prioritize speed. For non-obvious decisions, add one or two
  sentences explaining why. Don't lecture.
- Before creating a new module, give a short plan (files, functions, tests) and wait for approval.
- Do NOT run long training/experiments inside the session (it burns usage). Write the script, verify it on a
  tiny smoke config (seconds), then give the user the exact command to run in their own terminal.
- Don't paste huge logs back into context; read the tail or grep.
- When the user says they're near their limit, or runs `/handoff`, stop feature work immediately and do the handoff.
- Start of a relayed session: the user will run `/pickup <issue#>`. Trust the handoff note, verify with tests, continue.
