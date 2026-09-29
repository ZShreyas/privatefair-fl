# PrivateFair-FL — context for Gemini (Members C and D)

You're helping a 4-person student team. Two leads build the core system with Claude Code; the Gemini users
(Members C and D) do parallel work that must never block or collide with them.

## Project in brief
Simulated cross-silo federated learning for medical images. A coordinator picks which simulated hospitals
train each round, using only three privatized categorical signals per site (K-ary randomized response =
local differential privacy), decoded with Bayes' rule, plus a fairness rule so no site is skipped too long.
Full spec: `docs/PROJECT_REPORT.md`. Team tasks: `docs/TEAM_PLAN.md` (Member C = tasks C1–C5, Member D = D1–D5).

## Rules
- Work ONLY in your task's output paths (listed in TEAM_PLAN). Never edit `src/privatefair/interfaces.py`,
  `CLAUDE.md`, or the leads' folders (`data/`, `sim/`, `fl/`, `privacy/`, `coordinator/`, `baselines/`).
- C1 must be derived from the math in PROJECT_REPORT.md section 4 only, not from the leads' code (it's an
  independent cross-check).
- Code style: Python 3.11, type hints, `ruff`, `pytest`; pass `numpy.random.Generator` explicitly for randomness.
- Analysis code reads logs in the `RoundLog` format defined in `src/privatefair/interfaces.py`.
- Never commit datasets, checkpoints or run outputs.
- Branch `c/<task>-<slug>` or `d/<task>-<slug>`; small PRs; a lead reviews.
- For references (D1): never invent a citation. If a paper can't be verified, say so explicitly.
