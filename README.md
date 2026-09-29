# PrivateFair-FL

**Privacy-limited adaptive client coordination for fair cross-silo medical-image federated learning.**

## What this is, in plain English
Hospitals can't share patient scans, but together they'd train a better AI model. **Federated learning** sends the
model to the data instead. Each hospital trains a copy locally, and a coordinator averages the updates (**FedAvg**).

Each round, the coordinator picks *which* hospitals train. Picking well needs information about them (is the model
failing there? is their machine fast? are their scans unusual?), but that information is itself sensitive. So each
hospital sends only **three coarse signals**, scrambled with **randomized response**: sometimes it deliberately
reports a random value, with known odds. That gives formal **local differential privacy** (strength set by
epsilon). The coordinator uses **Bayes' rule** to turn the noisy signals into probabilities, and a **fairness
rule** guarantees slow or unusual hospitals aren't left out for long.

The research question: how much selection quality and worst-hospital accuracy do we keep as privacy gets
stronger, compared with random selection and with a coordinator that sees the raw, un-noised data?

Datasets: **PathMNIST** (fast development, synthetic hospitals) and **Fed-ISIC2019** (6 real hospital/scanner sites).

## Where to start
| You are | Read |
|---|---|
| Anyone, first day | `docs/SETUP.md` |
| A lead (Claude Code) | `docs/CLAUDE_CODE_GUIDE.md`, `docs/RELAY.md`, `docs/TEAM_PLAN.md` |
| Member C or D (Gemini) | `GEMINI.md`, your rows in `docs/TEAM_PLAN.md` |
| Want the full research spec | `docs/PROJECT_REPORT.md` |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,exp]"      # add ,ml for PyTorch/datasets
pytest -q
```

## Repo map
```
CLAUDE.md                     instructions auto-loaded by Claude Code
GEMINI.md                     context for Gemini users
src/privatefair/interfaces.py the shared contract between all modules (change only via dedicated PR)
tests/                        includes the privacy-boundary guard
.claude/commands/             /handoff /pickup /review-pr /status
.github/                      CI, CODEOWNERS, PR template
scripts/create_issues.sh      creates the whole task board on GitHub
docs/                         setup, relay protocol, team plan, Claude Code guide, original report
```
