---
description: Review a teammate's PR against the project rules (argument: PR number)
---
Review PR #$ARGUMENTS. Be efficient.

1. `gh pr view $ARGUMENTS` and `gh pr diff $ARGUMENTS`.
2. Check against CLAUDE.md hard rules: interfaces.py untouched (unless it's an interfaces-only PR),
   privacy boundary intact, no raw telemetry fields, equal-weight FedAvg, tests exist for new logic,
   no data/checkpoints committed.
3. Look for real bugs: wrong math (randomized-response probabilities, Bayes normalization, epsilon
   accounting), off-by-one in participation age, RNG not seeded/passed in, silent exception swallowing.
4. Optionally `gh pr checkout $ARGUMENTS && pytest -q 2>&1 | tail -15`, then switch back to my previous branch.
5. Report: **Blocking** issues, **Non-blocking** suggestions, and a verdict.
   Do NOT approve it yourself. If it looks good, tell me to run:
   `gh pr review $ARGUMENTS --approve` (or leave comments with `gh pr review $ARGUMENTS --comment -b "..."`).
