## Task
Closes #<issue number>  —  Task ID: <e.g. B2>

## What changed (2–5 lines)

## How to verify
```
pytest -q
```

## Checklist
- [ ] Tests pass locally and new logic has tests
- [ ] `interfaces.py` untouched (or this PR is ONLY an interfaces change, approved by both leads)
- [ ] No raw metrics / sample counts / hardware info added to anything the coordinator receives
- [ ] Any long experiment is runnable by a command written below, not hidden in a notebook
- [ ] Handoff notes on the issue are up to date (if this task was relayed)
