# Getting the most out of Claude Code (for the two leads)

## The mental model
Claude Code is Claude running in your terminal, inside this repo. It can read and edit files, run commands
(tests, git, `gh`, Python) and fix its own errors. It asks permission before running things. Read those
prompts; they're your chance to catch mistakes.

It starts every session with **no memory** except what's in the repo: `CLAUDE.md` (auto-loaded), the code, and
whatever you tell it. That's why this repo has `CLAUDE.md`, and why relay handoffs go into GitHub issues.

## Settings worth turning on
**Output style.** This changes how Claude talks to you, not what it knows.
- `/output-style explanatory` (**recommended for you**): normal speed, plus short "Insights" explaining *why*
  it's doing things. You learn as it works.
- `/output-style learning`: Claude occasionally asks *you* to write small pieces of code. Slower. Use it for
  the parts you'll need to defend in the viva (randomized response, posterior, coordinator score).
- `/output-style default`: pure efficiency.

The style is saved per person on your machine (`.claude/settings.local.json`, git-ignored), so each lead chooses their own.

**Plan mode.** Press **Shift+Tab** to cycle into it. Claude proposes a plan and touches nothing until you
approve. Use it at the start of every new task. It's cheap, and it catches wrong directions before any code exists.

## This repo's commands
| Command | When |
|---|---|
| `/pickup <issue#>` | Starting a task the other lead handed off |
| `/handoff` | Your usage is running low, or you're stopping for the day |
| `/review-pr <PR#>` | The other lead asked for your review |
| `/status` | "What's going on and what should I do next?" (cheap) |

Useful built-ins: `/clear` (fresh context, which you should use between tasks), `/compact` (summarize a long session to
free space), `/model` (switch models; a lighter model uses less of your limit on routine work), `/help`.

## How to prompt it
Good:
> Work on issue #9 (B2, posterior decoding). Read TEAM_PLAN row B2 and report section 4. Plan first.

> The smoke test fails with this error: <paste only the last ~20 lines>. Find the cause before changing anything.

> Explain src/privatefair/coordinator/posterior.py like I know Python but not probability. Include an ASCII diagram.

Bad:
> Build the project.  *(too big, and it'll drift and burn your whole window)*

> Fix it.  *(fix what? paste the error)*

Pattern for every task: **pickup/plan → approve → implement with tests → run tests → PR** (or `/handoff`).

## Saving usage (your real bottleneck)
- **One task per session, then `/clear`.** Long sessions re-read a huge context every turn and eat usage fast.
- **Don't let Claude run long jobs.** It writes the script and runs a seconds-long smoke test; *you* run the
  real experiment in a second terminal. Waiting on training costs nothing when Claude isn't watching.
- **Paste the tail of errors, not whole logs.**
- **Point at files.** "Look at `sim/loop.py`, function `run_round`" beats "find where rounds happen."
- **Small PRs.** Faster to write, faster to review (even by hand), easier to relay.
- **Hand off early** (see `RELAY.md`).

## Learning on the go, cheaply
- After a PR merges, ask in a fresh session: "Walk me through what PR #12 does and why, in 10 lines."
- Ask "what would break if I removed this line?" It's the fastest way to understand code.
- Ask for ASCII diagrams of data flow.
- Offload big explanations to the Gemini members, and use your Claude usage for building.
- `docs/GLOSSARY_AND_VIVA.md` (Member D) is your cheat sheet for the concepts.

## Safety net
- Git is your undo button. Claude commits often on task branches. If something goes badly wrong:
  `git log --oneline` then `git reset --hard <good sha>` (on your own task branch only).
- `main` is protected. Nothing lands without CI passing and the other lead approving.
- If Claude wants to edit `interfaces.py` or weaken a test, stop and ask why. `CLAUDE.md` tells it not to.
