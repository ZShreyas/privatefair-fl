# The relay: two Claude accounts, one codebase

Each lead's Claude Pro account has a usage window that resets every ~5 hours, plus a weekly cap.
Instead of both people stalling, the leads work in one of two modes.

**Parallel mode** (both have usage): Lead A works the A-track, Lead B works the B-track (see `TEAM_PLAN.md`).
The tracks are designed not to touch the same files until the integration task A6.

**Relay mode** (one is out): the person with usage either continues their own track or picks up the other's
handed-off task. Choose by this rule:

> Continue your own track if it has unblocked work. Pick up the other person's task only if it is on the
> **critical path** (it blocks integration) or your own track is blocked.

Critical path: `A1 → A2 → A3 → A6` and `B1 → B2 → B5 → A6`. After A6: `A7`, `X1`, `X2`, `X3`.

---

## The five rules that keep this from getting messy

1. **One baton per branch.** Whoever is *assigned* the GitHub issue is the only person who pushes to that task's
   branch. `/pickup` reassigns it to you; `/handoff` gives it back. Never both at once.
2. **Hand off early.** Run `/handoff` when Claude Code shows the usage warning, or when you estimate
   ~15 minutes left. Writing the handoff costs usage too; if you hit zero mid-task, the note never gets written.
3. **Tasks are small.** Every task in the plan is sized to finish in roughly one usage window. A half-finished
   small task is easy to continue. A half-finished giant one is not.
4. **The note is the memory.** The next Claude has zero memory of your session. If a decision, a dead end or a
   weird bug isn't in the handoff comment, it's lost.
5. **Rebase before PR.** Before opening a PR, run `git pull --rebase origin main`. Claude can resolve conflicts
   if you ask.

---

## The mechanics

### Handing off (you're running low)
```
/handoff            (or: /handoff 12   to name the issue explicitly)
```
Claude runs the tests, commits (as `WIP:` if unfinished), pushes, posts a structured "🔁 Handoff" comment on
the issue, reassigns it, and prints one line for you to paste in the team chat:

> Handed off #12 (B2) — branch task/B2-posterior — next: add the hand-computed test case

### Picking up (the other lead just ran out)
```
/pickup 12
```
Claude reads the latest handoff, checks out the branch, runs the tests, tells you in ≤8 lines what state it's
in, and waits for your go-ahead.

### Reviewing (every PR needs the other lead's approval)
```
/review-pr 15
```
Claude reads the diff against the project rules and lists blocking vs non-blocking issues. **You** approve,
not Claude: `gh pr review 15 --approve`.

If the reviewer is out of usage, they can still review by reading the diff on GitHub. PRs are small on
purpose, and CI already checked the tests.

### Who approves a relayed PR?
GitHub only blocks the PR's *author* from approving. So if A opened the PR and B added commits during a relay,
B can still approve it. The rule is simply that the non-author lead reviews the **whole** diff, including
their own commits.

---

## What to do while you're out of usage
You're not blocked, only your Claude is.
- Run the long experiments Claude prepared (the commands are in the PR or handoff note).
- Review the other lead's open PRs by hand on GitHub.
- Read the code that was just merged, and ask the team's Gemini members to explain parts you don't get.
- Check on C's and D's work and answer their questions.

---

## A typical day
| Time  | Lead A                                   | Lead B                                      |
|-------|------------------------------------------|---------------------------------------------|
| 09:00 | Claude: A2 (model + local training)      | Claude: B1 (randomized response)            |
| 12:30 | ⚠ limit near → `/handoff` A2             | finishes B1 → opens PR                      |
| 12:45 | reviews B1 PR by hand, approves          | `/pickup` A2 (critical path) and continues  |
| 14:00 | runs a long smoke experiment             | still on A2                                 |
| 15:30 | usage resets                              | ⚠ limit near → `/handoff` A2                |
| 15:40 | `/pickup` A2, finishes, opens PR          | reviews A2 PR by hand, approves             |

It works best if your usage windows are **staggered**, so one person starts ~2 hours after the other.
Then one of you almost always has a fresh window.
