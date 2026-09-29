---
description: Continue a task the other lead handed off (argument: issue number)
---
Pick up relayed work on issue #$ARGUMENTS.

1. `gh issue view $ARGUMENTS --comments` and find the most recent "🔁 Handoff" comment.
2. `git fetch origin`, then check out and pull the branch named in the note.
3. Assign the issue to me: `gh issue edit $ARGUMENTS --add-assignee @me`.
4. Run the "Verify with" command and `pytest -q 2>&1 | tail -15`.
5. Reply with at most 8 lines: current state, whether tests match what the note claims, and the next step
   you propose. If anything in the note looks wrong or risky, say so.
6. Wait for my go-ahead before writing code.
