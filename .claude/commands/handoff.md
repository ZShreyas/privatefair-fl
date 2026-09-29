---
description: Save all work and post a handoff note so the other lead can continue (run BEFORE you hit your limit)
---
Stop feature work now and hand off. Be brief to save usage. Task/issue hint: $ARGUMENTS

1. Work out the issue number: use $ARGUMENTS if given, otherwise infer the task ID from the current branch
   name (`task/<ID>-...`) and find it with `gh issue list --search "<ID>" --state open`.
2. Run `ruff check . --fix -q; ruff format . -q; pytest -q 2>&1 | tail -15` and note the result.
3. `git add -A` then commit. If tests pass: a normal message. If not: `WIP: <what's half-done>`.
4. `git push -u origin HEAD`.
5. Post the note with `gh issue comment <N> --body-file -` using exactly this template:

   ## 🔁 Handoff — <date/time> — from <git user.name>
   **Branch:** `<branch>` @ `<short sha>`   **Tests:** <passing / N failing (names)>
   **Done:** <bullets, 1 line each>
   **In progress:** <what's half-finished, with file:function>
   **Next steps (in order):** <numbered, concrete, each doable in one sitting>
   **Decisions & gotchas:** <anything the next person must know, including dead ends already tried>
   **Verify with:** `<exact command>`

6. Reassign the issue to the other lead if you know their username: `gh issue edit <N> --add-assignee <user> --remove-assignee @me`.
7. Print one short line the user can paste into the team chat, like:
   "Handed off #<N> (<task ID>) — branch <branch> — next: <first next step>".
