# Day-one setup

Do these in order. Total time: about an hour, mostly waiting on installs.

## 1. Everyone (A, B, C, D)
- [ ] GitHub account. Students: apply for the **GitHub Student Developer Pack** (free GitHub Pro), which is needed
      for branch protection on a *private* repo. If nobody gets it in time, make the repo public instead.
- [ ] Join the team chat and post your GitHub username.

## 2. Lead A (repo owner) — task P0
1. Create an **empty** repo on GitHub (no README). Private if you have Pro, else public.
2. Unzip this kit into a folder and push it **before** enabling protection (the first push goes straight to `main`):
   ```bash
   cd privatefair-fl
   git init -b main && git add -A && git commit -m "Starter kit"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
3. Edit `.github/CODEOWNERS` and put in both leads' usernames. This gets committed via the first PR later.
4. Settings → Collaborators → invite **B, C and D** (write access).
5. Wait for the **Actions** tab to show the `tests` job passing once. It has to run once before you can require it.
6. Settings → **Rules → Rulesets → New branch ruleset** (older UI: Settings → Branches → Add rule):
   - Name `protect-main`, Enforcement **Active**, Target: **default branch**
   - **Bypass list: empty**, so nobody (including you, the owner) can skip the rules
   - ✅ Restrict deletions  ✅ Block force pushes
   - ✅ Require a pull request before merging:
     required approvals **1**, ✅ dismiss stale approvals on new commits,
     ✅ require review from **Code Owners**, ✅ require conversation resolution
   - ✅ Require status checks to pass → add **`tests`** → ✅ require branches to be up to date
   - Save. The wording differs slightly between GitHub UI versions; the intent is what matters.
7. Create all task issues:
   ```bash
   gh auth login
   bash scripts/create_issues.sh
   ```
8. First PR as a test of the whole flow: branch `task/P0-codeowners`, commit the CODEOWNERS edit, and open the PR.
   Lead B approves it. If B *can* approve and you *can't* merge without that, the setup works.

## 3. Both leads — local tools
- [ ] **Git** and the **GitHub CLI** (`gh`): https://cli.github.com → then `gh auth login`
- [ ] **Python 3.11** (python.org or `uv`)
- [ ] **Claude Code**: follow https://code.claude.com/docs (install, then log in with your Pro account)
- [ ] Clone and install:
  ```bash
  git clone https://github.com/<owner>/<repo>.git && cd <repo>
  python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
  pip install -e ".[dev,exp]"          # core; add ,ml when you reach A2 (installs PyTorch, big)
  pytest -q                            # should say "12 passed"
  ```
- [ ] Start Claude Code **from the repo root**: `claude`. It reads `CLAUDE.md` automatically.
- [ ] Pick an output style: `/output-style explanatory` (see `CLAUDE_CODE_GUIDE.md`).
- [ ] Type `/` and confirm you see `/handoff`, `/pickup`, `/review-pr`, `/status`.
- [ ] Agree on **staggered start times** so your usage windows don't reset at the same moment (see `RELAY.md`).

## 4. Member C
- [ ] Git + GitHub account, and accept the collaborator invite.
- [ ] A Google account with **Colab** (free GPU runtime) for C2 and C4.
- [ ] Optional: **Gemini CLI** in the cloned repo (it reads `GEMINI.md`). Otherwise use Gemini on the web and
      paste in the specific files you're working on.
- [ ] Clone the repo and run `pytest -q` once, in Colab or locally.
- [ ] Workflow: branch `c/<task>-<slug>` → commit → PR → a lead reviews.

## 5. Member D
- [ ] Accept the collaborator invite. You'll use the GitHub **website** (Issues tab, and "Add file → Upload" for final docs).
- [ ] Create a shared Google Drive folder for the team: report draft, slides, meeting notes, references.
- [ ] Start D1 and D2 right away. They don't depend on anyone.

## Where each thing lives
| Thing | Where |
|---|---|
| Code, tests, configs | GitHub repo |
| Who's doing what | GitHub Issues (one per task) |
| Handoff notes between leads | Comments on the task's issue (`/handoff` writes them) |
| Datasets, checkpoints, run outputs | Local disk or Google Drive, **never git** |
| Report, slides, meeting notes (drafts) | Shared Google Drive |
