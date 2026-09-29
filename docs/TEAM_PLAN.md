# Team plan

Speed-first. No calendar dates, only dependencies. A task starts the moment its dependencies are merged.

## Roles
| Who | Tools | Owns | Fill in name / GitHub |
|---|---|---|---|
| **Lead A** | Claude Code (Pro) | A-track: data, model, training loop, simulator, integration | |
| **Lead B** | Claude Code (Pro) | B-track: privacy, coordinator, policies, baselines | |
| **Member C** (technical) | Gemini Pro (web, Gemini CLI, Colab) | Independent verification, analysis/plots, Colab compute, reproducibility | |
| **Member D** | Gemini Pro (web, Deep Research) | References, glossary and viva prep, project board, report and slides | |

Leads relay each other's tasks when one runs out of usage (see `RELAY.md`).
C and D work in **their own folders and documents**, so they never block or collide with the leads.

---

## Leads' task board

Size: every task fits in roughly one Claude usage window. **Gate** = a check that must pass before moving on.

### Phase 0 — Kickoff
| ID | Task | Depends | Done when |
|---|---|---|---|
| P0 | Push this kit, set up branch protection, CI green, create issues (`scripts/create_issues.sh`) | — | CI passes on `main`; issues exist |

### Phase 1 — Foundations (A and B run fully in parallel)
| ID | Task | Depends | Done when |
|---|---|---|---|
| A1 | `data/`: PathMNIST auto-download; split into N sites (Dirichlet label skew, unequal sizes); synthetic acquisition shift (contrast/blur/noise) on a chosen subset of "shifted" sites; per-site train/val/test; shift-group labels for evaluation only | P0 | Same seed → same partition; tests on sizes and determinism |
| A2 | `sim/`: pretrained ResNet-18 adapted to the dataset; `train_local(...)`, `evaluate(...)` returning balanced accuracy and loss | P0 | CPU smoke test (<1 min, marked `ml`) |
| A3 | Plain FedAvg loop (all sites every round), `fl/fedavg.py`, per-site eval at checkpoints, RoundLog JSONL, YAML config, `scripts/run_experiment.py` | A1, A2 | **Gate G1:** same seed twice → identical logs |
| B1 | `privacy/rr.py`: K-ary randomized response implementing `Privatizer`, per-signal epsilon | P0 | Empirical frequencies match the formula; C1 vectors pass (when available) |
| B2 | `coordinator/posterior.py`: Bayes decode (report + known channel + prior → `Posterior`) | B1 | Hand-computed case; large epsilon → near one-hot; C1 vectors pass |
| B3 | `privacy/ledger.py`: per-site/signal/epoch accounting, basic composition, hard budget cap that fails the run | B1 | Tests for composition and cap |

### Phase 2 — Heterogeneity and policies
| ID | Task | Depends | Done when |
|---|---|---|---|
| A4 | Systems simulation: per-site speed, seeded replayable availability traces, deadlines (`completed=False`), simulated round time and bytes | A3 | Traces replay identically; deadline misses logged |
| A5 | Simulator-side `TrueBins`: utility (clipped val-loss change → 5 bins), readiness (4), shift via frozen-encoder distance (3) | A3 | Shifted sites from A1 mostly land in shift bin 2 |
| B4 | `coordinator/`: server-side participation age via `observe()`; `RandomPolicy`, `CoverageRandomPolicy` | B2 | **Gate G2:** max age provably bounded under a test trace |
| B5 | `coordinator/privatefair.py`: posterior score, capacity, mandatory overdue sites, shifted-site slot, full/compressed/deferred, logged explanations | B3, B4 | Selection responds correctly to controlled risk/readiness/shift perturbations |
| B6 | `baselines/` (outside `coordinator/`, since the oracle sees truth by design): raw-oracle, quantized non-private, naive decoding, drop-one-signal, no-coverage ablations | B5 | Each variant runs through the same `Coordinator` protocol |

### Phase 3 — Integration, real data, experiments
| ID | Task | Depends | Done when |
|---|---|---|---|
| A6 | **Integration:** TrueBins → privatize → select → train with modes (compressed = fewer steps and/or smaller payload) → aggregate → observe → log | A4, A5, B5 | **Gate G3:** integration test proves the coordinator only ever receives `TelemetryReport` |
| A7 | Fed-ISIC2019 via `flwr-datasets` (6 natural sites) + EfficientNet-B0 behind the same data interface | A6 | GPU smoke run works (C2 confirms it on Colab) |
| X1 | Sweep runner: policy × epsilon grid × heterogeneity scenario × ≥3 seeds; resumable; `runs/<run_id>/log.jsonl` | A6, B6 | Dry-run lists the whole matrix; smoke sweep finishes |
| X2 | Run the sweep (humans run commands; C runs a share on Colab) | X1, C4 | All runs present |
| X3 | Merge C's analysis, final figures/tables, paired seed statistics, Pareto plots | X2, C3 | Every headline claim has per-site, fairness, privacy and cost numbers |

`X` tasks can be taken by either lead.

---

## Member C (technical, Gemini Pro) — parallel, non-blocking

Everything lives in C's own paths, so there are no conflicts with the leads. C opens PRs, and a lead reviews them
(CODEOWNERS). Each task has a "needed by", not a deadline, so there's plenty of slack.

| ID | Task | Start | Needed by | Output |
|---|---|---|---|---|
| C1 | **Independent test vectors.** From the math in `PROJECT_REPORT.md` section 4 only (do NOT read Lead B's code), compute K-ary randomized-response probabilities and Bayes posteriors for ~20 cases (various K, epsilon, priors, observed values). A second, independent implementation catches math bugs. | Now | B2 | `tests/fixtures/rr_vectors.json` + a short README of the format |
| C2 | **Dataset due diligence.** Licenses/citations for PathMNIST and Fed-ISIC2019. In **Google Colab**, load Fed-ISIC2019 via `flwr-datasets`, and record per-site image counts, class distributions and sample images. | Now | A7 | `docs/DATASET_CARD.md` + `notebooks/fed_isic_eda.ipynb` |
| C3 | **Analysis toolkit.** Using the `RoundLog` schema in `src/privatefair/interfaces.py`, write a synthetic-log generator and plotting functions: worst-site and mean accuracy over rounds, per-site final accuracy, max participation age over time, epsilon spent, time-to-target, and a Pareto plot (worst-site accuracy vs epsilon, per policy). | After P0 | X3 | `src/privatefair/analysis/`, `scripts/make_plots.py`, tests using synthetic logs |
| C4 | **Colab runner.** A notebook that clones the repo, installs `.[ml,exp]`, runs a given config on Colab's GPU and saves `runs/` to Google Drive, so experiments offload from the leads' machines. | After A3 | X2 | `notebooks/colab_runner.ipynb` |
| C5 | **Reproduction check.** Fresh clone, follow only the docs, reproduce one headline run, and file issues for anything broken or unclear. | After X1 | Final | Issues + a short `docs/REPRODUCTION.md` |

How C should use Gemini: open the repo files in Gemini (or use Gemini CLI inside the repo, which reads
`GEMINI.md`). Give it the specific file (interfaces, report section) plus the task row above. Run everything
in Colab if your laptop is weak.

---

## Member D (less technical, Gemini Pro) — parallel, no code required

D works in Google Docs/Slides and the GitHub website. Final versions get committed to `docs/` by uploading
through the GitHub web UI (a lead approves), or a lead commits them.

| ID | Task | Start | Needed by | Output |
|---|---|---|---|---|
| D1 | **Verify every reference.** The report was drafted with an AI research tool (Elicit), so each of its 16 references must be checked: does the paper exist, and are the title, authors, venue and year right? Get the link, write a 3-line plain-English summary, and **flag any that can't be found**. Gemini Deep Research helps, but confirm each one on Google Scholar yourself. | Now | Report | `docs/REFERENCES_VERIFIED.md` |
| D2 | **Glossary and viva question bank.** Plain-English definitions (federated learning, FedAvg, non-IID, local differential privacy, epsilon, randomized response, posterior/Bayes, worst-site performance, participation age, Pareto frontier, ablation), then 30+ likely examiner questions with short answers. Update it as the project evolves; the whole team uses it to prepare. | Now | Viva | `docs/GLOSSARY_AND_VIVA.md` |
| D3 | **Project board keeper.** Keep GitHub Issues tidy (labels, assignees), post a short weekly status in the team chat from the board, and take meeting notes. Light, ongoing. | After P0 | Ongoing | Team chat + `docs/meetings/` |
| D4 | **Report and slides skeleton.** Build the final report outline from the project report's sections and fill Introduction and Related Work (from D1). Redraw the architecture diagram cleanly (Google Slides/draw.io) and start the slide deck. Leave results as placeholders. | After D1 | Final | Google Doc + Slides; diagram PNG in `docs/figures/` |
| D5 | **Results and poster.** Drop in the leads' figures from X3, write captions and plain-English takeaways (ask Gemini to explain the plots, then have a lead check them), and make the poster/demo slides. | After X3 | Final | Final report, slides, poster |

---

## Communication
- One team chat. Leads post the one-line handoff message there every time.
- GitHub Issues are the source of truth for who holds what. If it's not on the board, it's not happening.
- Anyone blocked for more than a few hours says so in chat.
