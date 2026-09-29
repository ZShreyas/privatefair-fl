#!/usr/bin/env bash
# Creates labels + one GitHub issue per task in docs/TEAM_PLAN.md. Run ONCE, from the repo root, after `gh auth login`.
#   bash scripts/create_issues.sh
set -euo pipefail

gh label create track-A   --color 1f77b4 --description "Lead A track (data/model/sim)"          --force
gh label create track-B   --color ff7f0e --description "Lead B track (privacy/coordinator)"     --force
gh label create track-X   --color 2ca02c --description "Either lead"                             --force
gh label create member-C  --color 9467bd --description "Member C (Gemini, technical)"           --force
gh label create member-D  --color 8c564b --description "Member D (Gemini, docs/report)"         --force
gh label create gate      --color d62728 --description "Has a gate check in TEAM_PLAN"          --force
gh label create critical-path --color e377c2 --description "Blocks integration; relay this first" --force

mk() {  # mk "ID" "title" "labels" "body"
  gh issue create --title "[$1] $2" --label "$3" --body "$4

Details and done-when: see docs/TEAM_PLAN.md (task $1). Relay rules: docs/RELAY.md." >/dev/null
  echo "created $1"
}

mk A1 "Data: PathMNIST loader + site partitioner + synthetic shift" "track-A,critical-path" "Depends: P0"
mk A2 "Model: pretrained ResNet-18, train_local / evaluate"        "track-A,critical-path" "Depends: P0"
mk A3 "Plain FedAvg loop + RoundLog JSONL + run_experiment.py"      "track-A,critical-path,gate" "Depends: A1, A2. Gate G1: same seed -> identical logs"
mk A4 "Systems sim: speed, availability traces, deadlines"          "track-A" "Depends: A3"
mk A5 "Simulator-side TrueBins (utility/readiness/shift)"           "track-A" "Depends: A3"
mk A6 "INTEGRATION: telemetry -> coordinator -> training loop"      "track-A,critical-path,gate" "Depends: A4, A5, B5. Gate G3"
mk A7 "Fed-ISIC2019 via flwr-datasets + EfficientNet-B0"            "track-A" "Depends: A6"
mk B1 "K-ary randomized response (Privatizer)"                       "track-B,critical-path" "Depends: P0"
mk B2 "Posterior decoding (Bayes)"                                   "track-B,critical-path" "Depends: B1"
mk B3 "Privacy ledger + budget cap"                                  "track-B" "Depends: B1"
mk B4 "Participation age + Random / CoverageRandom policies"         "track-B,gate" "Depends: B2. Gate G2"
mk B5 "PrivateFair coordinator (score + constraints + modes)"        "track-B,critical-path" "Depends: B3, B4"
mk B6 "Baselines & ablations (oracle, quantized, naive, drop-one)"   "track-B" "Depends: B5"
mk X1 "Sweep runner (policy x epsilon x scenario x seeds)"           "track-X" "Depends: A6, B6"
mk X2 "Run the sweep"                                                "track-X" "Depends: X1, C4"
mk X3 "Final analysis: figures, tables, paired stats"                "track-X" "Depends: X2, C3"
mk C1 "Independent RR/posterior test vectors"                        "member-C" "Start now. Needed by B2."
mk C2 "Dataset card + Fed-ISIC EDA on Colab"                         "member-C" "Start now. Needed by A7."
mk C3 "Analysis toolkit on synthetic RoundLogs"                      "member-C" "Start after P0. Needed by X3."
mk C4 "Colab runner notebook"                                        "member-C" "Start after A3. Needed by X2."
mk C5 "Fresh-clone reproduction check"                               "member-C" "Start after X1."
mk D1 "Verify all 16 references"                                     "member-D" "Start now."
mk D2 "Glossary + viva question bank"                                "member-D" "Start now."
mk D3 "Project board keeper + weekly status"                         "member-D" "Ongoing after P0."
mk D4 "Report + slides skeleton, architecture diagram"               "member-D" "Start after D1."
mk D5 "Results write-up + poster"                                    "member-D" "Start after X3."
echo "Done. Assign issues on GitHub (or: gh issue edit <N> --add-assignee <user>)."
