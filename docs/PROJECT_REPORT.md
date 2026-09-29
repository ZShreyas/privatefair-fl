# Project report: privacy-limited adaptive federated learning for medical imaging

# Privacy-limited adaptive client coordination for fair cross-silo medical-image federated learning

## Executive definition

This project tests whether an FL coordinator can use **only privacy-limited, low-dimensional client telemetry** to choose whom to train with and how to schedule them, while preventing slow or clinically distinctive sites from being persistently excluded. It does **not** claim novelty for federated learning, adaptive client selection, differential privacy, or fairness individually. Its claim is narrower: a coordinator can quantify and manage the trade-off among decision-time telemetry privacy, system efficiency, participation coverage, and worst-site medical-image performance. The formulation is motivated by the fact that existing medical/client-selection systems couple some—but not all—of privacy, selection, resource allocation, and fairness. [^1][^2][^3]

---

## 1. Project title

**Privacy-limited adaptive client coordination for fair cross-silo medical-image federated learning**

*A shorter implementation title:* **PrivateFair-FL**

---

## 2. Objectives of the project

### Primary objective

Design and evaluate a coordinator that selects and schedules cross-silo medical-imaging clients from a small locally privatized telemetry vector, while enforcing a participation-coverage constraint and measuring protection of worst-site model performance.

### Specific objectives

1. **Implement a minimal decision-telemetry interface.** Each site releases only three categorical signals: a utility/tail-risk status, a readiness/completion-time status, and a distribution-shift status. The coordinator derives participation history itself rather than requesting it from clients.
2. **Provide formal privacy for decision telemetry.** Apply $$K$$-ary randomized response independently to each categorical signal and track the composed per-site local-DP budget across decision epochs.
3. **Build an uncertainty-aware selection and scheduling policy.** Convert perturbed categories into posterior probabilities, then choose a feasible cohort using risk, representativeness, readiness, and exact participation age.
4. **Prevent participation starvation.** Enforce an age/coverage constraint so that efficiency-driven selection cannot indefinitely omit slow or unusual sites; record explicit deferrals when a site is unavailable.
5. **Protect tail clinical performance.** Evaluate worst-site and distribution-shifted-site performance rather than reporting only pooled/global accuracy.
6. **Measure the privacy–adaptation–fairness trade-off.** Compare raw-telemetry, private-telemetry, and non-adaptive baselines at multiple privacy budgets, noise levels, communication conditions, and heterogeneity regimes.
7. **Charge the coordinator for its cost.** Report telemetry bytes, controller compute time, extra validation time, total traffic, round duration, and time-to-target quality.

This scope deliberately combines the system/resource and fairness criteria used by client schedulers with medical-model site heterogeneity, rather than treating a mean task score as the sole outcome. [^4][^5][^6]

The objectives are motivated by medical selection methods that use distribution/performance information and by scheduling work that uses resource, behavior, and data-quality signals. Those methods establish the utility of adaptive signals, but typically do not restrict them to a small, formally privatized decision interface. [^2][^4][^1]

---

## 3. Scope of the project

### In scope

- **Setting:** synchronous or deadline-bounded federated learning across $$N=5\text{–}20$$ simulated hospital/site clients; no patient images leave a site.
- **Task:** one classification task and, if time permits, one segmentation task using public multi-site or deliberately site-partitioned medical-image data.
- **Heterogeneity:** joint label/prevalence skew, feature/domain shift, unequal site size, heterogeneous compute/network speed, and stochastic availability.
- **Adaptive decisions:** cohort selection, a simple readiness-dependent configuration (full/compressed/deferred), and an optional switch between a standard shared model mode and a personalized/robust mode.
- **Privacy:** local differential privacy for the *three decision signals*, not a new cryptographic primitive and not a claim of patient-level DP for the entire learning pipeline.
- **Fairness:** hard participation coverage, participation-age distribution, coverage of estimated shifted sites, and worst-site clinical performance.
- **Reproducibility:** a public experimental emulator, fixed seeds, recorded availability/resource traces, privacy ledger, and released configuration files.

### Explicitly out of scope

- Hospital deployment, clinical validation, or a claim of regulatory compliance.
- New secure-aggregation, homomorphic-encryption, MPC, or differential-privacy primitives.
- Defending against malicious telemetry manipulation, poisoning, or Byzantine updates in the first prototype.
- Claiming that local-DP telemetry protects the complete model update or provides end-to-end patient-level privacy. Client-level DP for updates is a separate, more expensive comparator/extension. Medical imaging work shows why DP becomes difficult when few hospitals participate. [^7]
- A learned reinforcement-learning controller. A transparent constrained score/optimization policy is the appropriate first controller because it makes privacy and fairness ablations interpretable.

These boundaries distinguish the study from prior work on client-level DP for medical updates, DP/resource-selection systems, and general fair selection. [^7][^1][^8]

### Research hypotheses

- **H1 — utility under privacy:** at moderate telemetry privacy budgets, the private coordinator improves worst-site performance and/or time-to-target quality relative to random selection with the same coverage requirement.
- **H2 — fairness:** coverage-constrained selection reduces maximum participation age and improves shifted-site inclusion relative to efficiency-only selection, with an explicit efficiency cost.
- **H3 — observability cost:** stronger local privacy or coarser telemetry reduces agreement with a raw-telemetry oracle and eventually removes the adaptive advantage; identifying that threshold is a principal result.
- **H4 — robustness:** the controller's relative benefit is largest when data, system, and availability heterogeneity occur together—not merely under a synthetic label-skew partition.

These are preregisterable empirical hypotheses, not claims already established by the cited papers. The prior work supplies the relevant ingredients—heterogeneity-aware clustering, fair scheduling, and private selection—but has not established the full intersection in cross-silo imaging. [^9][^4][^1]

---

## 4. Block diagram / system architecture

```text
                         Public initialization
              model w0, policy weights, public encoder/reference
                                      |
                                      v
+--------------------------------------------------------------------------+
|                   SERVER / COORDINATOR                                  |
|                                                                          |
|  Server state: participation age ai, privacy ledger, posterior beliefs  |
|                                                                          |
|  1. Receive three privatized categorical reports YiU, YiR, YiD          |
|  2. Bayesian posterior update: risk, readiness, shift probabilities     |
|  3. Enforce coverage / shift-representation constraints                  |
|  4. Select feasible cohort S and schedule: full / compressed / deferred |
|  5. Broadcast model/configuration                                        |
|  6. Aggregate selected model updates (initially equal-weight FedAvg)    |
+--------------------------------------------------------------------------+
             ^                                  |                 |
             | privatized telemetry             | model/config     | local update
             |                                  v                 ^
+-----------------------------+        +-----------------------------+
| HOSPITAL / SITE i           |  ...   | HOSPITAL / SITE n           |
| private images, labels, Vi  |        | private images, labels, Vn  |
|                             |        |                             |
| Local evaluation:           |        | Local evaluation:           |
|  - deterioration bin BU     |        |  - deterioration bin BU     |
|  - readiness bin BR         |        |  - readiness bin BR         |
|  - shift bin BD             |        |  - shift bin BD             |
|                             |        |                             |
| K-ary randomized response   |        | K-ary randomized response   |
| on only BU, BR, BD           |        | on only BU, BR, BD           |
|                             |        |                             |
| Local training; upload      |        | Local training; upload      |
| selected model update       |        | selected model update       |
+-----------------------------+        +-----------------------------+
```

### Minimal telemetry and control path

| Signal | Local construction | What the coordinator receives | Main decision it enables |
|---|---|---|---|
| Utility/tail-risk status | Five-level clipped change in local validation loss or calibration error | One locally randomized ordinal category | Prioritize a site at likely local deterioration; trigger robust/personalized mode |
| Readiness status | Four-level expected-completion/availability bucket | One locally randomized category | Form a feasible cohort; choose full, compressed, or deferred participation |
| Distribution-shift status | Three-level shift flag from a local frozen-encoder distance or prespecified shift detector | One locally randomized category | Ensure representation of atypical domains; activate local normalization/personalization |
| Participation age | Computed from the coordinator's own schedule log | Not transmitted | Hard maximum-age/coverage constraint |

This deliberately avoids raw accuracy/IoU, class histograms, sample counts, scanner identifiers, gradients, and detailed hardware/network profiles. It is a proposed minimization rule, not a claim that these fields are never useful. Medical clustering methods demonstrate why distribution information is useful, while general schedulers show that raw resources and behavior are commonly exposed to make decisions. [^2][^4]

### Privacy mechanism and decision rule

For a true $$K$$-class telemetry bin $$B$$, a site emits $$Y$$ through $$K$$-ary randomized response with privacy budget $$\epsilon$$:

$$
\Pr(Y=y\mid B=b)=
\begin{cases}
\frac{e^{\epsilon}}{e^{\epsilon}+K-1},&y=b\\
\frac{1}{e^{\epsilon}+K-1},&y\ne b.
\end{cases}
$$

This gives pure $$\epsilon$$-local DP **for that report**. The coordinator uses the known channel probabilities and a prior to calculate posterior probabilities for risk, readiness, and shift. This proposed inference step is designed to avoid treating a randomized categorical report as an exact measurement. It then selects sites by a constrained score:

$$
\max_{x_i\in\{0,1\}}\sum_i x_i\left[\alpha\,\widehat{risk}_i+\beta\,\Pr(shift_i)+\gamma\,age_i+\delta\,\Pr(ready_i)\right]
$$

subject to cohort capacity, mandatory coverage of eligible overdue sites, and at least one shifted-site slot when such a site is available. The initial design aggregates selected shared parameters equally; it does **not** use noisy telemetry as an aggregation weight. Existing DP-selection work supports coupling privacy and selection, while fair-selection work supports explicit long-run participation control. [^10][^11][^8]

---

## 5. Literature review: 16 core papers

The papers below form a focused literature set for the **finalized problem**, not a general survey of medical federated learning. A few are preprints; they are included because they are direct technical comparators and are labelled accordingly. The review separates four building blocks: medical-imaging heterogeneity, privacy-preserving FL, adaptive/fair selection, and secure/private scheduling.

### A. Medical-imaging and healthcare comparators

1. **Jiang et al. (2023), “Client-Level Differential Privacy via Adaptive Intermediary in Federated Medical Imaging.”** Establishes that client-level DP is especially difficult in small hospital federations and provides an adaptive intermediary mechanism plus open code. It is the best update-privacy comparator, but it does not build a fairness-constrained coordinator from privatized utility telemetry. [^7]

2. **Messinis, Protonotarios & Doulamis (2024), “Differentially Private Client Selection and Resource Allocation in Federated Learning for Medical Applications Using Graph Neural Networks.”** DPS-GAT jointly addresses DP, client selection, and resource allocation on a medical dataset. It is the closest healthcare scheduling precedent; the proposed work differs by restricting coordinator observability to a small privatized telemetry vector and testing coverage plus worst-site outcomes. [^1]

3. **Huang et al. (2024), “Personalized Federated Learning Using Client Clustering for Medical Image Classification.”** pFedCM estimates class distributions, clusters/selects clients, shares partial parameters, and uses adaptive aggregation. It shows the decision value of distribution information, while also motivating its privacy-limited replacement because the coordinator still obtains estimated distribution-related information. [^2]

4. **Lin et al. (2023), “Unifying and Personalizing Weakly-supervised Federated Medical Image Segmentation via Adaptive Representation and Aggregation.”** FedICRA treats domain and label/annotation heterogeneity with personalized representation and adaptive aggregation. It is a strong downstream model-mode baseline, not a private decision-telemetry coordinator. [^6]

5. **Wang et al. (2023), “FedDP: Dual Personalization in Federated Medical Image Segmentation.”** Uses local-query personalization and prediction inconsistency to address site uniqueness. It justifies a personalized/robust response after selecting a high-shift site, but does not solve fairness-aware private site selection. [^12]

### B. Privacy-aware client selection and scheduling

6. **Xie & Zhang (2022), “Federated Learning With Personalized Differential Privacy Combining Client Selection.”** Couples a client-specific privacy requirement with selection and a loss/privacy score. It is central evidence that privacy and selection cannot be optimized independently, but it uses raw local-loss-related ranking rather than a privacy-limited coordinator interface. [^10]

7. **Alam, Shukla & Rao (2023), “Near-optimal Differentially Private Client Selection in Federated Settings.”** Provides an iterative DP selection mechanism under local computation and probabilistic intent, with a long-run participation objective. It is a theoretical/algorithmic comparator for privacy-aware participation, but lacks medical-image tail-performance evaluation. [^11]

8. **Xu, Zhang & Huang (2024), “Joint Client Selection and Privacy Compensation for Differentially Private Federated Learning.”** Formulates selection jointly with compensation for privacy leakage accumulating through repeated participation. It informs the project privacy ledger and participation-cost discussion, but its fairness notion is economic/selection-probability oriented rather than clinical worst-site protection. [^13]

9. **Zhao et al. (2025, preprint), “Enhancing Convergence, Privacy and Fairness for Wireless Personalized Federated Learning: Quantization-Assisted Min-Max Fair Scheduling.”** The strongest general technical precedent: it couples quantization-assisted Gaussian DP with client selection, channel allocation, power control, and a min–max objective on individual-model bounds. It differs fundamentally in setting (wireless personalized FL rather than cross-silo imaging) and does not demonstrate a coordinator driven by locally privatized clinical utility categories. [^3]

10. **Jiang et al. (2024, preprint), “Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning.”** Shows that a malicious server can manipulate participation to break privacy assumptions, and proposes verifiable random/informed-pool selection. It is a security-boundary reference; adversarial-server protection is an extension, not a first-prototype objective. [^14]

### C. Fairness and coverage in adaptive selection

11. **Zhang et al. (2023, preprint), “Multi-Criteria Client Selection and Scheduling with Fairness Guarantee for Federated Learning Service.”** Uses resources, data quality, and behavior for pool selection, then schedules subsets so every client is selected at least once. It directly motivates an explicit coverage constraint, but it assumes readable multi-criteria information and does not use medical site-level outcome protection. [^4]

12. **Shi et al. (2023, preprint), “Fairness-Aware Client Selection for Federated Learning.”** FairFedCS adjusts selection probabilities using reputation, participation history, and model contribution. The project takes the participation-history idea but avoids collecting client-reported history and replaces unrestricted contribution signals with privatized ordinal status. [^8]

13. **Huang et al. (2020), “An Efficiency-Boosting Client Selection Scheme for Federated Learning With Fairness Guarantee.”** Uses estimated model-exchange time and a fairness-guaranteed selection formulation. It is a baseline for efficiency–coverage trade-offs; it predates the decision-time privacy question. [^5]

14. **Wolfrath et al. (2022), “HACCS: Heterogeneity-Aware Clustered Client Selection for Accelerated Federated Learning.”** Represents distinguishable data distributions rather than individual devices, uses privacy-preserving distribution estimation and clustering, and uses clusters for scheduling. It is close to the representativeness component but does not combine local-DP telemetry, hard coverage, and worst-site medical performance. [^9]

15. **Albaseer et al. (2023, preprint), “Fair Selection of Edge Nodes to Participate in Clustered Federated Multitask Learning.”** Couples fairness, clustering, device-delay scheduling, and specialized models. It demonstrates the value—and the sensitivity—of client distribution and resource signals, which the proposed interface deliberately coarsens and privatizes. [^15]

16. **Ren et al. (2022), “Client Selection Based on Diversity Scaling for Federated Learning on Non-IID Data.”** Dynamically adjusts selection weights using data diversity to trade off client count, communication cost, and convergence. It is a non-private diversity-selection baseline; the project asks whether a coarsened privatized shift signal retains enough of that value. [^16]

### Synthesis of the literature

The literature already supports each isolated component: DP-protected FL, distribution-aware medical selection, resource-aware scheduling, and participation-fair selection. [^7][^2][^4] The unresolved experimental intersection is whether a coordinator can make **clinically useful, site-inclusive** choices after observability is deliberately constrained by local privacy, rather than simply assuming access to exact loss, class distribution, resource, or update-quality signals. The most direct general comparator is Zhao et al.'s privacy/fair wireless scheduler; the most direct medical comparator is DPS-GAT. Neither removes the need to test site-level clinical performance and privacy-limited decision telemetry under realistic cross-silo heterogeneity. [^3][^1]

---

## 6. Current limitations in the domain

The project is motivated by a narrower limitation than “federated learning is not private enough.” In the closest selection and scheduling systems, the coordinator often needs exactly the information a cross-silo hospital may be reluctant to disclose: local loss or contribution, class/distribution characteristics, availability, network state, or hardware capacity. The resulting problem is one of **decision-time observability**: keeping images local does not by itself limit what a central controller learns from the metadata it uses to prioritize sites. Distribution-aware medical methods demonstrate the utility of site-distribution and performance signals, while scheduling methods routinely use resource and behavior signals; neither pattern by itself supplies a privacy-minimized clinical coordination interface. [^2][^4]

### 6.1 Technical limitations

- **Privacy is usually applied to updates, not the decision loop.** Client-level DP and secure aggregation can protect model updates, but they do not tell a coordinator how to select an individual hospital when the useful per-site state is hidden. Medical client-level DP work also shows that the small number of silos makes the privacy–utility trade-off particularly acute. [^7]
- **Raw telemetry has an under-specified disclosure risk.** Exact local validation scores, class-distribution estimates, sample counts, scanner/domain descriptors, and fine-grained resource traces can reveal operational or clinically meaningful properties of a site. Existing medical clustering/selection uses distribution-related information, but does not establish a formal local-DP protection mechanism for that telemetry. [^2]
- **Fair selection and clinical fairness are different objectives.** A scheduler can equalize selection counts or bound participation delay yet still leave the lowest-performing or most shifted hospital with a poor model. Conversely, prioritizing high-loss sites can starve slow sites. Fair-selection literature provides participation mechanisms, but not a sufficient site-level clinical protection criterion. [^4][^8]
- **Current evaluations isolate heterogeneity sources.** Many studies vary only data skew or only resource delay. The project-relevant failure mode arises when domain/label shift, unequal data volume, availability, and stragglers coexist; performance under such joint heterogeneity is less well characterized in the closest methods. [^6][^9]
- **Privacy noise can destabilize adaptation.** A noisy signal may cause cohort churn, missed high-risk sites, or scheduling decisions that are worse than coverage-matched random selection. Reporting only a nominal $$\epsilon$$ or an average accuracy score does not reveal that observability failure.
- **Control-plane cost is commonly omitted.** Telemetry derivation, randomized reporting, posterior inference, and constrained selection consume client and coordinator time. Resource-aware systems motivate accounting for time and communication, but the incremental cost of a privacy-limited telemetry control loop should be measured separately. [^9][^5]

### 6.2 Experimental and translational limitations

- **Small-silo statistics are fragile.** With only 5–20 sites, one excluded or poorly served hospital can strongly affect both worst-site metrics and privacy composition. Results should therefore be replicated across site partitions and availability seeds rather than presented as one final-round number.
- **Simulated sites are not clinical deployment.** Public datasets partitioned into synthetic hospitals cannot fully capture governance rules, scanner procurement history, workflow differences, or true outage patterns. The study should frame its contribution as a reproducible systems evaluation, not evidence of hospital deployability.
- **LDP protects the released category, not the entire pipeline.** The proposed mechanism does not protect model updates, membership in a participating consortium, a malicious client, or fabricated telemetry. These remain explicit extension/security boundaries; adversarial-server participant-selection attacks are separately studied in Lotto. [^14]

## 7. Methodology used by the project team

### 7.1 Study design

The team will conduct a controlled simulation study of cross-silo medical-image FL. A common task, model architecture, site partition, optimization budget, deadline, and availability trace will be held fixed while only the coordination policy and telemetry visibility change. This design makes the primary causal contrast clear: **what is gained or lost when adaptive coordination sees a small locally privatized telemetry vector rather than raw site metadata?**

The study will use $$N=5\text{–}20$$ simulated sites. Start with one public multi-site classification dataset; add one segmentation task only after the first experiment family is stable. Create a factorial heterogeneity generator with: (i) label/prevalence imbalance, (ii) feature or acquisition-domain shift, (iii) unequal site sample sizes, (iv) heterogeneous compute/network completion times, and (v) stochastic availability. Run a clean homogeneous scenario first as a debugging control, then data-only, systems-only, and joint-heterogeneity conditions. This design combines the distribution heterogeneity addressed by medical personalization/clustering studies with the resource and behavior heterogeneity used in client schedulers. [^2][^6][^4]

### 7.2 Per-round workflow

1. **Local measurement.** At the beginning of each decision epoch, each eligible site computes three bounded internal categories: deterioration/tail-risk status, readiness/completion-time status, and domain-shift status. These calculations use local validation data and local runtime observations; the underlying numeric values remain local. The categories retain the performance/distribution/resource roles used by prior adaptive methods while intentionally avoiding their direct numeric inputs. [^2][^4][^1]
2. **Telemetry minimization and privatization.** Each category is clipped to its fixed alphabet and passed through $$K$$-ary randomized response. Release only one report per signal per epoch, with a recorded per-signal privacy budget and a composed per-site ledger. Optionally quantize before privatization, but do not add raw metrics to the message.
3. **Uncertainty-aware coordination.** The server uses the known randomized-response channel and predeclared priors to calculate posterior probabilities for risk, readiness, and shift. It combines those posterior estimates with its exact, server-derived participation age; it never requests participation history from a client.
4. **Constrained selection and scheduling.** The coordinator first reserves capacity for eligible overdue sites and, when feasible, a shifted-site representation slot. It fills remaining capacity by the posterior score specified in Section 4, then assigns full, compressed, or deferred participation according to readiness/deadline status.
5. **Training and aggregation.** Selected clients perform a fixed local training budget and return a model update. The initial implementation uses equal-weight FedAvg aggregation to isolate the value of coordination; robust or personalized aggregation is a controlled extension rather than a confound.
6. **Evaluation and logging.** Evaluate the shared model on every site at prespecified checkpoints, not only participating sites. Log true internal bins in the simulator for offline evaluation only, privatized reports, posterior estimates, cohort decisions, deferral reasons, bytes, timing, and privacy consumption.

### 7.3 Comparator and ablation plan

The primary comparisons are: random FedAvg, coverage-constrained random selection, raw-telemetry adaptive oracle, quantized non-private telemetry, and the proposed private telemetry controller. The ablations remove one signal at a time, remove the coverage constraint, vary $$\epsilon^{tel}$$ and reporting cadence, and compare posterior inference against naively treating a noisy category as true. These comparisons separate the effects of adaptivity, coarsening, randomized-response noise, and fairness constraints.

Where an external method can be faithfully reproduced, include it as a secondary comparator; otherwise describe it as a conceptual reference rather than claiming a direct benchmark. DPS-GAT, privacy-aware selection, and fairness schedulers motivate these design axes but differ in information access, setting, or outcome definition. [^1][^10][^4]

### 7.4 Experimental protocol and analysis

- Predefine the task metric, coverage threshold, cohort capacity, deadline, privacy-budget grid, and stopping rule before the final runs.
- Use at least 3–5 random seeds per condition; preserve identical partitions, availability traces, and initialization across paired policies.
- Report mean, dispersion, per-site distributions, worst-site result, and paired differences—not only the best seed or final global average.
- Plot the Pareto frontier of worst-site quality, coverage, wall-clock time, traffic, and telemetry privacy. A non-monotone or negative result is reportable evidence about the observability limit; this makes visible the privacy–performance trade-off emphasized in medical client-level DP work and in private fair scheduling. [^7][^3]
- Maintain configuration files, deterministic seeds where feasible, a privacy ledger, and machine-readable round logs. Perform a dry run that verifies the telemetry message contains no raw metric, histogram, sample count, or hardware identifier.

## 8. System requirements

### 8.1 Recommended development environment

| Layer | Minimum practical requirement | Recommended project configuration |
|---|---|---|
| Operating system | Linux, macOS, or Windows with a Linux-compatible environment | Ubuntu 22.04/24.04 LTS for reproducible CUDA and container support |
| Language and environment | Python 3.10+; isolated virtual environment | Python 3.11, `uv` or Conda environment, pre-commit formatting and pinned dependency lockfile |
| FL framework | Any framework that supports custom client selection and simulation | Flower for rapid custom control logic; optionally Fed-BioMed only if its healthcare-specific workflow is required |
| Deep-learning stack | PyTorch 2.x and torchvision/MONAI as task requires | PyTorch 2.x + MONAI for imaging transforms, metrics, and segmentation support |
| Privacy module | A small local randomized-response implementation; no cryptographic library required | Team-written, unit-tested categorical LDP module with deterministic test vectors and a per-client privacy ledger |
| Optimization/data tools | NumPy, pandas, scikit-learn, matplotlib/seaborn | Add Hydra or YAML configuration management and MLflow/W&B local logging if the team can maintain it |
| Reproducibility | Git repository and fixed requirements | Git + Git LFS/DVC only when data artefacts require it; Docker or Apptainer image for final replication |

### 8.2 Compute and storage

| Resource | Minimum for development | Recommended for final experiments |
|---|---|---|
| Developer machines | 16 GB RAM; 4 CPU cores | 32 GB RAM; 8+ CPU cores for parallel site simulation |
| GPU | One CUDA-capable GPU with 8 GB VRAM, or CPU-only for toy tests | One GPU with 16–24 GB VRAM; access to a second GPU or shared cluster materially shortens repeated-seed experiments |
| Storage | 100 GB free SSD | 250–500 GB SSD for datasets, checkpoints, per-round logs, and repeated experiments |
| Network | Standard internet for package/data acquisition | No special network hardware; emulate bandwidth, latency, availability, and deadlines in software |
| Runtime planning | Short smoke tests on a reduced dataset | Budget a pilot run, an ablation sweep, and final 3–5-seed reruns; use checkpoints and resumable logs |

A single 16–24 GB GPU is sufficient for a modest prototype if the team begins with 2D classification and small cohorts. Full-resolution 3D segmentation or multiple concurrent client processes can exceed that budget; use sequential simulation, patch-based training, mixed precision, and smaller task variants rather than silently reducing the evaluation to one seed.

### 8.3 Functional software requirements

The implementation should expose the following modules with clean interfaces:

1. **Data/site-partition module:** creates reproducible site partitions and labels each client’s shift/severity group for evaluation only.
2. **Client simulator:** models local training, local validation, completion time, stochastic availability, and deadline misses.
3. **Telemetry module:** derives the three true categories, applies randomized response, validates message size, and writes privacy-ledger entries.
4. **Coordinator module:** stores posterior beliefs and exact participation age, enforces feasibility/coverage rules, scores candidates, schedules the cohort, and records its decision explanation.
5. **Training/aggregation module:** implements FedAvg first, with a clear extension point for personalization or robust aggregation.
6. **Evaluation module:** computes global and per-site medical metrics, fairness/coverage, privacy, traffic, controller cost, and selection stability.
7. **Experiment manager:** reads versioned configuration files, assigns seeds, replays traces, saves checkpoints, and exports plots/tables.

### 8.4 Quality, privacy, and security requirements

- **No raw decision telemetry leaves the client process.** Enforce schema validation so the outbound telemetry payload contains only three categorical reports and an epoch identifier.
- **Separate private signals from simulator ground truth.** True bins and raw local metrics may be logged only in the offline simulator namespace for evaluation; the coordinator interface must not receive them.
- **Privacy accounting is auditable.** Store per-client, per-signal, per-epoch $$\epsilon$$ values and the reported release count; fail a run if a configured budget is exceeded.
- **Policy decisions are reproducible.** Log posteriors, mandatory-coverage flags, candidate scores (or rank), selected cohort, availability, and deferral reasons.
- **Baselines are budget-matched.** Give compared policies the same cohort capacity, deadlines, local epochs, model architecture, partitions, traces, and total round budget unless the comparison explicitly studies one of these variables.
- **Unit and integration tests are required.** Test randomized-response probabilities, posterior updates, privacy composition accounting, coverage enforcement, unavailable-client handling, and the guarantee that server state alone produces participation age.

## 9. Six-month project timeline

The schedule assumes a team of 3–5 undergraduate researchers, a weekly technical meeting, and a shared codebase. Each month ends with a concrete review gate; work should not progress to the next experimental layer until the preceding gate is reproducible.

| Month | Main work | Deliverables and decision gate |
|---|---|---|
| **1 — specification and baseline environment** | Lock the research question, hypotheses, datasets, task metric, site-count range, and heterogeneity factors. Set up the repository, environment, data pipeline, and plain FedAvg simulator. Reproduce a single-site and homogeneous multi-site training run. | One-page experiment specification; reproducible FedAvg baseline; dataset card and initial risk/privacy statement. **Gate:** identical seed reproduces a reference result. |
| **2 — heterogeneity emulator and coverage baseline** | Implement deterministic site partitions, availability/completion-time traces, deadline handling, and server-side participation age. Add coverage-constrained random selection and core logging. | Joint-heterogeneity generator; random and coverage-constrained baselines; per-site metric dashboard. **Gate:** coverage policy demonstrably bounds maximum age in controlled traces. |
| **3 — minimal telemetry and privacy layer** | Implement local binning, $$K$$-ary randomized response, privacy ledger, schema checks, and posterior decoding. Run unit tests and small privacy/utility smoke tests. | Telemetry API; tested LDP module; privacy-composition plots; message-size report. **Gate:** coordinator code cannot access raw telemetry in integration tests. |
| **4 — adaptive coordinator and ablations** | Implement constrained posterior-score policy, full/compressed/deferred scheduling, raw-oracle mode, quantized non-private mode, and one-signal-removal ablations. Tune only on designated development traces. | Functional proposed controller and four primary baselines. **Gate:** controller changes selection in response to controlled risk/readiness/shift perturbations while satisfying coverage. |
| **5 — full experimental sweep** | Execute privacy-budget, heterogeneity, and availability sweeps; run repeated seeds; profile timing, traffic, and controller overhead. Investigate failures and rerun only from predeclared criteria. | Complete experiment matrix; preliminary Pareto plots; failure analysis; reproducible checkpoints. **Gate:** every headline comparison has per-site, fairness, and systems metrics. |
| **6 — validation, analysis, and reporting** | Freeze code/configurations; rerun final selected conditions from clean environments; perform statistical summaries and sensitivity analysis; prepare final report, poster/demo, and repository documentation. | Final tables/figures; methods appendix; reproducibility package; presentation/demo. **Gate:** an independent team member can reproduce at least one headline run from the documentation. |

### Weekly operating rhythm

- **Week start:** choose a bounded issue list and identify the single experiment needed to retire the highest-risk assumption.
- **Midweek:** code review and a short experiment readout; reject runs missing site-level or privacy-ledger logs.
- **Week end:** merge only configuration-backed results, update the experiment register, and record failed hypotheses as results rather than deleting them.

## 10. Performance metrics identified

### A. Medical-model quality — primary outcomes

These outcomes operationalize site-level quality and tail performance as project endpoints; they are proposed measurement choices motivated by the site heterogeneity addressed in medical personalization methods. [^6][^12]

| Metric | Definition / reporting rule | Why it matters |
|---|---|---|
| Per-site task quality | Classification: AUROC, macro-F1, balanced accuracy; segmentation: Dice and 95th-percentile Hausdorff distance | A pooled score can hide a failed hospital/site |
| Worst-site quality | $$\min_i Q_i$$; report the bottom-10th percentile when $$N\ge10$$ | Direct protection target for the project |
| Mean and macro site quality | $$N^{-1}\sum_i Q_i$$ and unweighted mean across sites | Prevents a large site from dominating the assessment |
| Site-performance disparity | Standard deviation, range, and $$\max_i Q_i-\min_i Q_i$$ | Measures outcome inequality |
| Shifted-site quality | Mean and worst quality within moderate/high-shift sites | Tests whether unusual sites were represented rather than excluded |
| Calibration (classification) | Expected calibration error and Brier score per site | Clinical confidence can fail even when discrimination is acceptable |
| Time-to-target quality | Wall-clock time and communication rounds to a prespecified quality threshold | Separates fast convergence from useful deployment time |

### B. Participation fairness and representativeness — co-primary outcomes

The coverage metrics reflect the long-run participation and fairness concerns explicit in fair schedulers; the clinical site-performance endpoint remains a project-specific extension. [^4][^8][^5]

| Metric | Definition / reporting rule | Desired direction |
|---|---|---|
| Maximum participation age | Largest number of decision epochs since an eligible site last participated | Lower, bounded by the coverage policy |
| Selection-count distribution | Per-site counts, min/max, standard deviation, and Jain's fairness index | Higher fairness index; do not interpret alone |
| Coverage-constraint satisfaction | Fraction of eligible overdue sites selected; number and reason for coverage deferrals | Near 100% among feasible sites; explicit deferrals |
| Shift-group coverage | Proportion of cohorts containing an estimated shifted site; selection rate by shift group | Demonstrates representativeness |
| Tail selection rate | Selection rate for high posterior-risk sites versus low-risk sites | Confirms that risk telemetry affects decisions without permanent exclusion |
| Worst-client loss | Maximum local validation loss / worst-site error across all sites, not only selected sites | Prevents selective reporting of participating clients |

### C. Privacy and telemetry quality

The telemetry measures make the privacy–observability trade-off visible. Existing medical DP work and privacy-aware schedulers motivate reporting both protection and performance loss rather than only a nominal privacy parameter. [^7][^3]

| Metric | Definition / reporting rule | Why it matters |
|---|---|---|
| Telemetry privacy budget | Per-signal and composed per-site $$\epsilon_i^{tel}$$ after all epochs | Makes sequential local-DP loss explicit |
| Truthful-report probability | $$e^{\epsilon}/(e^{\epsilon}+K-1)$$ for each signal alphabet | Converts $$\epsilon$$ to an interpretable observability level |
| Telemetry reconstruction quality | Agreement/confusion matrix between true bins and coordinator posterior/MAP bins | Measures information retained after privatization |
| Raw-vs-private selection overlap | Jaccard overlap with an oracle using raw bins; also compare resulting outcomes | Direct measure of adaptation lost to privacy |
| Privacy–utility Pareto curve | Worst-site quality, mean quality, coverage, and time versus $$\epsilon^{tel}$$ | Main contribution; avoids one arbitrary privacy setting |
| Release frequency | Decision epochs and signals released per site | Reporting cadence is part of privacy cost |

### D. Systems and controller overhead

The overhead measures reflect the resource and time constraints that motivate adaptive client selection in the first place. [^9][^5]

| Metric | Definition / reporting rule | Why it matters |
|---|---|---|
| Telemetry traffic | Bytes/site/epoch and total bytes; report separately from model updates | Shows whether telemetry is actually lightweight |
| Model-update traffic | Downlink + uplink bytes, compression ratio, and total training traffic | Captures cost of selected configuration |
| Round duration / straggler rate | Mean, median, p95 round time; deadline misses and aborted rounds | Tests readiness scheduling |
| Controller runtime | Posterior update + optimization time per decision epoch | Prevents treating coordination as free |
| Extra local evaluation cost | Validation examples processed, GPU/CPU seconds where available | Captures the cost of deriving telemetry |
| Time-to-quality under a fixed budget | Quality reached given fixed time, bytes, and cohort budget | Allows fair comparison with random/FedAvg-style selection |
| Selection stability | Consecutive-epoch selection overlap and sensitivity to telemetry noise | Detects noise-driven policy thrashing |

### E. Experimental comparisons and ablations

At minimum, report the following conditions under identical task, availability trace, deadline, and total privacy/communication budget. This comparison isolates the components emphasized separately in prior private, fair, and heterogeneity-aware selection work. [^10][^4][^9]

1. **Random selection + FedAvg** (no adaptive telemetry).
2. **Coverage-constrained random selection** (isolates the cost/benefit of fairness).
3. **Raw-telemetry adaptive oracle** (upper bound; not privacy-preserving).
4. **Quantized but non-private telemetry** (isolates coarsening from randomized-response noise).
5. **Private telemetry + adaptive policy** (proposed system), at low/medium/high $$\epsilon^{tel}$$.
6. **Private telemetry without hard coverage** (isolates fairness constraint).
7. **Private telemetry without risk or shift signal** (one ablation per signal).
8. **DPS-GAT / HACCS / fairness scheduler reproduction where feasible**, or clearly labelled conceptual baselines if exact reproduction is unavailable.

Include at least one mid-training domain/availability change. The decisive result is not a single mean-accuracy gain; it is whether private adaptation shifts the measured Pareto frontier for worst-site quality, coverage, and time/traffic compared with the baselines.

---

## 11. References (IEEE style)

[1] M. Jiang, Y. Zhong, A. Le, X. Li, and Q. Dou, “Client-Level Differential Privacy via Adaptive Intermediary in Federated Medical Imaging,” 2023. Available: https://arxiv.org/abs/2307.12542.

[2] S. Messinis, N. E. Protonotarios, and N. Doulamis, “Differentially Private Client Selection and Resource Allocation in Federated Learning for Medical Applications Using Graph Neural Networks,” *Sensors*, vol. 24, no. 16, Art. no. 5142, 2024.

[3] L. Huang, S. Gou, S. Cao, W. Liu, and K. Jiang, “Personalized Federated Learning Using Client Clustering for Medical Image Classification,” 2024.

[4] L. Lin, J. Wu, Y. Liu, K. K. Y. Wong, and X. Tang, “Unifying and Personalizing Weakly-supervised Federated Medical Image Segmentation via Adaptive Representation and Aggregation,” 2023. Available: https://arxiv.org/abs/2304.05635.

[5] J. Wang, Y. Jin, D. Stoyanov, and L.-C. Wang, “FedDP: Dual Personalization in Federated Medical Image Segmentation,” 2023.

[6] Y. Xie and L. Zhang, “Federated Learning With Personalized Differential Privacy Combining Client Selection,” 2022.

[7] S. E. Alam, D. Shukla, and S. Rao, “Near-optimal Differentially Private Client Selection in Federated Settings,” 2023. Available: https://arxiv.org/abs/2310.09370.

[8] R. Xu, Y.-J. A. Zhang, and J. Huang, “Joint Client Selection and Privacy Compensation for Differentially Private Federated Learning,” 2024.

[9] X. Zhao, Q. Cui, Z. Du, W. Ni, W. Li, X. Yu, J. Zhang, X. Tao, and P. Zhang, “Enhancing Convergence, Privacy and Fairness for Wireless Personalized Federated Learning: Quantization-Assisted Min-Max Fair Scheduling,” preprint, 2025. Available: https://arxiv.org/abs/2506.02422.

[10] Z. Jiang, P. Ye, S. He, W. Wang, R. Chen, and B. Li, “Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning,” preprint, 2024. Available: https://arxiv.org/abs/2401.02880.

[11] M. Zhang, H. Zhao, S. C. Ebron, R. Xie, and K. Yang, “Multi-Criteria Client Selection and Scheduling with Fairness Guarantee for Federated Learning Service,” preprint, 2023. Available: https://arxiv.org/abs/2312.14941.

[12] Y. Shi, Z. Liu, Z. Shi, and H. Yu, “Fairness-Aware Client Selection for Federated Learning,” preprint, 2023. Available: https://arxiv.org/abs/2307.10738.

[13] T. Huang, W. Lin, W. Wu, L. He, K. Li, and A. Y. Zomaya, “An Efficiency-Boosting Client Selection Scheme for Federated Learning With Fairness Guarantee,” 2020. Available: https://arxiv.org/abs/2011.01783.

[14] J. Wolfrath, N. Sreekumar, D. Kumar, Y. Wang, and A. Chandra, “HACCS: Heterogeneity-Aware Clustered Client Selection for Accelerated Federated Learning,” 2022.

[15] A. Albaseer, M. Abdallah, A. Al-Fuqaha, A. Mohammed, A. Erbad, and O. Dobre, “Fair Selection of Edge Nodes to Participate in Clustered Federated Multitask Learning,” preprint, 2023. Available: https://arxiv.org/abs/2304.13423.

[16] Y. Ren, A. Sajjanhar, S. Gao, and S. Loke, “Client Selection Based on Diversity Scaling for Federated Learning on Non-IID Data,” 2022.

---

## Proposed success criterion

The project succeeds if, across more than one joint-heterogeneity scenario, the private coordinator provides a transparent, measured trade-off: it keeps coverage violations bounded and retains or improves worst-site performance versus coverage-matched random selection, while its loss relative to the raw-telemetry oracle is reported rather than hidden. This is a project decision criterion, informed by the fact that present systems separately optimize privacy/resource selection, fairness, or distribution-aware selection. [^1][^3][^2] A negative result—privacy noise eliminates the adaptive advantage at a practical budget—is also scientifically useful, because it locates the observability limit that existing raw-telemetry schedulers leave unmeasured.



[^1]: Messinis et al., 2024. Differentially Private Client Selection and Resource Allocation in Federated Learning for Medical Applications Using Graph Neural Networks. Italian National Conference on Sensors.

[^2]: Huang et al., 2024. Personalized Federated Learning Using Client Clustering for Medical Image Classification. Proceedings of the 2024 9th International Conference on Biomedical Imaging, Signal Processing.

[^3]: Zhao et al., 2025. Enhancing Convergence, Privacy and Fairness for Wireless Personalized Federated Learning: Quantization-Assisted Min-Max Fair Scheduling. IEEE Transactions on Mobile Computing.

[^4]: Zhang et al., 2023. Multi-Criteria Client Selection and Scheduling with Fairness Guarantee for Federated Learning Service. arXiv.org.

[^5]: Huang et al., 2020. An Efficiency-Boosting Client Selection Scheme for Federated Learning With Fairness Guarantee. IEEE Transactions on Parallel and Distributed Systems.

[^6]: Lin et al., 2023. Unifying and Personalizing Weakly-supervised Federated Medical Image Segmentation via Adaptive Representation and Aggregation. MLMI@MICCAI.

[^7]: Jiang et al., 2023. Client-Level Differential Privacy via Adaptive Intermediary in Federated Medical Imaging. International Conference on Medical Image Computing and Computer-Assisted Intervention.

[^8]: Shi et al., 2023. Fairness-Aware Client Selection for Federated Learning. IEEE International Conference on Multimedia and Expo.

[^9]: Wolfrath et al., 2022. HACCS: Heterogeneity-Aware Clustered Client Selection for Accelerated Federated Learning. IEEE International Parallel and Distributed Processing Symposium.

[^10]: Xie & Zhang, 2022. Federated Learning With Personalized Differential Privacy Combining Client Selection. International Conference on Big Data Computing and Communications.

[^11]: Alam et al., 2023. Near-optimal Differentially Private Client Selection in Federated Settings. Allerton Conference on Communication, Control, and Computing.

[^12]: Wang et al., 2023. FedDP: Dual Personalization in Federated Medical Image Segmentation. IEEE Transactions on Medical Imaging.

[^13]: Xu et al., 2024. Joint Client Selection and Privacy Compensation for Differentially Private Federated Learning. Conference on Computer Communications Workshops.

[^14]: Jiang et al., 2024. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning. USENIX Security Symposium.

[^15]: Albaseer et al., 2023. Fair Selection of Edge Nodes to Participate in Clustered Federated Multitask Learning. IEEE Transactions on Network and Service Management.

[^16]: Ren et al., 2022. Client Selection Based on Diversity Scaling for Federated Learning on Non-IID Data. International Conference on Broadband Communications, Networks and Systems.