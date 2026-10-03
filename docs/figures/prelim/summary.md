# Summary

_Preliminary — single seed, PathMNIST_

| run | rounds | mean_acc | worst_site_acc | mean_acc_last3 | worst_site_acc_last3 | shifted_mean_acc | reports_per_site | eps_per_site_max | eps_per_site_mean | sim_seconds_total | MB_total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FedAvg (all sites) | 50 | 0.716 | 0.380 | 0.716 | 0.388 | 0.381 | 0.000 | 0.000 | 0.000 | 2594.325 | 12971.700 |
| Random | 50 | 0.723 | 0.318 | 0.707 | 0.369 | 0.321 | 0.000 | 0.000 | 0.000 | 2382.584 | 5483.700 |
| Coverage-random | 50 | 0.610 | 0.513 | 0.676 | 0.432 | 0.519 | 0.000 | 0.000 | 0.000 | 2330.813 | 5464.800 |
| PrivateFair (RR, ε=1) | 50 | 0.648 | 0.421 | 0.645 | 0.409 | 0.428 | 44.250 | 144.000 | 132.750 | 2204.700 | 5540.400 |
| PrivateFair (non-private) | 50 | 0.668 | 0.438 | 0.618 | 0.485 | 0.443 | 44.250 | 0.000 | 0.000 | 2422.395 | 5370.200 |

Accuracies are per-site test **balanced accuracy**: `mean_acc`, `worst_site_acc` and `shifted_mean_acc` are at the final round; `*_last3` average the mean / worst-site accuracy over the last 3 evaluations (rounds 40, 45, 50). `reports_per_site` = telemetry reports (one per signal triple) a site released over the run, on average. ε is the composed budget per site (sum over all released reports and signals; 0 when nothing is privatized). Time is simulated seconds.
