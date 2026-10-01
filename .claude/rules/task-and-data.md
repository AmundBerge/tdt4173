# Task and data facts (from ML_Task_TDT4173.pdf, Dataset_definitions_and_explanation.pdf, kaggle_metric.py)

**Problem:** predict hourly ON/OFF (probability in [0,1]) of every generating unit for a 168-hour (one-week) rolling-horizon case. System: Tokke–Vinje, 17 reservoirs, 9 plants, 14 units.

**Targets:** `data/kernel/Unit_commitment_decisions.csv` — 2922 training cases (`Run No` 1..2922, `starttime` 2015-01-01..2022-12-31, plus `total_calculation_time`), 2352 binary columns `result_committed_<Plant>_<Unit>_t<0..167>` (14 units × 168 h). No constant columns; mean ON-rate per column ≈ 0.52 (range 0.13–0.77).

**Test:** `prediction_mapping.csv` — 725 cases, `Run No` 2923..3647, one case per day 2023-01-01..2024-12-25. `sample_submission.csv` = `Run No` + the same 2352 columns, in that order. Public leaderboard = 2023-01-01..2024-01-01, private = 2024-01-01..2024-12-25 (**course base points use private**). Train split per slides: 2015-01-01..2023-01-01.

**Metric:** `kaggle_metric.py` — micro-average ROC-AUC over the whole (cases × 2352) matrix flattened. Consequences: only the *ranking* of all predictions together matters; calibration per column doesn't, but cross-column scale does (a unit with a high base rate vs. low). Output probabilities, never hard 0/1. Always evaluate locally with `kaggle_metric.score`.

**Input data (`data/kernel/`):**
- `Historical_day_ahead_price_2015_2025.csv` — hourly UTC, EUR/MWh. (Last timestamp ends `:45` → **check the time resolution** before assuming hourly.)
- `Historical_inflow_1958_2025.csv` — hourly, m³/s, 17 reservoirs + `r_Vest_Vassdraget` river.
- `Historical_volume_2015_2024.csv` — **daily** reservoir storage at start of day, Mm³ (Hyljelihyl, Vatjern are a fake constant 50%).
- `Synthetic_water_value_2015_2024.csv` — daily marginal water value per reservoir, EUR/MWh. For a case starting day D: initial volume = value at D 00:00; final water value = value at D+7 00:00.
- `Constraint_min_volume.csv` (3 reservoirs), `Constraint_min_flow.csv` (6 rivers) — hourly, 0 = no restriction.
- `data/extended/` — `Tokke_Vinje_topology.yaml` (SHOP format), `Tokke_topology.pdf`. Optional context for features (e.g. which reservoir feeds which plant).

**Case window for case starting day D:** hours D 00:00 … D+6 23:00 (t0..t167). Inputs for that window (prices, inflows, min-constraints) are legitimately known for the test period — the test set is features-only, labels hidden.

**Grading context:** better private score → more Virtual Teams beaten → higher grade. Report + clean code also graded (read the Canvas submission guide; ask the team for it if details are needed).
