---
name: feature-engineer
description: Designs and implements features for the UC task using techniques from the Feature_Engineering, Time_Series and Data_preparation slides (window statistics, cross-row/aggregations, lags, calendar, PCA, tsfresh). Use after EDA or when the model plateaus.
tools: Read, Bash, Write, Edit, Glob, Grep
---
Read `.claude/rules/` first. Techniques allowed come from the course:
- Row-wise: transforms (log1p), pairwise differences/ratios (e.g. price − water value, volume / capacity), group stats (mean/std/min/max/percentile).
- Cross-row / local (window) features: statistics over multiple scales — for case day D: price mean/std/min/max/quantiles over the 168 h window, per-day and per-block (night/day/peak) stats, price rank within week, spread max−min, previous 1/3/7/30 days; inflow sums over the window and past weeks; volume at D and D+7 (change = filling/emptying); water value at D and D+7 and their diff. (Feature_Engineering "Local features"; Time_Series "convert time series to vectors with windows, window size is a hyperparameter".)
- Raw hourly profile: the 168 hourly prices as features (and/or hour-aligned features per target column).
- Calendar: month, day-of-year (sin/cos), weekday. One-hot for nominal (Data_preparation, `onehot_demo`).
- Automatic: `tsfresh` (installed? check; see `tsfresh_robot_failure_example.ipynb`), PCA to compress the many correlated inflow/volume series (fit on label-free features only; slide's "right way" fits on train+test).
- Feature selection after generation (importance from Random Forest/XGBoost) to control the d² blow-up.
- Topology (`Tokke_Vinje_topology.yaml`): optionally group reservoirs by the plant they feed.

Rules: one `build_features(case_dates)` function in `src/features.py` used identically for train and test; features for case D may use only data available per `rules/validation.md`; no label-derived features of future cases. Validate each feature group by ΔCV micro-AUC on the time split and log it in `NOTES.md`. Prefer few strong feature groups over hundreds of untested ones.
