# Validation and leakage (project hint: "Time-aware validation")

- **Never random-shuffle** cases for validation — adjacent daily cases overlap by 6/7 of their window (rolling horizon) and shuffling leaks. Course source: Time_Series slides ("avoid random shuffle splitting; test set should cover relevant seasons") and Project Intro hints.
- Mirror the real split: **train 2015–2022 → validate on 2023** (matches the public leaderboard period). Better: expanding-window folds by year (train ≤ Y−1, validate Y) using `sklearn.model_selection.TimeSeriesSplit` or manual year folds. Keep a gap of ≥ 7 days between train end and validation start so overlapping windows don't leak.
- Seasonality: validate on whole years so all seasons appear.
- Score with the official `kaggle_metric.score` (micro ROC-AUC over all 2352 columns). Don't substitute per-column AUC as the headline number; it is a diagnostic only.
- Feature leakage checklist for case D: use only information the test set also has (prices/inflow/min-constraints for the window, volume & water value at D and D+7, calendar). Never use another case's **labels** from the test period, never features derived from `Unit_commitment_decisions` of future dates, never `total_calculation_time` (not available at test time).
- Fit scalers/encoders/target stats on training folds only. (Unsupervised PCA on train+test features is the one thing the Feature_Engineering slides show as acceptable — only for label-free features.)
- Public leaderboard is only an estimate; trust the local time-split CV, don't overfit to public score.
- Tune with CV (course: grid search / `skopt` BayesSearch from the stacking notebook), then refit on all training data (Supervised_Learning slides).
