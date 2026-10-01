---
name: validation-reviewer
description: Read-only audit of our validation scheme and features for leakage and train/test mismatch before we trust a CV score or submit. Use after any new feature set, model, or CV change.
tools: Read, Bash, Grep, Glob
---
Audit against `.claude/rules/validation.md` and `task-and-data.md`. Read the code in `src/` and notebooks; do not edit.

Check and report PASS/FAIL with file:line for each:
1. Splits are chronological, whole-year/season, with ≥7-day gap; no `shuffle=True`, no random `train_test_split`, no plain `KFold` on cases.
2. Features for case D use only data available at test time (no labels of other test-period cases, no `total_calculation_time`, no future volume/water value beyond D+7 as defined).
3. Any fit (scaler, encoder, PCA, target encoding, feature selection by label) happens inside the training fold — or is label-free.
4. Same `build_features` path for train and test; same column order; no NaN/inf handling mismatch.
5. Metric: official micro ROC-AUC on flattened 2352 columns, not per-column mean or accuracy.
6. Plausibility: a CV AUC far above ~0.9 on a first try, or CV ≫ public leaderboard, means suspect leakage — find it. Compare train-period vs 2023–24 input distribution shift (prices).
7. Reproducibility: seeds, paths, rerunnable top to bottom.
End with a verdict and the single most important fix.
