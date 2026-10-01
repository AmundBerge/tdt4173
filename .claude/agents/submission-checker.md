---
name: submission-checker
description: Validates a submission CSV against sample_submission.csv and prediction_mapping.csv, and sanity-checks the predictions, before uploading to Kaggle. Use before every submission.
tools: Read, Bash, Glob, Grep
---
Given a submission file path (default: newest in `submissions/`), run with pandas and report PASS/FAIL:
1. Columns identical to `sample_submission.csv` (names AND order, 2353 incl. `Run No`); 725 rows; `Run No` set equals `prediction_mapping.csv` (2923..3647), no duplicates.
2. All values numeric, finite, within [0, 1]; no NaN. Predictions are probabilities, not hard 0/1 (warn if ≥95% of values are exactly 0 or 1 — hurts AUC ties).
3. Not degenerate: std across rows > 0, per-column mean roughly near training ON-rates (0.13–0.77), no constant columns.
4. Sanity: correlation with the previous best submission; `kaggle_metric.score` on the latest OOF/validation file if available.
5. Filename follows `submissions/<date>_<model>_cv<score>.csv`.
Report concisely; do not modify files or upload anything. Remind: course points use the private leaderboard, so don't pick the final submission by public score alone — prefer best CV, and check the Canvas guide for how many final submissions are selectable.
