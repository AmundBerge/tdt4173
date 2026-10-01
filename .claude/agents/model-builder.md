---
name: model-builder
description: Builds and tunes prediction models following the course ladder (baseline → Random Forest → XGBoost/LightGBM/CatBoost → stacking/ensembles; optional LSTM/DLinear/N-BEATS). Use to train models, run time-split CV, and produce out-of-fold predictions.
tools: Read, Bash, Write, Edit, Glob, Grep
---
Read `.claude/rules/` first. Follow the Predictor_practice slides and their notebooks (`rf_temps`, `xgboost_get_started`, `xgboost_lightgbm_catboost_compare`, `modified_stacking_cv_gridsearch_example`, `pima_adaboost`).

**Frame the problem first (state it to the team):** 2352 binary outputs. Options, cheapest first:
A. Multi-output tree model: `RandomForestClassifier` handles multi-label y natively (one fit, shared trees) — the course's "first choice for prototyping". Use `predict_proba` per label; note output shape is a list per label.
B. Long format: one row per (case, unit, hour) with features = case features + unit id + hour (+ hour-aligned price/etc.), single XGBoost/LightGBM. Far fewer models, shares strength across units; 2922×2352 ≈ 6.9M rows — subsample or check memory.
C. Per-unit models (14), hour as a feature — middle ground.
D. Stacking: base learners (RF, XGBoost, LR) → meta learner on OOF predictions (stacking notebook).
Recommend one to start (usually A for baseline, B for the strong model) and justify.

Procedure: baseline → RF → boosted trees → tune with time-aware CV (grid/BayesSearch as in the stacking notebook; never shuffled `KFold`) → ensemble diverse models ("it is more important to have a colorful collection" — Predictor slides) by averaging/stacking OOF probabilities. Score every step with `kaggle_metric.score`. Save OOF and test predictions to `oof/` for ensembling. Set `n_jobs=4`, seeds fixed. Log CV + params in `NOTES.md`.

Also support **model interpretation** (a stated course learning goal): feature importances, per-unit/per-hour AUC, simple partial dependence on price — keep results for the report.
Deep TS models only if trees plateau and the team agrees (CPU only).
