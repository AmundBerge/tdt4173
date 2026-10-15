# Experiment notes

## 2026-10-15 – XGBoost baseline (GPU run)

- Code: `train_xgb_baseline.py` (script twin of `xgboost_baseline.ipynb`), run on the RTX 2070 desktop with `--device cuda`.
- Data: one row per (case, hour), 85 features (calendar, price + week stats, hourly inflow, start volumes, start/end water values), 14 binary labels. Multi-label `XGBClassifier` = one booster per unit.
- Params: lr 0.05, depth 6, subsample 0.8, colsample 0.8, min_child_weight 5, early stopping 100 on 2022 hold-out.
- Result: hold-out 2022 micro ROC AUC **0.8623**, best iteration 1637. Per-unit AUC from 0.72 (Lio_G1) to 0.92 (Tokke_G2).
- Timing: fit 192 s + refit 197 s = 402 s total on GPU (~0.11 s/round vs ~0.46 s/round on the laptop CPU).
- Top features by gain: end-of-week water values (Songavatn, Kjelavatn, Vinjevatn, ...), then hourly price.
- Submission: `submissions/2026-10-15_xgb_baseline_cv0.8623.csv` (not yet uploaded to Kaggle).
- Next: price minus water value per plant, reservoir fill fraction from the topology yaml, min-flow/min-volume constraints, lower lr with more rounds.

## 2026-10-15 – price minus water value features (GPU run)

- Code: `xgboost_price_minus_wv.ipynb`. Baseline features + 27 margin features: for each of the 9 plants,
  `price − wv_start`, `price − wv_end`, `price_week_mean − wv_end` of the upstream reservoir
  (mapping from `Tokke_Vinje_topology.yaml` connections; Songa uses Songavatn).
- Same split and params as baseline (lr 0.05, depth 6, early stopping 100 on 2022).
- Result: hold-out 2022 micro ROC AUC **0.9792** (baseline 0.8623, +0.117). Every unit improved; Lio_G1 0.72 → 0.91,
  Kjela_G1 0.75 → 0.98, Byrte_G1 0.86 → 0.99. Best iteration 249 (was 1637): the signal is much more direct.
- Timing: fit 49 s + refit 38 s on GPU.
- Submission: `submissions/2026-10-15_xgb_margin_cv0.9792.csv` (not yet uploaded).
- Next: fill fraction, min-flow/min-volume constraints, rolling validation over 2020–2022 to confirm the gain.

## 2026-10-15 – one-at-a-time parameter sweep (GPU, margin features, 2022 hold-out)

- Code: `sweep_params.py`, results in `results/sweep_params.csv`. Base = lr 0.05, depth 6, mcw 5, lambda 1, subsample 0.8, colsample 0.8 (0.9792).
- All regularising changes help slightly, all within +0.0013: mcw 50 → 0.9805, depth 4 → 0.9805, lambda 20 → 0.9803, lambda 5 → 0.9800.
  Less regularisation hurts: mcw 1 → 0.9786. depth 8/10, colsample 0.5, subsample 1.0 ≈ no change.
- Takeaway: the model is slightly over-fitted; shallower / heavier-regularised trees are marginally better and faster. Gains are ~10× smaller than the margin-feature gain.
- Not yet tried: combining depth 4 + mcw 50 + lambda 20.

## 2026-10-15 – validation protocol fixed (after leakage audit), GPU re-runs

- Audit (read-only agent): no confirmed leak. End-of-week water value is a documented optimiser input, mapping and
  windows verified, test inputs complete. Issues: early stopping on the scored year (≈ +0.001 optimism), 7-fold
  overlapping weeks make in-sample AUC meaningless, price_prev/next wrapped within the case (fixed, edge padding).
- New protocol in `train_xgb_baseline.py` (`split_masks`, `honest_fit`): rounds chosen by early stopping on 2021
  (train < 2021), reported score from train < 2022 scored on 2022, one-week gap before each validation year.
  Final model: all years, 2021-chosen rounds. Notebook `xgboost_price_minus_wv.ipynb` and `sweep_params.py` use it.
- Baseline features under the new protocol: 2021 tune AUC 0.8926, rounds 411, **2022 report AUC 0.8403**
  (was 0.8623 with ES on 2022 and 1637 rounds; 2022 is a different price regime, so round choice on 2021 transfers poorly).
- Margin features under the new protocol: 2021 tune AUC 0.9748, rounds 222, **2022 report AUC 0.9786** (was 0.9792).
  Gain over baseline +0.138. Margin features are far less sensitive to the price-level shift between years.
- Submission: `submissions/2026-10-15_xgb_margin_honest_cv0.9786.csv` (222 rounds, all years). Not yet uploaded.

## 2026-10-15 – combined regularisers, honest two-stage protocol (GPU, margin features)

- Code: `sweep_params.py --params @combo.json --tag combo`, results in `results/sweep_combo.csv`. Rounds chosen by early stopping on 2021,
  then refit on all cases before 2022 and scored on 2022 (micro ROC AUC). Base = lr 0.05, depth 6, mcw 5, lambda 1, subsample 0.8, colsample 0.8.
- Note: under this protocol the base scores 0.9786, not the 0.9792 of the earlier sweep (which tuned rounds on 2022 itself).

| change                          | n_rounds | AUC 2021 (tune) | AUC 2022 (report) | s   |
|---------------------------------|---------:|----------------:|------------------:|----:|
| base                            |      222 |          0.9748 |            0.9786 | 185 |
| mcw 50 + lambda 20              |      423 |          0.9772 |        **0.9806** | 311 |
| depth 4 + mcw 50 + lambda 20    |      648 |          0.9759 |            0.9801 | 313 |
| depth 4 + lambda 20             |      624 |          0.9753 |            0.9797 | 203 |
| depth 4 + mcw 50                |      533 |          0.9758 |            0.9796 | 144 |

- Conclusion: combining regularisers helps a little; every combination beats base on both years, and the best is mcw 50 + lambda 20
  at depth 6 (+0.0020 on 2022, +0.0024 on 2021). Adding depth 4 on top does not help (0.9801 vs 0.9806). Gains stay ~10× smaller than the margin-feature gain.

## 2026-10-15 – tuned notebook (GPU, honest protocol)

- Code: `xgboost_tuned.ipynb` = margin features + max_depth 4, min_child_weight 50, reg_lambda 20 (lr 0.05).
- Rounds chosen on 2021: 648 (default trees: 222); 2021 tune AUC 0.9759; **2022 report AUC 0.9801** (default trees 0.9786, +0.0015).
- Per unit: changes within ±0.005, no unit moves materially; the gain is spread thinly.
- Fit 333 s + refit 131 s on GPU. Submission: `submissions/2026-10-15_xgb_tuned_cv0.9801.csv` (not yet uploaded).

## 2026-10-15 – tuned notebook re-run with the combo-sweep winner (depth 6, mcw 50, lambda 20)

- `xgboost_tuned.ipynb` now uses max_depth 6, min_child_weight 50, reg_lambda 20 (combo sweep best, 0.9806 on 2022).
- Submission: `submissions/2026-10-15_xgb_tuned_d6_cv*.csv`. The earlier depth-4 variant (cv 0.9801) scored slightly
  *below* the untuned margin model on the Kaggle public leaderboard: parameter gains of ~0.002 on one hold-out year do not transfer.
- Decision: treat tuned and untuned margin models as equivalent; further tuning only with multi-year (2020/2021/2022) validation.

## 2026-10-15 – error analysis of the margin model on 2022 (`error_analysis_2022.ipynb`)

- Lio_G1 (AUC 0.891) + Haukeli_G1 (0.919) carry 32% of squared error; both over-predicted (mean pred 0.61 vs 0.55, 0.23 vs 0.14).
- Hours within 2 h of a start/stop: 12% of hours, 33% of error; first hour of a run Brier 0.20. Short runs (≤5 h): 4% of hours, 12% of error.
- Smoothing predictions over neighbouring hours hurts (3 h: −0.0009, 9 h: −0.007). Not a post-processing problem.
- Margin < −40 bin: 51% of hours, 7% ON, 42% of error → forced running (min flow / spill / full reservoir). Worst months April (0.941), November (0.949).
- Lio min-flow constraint is >0 all year (5/8/14 m³/s): need level + volume/inflow context, not a flag.
- Cross-unit calibration: isotonic on 2021 → 0.9772 (worse); in-sample upper bound 0.9824. Dropped as a lever.
- Next: forced-operation features (fill fraction, inflow vs capacity, min-flow level, distance to min volume) + local margin context (rolling ±3/6/12 h, positive-margin hours per day).

## 2026-10-15 – constraint features (`xgboost_constraints.ipynb`, `features_constraints.py`, GPU, honest protocol)

- +21 features from data/extended: min flow per river at hour t and weekly max (6 rivers); min volume, start-volume slack
  and weekly worst slack for Totak, Byrtevatn, Staavatn. 133 features total.
- Rounds on 2021: 229; 2021 AUC 0.9766; **2022 report AUC 0.9792** (margin only 0.9786, +0.0006).
- Lio_G1 0.891 → 0.912 (+0.021), Songa_G1 +0.005; Haukeli_G1 −0.004; all others within ±0.002.
- In top-25 gain: minflow_b_Vinjevatn, minvol_Totak, minflow_r_Vest_Vassdraget.
- Lio ON share by min-flow level: 5 m³/s 0.76, 8 → 0.66, 14 → 0.85 (not monotone; level is a season proxy as much as a constraint).
- Submission: `submissions/2026-10-15_xgb_constraints_cv0.9792.csv`. Haukeli (tiny Vatjern reservoir) still unexplained → fill fraction / inflow-vs-capacity next.

## 2026-10-15 – margin features, lr 0.01, no early stopping (GPU)

- Code: `run_margin_fixed.py` (3000 fixed rounds, train < 2022, score 2022). Params as `xgboost_price_minus_wv.ipynb` except lr 0.01.
- Result: 2022 micro ROC AUC **0.9788** at 3000 rounds; plateau from ~1000 rounds (0.9788 -> 0.9789). Compare lr 0.05 honest: 0.9792.
- lr 0.001 with early stopping (patience 100) stopped at round 35 (AUC 0.635): an early-stopping artifact, not a real result. Not rerun.
- Timing: 372 s on GPU. Lower lr gave no gain.

## 2026-10-15 – temporal context + cascade water values (`xgboost_context.ipynb`, `features_context.py`, GPU, honest protocol)

- Built on the constraints model (0.9792). Context: per plant, centred rolling mean/max/min of the margin over 3/6/12 h,
  positive-margin hours per day, rank in day, run lengths of positive/negative margin back and ahead (135 feats).
  Cascade: downstream reservoir wv (start/end), upstream−downstream wv difference, price − difference (28 feats). 296 total.
- Ablation (rounds on 2021 → score 2022): context only **0.9806** (+0.0014, 198 rounds); cascade only 0.9783 (−0.0009);
  both 0.9805 (+0.0013, 269 rounds). Context features carry 72% of total gain; cascade 3%.
- Per unit: Kjela_G1 +0.006, Tokke_G3/G4 +0.004, Vesle Kjela +0.003; Lio −0.002. Transition error share 0.33 → 0.37 of a
  smaller total (Brier 0.0546 → 0.0528): context helped more away from transitions than at them.
- Tokke_G1 ON share vs hours of positive margin behind/ahead: 0.38 at 0 h, 0.84 at 1 h, 0.96 at >24 h → strong, simple signal.
- Submission (both groups): `submissions/2026-10-15_xgb_context_cv0.9805.csv`. Decision: drop cascade, keep context.

## 2026-10-15 – margin features, lr 0.001, 12000 rounds, no early stopping (GPU)

- Code: `run_margin_fixed.py --lr 0.001 --rounds 12000`. Train < 2022, score 2022; other params as the notebook.
- Result: 2022 micro ROC AUC **0.9792** at 12000 rounds (same as lr 0.05 honest 0.9792, lr 0.01 0.9788). Still creeping up slowly: 5000 -> 0.9778, 8000 -> 0.9789, 10000 -> 0.9791.
- Timing: 1592 s on GPU. No gain over lr 0.05 / 249 rounds (49 s): the learning rate is not the bottleneck.

## 2026-10-15 – stacked stage-2 model (`notebooks/07_xgboost_stacked.ipynb`, GPU, honest protocol)

- Stage 1 = context model (268 feats, 198 rounds). Forward-chained OOF: stage-1 for year y trained on years < y
  (2016: 0.951, 2017: 0.978, 2018: 0.969, 2019: 0.981, **2020: 0.922**, 2021: 0.979, 2022: 0.981).
- Stage 2 = same features + 212 stacked: each unit's stage-1 prob at lags −6..+6, ±3 h / ±12 h means, units-on sums for Tokke/Vinje.
- Rounds on 2021: 185 (2021 AUC 0.9808); **2022 report AUC 0.9781** (stage 1 alone 0.9806, −0.0025). 12/14 units worse, Lio −0.024.
  Transition error share unchanged (0.38). Stacked features = 54% of stage-2 gain → over-reliance on stage 1 that does not transfer.
- Decision: drop stacking. Submission kept for the record: `submissions/2026-10-15_xgb_stacked_cv0.9781.csv`.
- Side finding: 2020 is a hard year (0.922) → do rolling validation over 2020/2021/2022 before trusting single-year gains.
