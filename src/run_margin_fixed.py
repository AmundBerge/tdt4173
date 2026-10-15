"""Margin-feature model, fixed number of rounds, no early stopping. Train < 2022, score 2022.

    py run_margin_fixed.py --lr 0.01 --rounds 3000 --device cuda
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from train_xgb_baseline import load, FeatureBuilder, H, SEED, split_masks
from features_margin import add_margin_features

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--rounds", type=int, default=3000)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
Path("results").mkdir(exist_ok=True)

uc, price, inflow, volume, wv = load("data/kernel")
sample = pd.read_csv("sample_submission.csv", nrows=1)
label_cols = [c for c in sample.columns if c != "Run No"]
units = list(dict.fromkeys(re.match(r"result_committed_(.+)_t\d+$", c).group(1) for c in label_cols))
X = add_margin_features(FeatureBuilder(price, inflow, volume, wv)(uc["starttime"]))
Y = uc[label_cols].to_numpy(dtype=np.int8).reshape(len(uc), len(units), H).transpose(0, 2, 1).reshape(-1, len(units))

tr, va = (np.repeat(m, H) for m in split_masks(uc["starttime"], 2022))
m = xgb.XGBClassifier(n_estimators=args.rounds, tree_method="hist", device=args.device, random_state=SEED,
                      learning_rate=args.lr, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=5)
t0 = time.time()
m.fit(X[tr], Y[tr])
secs = time.time() - t0
curve = {}
for r in sorted({100, 250, 500, 1000, 2000, 3000, 5000, 8000, 10000, args.rounds}):
    if r <= args.rounds:
        curve[r] = float(roc_auc_score(Y[va], m.predict_proba(X[va], iteration_range=(0, r)), average="micro"))
        print(f"rounds={r}: 2022 micro AUC {curve[r]:.4f}", flush=True)
final = curve[args.rounds]
print(f"lr={args.lr} final ({args.rounds} rounds, no early stopping): 2022 micro ROC AUC {final:.4f}  ({secs:.0f}s)")
json.dump(dict(lr=args.lr, rounds=args.rounds, report_2022_micro_auc=final, curve=curve, seconds=secs),
          open(f"results/margin_fixed_lr{args.lr}.json", "w"), indent=1)
