"""Margin-feature model (xgboost_price_minus_wv.ipynb) at a custom learning rate, honest protocol.

    py run_margin_lr.py --lr 0.001 --device cuda
Writes results/margin_lr<lr>.json
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np, pandas as pd
from train_xgb_baseline import load, FeatureBuilder, H, honest_fit
from features_margin import add_margin_features

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=0.001)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-rounds", type=int, default=40000)
args = ap.parse_args()
Path("results").mkdir(exist_ok=True)

uc, price, inflow, volume, wv = load("data/kernel")
sample = pd.read_csv("sample_submission.csv", nrows=1)
label_cols = [c for c in sample.columns if c != "Run No"]
units = list(dict.fromkeys(re.match(r"result_committed_(.+)_t\d+$", c).group(1) for c in label_cols))
X = add_margin_features(FeatureBuilder(price, inflow, volume, wv)(uc["starttime"]))
Y = uc[label_cols].to_numpy(dtype=np.int8).reshape(len(uc), len(units), H).transpose(0, 2, 1).reshape(-1, len(units))

params = dict(learning_rate=args.lr, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=5)
t0 = time.time()
res = honest_fit(X, Y, uc["starttime"], params, tune_year=2021, report_year=2022, device=args.device, max_rounds=args.max_rounds)
secs = time.time() - t0
print(f"lr={args.lr} rounds chosen on 2021: {res['n_rounds']} (2021 AUC {res['tune_auc']:.4f})")
print(f"2022 report micro ROC AUC: {res['report_auc']:.4f}  ({secs:.0f}s)")
json.dump(dict(lr=args.lr, n_rounds=res["n_rounds"], tune_2021_auc=res["tune_auc"], report_2022_micro_auc=res["report_auc"],
               per_unit_auc=dict(zip(units, res["per_unit"])), seconds=secs, max_rounds=args.max_rounds),
          open(f"results/margin_lr{args.lr}.json", "w"), indent=1)
