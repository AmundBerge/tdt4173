"""Parameter sweep with the honest protocol (rounds chosen on 2021, score reported on 2022), baseline + margin features.

    py sweep_params.py --device cuda                                   # built-in one-at-a-time candidates
    py sweep_params.py --device cuda --params '{"max_depth": 4, "min_child_weight": 50}' --tag combo
Writes results/sweep_<tag>.csv (one row per candidate, sorted by 2022 AUC).
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from kaggle_metric import score
from train_xgb_baseline import load, FeatureBuilder, H, SEED, honest_fit
from features_margin import add_margin_features

BASE = dict(learning_rate=0.05, max_depth=6, min_child_weight=5, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8)
CANDIDATES = [
    {}, {"max_depth": 4}, {"max_depth": 8}, {"max_depth": 10},
    {"min_child_weight": 1}, {"min_child_weight": 20}, {"min_child_weight": 50},
    {"reg_lambda": 5.0}, {"reg_lambda": 20.0},
    {"colsample_bytree": 0.5}, {"subsample": 1.0},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="results")
    ap.add_argument("--params", default=None, help="JSON dict of param overrides, or @path to a JSON file holding a dict or list of dicts")
    ap.add_argument("--tag", default="params")
    args = ap.parse_args()
    Path(args.out).mkdir(exist_ok=True)

    uc, price, inflow, volume, wv = load("data/kernel")
    sample = pd.read_csv("sample_submission.csv", nrows=1)
    label_cols = [c for c in sample.columns if c != "Run No"]
    units = list(dict.fromkeys(re.match(r"result_committed_(.+)_t\d+$", c).group(1) for c in label_cols))
    n_units = len(units)
    X = add_margin_features(FeatureBuilder(price, inflow, volume, wv)(uc["starttime"]))
    Y = uc[label_cols].to_numpy(dtype=np.int8).reshape(len(uc), n_units, H).transpose(0, 2, 1).reshape(-1, n_units)
    if args.params and args.params.startswith("@"):          # @file.json: a dict or a list of dicts
        loaded = json.load(open(args.params[1:]))
        candidates = loaded if isinstance(loaded, list) else [loaded]
    elif args.params:
        candidates = [json.loads(args.params)]
    else:
        candidates = CANDIDATES
    print(f"X {X.shape}; {len(candidates)} candidates", flush=True)

    rows = []
    for i, change in enumerate(candidates):
        p = {**BASE, **change}
        t0 = time.time()
        res = honest_fit(X, Y, uc["starttime"], p, tune_year=2021, report_year=2022, device=args.device)
        rows.append({"change": str(change) if change else "base", **p, "n_rounds": res["n_rounds"],
                     "auc_2021_tune": res["tune_auc"], "auc_2022_report": res["report_auc"], "seconds": round(time.time() - t0)})
        print(f"[{i + 1}/{len(candidates)}] {rows[-1]['change']:50s} 2022={res['report_auc']:.5f} 2021={res['tune_auc']:.5f} rounds={res['n_rounds']} ({rows[-1]['seconds']}s)", flush=True)
        pd.DataFrame(rows).sort_values("auc_2022_report", ascending=False).to_csv(Path(args.out) / f"sweep_{args.tag}.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
