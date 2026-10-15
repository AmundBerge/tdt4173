"""XGBoost baseline for the SINTEF unit-commitment task.

Script version of xgboost_baseline.ipynb so it can run headless on the GPU desktop.

Validation protocol: rounds are chosen by early stopping on 2021 (training on cases before 2021);
the reported score is then from a model trained on cases before 2022 and scored on 2022, so the
reported year never influences a modelling decision. A one-week gap keeps overlapping cases apart.

    py train_xgb_baseline.py --device cuda          # GPU
    python train_xgb_baseline.py --device cpu       # laptop

Outputs (in --out, default ./results):
    submission_xgb_baseline.csv   Kaggle submission (test cases 2023-2024)
    val_pred_2022.npy             hold-out probabilities, shape (365*168, 14)
    metrics.json                  hold-out micro AUC, per-unit AUC, best iteration, timings
    feature_importance.csv        gain per feature (summed over the 14 boosters)
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from kaggle_metric import score

H = 168
SEED = 42


def load(data_dir):
    data_dir = Path(data_dir)
    uc = pd.read_csv(
        data_dir / "Unit_commitment_decisions.csv", parse_dates=["starttime"]
    )
    price = (
        pd.read_csv(
            data_dir / "Historical_day_ahead_price_2015_2025.csv", parse_dates=["date"]
        )
        .set_index("date")["Day-ahead price (EUR/MWh)"]
        .resample("h")
        .mean()
    )
    inflow = pd.read_csv(
        data_dir / "Historical_inflow_1958_2025.csv", parse_dates=["date"]
    ).set_index("date")
    inflow = inflow[inflow.index >= "2014-12-01"].resample("h").mean()
    volume = pd.read_csv(
        data_dir / "Historical_volume_2015_2024.csv", parse_dates=["date"]
    ).set_index("date")
    water_value = pd.read_csv(
        data_dir / "Synthetic_water_value_2015_2024.csv", parse_dates=["date"]
    ).set_index("date")
    return uc, price, inflow, volume, water_value


class FeatureBuilder:
    def __init__(self, price, inflow, volume, water_value):
        self.price, self.inflow, self.volume, self.wv = (
            price,
            inflow,
            volume,
            water_value,
        )

    @staticmethod
    def window(hourly_df, starts, horizon=H):
        idx = hourly_df.index.get_indexer(starts)
        assert (idx >= 0).all(), (
            "some case start times are missing from the hourly series"
        )
        pos = idx[:, None] + np.arange(horizon)[None, :]
        return hourly_df.to_numpy(dtype=np.float32)[pos]

    def __call__(self, starts):
        starts = pd.DatetimeIndex(starts)
        if starts.tz is None:
            starts = starts.tz_localize("UTC")
        n = len(starts)
        P = self.window(self.price.to_frame(), starts)[:, :, 0]
        I = self.window(self.inflow, starts)
        V = self.volume.reindex(starts).to_numpy(dtype=np.float32)
        W = self.wv.reindex(starts).to_numpy(dtype=np.float32)
        W_end = self.wv.reindex(starts + pd.Timedelta(days=7)).to_numpy(
            dtype=np.float32
        )
        hours = pd.DatetimeIndex(
            (
                starts.values[:, None] + np.arange(H)[None, :] * np.timedelta64(1, "h")
            ).ravel()
        )
        rep = lambda a: np.repeat(a, H)

        f = pd.DataFrame(
            {
                "t": np.tile(np.arange(H), n),
                "hour": hours.hour,
                "dow": hours.dayofweek,
                "month": hours.month,
                "doy": hours.dayofyear,
                "price": P.ravel(),
                "price_week_mean": rep(P.mean(1)),
                "price_week_std": rep(P.std(1)),
                "price_week_min": rep(P.min(1)),
                "price_week_max": rep(P.max(1)),
            }
        )
        f["price_rel"] = f["price"] - f["price_week_mean"]
        f["price_rank"] = (P.argsort(1).argsort(1) / (H - 1)).ravel()
        f["price_day_mean"] = np.repeat(P.reshape(n, 7, 24).mean(2), 24, axis=1).ravel()
        f["price_prev"] = np.concatenate([P[:, :1], P[:, :-1]], axis=1).ravel()   # edge-padded, no wrap
        f["price_next"] = np.concatenate([P[:, 1:], P[:, -1:]], axis=1).ravel()
        for j, c in enumerate(self.inflow.columns):
            f[f"inflow_{c}"] = I[:, :, j].ravel()
        f["inflow_total"] = I.sum(2).ravel()
        for j, c in enumerate(self.volume.columns):
            f[f"vol_{c}"] = rep(V[:, j])
        for j, c in enumerate(self.wv.columns):
            f[f"wv_{c}"] = rep(W[:, j])
        for j, c in enumerate(self.wv.columns):
            f[f"wv_end_{c}"] = rep(W_end[:, j])
        return f


def split_masks(starts, val_year, gap_days=7):
    """Case-level masks. train: cases whose whole week ends before val_year (so no hour is shared with the
    validation cases); val: cases starting in val_year. Returned per case, use np.repeat(mask, H) for rows."""
    starts = pd.DatetimeIndex(starts)
    val_start = pd.Timestamp(f"{val_year}-01-01")
    train = np.asarray(starts < val_start - pd.Timedelta(days=gap_days - 1))
    val = np.asarray(starts.year == val_year)
    return train, val


def honest_fit(X, Y, starts, params, tune_year=2021, report_year=2022, device="cuda", verbose=False, max_rounds=3000):
    """Two-stage validation so the reported year is never used to make a modelling decision.

    stage 1: train on cases before tune_year, early-stop on tune_year  -> number of rounds
    stage 2: train on cases before report_year with that many rounds   -> score on report_year
    Returns a dict with n_rounds, tune_auc, report_auc, per-unit AUC, the stage-2 model and its predictions.
    """
    common = dict(tree_method="hist", device=device, random_state=SEED, **params)
    tr1, va1 = (np.repeat(m, H) for m in split_masks(starts, tune_year))
    m1 = xgb.XGBClassifier(n_estimators=max_rounds, eval_metric="auc", early_stopping_rounds=100, **common)
    m1.fit(X[tr1], Y[tr1], eval_set=[(X[va1], Y[va1])], verbose=verbose)
    n_rounds = int(m1.best_iteration) + 1
    tune_auc = roc_auc_score(Y[va1], m1.predict_proba(X[va1]), average="micro")

    tr2, va2 = (np.repeat(m, H) for m in split_masks(starts, report_year))
    m2 = xgb.XGBClassifier(n_estimators=n_rounds, **common).fit(X[tr2], Y[tr2])
    pred = m2.predict_proba(X[va2])
    report_auc = roc_auc_score(Y[va2], pred, average="micro")   # identical to kaggle_metric.score
    per_unit = [roc_auc_score(Y[va2][:, j], pred[:, j]) for j in range(Y.shape[1])]
    return dict(n_rounds=n_rounds, tune_auc=float(tune_auc), report_auc=float(report_auc),
                per_unit=per_unit, model=m2, pred=pred, val_mask=va2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="folder with kaggle_metric.py, prediction_mapping.csv, data/kernel",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    root = Path(args.root)
    out = root / args.out
    out.mkdir(exist_ok=True)
    t_start = time.time()

    uc, price, inflow, volume, water_value = load(root / "data" / "kernel")
    test_map = pd.read_csv(root / "prediction_mapping.csv", parse_dates=["starttime"])
    sample = pd.read_csv(root / "sample_submission.csv", nrows=1)
    label_cols = [c for c in sample.columns if c != "Run No"]
    assert label_cols == [c for c in uc.columns if c.startswith("result_committed_")]
    units = list(
        dict.fromkeys(
            re.match(r"result_committed_(.+)_t\d+$", c).group(1) for c in label_cols
        )
    )
    n_units = len(units)

    def to_wide(pred_long, run_ids):
        wide = (
            pred_long.reshape(len(run_ids), H, n_units)
            .transpose(0, 2, 1)
            .reshape(len(run_ids), -1)
        )
        df = pd.DataFrame(wide, columns=label_cols)
        df.insert(0, "Run No", np.asarray(run_ids))
        return df

    build_features = FeatureBuilder(price, inflow, volume, water_value)
    X = build_features(uc["starttime"])
    Y = (
        uc[label_cols]
        .to_numpy(dtype=np.int8)
        .reshape(len(uc), n_units, H)
        .transpose(0, 2, 1)
        .reshape(-1, n_units)
    )
    years = np.repeat(uc["starttime"].dt.year.to_numpy(), H)
    print(
        f"features {X.shape}, targets {Y.shape}, built in {time.time() - t_start:.1f}s",
        flush=True,
    )

    params = dict(
        learning_rate=args.lr, max_depth=args.max_depth,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    )

    t0 = time.time()
    res = honest_fit(X, Y, uc["starttime"], params, tune_year=2021, report_year=2022, device=args.device,
                     verbose=100, max_rounds=args.n_estimators)
    fit_time = time.time() - t0
    micro_auc = res["report_auc"]
    per_unit = {u: float(a) for u, a in zip(units, res["per_unit"])}
    print(f"rounds chosen on 2021: {res['n_rounds']} (2021 AUC {res['tune_auc']:.4f})  "
          f"2022 report micro AUC={micro_auc:.4f}  ({fit_time:.0f}s)", flush=True)
    for u, a in sorted(per_unit.items(), key=lambda kv: kv[1]):
        print(f"  {u:16s} {a:.3f}")
    np.save(out / "val_pred_2022.npy", res["pred"].astype(np.float32))
    imp = pd.Series(res["model"].get_booster().get_score(importance_type="gain"), name="gain").sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv")

    # refit on everything with the number of rounds chosen on 2021
    t0 = time.time()
    final = xgb.XGBClassifier(n_estimators=res["n_rounds"], tree_method="hist", device=args.device,
                              random_state=SEED, **params).fit(X, Y)
    refit_time = time.time() - t0
    submission = to_wide(
        final.predict_proba(build_features(test_map["starttime"])), test_map["Run No"]
    )
    assert list(submission.columns) == list(sample.columns)
    vals = submission[label_cols].to_numpy()
    assert ((vals >= 0) & (vals <= 1)).all()
    submission.to_csv(out / "submission_xgb_baseline.csv", index=False)

    metrics = {
        "report_2022_micro_auc": float(micro_auc),
        "per_unit_auc": per_unit,
        "n_rounds": res["n_rounds"], "tune_2021_auc": res["tune_auc"],
        "params": {k: v for k, v in params.items()},
        "fit_seconds": fit_time,
        "refit_seconds": refit_time,
        "total_seconds": time.time() - t_start,
        "n_features": int(X.shape[1]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
