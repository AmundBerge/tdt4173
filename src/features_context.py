"""Temporal-context and cascade features on top of the margin features (see xgboost_context.ipynb).

Context (per plant, from margin_end_<plant> = price - end-of-week water value of the upstream reservoir):
  centred rolling mean / max / min over 3, 6 and 12 hours, number of positive-margin hours in the same day,
  rank of the margin within the day, and run lengths: consecutive positive (negative) margin hours up to t and from t on.
Cascade (per plant with a downstream reservoir): the downstream reservoir's water value at case start and end,
  the upstream-minus-downstream difference, and price minus that difference ("cascade margin").
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d

from features_margin import UPSTREAM

H = 168
# plant -> reservoir the plant discharges into (plant -> reservoir connections in Tokke_Vinje_topology.yaml)
DOWNSTREAM = {"Tokke": "Bandak", "Vinje": "Vinjevatn", "Haukeli": "Vinjevatn", "Byrte": "Byrtevatn",
              "Kjela": "Hyljelihyl", "Vesle Kjela": "Foersvatn", "Songa": "Totak"}   # Lio -> river, Hogga -> outlet
WINDOWS = (3, 6, 12)


def _run_lengths(pos):
    """pos: bool (n, H). Returns consecutive-True counts ending at t and starting at t."""
    n = pos.shape[0]
    back = np.zeros((n, H), dtype=np.float32); ahead = np.zeros((n, H), dtype=np.float32)
    for t in range(H):
        back[:, t] = np.where(pos[:, t], (back[:, t - 1] if t else 0) + 1, 0)
    for t in range(H - 1, -1, -1):
        ahead[:, t] = np.where(pos[:, t], (ahead[:, t + 1] if t < H - 1 else 0) + 1, 0)
    return back, ahead


def add_context_features(f):
    n = len(f) // H
    out = {}
    for plant in UPSTREAM:
        M = f[f"margin_end_{plant}"].to_numpy(dtype=np.float32).reshape(n, H)
        for w in WINDOWS:
            out[f"ctx_mean{w}_{plant}"] = uniform_filter1d(M, size=w, axis=1, mode="nearest").ravel()
            out[f"ctx_max{w}_{plant}"] = maximum_filter1d(M, size=w, axis=1, mode="nearest").ravel()
            out[f"ctx_min{w}_{plant}"] = minimum_filter1d(M, size=w, axis=1, mode="nearest").ravel()
        day = M.reshape(n, 7, 24)
        out[f"ctx_poshours_day_{plant}"] = np.repeat((day > 0).sum(2), 24, axis=1).ravel()
        out[f"ctx_rank_day_{plant}"] = (day.argsort(2).argsort(2) / 23.0).reshape(n, H).ravel()
        pos_back, pos_ahead = _run_lengths(M > 0)
        neg_back, neg_ahead = _run_lengths(M <= 0)
        out[f"ctx_posrun_back_{plant}"] = pos_back.ravel(); out[f"ctx_posrun_ahead_{plant}"] = pos_ahead.ravel()
        out[f"ctx_negrun_back_{plant}"] = neg_back.ravel(); out[f"ctx_negrun_ahead_{plant}"] = neg_ahead.ravel()
    return pd.concat([f.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def add_cascade_features(f):
    out = {}
    for plant, down in DOWNSTREAM.items():
        up = UPSTREAM[plant]
        out[f"casc_wv_down_{plant}"] = f[f"wv_{down}"].to_numpy()
        out[f"casc_wv_down_end_{plant}"] = f[f"wv_end_{down}"].to_numpy()
        diff_end = f[f"wv_end_{up}"].to_numpy() - f[f"wv_end_{down}"].to_numpy()
        out[f"casc_wvdiff_end_{plant}"] = diff_end
        out[f"casc_margin_{plant}"] = f["price"].to_numpy() - diff_end
    return pd.concat([f.reset_index(drop=True), pd.DataFrame(out)], axis=1)
