"""Environmental-constraint features from data/extended (see xgboost_constraints.ipynb).

Hourly minimum river flows (6 rivers) and minimum reservoir volumes (3 reservoirs), plus the distance between the
reservoir's volume at case start and its minimum. All are inputs the optimiser receives for the case week.
"""
from pathlib import Path
import numpy as np
import pandas as pd

H = 168


def load_constraints(data_dir="data/extended"):
    data_dir = Path(data_dir)
    minflow = pd.read_csv(data_dir / "Constraint_min_flow.csv", parse_dates=["date"]).set_index("date").resample("h").mean()
    minvol = pd.read_csv(data_dir / "Constraint_min_volume.csv", parse_dates=["date"]).set_index("date").resample("h").mean()
    return minflow, minvol


class ConstraintFeatures:
    def __init__(self, minflow, minvol, volume):
        self.minflow, self.minvol, self.volume = minflow, minvol, volume

    @staticmethod
    def window(hourly_df, starts):
        idx = hourly_df.index.get_indexer(starts)
        assert (idx >= 0).all(), "case start missing from constraint series"
        return hourly_df.to_numpy(dtype=np.float32)[idx[:, None] + np.arange(H)[None, :]]   # (n, H, cols)

    def __call__(self, starts):
        starts = pd.DatetimeIndex(starts)
        if starts.tz is None:
            starts = starts.tz_localize("UTC")
        n = len(starts)
        rep = lambda a: np.repeat(a, H)
        MF = self.window(self.minflow, starts)
        MV = self.window(self.minvol, starts)
        V = self.volume.reindex(starts)

        f = {}
        for j, river in enumerate(self.minflow.columns):
            f[f"minflow_{river}"] = MF[:, :, j].ravel()                 # requirement at hour t
            f[f"minflow_{river}_wkmax"] = rep(MF[:, :, j].max(1))       # tightest requirement during the week
        for j, res in enumerate(self.minvol.columns):
            v0 = V[res].to_numpy(dtype=np.float32)
            f[f"minvol_{res}"] = MV[:, :, j].ravel()
            f[f"volgap_{res}"] = rep(v0) - MV[:, :, j].ravel()           # volume at case start minus minimum at hour t
            f[f"volgap_{res}_wkmin"] = rep(v0 - MV[:, :, j].max(1))      # worst-case slack during the week
        return pd.DataFrame(f)


def add_constraint_features(f, starts, constraint_features):
    extra = constraint_features(starts)
    assert len(extra) == len(f)
    return pd.concat([f.reset_index(drop=True), extra], axis=1)
