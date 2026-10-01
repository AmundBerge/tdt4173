# Code and notebooks (hint: "✗ messy notebooks and dirty code")

- Layout: `notebooks/` numbered (`01_eda.ipynb`, `02_features.ipynb`, …), reusable code in `src/` (`data.py`, `features.py`, `cv.py`, `models.py`), outputs in `submissions/` and `oof/`. Don't commit large data/model files; keep `data/` as given.
- Notebooks: top-to-bottom reproducible, markdown headings per step, no dead cells, set seeds (`random_state=42`).
- Put anything used twice in `src/`; notebooks import it. Feature building must be one function used identically for train and test.
- Vectorize with pandas/numpy; the target has 2352 columns — avoid per-column Python loops where a multi-output model or reshaping to long format (case, unit, hour) works. Think about *shape of the problem* before coding (see `agents/model-builder.md`).
- Respect 4 CPU / 32 GB: no giant dense 2922×2352×N feature blow-ups; check memory.
- Save every submission as `submissions/<date>_<model>_cv<score>.csv` with the exact `sample_submission.csv` columns and row order; run `submission-checker` before uploading.
- Commits: small, descriptive. Never commit secrets or Kaggle tokens.
