# Ground everything in the course

Every method, library and design choice must trace to `learningmaterials/`. If it doesn't, say so and ask before using it.

## Where things are
| Topic | Slides | Demo notebooks (`learningmaterials/...`) |
|---|---|---|
| Project, hints, splits, grading | `TDT4173_Course_Project_Introduction.pdf` | — |
| Supervised learning, over/underfitting, validation, trees, LR, NB | `Supervised_Learning_slides.pdf` | — |
| Data prep, one-hot, pandas, tsfresh, TS2Vec | `Data_prepration_slides.pdf` | `data_preparation_related materials/` (`onehot_demo`, `pandas_demo`, `tsfresh_robot_failure_example`) |
| Feature engineering (row-wise, cross-row, local/window features, PCA, CCA, KNN features, selection) | `Feature_Engineering_slides.pdf` | — |
| Time series (windows, ARIMA baseline, RNN/LSTM, DLinear, N-BEATS, metrics, time-based splits) | `Time_Series_slides.pdf` | `time series related materials/` (`ARIMA_demo`, `LSTM_ts_demo`, `DLinear_demo`, `N-BEATS_demo_AirPassengers`) |
| Ensembles: bagging, boosting, AdaBoost, Random Forest, XGBoost, CatBoost/LightGBM, stacking, CV grid search | `TDT4173_Predictor_practice_slides.pdf` | `predictor_practice_related_materials/` (`rf_iris`, `rf_temps`, `pima_adaboost`, `xgboost_get_started`, `xgboost_lightgbm_catboost_compare`, `modified_stacking_cv_gridsearch_example`) |

## Behaviour
- Before proposing an approach, **read the relevant slide/notebook** (use `pdftotext -layout file.pdf -`) and cite it: "Predictor_practice slides, Random Forest section".
- Prefer the course's own recipes: Random Forest as first prototype → XGBoost/LightGBM/CatBoost → stacking/ensembling; window + statistical features for time series; time-based validation.
- Course-endorsed baselines to beat first: constant/mean-rate prediction, then logistic regression/decision tree, then Random Forest.
- Deep time-series models (LSTM, DLinear, N-BEATS) are in the curriculum but only worth it if tabular models plateau; 4 CPUs/32GB, no GPU.
- **No external data** (project hint explicitly forbids it). Only `data/` in this repo.
- Allowed to look at other teams' approaches in the discussion sense ("spy on other teams" is a course hint) but never copy code we don't understand.
