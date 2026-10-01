# TDT4173 Course Project — Unit Commitment Prediction (SINTEF Tokke–Vinje)

We are students. The goal is to **learn the course material and score well on Kaggle**, in that order of honesty: a good score we can't explain is a bad outcome.

Rules in `.claude/rules/` load automatically. Read them. Key points:
- Everything must be grounded in `learningmaterials/` (slides + demo notebooks). No external data. See `rules/course-grounding.md`.
- Task facts, data shapes, metric: `rules/task-and-data.md`.
- Time-aware validation only; no leakage: `rules/validation.md`.
- Teach while building: `rules/learning-mode.md`. Clean notebooks/code: `rules/code-style.md`.

Agents in `.claude/agents/`: `course-tutor`, `eda-analyst`, `feature-engineer`, `model-builder`, `validation-reviewer`, `submission-checker`.

Environment: python3 with pandas, scikit-learn, xgboost, scikit-optimize, matplotlib. `lightgbm`/`catboost` are NOT installed (ask before installing; both are in the course slides).
