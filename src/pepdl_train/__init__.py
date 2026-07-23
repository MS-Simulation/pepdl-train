"""pepdl-train — training for pepdl predictors (datasets, loops, losses). Importing this attaches the
fine_tune_* methods back onto the pepdl predictor classes. Inference lives in the separate `pepdl`."""
from pepdl_train import _extensions  # noqa: F401  (side effect: monkeypatch fine_tune_* onto predictors)
