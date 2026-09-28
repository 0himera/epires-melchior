"""Trusted adapter. The child receives training labels only, never test labels."""
import os
import runpy
import sys

if os.name != "nt":
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))

import numpy as np

code, data_path, output_path = sys.argv[1:]
data = np.load(data_path, allow_pickle=False)
namespace = runpy.run_path(code, run_name="candidate")
function = namespace.get("fit_predict")
if not callable(function):
    raise ValueError("Candidate must define fit_predict(X_train, y_train, X_test)")
predictions = np.asarray(function(data["X_train"], data["y_train"], data["X_test"]))
if predictions.ndim != 1 or len(predictions) != len(data["X_test"]):
    raise ValueError("fit_predict must return a vector with one prediction per test row")
if predictions.dtype.kind not in "biuf" or not np.isfinite(predictions).all():
    raise ValueError("Predictions must be finite numbers")
np.save(output_path, predictions, allow_pickle=False)
