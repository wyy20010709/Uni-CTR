from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def compute_auc_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = 0.5
    out["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    return out
