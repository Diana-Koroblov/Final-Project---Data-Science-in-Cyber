"""
evaluation.py — Model evaluation helpers for the ACR'25 reproduction study.

Functions
---------
evaluate_model   : Compute Accuracy, Precision, Recall, F1, Fβ, MCC, ROC-AUC
plot_cm          : Plot a single confusion matrix
get_cm_row       : Return a dict of TP/TN/FP/FN/FPR/FNR for a fitted model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)


def evaluate_model(name: str, y_true, y_pred, y_prob=None, beta: int = 2) -> dict:
    """
    Compute a full suite of classification metrics.

    Parameters
    ----------
    name    : Label for the model (used as the 'Model' key in the returned dict).
    y_true  : Ground-truth binary labels.
    y_pred  : Predicted binary labels.
    y_prob  : Predicted positive-class probabilities (required for ROC-AUC).
    beta    : Beta value for Fβ score (default 2 → weights Recall twice over Precision).

    Returns
    -------
    dict with keys: Model, Accuracy, Precision, Recall, F1, F{beta}, MCC, ROC-AUC (if y_prob given).
    """
    r = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        f"F{beta}":  fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        r["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return r


def plot_cm(y_true, y_pred, title: str, labels=("Normal", "Attack")):
    """
    Plot a confusion matrix using sklearn's ConfusionMatrixDisplay.

    Parameters
    ----------
    y_true  : Ground-truth binary labels.
    y_pred  : Predicted binary labels.
    title   : Plot title.
    labels  : Display labels for the two classes (default: Normal / Attack).

    Returns
    -------
    cm : numpy array — the raw confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return cm


def get_cm_row(dataset: str, name: str, model, X_te, y_te) -> dict:
    """
    Predict on X_te and return a confusion-matrix summary row.

    Parameters
    ----------
    dataset : Dataset label (e.g. 'KDDCUP99').
    name    : Model label (e.g. 'XGBoost').
    model   : Fitted sklearn-compatible model with a .predict() method.
    X_te    : Test feature matrix.
    y_te    : Test labels.

    Returns
    -------
    dict with keys: Dataset, Model, TP, TN, FP, FN, Total, FPR %, FNR %.
    """
    yp = model.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_te, yp).ravel()
    total = tn + fp + fn + tp
    return {
        "Dataset": dataset,
        "Model":   name,
        "TP":      tp,
        "TN":      tn,
        "FP":      fp,
        "FN":      fn,
        "Total":   total,
        "FPR %":   round(fp / (fp + tn) * 100, 3),  # false alarm rate
        "FNR %":   round(fn / (fn + tp) * 100, 3),  # miss rate
    }
