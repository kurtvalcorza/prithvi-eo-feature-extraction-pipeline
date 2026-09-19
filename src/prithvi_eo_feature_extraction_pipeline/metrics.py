"""Pixel-level segmentation metrics from a confusion matrix over the labelled pixels (ignore index excluded):
per-class IoU, mean IoU, overall accuracy, and the positive class's precision, recall and F1 — plus the
"not burned" baseline that predicts class 0 everywhere, scored on exactly the same pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .pipeline import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(predictions: Sequence[Any], labels: Sequence[Any]) -> Any:
    """(NUM_CLASSES, NUM_CLASSES) counts, rows = truth, columns = prediction; ignore-index pixels are skipped."""
    import numpy as np

    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must be non-empty sequences of equal length")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for pred, label in zip(predictions, labels, strict=True):
        pred = np.asarray(pred).astype(np.int64)
        label = np.asarray(label).astype(np.int64)
        if pred.shape != label.shape:
            raise ValueError(f"prediction shape {pred.shape} != label shape {label.shape}")
        keep = label != IGNORE_INDEX
        if not np.all((pred[keep] >= 0) & (pred[keep] < NUM_CLASSES)) or not np.all(label[keep] < NUM_CLASSES):
            raise ValueError("class ids outside 0..NUM_CLASSES-1")
        matrix += np.bincount(label[keep] * NUM_CLASSES + pred[keep], minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return matrix


def metrics_from_confusion(matrix: Any) -> dict[str, Any]:
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    if total == 0:
        raise ValueError("no labelled pixel to score")
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    union = tp + fp + fn
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    present = matrix.sum(axis=1) > 0
    pos = NUM_CLASSES - 1
    precision = tp[pos] / (tp[pos] + fp[pos]) if tp[pos] + fp[pos] > 0 else 0.0
    recall = tp[pos] / (tp[pos] + fn[pos]) if tp[pos] + fn[pos] > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "iou": {CLASS_NAMES[c]: (round(float(iou[c]), 4) if not np.isnan(iou[c]) else None) for c in range(NUM_CLASSES)},
        "mean_iou": round(float(np.nanmean(iou[present])), 4),
        "accuracy": round(float(tp.sum() / total), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "positive_class": CLASS_NAMES[pos],
        "labelled_pixels": int(total),
        "positive_fraction": round(float(matrix[pos].sum() / total), 4),
        "confusion": matrix.astype(int).tolist(),
    }


def segmentation_metrics(predictions: Sequence[Any], labels: Sequence[Any]) -> dict[str, Any]:
    """Metrics of predicted masks against labels over all chips at once (pixel-pooled, not chip-averaged)."""
    return metrics_from_confusion(confusion_matrix(predictions, labels))


def majority_baseline(labels: Sequence[Any]) -> dict[str, Any]:
    """The all-class-0 prediction scored on the same pixels: the number any model must beat on the positive class."""
    import numpy as np

    predictions = [np.zeros_like(np.asarray(label), dtype=np.int64) for label in labels]
    report = segmentation_metrics(predictions, labels)
    report["note"] = f"predicts '{CLASS_NAMES[0]}' for every pixel"
    return report
