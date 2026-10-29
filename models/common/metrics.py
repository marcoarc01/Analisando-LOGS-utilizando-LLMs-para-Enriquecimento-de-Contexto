import statistics

from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

SCALARS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy")
CELLS = ("tn", "fp", "fn", "tp")


def binary_metrics(y_true, scores, y_pred):
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "n": len(y_true),
        "positives": int(sum(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def by_condition(y_true, y_pred, conditions):
    out = {}
    for condition in sorted(set(conditions)):
        idx = [i for i, c in enumerate(conditions) if c == condition]
        positives = sum(y_true[i] for i in idx)
        tp = sum(y_true[i] == 1 and y_pred[i] == 1 for i in idx)
        out[condition] = {"windows": len(idx), "positives": positives, "true_positives": tp,
                          "predicted_positive": sum(y_pred[i] for i in idx),
                          "recall": tp / positives if positives else None}
    return out


def mean_std(values):
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}


def summarize(per_seed):
    out = {m: mean_std([p[m] for p in per_seed]) for m in SCALARS}
    out["confusion_matrix_sum"] = {c: sum(p["confusion_matrix"][c] for p in per_seed) for c in CELLS}
    return out


def summarize_conditions(per_seed):
    out = {}
    for condition in per_seed[0]:
        recalls = [p[condition]["recall"] for p in per_seed]
        out[condition] = {"recall": None if recalls[0] is None else mean_std(recalls),
                          "predicted_positive": mean_std([p[condition]["predicted_positive"] for p in per_seed])}
    return out
