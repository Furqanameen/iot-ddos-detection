# =============================================================================
# models/baseline_models.py
# Train and evaluate RF, XGBoost, SVM, KNN baselines
# Usage: python models/baseline_models.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib
import time
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score)
import xgboost as xgb

from config import MODELS_DIR, RESULTS_DIR, BASELINE_MODELS, EVAL
from scripts.preprocess import load_preprocessed


def train_baseline(name: str, clf, X_train, y_train, X_test, y_test):
    """Train a single baseline classifier and return metrics."""
    print(f"\n  Training {name}...")

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"    Train time : {train_time:.1f}s")

    # Inference latency
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = (time.perf_counter() - t0) / len(X_test) * 1000
    print(f"    Infer/sample: {infer_time:.4f} ms")

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    auc  = roc_auc_score(y_test, y_pred)

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign','Attack'])}")

    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "auc_roc":  round(auc, 4),
        "train_time_s": round(train_time, 2),
        "infer_ms_per_sample": round(infer_time, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return clf, metrics


def run_all_baselines(dataset_name: str = "CICDDoS2019_sample"):
    """Train all baseline models on a given dataset."""
    print(f"\n{'='*60}")
    print(f"  Baseline Models — {dataset_name}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        load_preprocessed(dataset_name)

    # Combine train+val for baseline models (no early stopping needed)
    X_tr = np.vstack([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])

    classifiers = {
        "random_forest": RandomForestClassifier(
            **BASELINE_MODELS["random_forest"], random_state=EVAL["random_state"]),
        "xgboost": xgb.XGBClassifier(
            **BASELINE_MODELS["xgboost"],
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=EVAL["random_state"]),
        "knn": KNeighborsClassifier(**BASELINE_MODELS["knn"]),
        # SVM is slow on large datasets — sub-sample for speed
        "svm": SVC(**BASELINE_MODELS["svm"],
                   probability=True, random_state=EVAL["random_state"]),
    }

    all_metrics = []
    out_dir = MODELS_DIR / "baseline" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # SVM: sub-sample to 10k for tractability
    n_svm = min(10000, len(X_tr))
    idx   = np.random.choice(len(X_tr), n_svm, replace=False)

    for name, clf in classifiers.items():
        _X = X_tr[idx] if name == "svm" else X_tr
        _y = y_tr[idx] if name == "svm" else y_tr
        trained_clf, metrics = train_baseline(name, clf, _X, _y, X_test, y_test)
        all_metrics.append(metrics)
        joblib.dump(trained_clf, out_dir / f"{name}.pkl")
        print(f"    [✓] Saved {name} → {out_dir / f'{name}.pkl'}")

    # Save all metrics
    results_path = RESULTS_DIR / f"baseline_metrics_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[✓] Baseline results saved → {results_path}")

    # Summary table
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'ms/sample':>12}")
    print("-" * 65)
    for m in all_metrics:
        print(f"{m['model']:<20} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f} "
              f"{m['auc_roc']:>10.4f} {m['infer_ms_per_sample']:>12.4f}")

    return all_metrics


if __name__ == "__main__":
    run_all_baselines("CICDDoS2019_sample")
