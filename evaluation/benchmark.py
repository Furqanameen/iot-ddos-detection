# =============================================================================
# evaluation/benchmark.py
# Full evaluation: cross-dataset, FGSM adversarial, latency benchmarks
# Usage: python evaluation/benchmark.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import time
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score)

from config import MODELS_DIR, RESULTS_DIR, EVAL
from scripts.preprocess import load_preprocessed


# ── Helpers ───────────────────────────────────────────────────────────────
def load_model_safe(dataset_name: str):
    path = MODELS_DIR / "hybrid" / dataset_name / "cnn_lstm_gru.keras"
    if not path.exists():
        print(f"  [!] No model at {path}. Run models/hybrid_model.py first.")
        return None
    return tf.keras.models.load_model(path)


def compute_full_metrics(model, X_test, y_test, model_name="CNN-LSTM-GRU"):
    """Compute all classification metrics + latency."""
    X3 = X_test.reshape(-1, X_test.shape[1], 1).astype('float32')

    # Latency
    t0      = time.perf_counter()
    y_proba = model.predict(X3, verbose=0)
    lat_ms  = (time.perf_counter() - t0) / len(X_test) * 1000
    y_pred  = y_proba.argmax(axis=1)

    metrics = {
        "model":       model_name,
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":    round(f1_score(y_test, y_pred, average='weighted'), 4),
        "auc_roc":     round(roc_auc_score(y_test, y_proba[:, 1]), 4),
        "infer_ms":    round(lat_ms, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, y_pred, y_proba


# ── 1. Cross-dataset evaluation ───────────────────────────────────────────
def cross_dataset_eval(train_ds: str = "CICDDoS2019_sample"):
    """Train on one dataset, test on others."""
    print(f"\n{'='*60}")
    print(f"  Cross-Dataset Generalisation — trained on {train_ds}")
    print(f"{'='*60}")

    model = load_model_safe(train_ds)
    if model is None:
        return []

    test_datasets = ["CICDDoS2019_sample", "CICIOT2023_sample", "NBaIoT_sample"]
    results = []

    for ds in test_datasets:
        proc_path = __import__('config').DATA_PROCESSED / ds
        if not proc_path.exists():
            print(f"  [!] {ds} not preprocessed — skipping")
            continue

        _, _, X_test, _, _, y_test, _ = load_preprocessed(ds)
        metrics, _, _ = compute_full_metrics(
            model, X_test, y_test, model_name=f"trained:{train_ds}→test:{ds}")
        results.append({"train": train_ds, "test": ds, **metrics})
        print(f"  {ds:<30} acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1_score']:.4f}  auc={metrics['auc_roc']:.4f}")

    path = RESULTS_DIR / f"cross_dataset_{train_ds}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [✓] Cross-dataset results → {path}")
    return results


# ── 2. FGSM Adversarial robustness ────────────────────────────────────────
def fgsm_attack(model, X, y, epsilon: float = 0.01):
    """Fast Gradient Sign Method adversarial perturbation."""
    X_tf = tf.constant(X.reshape(-1, X.shape[1], 1).astype('float32'))
    y_tf = tf.constant(y.astype(np.int64))

    with tf.GradientTape() as tape:
        tape.watch(X_tf)
        preds = model(X_tf, training=False)
        loss  = tf.keras.losses.sparse_categorical_crossentropy(y_tf, preds)

    grads    = tape.gradient(loss, X_tf)
    sign     = tf.sign(grads).numpy()
    X_adv    = X_tf.numpy() + epsilon * sign
    return X_adv


def adversarial_robustness_test(dataset_name: str = "CICDDoS2019_sample"):
    """Test model robustness against FGSM attacks at multiple epsilons."""
    print(f"\n{'='*60}")
    print(f"  FGSM Adversarial Robustness — {dataset_name}")
    print(f"{'='*60}")

    model = load_model_safe(dataset_name)
    if model is None:
        return []

    _, _, X_test, _, _, y_test, _ = load_preprocessed(dataset_name)
    # Sub-sample for speed
    idx    = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
    X_sub  = X_test[idx]
    y_sub  = y_test[idx]

    epsilons = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    results  = []

    print(f"  {'Epsilon':>10} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*44}")

    for eps in epsilons:
        if eps == 0:
            X_eval = X_sub.reshape(-1, X_sub.shape[1], 1).astype('float32')
        else:
            X_eval = fgsm_attack(model, X_sub, y_sub, epsilon=eps)

        y_proba = model.predict(X_eval, verbose=0)
        y_pred  = y_proba.argmax(axis=1)
        acc = accuracy_score(y_sub, y_pred)
        f1  = f1_score(y_sub, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_sub, y_proba[:, 1])

        results.append({"epsilon": eps, "accuracy": round(acc, 4),
                         "f1": round(f1, 4), "auc": round(auc, 4)})
        print(f"  {eps:>10.3f} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")

    # Plot
    out_dir = RESULTS_DIR / "adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    eps_vals = [r["epsilon"] for r in results]
    plt.plot(eps_vals, [r["accuracy"] for r in results], 'o-', label='Accuracy')
    plt.plot(eps_vals, [r["f1"]       for r in results], 's-', label='F1-score')
    plt.plot(eps_vals, [r["auc"]      for r in results], '^-', label='AUC-ROC')
    plt.xlabel("FGSM Epsilon"); plt.ylabel("Score")
    plt.title("Model Robustness Under FGSM Attack")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fgsm_robustness.png", dpi=150, bbox_inches='tight')
    plt.close()

    path = RESULTS_DIR / f"fgsm_robustness_{dataset_name}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [✓] Adversarial results → {path}")
    return results


# ── 3. Latency and resource benchmark ─────────────────────────────────────
def latency_benchmark(dataset_name: str = "CICDDoS2019_sample"):
    """Measure inference latency at different batch sizes."""
    print(f"\n{'='*60}")
    print(f"  Latency Benchmark — {dataset_name}")
    print(f"{'='*60}")

    model = load_model_safe(dataset_name)
    if model is None:
        return []

    _, _, X_test, _, _, y_test, _ = load_preprocessed(dataset_name)
    X3 = X_test.reshape(-1, X_test.shape[1], 1).astype('float32')

    batch_sizes = [1, 8, 32, 64, 128, 256, 512]
    results     = []

    print(f"  {'Batch':>8} {'ms/batch':>12} {'ms/sample':>12} {'CPU%':>8} {'RAM MB':>10}")
    print(f"  {'-'*56}")

    for bs in batch_sizes:
        X_batch = X3[:bs]
        times   = []

        # Warm-up
        _ = model.predict(X_batch, verbose=0)

        # Measure
        for _ in range(10):
            proc = psutil.Process()
            t0   = time.perf_counter()
            _    = model.predict(X_batch, verbose=0)
            times.append((time.perf_counter() - t0) * 1000)

        cpu_pct  = psutil.cpu_percent(interval=0.1)
        ram_mb   = proc.memory_info().rss / 1024 / 1024
        avg_ms   = np.mean(times)
        per_s    = avg_ms / bs

        results.append({
            "batch_size": bs,
            "ms_per_batch": round(avg_ms, 3),
            "ms_per_sample": round(per_s, 4),
            "cpu_percent": round(cpu_pct, 1),
            "ram_mb": round(ram_mb, 1),
        })
        marker = " ← target <100ms" if per_s < 100 else ""
        print(f"  {bs:>8} {avg_ms:>12.2f} {per_s:>12.4f} "
              f"{cpu_pct:>8.1f} {ram_mb:>10.1f}{marker}")

    path = RESULTS_DIR / f"latency_benchmark_{dataset_name}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  [✓] Latency results → {path}")
    return results


# ── 4. Model comparison summary ───────────────────────────────────────────
def comparison_summary(dataset_name: str = "CICDDoS2019_sample"):
    """Compare hybrid model vs all baselines."""
    print(f"\n{'='*60}")
    print(f"  Model Comparison Summary — {dataset_name}")
    print(f"{'='*60}")

    hybrid_path  = RESULTS_DIR / f"hybrid_metrics_{dataset_name}.json"
    baseline_path = RESULTS_DIR / f"baseline_metrics_{dataset_name}.json"

    all_results = []
    if hybrid_path.exists():
        with open(hybrid_path) as f:
            all_results.append(json.load(f))
    if baseline_path.exists():
        with open(baseline_path) as f:
            all_results.extend(json.load(f))

    if not all_results:
        print("  [!] No results found — run training first")
        return

    print(f"\n  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'ms/sample':>12}")
    print(f"  {'-'*64}")
    for m in sorted(all_results, key=lambda x: x.get('accuracy', 0), reverse=True):
        print(f"  {m['model']:<22} "
              f"{m.get('accuracy', 0):>10.4f} "
              f"{m.get('f1_score', 0):>8.4f} "
              f"{m.get('auc_roc', 0):>8.4f} "
              f"{m.get('infer_ms_per_sample', m.get('infer_ms', 0)):>12.4f}")

    # Bar chart
    if len(all_results) > 1:
        models = [m['model'] for m in all_results]
        accs   = [m.get('accuracy', 0) for m in all_results]
        f1s    = [m.get('f1_score', 0) for m in all_results]

        x   = np.arange(len(models))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + w/2, f1s,  w, label='F1-score',  color='coral')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_ylim(0.8, 1.01)
        ax.set_ylabel('Score'); ax.set_title('Model Comparison')
        ax.legend(); ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"model_comparison_{dataset_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  [✓] Comparison chart → {RESULTS_DIR}")


if __name__ == "__main__":
    ds = "CICDDoS2019_sample"
    comparison_summary(ds)
    cross_dataset_eval(ds)
    adversarial_robustness_test(ds)
    latency_benchmark(ds)
    print("\n[✓] All evaluations complete — check results/ directory")
