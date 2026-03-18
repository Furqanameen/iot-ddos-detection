# =============================================================================
# xai/shap_analysis.py
# SHAP explainability — global feature importance + local explanations
# Usage: python xai/shap_analysis.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

import tensorflow as tf
from config import MODELS_DIR, RESULTS_DIR, XAI
from scripts.preprocess import load_preprocessed


def run_shap(dataset_name: str = "CICDDoS2019_sample"):
    print(f"\n{'='*60}")
    print(f"  SHAP Analysis — {dataset_name}")
    print(f"{'='*60}")

    # Load data and model
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        load_preprocessed(dataset_name)

    model_path = MODELS_DIR / "hybrid" / dataset_name / "cnn_lstm_gru.keras"
    if not model_path.exists():
        print(f"  [!] Model not found. Run: python models/hybrid_model.py first.")
        return

    model = tf.keras.models.load_model(model_path)

    # Use small samples for speed
    n_bg   = XAI["shap_background_samples"]
    n_test = XAI["shap_test_samples"]

    bg_idx   = np.random.choice(len(X_train), n_bg,   replace=False)
    test_idx = np.random.choice(len(X_test),  n_test, replace=False)

    background = X_train[bg_idx].reshape(-1, X_train.shape[1], 1).astype('float32')
    test_data  = X_test[test_idx].reshape(-1, X_test.shape[1], 1).astype('float32')

    print(f"  Computing SHAP values ({n_bg} background, {n_test} test)...")
    t0 = time.perf_counter()

    explainer   = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_data)
    elapsed     = time.perf_counter() - t0
    overhead_pct = (elapsed / n_test * 1000)

    print(f"  SHAP computation: {elapsed:.1f}s  ({overhead_pct:.2f} ms/sample overhead)")

    # Attack class SHAP values (class 1)
    sv_attack = shap_values[1] if isinstance(shap_values, list) \
                else shap_values[:, :, 0, 1]
    sv_2d = sv_attack.reshape(n_test, -1)
    X_2d  = test_data.reshape(n_test, -1)

    out_dir = RESULTS_DIR / "xai" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Global: bar summary ────────────────────────────────────
    print("  Generating global feature importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_2d, X_2d,
                      feature_names=feature_names,
                      plot_type="bar",
                      max_display=20,
                      show=False)
    plt.title("SHAP — Top 20 Global Feature Importances (Attack Class)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Global: beeswarm ──────────────────────────────────────
    print("  Generating beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_2d, X_2d,
                      feature_names=feature_names,
                      max_display=15,
                      show=False)
    plt.title("SHAP — Feature Impact Distribution (Attack Class)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Local: waterfall for top 3 attack samples ──────────────
    y_pred = model.predict(test_data, verbose=0).argmax(axis=1)
    attack_idx = np.where(y_pred == 1)[0][:3]

    for i, idx in enumerate(attack_idx):
        print(f"  Generating waterfall for sample {idx}...")
        exp = shap.Explanation(
            values=sv_2d[idx],
            base_values=float(explainer.expected_value[1]),
            data=X_2d[idx],
            feature_names=feature_names)
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.title(f"SHAP Local Explanation — Attack Sample {i+1}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_waterfall_sample_{i+1}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Save top feature importances as JSON ───────────────────
    mean_abs = np.abs(sv_2d).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:20]
    top_features = [
        {"feature": feature_names[i], "importance": round(float(mean_abs[i]), 6)}
        for i in top_idx
    ]
    with open(out_dir / "top_features.json", 'w') as f:
        json.dump(top_features, f, indent=2)

    print(f"\n  Top 5 features by SHAP importance:")
    for item in top_features[:5]:
        print(f"    {item['feature']:<35} {item['importance']:.6f}")

    print(f"\n[✓] SHAP analysis saved → {out_dir}")
    print(f"  XAI overhead: {overhead_pct:.2f} ms/sample")

    return top_features


if __name__ == "__main__":
    run_shap("CICDDoS2019_sample")
