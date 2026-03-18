# =============================================================================
# xai/lime_analysis.py
# LIME explainability — local instance-level explanations
# Usage: python xai/lime_analysis.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lime
import lime.tabular
import tensorflow as tf

from config import MODELS_DIR, RESULTS_DIR, XAI
from scripts.preprocess import load_preprocessed


def run_lime(dataset_name: str = "CICDDoS2019_sample"):
    print(f"\n{'='*60}")
    print(f"  LIME Analysis — {dataset_name}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        load_preprocessed(dataset_name)

    model_path = MODELS_DIR / "hybrid" / dataset_name / "cnn_lstm_gru.keras"
    if not model_path.exists():
        print(f"  [!] Model not found. Run: python models/hybrid_model.py first.")
        return

    model = tf.keras.models.load_model(model_path)

    # Predict function: LIME passes 2D arrays, model needs 3D
    def predict_fn(x: np.ndarray) -> np.ndarray:
        x3d = x.reshape(-1, x.shape[1], 1).astype('float32')
        return model.predict(x3d, verbose=0)

    # Build LIME explainer
    explainer = lime.tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Benign', 'Attack'],
        mode='classification',
        random_state=42)

    out_dir = RESULTS_DIR / "xai" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explain 5 attack + 5 benign samples
    y_pred = model.predict(
        X_test.reshape(-1, X_test.shape[1], 1).astype('float32'),
        verbose=0).argmax(axis=1)

    attack_idx = np.where(y_pred == 1)[0][:5]
    benign_idx = np.where(y_pred == 0)[0][:5]

    all_explanations = []
    overheads = []

    for label, indices in [("attack", attack_idx), ("benign", benign_idx)]:
        for i, idx in enumerate(indices):
            print(f"  Explaining {label} sample {i+1}/5...")
            t0 = time.perf_counter()
            exp = explainer.explain_instance(
                X_test[idx],
                predict_fn,
                num_features=XAI["lime_num_features"],
                num_samples=XAI["lime_num_samples"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
            overheads.append(elapsed_ms)

            # Save plot
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME — {label.capitalize()} Sample {i+1} "
                         f"(pred: {'Attack' if y_pred[idx]==1 else 'Benign'})")
            fig.tight_layout()
            fig.savefig(out_dir / f"lime_{label}_sample_{i+1}.png",
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Collect feature weights
            weights = exp.as_list()
            all_explanations.append({
                "sample_idx": int(idx),
                "true_label": int(y_test[idx]),
                "pred_label": int(y_pred[idx]),
                "type": label,
                "top_features": [
                    {"feature": f, "weight": round(float(w), 6)}
                    for f, w in weights
                ],
                "prediction_proba": exp.predict_proba.tolist(),
                "elapsed_ms": round(elapsed_ms, 2),
            })

    avg_overhead = np.mean(overheads)
    print(f"\n  Average LIME overhead: {avg_overhead:.1f} ms/explanation")

    # Save all explanations
    with open(out_dir / "lime_explanations.json", 'w') as f:
        json.dump(all_explanations, f, indent=2)

    # Feature frequency across all explanations
    from collections import Counter
    feat_counter = Counter()
    for exp_data in all_explanations:
        for item in exp_data["top_features"]:
            feat_counter[item["feature"]] += 1

    print("\n  Most frequently explanatory features:")
    for feat, count in feat_counter.most_common(10):
        print(f"    {feat:<40} appeared in {count}/{len(all_explanations)} explanations")

    print(f"\n[✓] LIME analysis saved → {out_dir}")
    return all_explanations


if __name__ == "__main__":
    run_lime("CICDDoS2019_sample")
