# =============================================================================
# models/hybrid_model.py
# CNN-LSTM-GRU Hybrid Deep Learning Model
# Usage: python models/hybrid_model.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              accuracy_score)

from config import MODELS_DIR, RESULTS_DIR, LOGS_DIR, CNN_LSTM_GRU, EVAL
from scripts.preprocess import load_preprocessed


# ── Model architecture ────────────────────────────────────────────────────
def build_cnn_lstm_gru(input_shape: tuple, num_classes: int = 2) -> Model:
    """
    Hybrid CNN-LSTM-GRU architecture:
      CNN  → spatial feature extraction from traffic flow features
      LSTM → long-range temporal dependencies (sequence patterns)
      GRU  → short-range temporal dependencies (recent burst patterns)
    """
    cfg = CNN_LSTM_GRU
    inputs = tf.keras.Input(shape=input_shape, name="flow_features")

    # ── CNN block ──────────────────────────────────────────────
    x = layers.Conv1D(cfg["cnn_filters"], cfg["cnn_kernel"],
                      activation='relu', padding='same',
                      name='cnn_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Conv1D(cfg["cnn_filters"] * 2, cfg["cnn_kernel"],
                      activation='relu', padding='same',
                      name='cnn_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool_1')(x)
    x = layers.Dropout(cfg["dropout"], name='drop_cnn')(x)

    # ── LSTM block ─────────────────────────────────────────────
    x = layers.LSTM(cfg["lstm_units"], return_sequences=True,
                    name='lstm_1')(x)
    x = layers.Dropout(cfg["dropout"], name='drop_lstm')(x)

    # ── GRU block ──────────────────────────────────────────────
    x = layers.GRU(cfg["gru_units"], return_sequences=False,
                   name='gru_1')(x)
    x = layers.Dropout(cfg["dropout"], name='drop_gru')(x)

    # ── Classification head ────────────────────────────────────
    x = layers.Dense(cfg["dense_units"], activation='relu',
                     name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='output')(x)

    model = Model(inputs, outputs, name="CNN_LSTM_GRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=cfg["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')])
    return model


def reshape_for_model(X: np.ndarray) -> np.ndarray:
    """Add timestep dimension: (N, features) → (N, features, 1)"""
    return X.reshape(X.shape[0], X.shape[1], 1)


def train(dataset_name: str = "CICDDoS2019_sample"):
    print(f"\n{'='*60}")
    print(f"  CNN-LSTM-GRU Training — {dataset_name}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        load_preprocessed(dataset_name)

    # Reshape for Conv1D
    X_tr3 = reshape_for_model(X_train)
    X_v3  = reshape_for_model(X_val)
    X_te3 = reshape_for_model(X_test)

    print(f"  Input shape : {X_tr3.shape}")
    print(f"  Num features: {X_train.shape[1]}")

    model = build_cnn_lstm_gru(
        input_shape=(X_train.shape[1], 1), num_classes=2)
    model.summary()

    # ── Callbacks ──────────────────────────────────────────────
    out_dir   = MODELS_DIR / "hybrid" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir   = LOGS_DIR / "tensorboard" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CNN_LSTM_GRU["early_stop_patience"],
            restore_best_weights=True,
            verbose=1),
        callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1),
        callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1),
    ]

    # ── Training ───────────────────────────────────────────────
    t0 = time.perf_counter()
    history = model.fit(
        X_tr3, y_train,
        validation_data=(X_v3, y_val),
        epochs=CNN_LSTM_GRU["epochs"],
        batch_size=CNN_LSTM_GRU["batch_size"],
        callbacks=cb_list,
        verbose=1)
    train_time = time.perf_counter() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    # ── Evaluation ─────────────────────────────────────────────
    print("\n  Evaluating on test set...")
    t0       = time.perf_counter()
    y_proba  = model.predict(X_te3, verbose=0)
    inf_time = (time.perf_counter() - t0) / len(X_test) * 1000
    y_pred   = y_proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba[:, 1])

    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  F1-score        : {f1:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Infer ms/sample : {inf_time:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Benign','Attack'])}")

    # ── Save model and results ─────────────────────────────────
    model.save(out_dir / "cnn_lstm_gru.keras")

    metrics = {
        "model": "CNN_LSTM_GRU",
        "dataset": dataset_name,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "auc_roc":  round(auc, 4),
        "train_time_s": round(train_time, 2),
        "infer_ms_per_sample": round(inf_time, 4),
        "epochs_trained": len(history.history['accuracy']),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    results_path = RESULTS_DIR / f"hybrid_metrics_{dataset_name}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────
    _plot_history(history, out_dir)
    _plot_confusion(y_test, y_pred, out_dir)
    _plot_roc(y_test, y_proba[:, 1], auc, out_dir)

    print(f"\n[✓] Model saved → {out_dir}")
    print(f"[✓] Results   → {results_path}")
    print(f"[✓] TensorBoard → tensorboard --logdir {log_dir}")
    return model, metrics


def _plot_history(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.set_xlabel('Epoch')
    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend(); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_confusion(y_test, y_pred, out_dir):
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix — CNN-LSTM-GRU')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_roc(y_test, y_proba, auc, out_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — CNN-LSTM-GRU')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    train("CICDDoS2019_sample")
