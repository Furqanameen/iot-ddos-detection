# =============================================================================
# config.py — Central configuration for all modules
# =============================================================================

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_SYNTHETIC = BASE_DIR / "data" / "synthetic"
MODELS_DIR     = BASE_DIR / "models"
LOGS_DIR       = BASE_DIR / "logs"
RESULTS_DIR    = BASE_DIR / "results"

for d in [DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC,
          MODELS_DIR / "baseline", MODELS_DIR / "hybrid", MODELS_DIR / "tflite",
          LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset URLs (public mirrors) ─────────────────────────────────────────
DATASET_URLS = {
    "CICDDoS2019": {
        "info": "https://www.unb.ca/cic/datasets/ddos-2019.html",
        "gdrive_id": "1bwuXVHEsUBPXiJNcVH1s5oCnZ4NK0BQT",  # community mirror
        "filename": "CICDDoS2019_sample.csv"
    },
    "N-BaIoT": {
        "info": "https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot",
        "filename": "nbaiot_sample.csv"
    },
}

# ── Feature engineering ────────────────────────────────────────────────────
DROP_COLS = [
    'Flow ID', ' Source IP', ' Destination IP',
    ' Source Port', ' Timestamp', 'SimillarHTTP'
]

LABEL_COL  = ' Label'
BENIGN_STR = 'BENIGN'

# ── Model hyperparameters ──────────────────────────────────────────────────
CNN_LSTM_GRU = {
    "cnn_filters":    64,
    "cnn_kernel":     3,
    "lstm_units":     128,
    "gru_units":      64,
    "dense_units":    64,
    "dropout":        0.3,
    "learning_rate":  1e-3,
    "batch_size":     256,
    "epochs":         50,
    "early_stop_patience": 7,
}

BASELINE_MODELS = {
    "random_forest": {"n_estimators": 100, "max_depth": 20, "n_jobs": -1},
    "xgboost":       {"n_estimators": 200, "max_depth": 8,  "learning_rate": 0.1},
    "svm":           {"kernel": "rbf", "C": 10, "gamma": "scale"},
    "knn":           {"n_neighbors": 5, "n_jobs": -1},
}

# ── Federated learning ─────────────────────────────────────────────────────
FL = {
    "server_address": "0.0.0.0:8080",
    "num_rounds":     10,
    "num_clients":    3,
    "fraction_fit":   1.0,
    "local_epochs":   2,
}

# ── XAI ───────────────────────────────────────────────────────────────────
XAI = {
    "shap_background_samples": 200,
    "shap_test_samples":       100,
    "lime_num_features":       15,
    "lime_num_samples":        1000,
}

# ── SDN ───────────────────────────────────────────────────────────────────
SDN = {
    "block_threshold":    0.85,   # confidence to trigger block
    "idle_timeout":       60,     # seconds before flow rule expires
    "rate_limit_kbps":    100,    # traffic shaping limit
    "controller_port":    6633,
}

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL = {
    "cv_folds":        10,
    "test_size":       0.15,
    "val_size":        0.15,
    "random_state":    42,
    "target_accuracy": 0.98,
    "target_latency_ms": 100,
}

# ── Dashboard ─────────────────────────────────────────────────────────────
DASHBOARD = {
    "host": "0.0.0.0",
    "port": 8050,
    "debug": True,
}
