# iot-ddos-detection
# IoT-Enabled DDoS Detection — Complete Running Guide
## Ubuntu Setup & Execution Instructions

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS        | Ubuntu 20.04+ | Ubuntu 22.04 |
| RAM       | 8 GB    | 16 GB |
| CPU       | 4 cores | 8 cores |
| Disk      | 10 GB   | 30 GB (for real datasets) |
| Python    | 3.9+    | 3.11 |
| GPU       | Optional | NVIDIA (speeds up training 10×) |

---

## Step 1 — Clone / Navigate to Project

```bash
# If you received this as a folder, just navigate into it:
cd iot-ddos-detection

# OR unzip if you got a zip file:
unzip iot-ddos-detection.zip
cd iot-ddos-detection
```

---

## Step 2 — Automated Setup (Recommended)

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install system packages (Python, build tools, etc.)
- Create a Python virtual environment (`./venv`)
- Install all Python packages
- Register Jupyter kernel

---

## Step 3 — Manual Setup (If setup.sh fails)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core packages
pip install tensorflow==2.15.0 torch==2.2.0
pip install scikit-learn imbalanced-learn pandas numpy
pip install matplotlib seaborn plotly
pip install shap lime
pip install flwr xgboost lightgbm
pip install dash dash-bootstrap-components
pip install joblib tqdm tensorboard psutil optuna
pip install jupyter notebook

# Optional (may fail on some systems — skip if it does)
pip install tenseal
pip install ryu
```

---

## Step 4 — Activate Virtual Environment

**Every time you open a new terminal, run this first:**

```bash
source venv/bin/activate
# Your prompt will show: (venv) user@machine:~$
```

---

## Step 5 — Run the Full Pipeline

### Option A: Run Everything at Once

```bash
python scripts/run_pipeline.py
```

This runs all stages sequentially:
1. Downloads/generates datasets
2. Preprocesses all data
3. Trains baseline models (RF, XGBoost, SVM, KNN)
4. Trains CNN-LSTM-GRU hybrid model
5. Runs SHAP + LIME analysis
6. Runs full evaluation suite
7. Runs SDN simulation
8. Launches the dashboard

**Expected total time: ~20-40 minutes** (CPU only, no GPU)

---

### Option B: Run Individual Stages

```bash
# Stage 1: Generate datasets and preprocess
python scripts/run_pipeline.py --stage data

# Stage 2: Train baseline models
python scripts/run_pipeline.py --stage baseline

# Stage 3: Train CNN-LSTM-GRU (main model)
python scripts/run_pipeline.py --stage hybrid

# Stage 4: SHAP + LIME explainability
python scripts/run_pipeline.py --stage xai

# Stage 5: Evaluation + benchmarks
python scripts/run_pipeline.py --stage eval

# SDN simulation
python scripts/run_pipeline.py --stage sdn

# Launch dashboard only
python scripts/run_pipeline.py --stage dashboard
```

---

### Option C: Run Everything Except Dashboard

```bash
python scripts/run_pipeline.py --skip-dashboard
```

Then launch the dashboard separately when ready:

```bash
python dashboard/app.py
# Open: http://localhost:8050
```

---

## Step 6 — Federated Learning (Optional, Multi-Terminal)

Federated learning requires 4 terminals open simultaneously:

```bash
# Terminal 1 — Start the FL server FIRST
source venv/bin/activate
python federated/server.py

# Terminal 2 — Start client 0 (after server shows "Waiting...")
source venv/bin/activate
python federated/client.py --id 0

# Terminal 3 — Start client 1
source venv/bin/activate
python federated/client.py --id 1

# Terminal 4 — Start client 2
source venv/bin/activate
python federated/client.py --id 2
```

Server will run for 10 rounds (~5-10 minutes), then stop.
Results saved to: `results/federated_rounds.json`

---

## Step 7 — SDN Simulation (Without Ryu)

If Ryu isn't installed, the SDN simulation runs in standalone mode:

```bash
python sdn/ddos_controller.py
```

**With Ryu installed (full SDN):**

```bash
# Terminal 1 — Start Ryu controller
ryu-manager sdn/ddos_controller.py

# Terminal 2 — Start Mininet topology
sudo mn --controller=remote,ip=127.0.0.1,port=6633 \
        --topo=tree,2,4 --switch=ovsk,protocols=OpenFlow13

# In Mininet CLI, generate traffic:
mininet> h1 ping h2
mininet> iperf h1 h2
```

---

## Step 8 — View Results

```bash
# All output files are in:
ls results/

# Key files:
results/hybrid_metrics_CICDDoS2019_sample.json    # Main model performance
results/baseline_metrics_CICDDoS2019_sample.json  # Baseline comparison
results/fgsm_robustness_CICDDoS2019_sample.json   # Adversarial results
results/latency_benchmark_CICDDoS2019_sample.json # Latency data
results/xai/CICDDoS2019_sample/top_features.json  # SHAP importances
results/sdn_simulation.json                        # SDN mitigation log
results/adversarial/fgsm_robustness.png           # Robustness chart
models/hybrid/CICDDoS2019_sample/                 # Saved Keras model
models/hybrid/CICDDoS2019_sample/training_history.png
models/hybrid/CICDDoS2019_sample/confusion_matrix.png
models/hybrid/CICDDoS2019_sample/roc_curve.png
```

---

## Step 9 — TensorBoard (Training Monitoring)

```bash
# In a separate terminal:
source venv/bin/activate
tensorboard --logdir logs/tensorboard

# Open: http://localhost:6006
```

---

## Step 10 — Jupyter Notebooks (Exploration)

```bash
source venv/bin/activate
jupyter notebook

# Open: http://localhost:8888
# Navigate to notebooks/ folder
```

---

## Using Real Datasets (For Your Final Dissertation)

The pipeline uses synthetic data by default for immediate testing.
Replace with real datasets before your final experiments:

### CICDDoS2019 (Primary dataset)
```
1. Visit: https://www.unb.ca/cic/datasets/ddos-2019.html
2. Register free account and download CSV files
3. Place ALL CSV files in: ./data/raw/
4. Re-run: python scripts/run_pipeline.py --stage data
```

### N-BaIoT
```
https://archive.ics.uci.edu/dataset/442
```

### TON-IoT
```
https://research.unsw.edu.au/projects/toniot-datasets
```

### BOT-IoT
```
https://research.unsw.edu.au/projects/bot-iot-dataset
```

### CICIOT2023
```
https://www.unb.ca/cic/datasets/iotdataset-2023.html
```

The preprocessing script automatically handles any CSV with a `Label` column.

---

## Common Issues & Fixes

### "ModuleNotFoundError: No module named X"
```bash
source venv/bin/activate   # Make sure venv is active
pip install X
```

### TensorFlow GPU not detected
```bash
# Install CUDA 12.x + cuDNN 8.x from NVIDIA, then:
pip install tensorflow[and-cuda]==2.15.0
```

### "Permission denied" for Mininet
```bash
sudo mn --version   # Mininet requires sudo
```

### SHAP hangs on large datasets
Edit `config.py` and reduce `shap_background_samples` to 50 and
`shap_test_samples` to 50, then re-run.

### Out of memory during training
Edit `config.py` → `CNN_LSTM_GRU` → reduce `batch_size` to 64.

### Ryu not compatible with Python 3.11
```bash
# Ryu works best with Python 3.8:
python3.8 -m venv venv_ryu
source venv_ryu/bin/activate
pip install ryu==4.34
ryu-manager sdn/ddos_controller.py
```

---

## Quick Reference — Most Important Commands

```bash
# Setup (once)
./setup.sh && source venv/bin/activate

# Run full project
python scripts/run_pipeline.py

# Train model only
python scripts/run_pipeline.py --stage hybrid

# View dashboard
python dashboard/app.py          # → http://localhost:8050

# Monitor training
tensorboard --logdir logs/        # → http://localhost:6006

# Federated learning
python federated/server.py        # Terminal 1
python federated/client.py --id 0 # Terminal 2
python federated/client.py --id 1 # Terminal 3
python federated/client.py --id 2 # Terminal 4
```

---

*Muhammad Farqan | B01822365 | MSc Cyber Security | University of the West of Scotland*
