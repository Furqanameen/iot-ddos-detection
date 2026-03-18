# =============================================================================
# scripts/download_datasets.py
# Downloads or generates sample datasets for testing
# Usage: python scripts/download_datasets.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import zipfile

from config import DATA_RAW

# ── Attack type labels ────────────────────────────────────────────────────
ATTACK_TYPES = [
    'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
    'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP',
    'Syn', 'TFTP', 'UDP-lag', 'WebDDoS', 'BENIGN'
]

FEATURE_NAMES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]


def generate_synthetic_dataset(n_samples=50000, dataset_name="CICDDoS2019_sample"):
    """
    Generates a realistic synthetic dataset mimicking CICDDoS2019 statistics.
    Used when the real dataset is not yet downloaded.
    Real datasets: https://www.unb.ca/cic/datasets/ddos-2019.html
    """
    print(f"\n[+] Generating synthetic {dataset_name} ({n_samples:,} samples)...")
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n_features = len(FEATURE_NAMES)
    rows = []

    # 70% attack traffic, 30% benign
    for i in tqdm(range(n_samples), desc="  Generating rows"):
        label_idx = rng.choice(len(ATTACK_TYPES),
                               p=[0.06]*12 + [0.28])
        label = ATTACK_TYPES[label_idx]
        is_attack = (label != 'BENIGN')

        if is_attack:
            # Attack traffic: high packet rates, short durations
            row = {
                'Destination Port':          rng.choice([53, 80, 443, 123, 161]),
                'Flow Duration':             rng.integers(100, 50000),
                'Total Fwd Packets':         rng.integers(100, 5000),
                'Total Backward Packets':    rng.integers(0, 10),
                'Total Length of Fwd Packets': rng.integers(5000, 500000),
                'Total Length of Bwd Packets': rng.integers(0, 500),
                'Fwd Packet Length Max':     rng.integers(60, 1500),
                'Fwd Packet Length Min':     rng.integers(40, 100),
                'Fwd Packet Length Mean':    rng.uniform(50, 200),
                'Fwd Packet Length Std':     rng.uniform(0, 50),
                'Bwd Packet Length Max':     rng.integers(0, 100),
                'Bwd Packet Length Min':     rng.integers(0, 50),
                'Bwd Packet Length Mean':    rng.uniform(0, 50),
                'Bwd Packet Length Std':     rng.uniform(0, 20),
                'Flow Bytes/s':              rng.uniform(1e5, 1e8),
                'Flow Packets/s':            rng.uniform(1e3, 1e6),
                'Flow IAT Mean':             rng.uniform(0, 1000),
                'Flow IAT Std':              rng.uniform(0, 500),
                'Flow IAT Max':              rng.integers(100, 10000),
                'Flow IAT Min':              rng.integers(0, 100),
                'SYN Flag Count':            rng.integers(1, 200) if 'Syn' in label else 0,
                'ACK Flag Count':            rng.integers(0, 50),
                'FIN Flag Count':            rng.integers(0, 5),
                'RST Flag Count':            rng.integers(0, 20),
                'PSH Flag Count':            rng.integers(0, 30),
                'URG Flag Count':            0,
            }
        else:
            # Benign: normal web/IoT traffic patterns
            row = {
                'Destination Port':          rng.choice([80, 443, 22, 8080]),
                'Flow Duration':             rng.integers(10000, 2000000),
                'Total Fwd Packets':         rng.integers(2, 200),
                'Total Backward Packets':    rng.integers(2, 200),
                'Total Length of Fwd Packets': rng.integers(100, 50000),
                'Total Length of Bwd Packets': rng.integers(100, 50000),
                'Fwd Packet Length Max':     rng.integers(100, 1500),
                'Fwd Packet Length Min':     rng.integers(20, 100),
                'Fwd Packet Length Mean':    rng.uniform(200, 800),
                'Fwd Packet Length Std':     rng.uniform(50, 300),
                'Bwd Packet Length Max':     rng.integers(100, 1500),
                'Bwd Packet Length Min':     rng.integers(20, 100),
                'Bwd Packet Length Mean':    rng.uniform(200, 800),
                'Bwd Packet Length Std':     rng.uniform(50, 300),
                'Flow Bytes/s':              rng.uniform(1e2, 5e5),
                'Flow Packets/s':            rng.uniform(1, 500),
                'Flow IAT Mean':             rng.uniform(1000, 100000),
                'Flow IAT Std':              rng.uniform(500, 50000),
                'Flow IAT Max':              rng.integers(10000, 1000000),
                'Flow IAT Min':              rng.integers(100, 10000),
                'SYN Flag Count':            rng.integers(0, 2),
                'ACK Flag Count':            rng.integers(1, 100),
                'FIN Flag Count':            rng.integers(0, 3),
                'RST Flag Count':            rng.integers(0, 2),
                'PSH Flag Count':            rng.integers(0, 50),
                'URG Flag Count':            0,
            }

        # Fill remaining features with plausible noise
        for feat in FEATURE_NAMES:
            if feat not in row:
                row[feat] = max(0, rng.normal(50, 20))

        row[' Label'] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all columns exist
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    out_path = DATA_RAW / f"{dataset_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df):,} rows × {len(df.columns)} cols)")
    print(f"  Label distribution:\n{df[' Label'].value_counts().to_string()}")
    return out_path


def download_real_dataset_instructions():
    """Print instructions for downloading the real CICDDoS2019 dataset."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  How to download the REAL CICDDoS2019 dataset                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Visit: https://www.unb.ca/cic/datasets/ddos-2019.html   ║
║  2. Register for free and download the CSV files             ║
║  3. Place CSVs in:  ./data/raw/                              ║
║                                                              ║
║  Other datasets:                                             ║
║  N-BaIoT:  https://archive.ics.uci.edu/dataset/442          ║
║  TON-IoT:  https://research.unsw.edu.au/projects/toniot-datasets
║  BOT-IoT:  https://research.unsw.edu.au/projects/bot-iot-dataset
║  CICIOT2023: https://www.unb.ca/cic/datasets/iotdataset-2023.html
║                                                              ║
║  For now, a SYNTHETIC dataset has been generated for you     ║
║  to test the full pipeline immediately.                      ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print("=" * 60)
    print("  Dataset Setup")
    print("=" * 60)
    download_real_dataset_instructions()

    # Generate synthetic datasets for immediate use
    generate_synthetic_dataset(50000, "CICDDoS2019_sample")
    generate_synthetic_dataset(30000, "CICIOT2023_sample")
    generate_synthetic_dataset(20000, "NBaIoT_sample")

    print("\n[✓] Datasets ready — run: python scripts/run_pipeline.py")
