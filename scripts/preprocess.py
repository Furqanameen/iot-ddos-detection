# =============================================================================
# scripts/preprocess.py
# Loads, cleans, balances, and splits datasets for training
# Usage: python scripts/preprocess.py
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

from config import DATA_RAW, DATA_PROCESSED, EVAL, DROP_COLS, LABEL_COL, BENIGN_STR


def load_and_merge_csvs(directory: Path) -> pd.DataFrame:
    """Load all CSV files from a directory and merge them."""
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    dfs = []
    for f in csv_files:
        print(f"  Loading {f.name} ...", end=" ")
        df = pd.read_csv(f, low_memory=False)
        print(f"{len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(csv_files)} file(s)")
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulls, infs, and non-numeric columns."""
    initial = len(df)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Replace inf/-inf with NaN then drop
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)

    # Drop known non-feature columns
    drop = [c for c in DROP_COLS if c.strip() in df.columns]
    df.drop(columns=drop, inplace=True, errors='ignore')

    removed = initial - len(df)
    print(f"  Cleaned: removed {removed:,} invalid rows ({removed/initial*100:.1f}%)")
    return df


def encode_labels(df: pd.DataFrame, label_col: str = 'Label') -> tuple:
    """Binary encode: BENIGN=0, all attacks=1. Also return multiclass mapping."""
    label_col = next((c for c in df.columns if c.strip().upper() == 'LABEL'), None)
    if label_col is None:
        raise ValueError("No 'Label' column found in dataframe")

    raw_labels = df[label_col].str.strip()

    # Multiclass mapping
    le = LabelEncoder()
    y_multi = le.fit_transform(raw_labels)

    # Binary mapping
    y_binary = (raw_labels != BENIGN_STR).astype(int).values

    df.drop(columns=[label_col], inplace=True)

    print(f"  Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
    print(f"  Binary  — Benign: {(y_binary==0).sum():,}  Attack: {(y_binary==1).sum():,}")
    return y_binary, y_multi, le


def select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric columns."""
    X = df.select_dtypes(include=[np.number])
    print(f"  Features: {X.shape[1]} numeric columns retained")
    return X


def apply_smote(X, y, random_state=42):
    """Balance classes with SMOTE."""
    print(f"  Before SMOTE: {np.bincount(y)}")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  After  SMOTE: {np.bincount(y_res)}")
    return X_res, y_res


def split_and_scale(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """Train/val/test split + StandardScaler fit on train only."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        stratify=y,
        random_state=random_state)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"  Train: {len(X_train_s):,}  Val: {len(X_val_s):,}  Test: {len(X_test_s):,}")
    return (X_train_s, X_val_s, X_test_s,
            y_train, y_val, y_test, scaler)


def preprocess(dataset_name: str = "CICDDoS2019_sample", balance=True):
    """Full preprocessing pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Preprocessing: {dataset_name}")
    print(f"{'='*60}")

    # Load
    df = load_and_merge_csvs(DATA_RAW) if dataset_name == "all" \
         else pd.read_csv(DATA_RAW / f"{dataset_name}.csv", low_memory=False)

    # Clean
    df = clean_dataframe(df)

    # Labels
    y_binary, y_multi, le = encode_labels(df)

    # Features
    X = select_numeric_features(df)
    feature_names = list(X.columns)
    X = X.values

    # Balance
    if balance:
        X, y_binary = apply_smote(X, y_binary)

    # Split + Scale
    (X_train, X_val, X_test,
     y_train, y_val, y_test, scaler) = split_and_scale(X, y_binary)

    # Save
    out_dir = DATA_PROCESSED / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "y_test.npy",  y_test)

    joblib.dump(scaler,       out_dir / "scaler.pkl")
    joblib.dump(le,           out_dir / "label_encoder.pkl")
    joblib.dump(feature_names, out_dir / "feature_names.pkl")

    print(f"\n  [✓] Saved preprocessed data → {out_dir}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def load_preprocessed(dataset_name: str = "CICDDoS2019_sample"):
    """Load previously preprocessed data."""
    d = DATA_PROCESSED / dataset_name
    return (
        np.load(d / "X_train.npy"),
        np.load(d / "X_val.npy"),
        np.load(d / "X_test.npy"),
        np.load(d / "y_train.npy"),
        np.load(d / "y_val.npy"),
        np.load(d / "y_test.npy"),
        joblib.load(d / "feature_names.pkl"),
    )


if __name__ == "__main__":
    datasets = ["CICDDoS2019_sample", "CICIOT2023_sample", "NBaIoT_sample"]
    for ds in datasets:
        csv_path = DATA_RAW / f"{ds}.csv"
        if csv_path.exists():
            preprocess(ds)
        else:
            print(f"  [!] Skipping {ds} — file not found. Run download_datasets.py first.")

    print("\n[✓] Preprocessing complete")
