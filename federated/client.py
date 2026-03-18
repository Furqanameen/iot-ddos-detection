# =============================================================================
# federated/client.py
# Federated Learning Client using Flower
# Run AFTER server: python federated/client.py --id 0
#                   python federated/client.py --id 1
#                   python federated/client.py --id 2
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import flwr as fl
import tensorflow as tf
from typing import Dict, List, Tuple

from config import FL, EVAL
from models.hybrid_model import build_cnn_lstm_gru
from scripts.preprocess import load_preprocessed


class IoTDDoSClient(fl.client.NumPyClient):
    """
    Each client simulates an independent IoT network segment
    with its own local data partition (Non-IID simulation).
    """

    def __init__(self, client_id: int, n_features: int,
                 X_train, y_train, X_val, y_val):
        self.client_id = client_id
        self.model = build_cnn_lstm_gru(
            input_shape=(n_features, 1), num_classes=2)

        # Reshape for Conv1D
        self.X_train = X_train.reshape(-1, X_train.shape[1], 1)
        self.y_train = y_train
        self.X_val   = X_val.reshape(-1, X_val.shape[1], 1)
        self.y_val   = y_val

        print(f"  [Client {client_id}] Ready — "
              f"train: {len(X_train):,}  val: {len(X_val):,}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray],
            config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.model.set_weights(parameters)

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=FL["local_epochs"],
            batch_size=256,
            validation_data=(self.X_val, self.y_val),
            verbose=0)

        acc = history.history['val_accuracy'][-1]
        print(f"  [Client {self.client_id}] Round fit — val_acc: {acc:.4f}")
        return self.model.get_weights(), len(self.X_train), {"accuracy": acc}

    def evaluate(self, parameters: List[np.ndarray],
                 config: Dict) -> Tuple[float, int, Dict]:
        self.model.set_weights(parameters)
        loss, acc, *_ = self.model.evaluate(
            self.X_val, self.y_val, verbose=0)
        print(f"  [Client {self.client_id}] Eval — loss: {loss:.4f}  acc: {acc:.4f}")
        return loss, len(self.X_val), {"accuracy": float(acc)}


def partition_data(X, y, client_id: int, num_clients: int,
                   non_iid: bool = True):
    """
    Split data across clients.
    non_iid=True: each client gets a skewed label distribution
    non_iid=False: IID uniform split
    """
    n = len(X)
    if non_iid:
        # Sort by label then assign contiguous blocks (simulates non-IID)
        idx = np.argsort(y)
        X, y = X[idx], y[idx]
        chunk = n // num_clients
        start = client_id * chunk
        end   = start + chunk if client_id < num_clients - 1 else n
    else:
        chunk = n // num_clients
        start = client_id * chunk
        end   = start + chunk

    return X[start:end], y[start:end]


def run_client(client_id: int, dataset_name: str = "CICDDoS2019_sample"):
    print(f"\n{'='*60}")
    print(f"  FL Client {client_id} — {dataset_name}")
    print(f"  Server: {FL['server_address']}")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        load_preprocessed(dataset_name)

    # Partition data for this client
    X_part, y_part = partition_data(
        X_train, y_train,
        client_id=client_id,
        num_clients=FL["num_clients"],
        non_iid=True)

    # Use full val set for evaluation
    client = IoTDDoSClient(
        client_id=client_id,
        n_features=X_train.shape[1],
        X_train=X_part,
        y_train=y_part,
        X_val=X_val,
        y_val=y_val)

    fl.client.start_numpy_client(
        server_address=FL["server_address"],
        client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",      type=int, default=0,
                        help="Client ID (0, 1, 2, ...)")
    parser.add_argument("--dataset", type=str,
                        default="CICDDoS2019_sample",
                        help="Dataset name")
    args = parser.parse_args()
    run_client(args.id, args.dataset)
