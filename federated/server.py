# =============================================================================
# federated/server.py
# Federated Learning Server using Flower
# Run FIRST: python federated/server.py
# Then run clients in separate terminals
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import flwr as fl
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import tensorflow as tf
from config import FL, MODELS_DIR, RESULTS_DIR
from models.hybrid_model import build_cnn_lstm_gru


def get_initial_parameters(n_features: int) -> fl.common.Parameters:
    """Build a fresh model and return its weights as FL parameters."""
    model = build_cnn_lstm_gru(input_shape=(n_features, 1), num_classes=2)
    weights = model.get_weights()
    return fl.common.ndarrays_to_parameters(weights)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg with model saving after each round."""

    def __init__(self, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.round_metrics: List[Dict] = []
        self.out_dir = MODELS_DIR / "hybrid" / "federated"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            params, metrics = aggregated
            # Save aggregated model
            model = build_cnn_lstm_gru(
                input_shape=(self.n_features, 1), num_classes=2)
            ndarrays = fl.common.parameters_to_ndarrays(params)
            model.set_weights(ndarrays)
            model.save(self.out_dir / f"round_{server_round}_model.keras")
            print(f"\n  [FL] Round {server_round} — model saved")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            entry = {"round": server_round, "loss": round(loss, 4)}
            entry.update(metrics)
            self.round_metrics.append(entry)
            print(f"  [FL] Round {server_round} eval — loss: {loss:.4f}  "
                  f"acc: {metrics.get('accuracy', 'N/A')}")

            # Save progress
            path = RESULTS_DIR / "federated_rounds.json"
            with open(path, 'w') as f:
                json.dump(self.round_metrics, f, indent=2)
        return aggregated


def run_server(n_features: int = 78):
    min_clients = int(FL.get("min_clients", FL.get("num_clients", 2)))

    print(f"\n{'='*60}")
    print(f"  Federated Learning Server")
    print(f"  Address : {FL['server_address']}")
    print(f"  Rounds  : {FL['num_rounds']}")
    print(f"  Min clients: {min_clients}")
    print(f"{'='*60}\n")
    print("  Waiting for clients to connect...")
    print(f"  Start clients with: python federated/client.py --id 0")
    print(f"                      python federated/client.py --id 1")
    print(f"                      python federated/client.py --id 2\n")

    strategy = SaveModelStrategy(
        n_features=n_features,
        fraction_fit=FL["fraction_fit"],
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=get_initial_parameters(n_features),
    )

    fl.server.start_server(
        server_address=FL["server_address"],
        config=fl.server.ServerConfig(num_rounds=FL["num_rounds"]),
        strategy=strategy,
    )


if __name__ == "__main__":
    run_server()
