#!/usr/bin/env python3
# =============================================================================
# scripts/run_pipeline.py
# Master pipeline — runs all stages end-to-end
# Usage: python scripts/run_pipeline.py [--stage STAGE]
#
# Stages:
#   all        Run everything (default)
#   data       Download + preprocess datasets
#   baseline   Train baseline models (RF, XGBoost, SVM, KNN)
#   hybrid     Train CNN-LSTM-GRU model
#   xai        Run SHAP and LIME analysis
#   eval       Run full evaluation suite
#   sdn        Run SDN simulation
#   dashboard  Launch the dashboard
# =============================================================================

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  Intelligent Detection and Mitigation of IoT-Enabled DDoS   ║
║  Attacks Using Hybrid Security Models                        ║
║  MSc Cyber Security  |  Muhammad Farqan  |  B01822365        ║
╚══════════════════════════════════════════════════════════════╝
"""

def header(title):
    print(f"\n{'━'*60}")
    print(f"  ▶  {title}")
    print(f"{'━'*60}")

def ok(msg):    print(f"  ✓  {msg}")
def warn(msg):  print(f"  ⚠  {msg}")
def fail(msg):  print(f"  ✗  {msg}")


def stage_data():
    header("Stage 1/5: Dataset Download & Preprocessing")
    from scripts.download_datasets import generate_synthetic_dataset
    from scripts.preprocess import preprocess
    from config import DATA_RAW

    datasets = ["CICDDoS2019_sample", "CICIOT2023_sample", "NBaIoT_sample"]
    for ds in datasets:
        csv = DATA_RAW / f"{ds}.csv"
        if not csv.exists():
            warn(f"{ds}.csv not found — generating synthetic data")
            n = 50000 if "CICDDoS2019" in ds else 30000
            generate_synthetic_dataset(n, ds)
        preprocess(ds)
    ok("All datasets preprocessed")


def stage_baseline():
    header("Stage 2/5: Baseline Model Training")
    from models.baseline_models import run_all_baselines
    run_all_baselines("CICDDoS2019_sample")
    ok("Baseline models trained")


def stage_hybrid():
    header("Stage 3/5: CNN-LSTM-GRU Hybrid Model Training")
    from models.hybrid_model import train
    model, metrics = train("CICDDoS2019_sample")
    ok(f"Hybrid model trained — accuracy: {metrics['accuracy']:.4f}")

    target = 0.98
    if metrics['accuracy'] < target:
        warn(f"Accuracy {metrics['accuracy']:.4f} below target {target:.2f}")
        warn("Consider more epochs or hyperparameter tuning")


def stage_xai():
    header("Stage 4/5: Explainable AI (SHAP + LIME)")
    from xai.shap_analysis import run_shap
    from xai.lime_analysis import run_lime

    top_features = run_shap("CICDDoS2019_sample")
    ok(f"SHAP complete — top feature: {top_features[0]['feature'] if top_features else 'N/A'}")

    run_lime("CICDDoS2019_sample")
    ok("LIME complete")


def stage_eval():
    header("Stage 5/5: Full Evaluation & Benchmarking")
    from evaluation.benchmark import (
        comparison_summary, cross_dataset_eval,
        adversarial_robustness_test, latency_benchmark)

    comparison_summary("CICDDoS2019_sample")
    cross_dataset_eval("CICDDoS2019_sample")
    adversarial_robustness_test("CICDDoS2019_sample")
    latency_benchmark("CICDDoS2019_sample")
    ok("All evaluations complete")


def stage_sdn():
    header("Stage SDN: Software-Defined Networking Simulation")
    from sdn.ddos_controller import SDNSimulator
    sim = SDNSimulator()
    sim.simulate_traffic(n_packets=500)
    ok("SDN simulation complete")


def stage_dashboard():
    header("Dashboard: Launching Plotly Dash")
    print("  Open your browser: http://localhost:8050")
    print("  Press Ctrl+C to stop\n")
    from dashboard.app import app
    from config import DASHBOARD
    app.run(host=DASHBOARD["host"],
            port=DASHBOARD["port"],
            debug=False)


STAGES = {
    "data":      stage_data,
    "baseline":  stage_baseline,
    "hybrid":    stage_hybrid,
    "xai":       stage_xai,
    "eval":      stage_eval,
    "sdn":       stage_sdn,
    "dashboard": stage_dashboard,
}


if __name__ == "__main__":
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="IoT-DDoS Detection Pipeline")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=list(STAGES.keys()) + ["all"],
        help="Pipeline stage to run (default: all)")
    parser.add_argument(
        "--skip-dashboard", action="store_true",
        help="Skip dashboard launch when running 'all'")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.stage == "all":
        run_order = ["data", "baseline", "hybrid", "xai", "eval", "sdn"]
        if not args.skip_dashboard:
            run_order.append("dashboard")
        for stage in run_order:
            try:
                STAGES[stage]()
            except Exception as e:
                fail(f"Stage '{stage}' failed: {e}")
                import traceback; traceback.print_exc()
                print(f"  Continuing with next stage...\n")
    else:
        try:
            STAGES[args.stage]()
        except Exception as e:
            fail(f"Stage '{args.stage}' failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'━'*60}")
    print(f"  Pipeline completed in {elapsed:.0f}s")
    print(f"  Results: ./results/")
    print(f"{'━'*60}\n")
