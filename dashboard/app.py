# =============================================================================
# dashboard/app.py
# Interactive Plotly Dash dashboard for results visualisation
# Usage: python dashboard/app.py
# Open: http://localhost:8050
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from config import RESULTS_DIR, DASHBOARD

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="IoT-DDoS Detection Dashboard"
)


# ── Data loaders ──────────────────────────────────────────────────────────
def load_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default or []


def get_datasets():
    dirs = [d.name for d in (RESULTS_DIR).iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and d.name != "adversarial" and d.name != "xai"]
    return [{"label": d, "value": d} for d in dirs] or \
           [{"label": "CICDDoS2019_sample", "value": "CICDDoS2019_sample"}]


# ── Layout ────────────────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🛡 IoT-DDoS Detection Dashboard",
                    className="text-center mt-3 mb-1"),
            html.P("CNN-LSTM-GRU · Federated Learning · SHAP/LIME · SDN Mitigation",
                   className="text-center text-muted mb-3"),
        ])
    ]),

    # Metric cards
    dbc.Row(id="metric-cards", className="mb-3"),

    # Charts row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Comparison"),
                dbc.CardBody(dcc.Graph(id="model-comparison-chart"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Training History"),
                dbc.CardBody(dcc.Graph(id="training-history-chart"))
            ])
        ], width=6),
    ], className="mb-3"),

    # Charts row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("FGSM Adversarial Robustness"),
                dbc.CardBody(dcc.Graph(id="adversarial-chart"))
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Latency Benchmark"),
                dbc.CardBody(dcc.Graph(id="latency-chart"))
            ])
        ], width=6),
    ], className="mb-3"),

    # XAI row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top SHAP Feature Importances"),
                dbc.CardBody(dcc.Graph(id="shap-chart"))
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Federated Learning Rounds"),
                dbc.CardBody(dcc.Graph(id="fl-chart"))
            ])
        ], width=4),
    ], className="mb-3"),

    # Refresh
    dbc.Row([
        dbc.Col([
            dbc.Button("🔄 Refresh Data", id="refresh-btn",
                       color="primary", className="me-2"),
            html.Span("Dashboard auto-refreshes on button click",
                      className="text-muted small"),
        ], className="text-center mb-4")
    ]),

    dcc.Interval(id='interval', interval=30000, n_intervals=0),
    dcc.Store(id='ds', data="CICDDoS2019_sample"),
])

DS = "CICDDoS2019_sample"


# ── Callbacks ─────────────────────────────────────────────────────────────
@app.callback(
    Output("metric-cards", "children"),
    Input("interval", "n_intervals"),
    Input("refresh-btn", "n_clicks"),
)
def update_cards(*_):
    hybrid  = load_json(RESULTS_DIR / f"hybrid_metrics_{DS}.json", {})
    fgsm    = load_json(RESULTS_DIR / f"fgsm_robustness_{DS}.json", [])
    latency = load_json(RESULTS_DIR / f"latency_benchmark_{DS}.json", [])

    clean_acc  = hybrid.get("accuracy", 0)
    clean_f1   = hybrid.get("f1_score", 0)
    clean_auc  = hybrid.get("auc_roc", 0)
    base_acc   = fgsm[0]["accuracy"] if fgsm else 0
    adv_acc    = fgsm[-1]["accuracy"] if fgsm else 0
    best_lat   = min((r["ms_per_sample"] for r in latency), default=0)

    def card(title, value, subtitle, color):
        return dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5(title, className="text-muted small mb-1"),
                html.H3(value, className=f"text-{color} mb-1"),
                html.P(subtitle, className="text-muted small mb-0"),
            ])
        ], className="text-center"), width=2)

    return [
        card("Accuracy",    f"{clean_acc:.2%}", "CNN-LSTM-GRU",       "success"),
        card("F1-Score",    f"{clean_f1:.4f}",  "Weighted",            "info"),
        card("AUC-ROC",     f"{clean_auc:.4f}", "Test set",            "primary"),
        card("Clean Acc",   f"{base_acc:.2%}",  "No attack",           "warning"),
        card("Adv Acc",     f"{adv_acc:.2%}",   "FGSM ε=0.1",          "danger"),
        card("Latency",     f"{best_lat:.2f}ms","Per sample (best)",   "secondary"),
    ]


@app.callback(
    Output("model-comparison-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_comparison(_):
    hybrid  = load_json(RESULTS_DIR / f"hybrid_metrics_{DS}.json",  {})
    base    = load_json(RESULTS_DIR / f"baseline_metrics_{DS}.json", [])

    all_m   = [hybrid] + base if hybrid else base
    if not all_m:
        return go.Figure().add_annotation(text="Run training first",
                                          showarrow=False)
    models  = [m.get("model","?") for m in all_m]
    acc     = [m.get("accuracy",0) for m in all_m]
    f1      = [m.get("f1_score",0) for m in all_m]

    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=models, y=acc, marker_color="steelblue")
    fig.add_bar(name="F1-Score", x=models, y=f1,  marker_color="coral")
    fig.update_layout(barmode='group', yaxis_range=[0.8, 1.01],
                      template="plotly_dark", margin=dict(t=10, b=80),
                      height=300)
    return fig


@app.callback(
    Output("adversarial-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_adversarial(_):
    data = load_json(RESULTS_DIR / f"fgsm_robustness_{DS}.json")
    if not data:
        return go.Figure().add_annotation(text="Run evaluation/benchmark.py",
                                          showarrow=False)
    eps  = [d["epsilon"] for d in data]
    acc  = [d["accuracy"] for d in data]
    f1   = [d["f1"]       for d in data]

    fig = go.Figure()
    fig.add_scatter(x=eps, y=acc, mode='lines+markers',
                    name='Accuracy', line=dict(color='steelblue'))
    fig.add_scatter(x=eps, y=f1,  mode='lines+markers',
                    name='F1-Score', line=dict(color='coral'))
    fig.add_hline(y=0.98, line_dash="dot",
                  annotation_text="98% target", line_color="green")
    fig.update_layout(xaxis_title="FGSM ε", yaxis_title="Score",
                      template="plotly_dark", margin=dict(t=10),
                      height=300)
    return fig


@app.callback(
    Output("latency-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_latency(_):
    data = load_json(RESULTS_DIR / f"latency_benchmark_{DS}.json")
    if not data:
        return go.Figure().add_annotation(text="Run evaluation/benchmark.py",
                                          showarrow=False)
    bs  = [d["batch_size"]    for d in data]
    ms  = [d["ms_per_sample"] for d in data]

    fig = go.Figure()
    fig.add_scatter(x=bs, y=ms, mode='lines+markers',
                    fill='tozeroy', fillcolor='rgba(70,130,180,0.2)',
                    line=dict(color='steelblue'))
    fig.add_hline(y=100, line_dash="dot",
                  annotation_text="100ms target", line_color="red")
    fig.update_layout(xaxis_title="Batch size", yaxis_title="ms/sample",
                      template="plotly_dark", margin=dict(t=10),
                      height=300)
    return fig


@app.callback(
    Output("shap-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_shap(_):
    data = load_json(RESULTS_DIR / "xai" / DS / "top_features.json")
    if not data:
        return go.Figure().add_annotation(text="Run xai/shap_analysis.py",
                                          showarrow=False)
    data = data[:15]
    feats = [d["feature"]    for d in reversed(data)]
    imps  = [d["importance"] for d in reversed(data)]

    fig = go.Figure(go.Bar(x=imps, y=feats, orientation='h',
                           marker_color='teal'))
    fig.update_layout(xaxis_title="Mean |SHAP|",
                      template="plotly_dark",
                      margin=dict(t=10, l=200), height=350)
    return fig


@app.callback(
    Output("fl-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_fl(_):
    data = load_json(RESULTS_DIR / "federated_rounds.json")
    if not data:
        return go.Figure().add_annotation(text="Run federated learning",
                                          showarrow=False)
    rounds = [d["round"]    for d in data]
    accs   = [d.get("accuracy", 0) for d in data]
    losses = [d.get("loss",     0) for d in data]

    fig = go.Figure()
    fig.add_scatter(x=rounds, y=accs, mode='lines+markers',
                    name='Accuracy', line=dict(color='green'))
    fig.update_layout(xaxis_title="Round", yaxis_title="Accuracy",
                      template="plotly_dark", margin=dict(t=10),
                      height=300)
    return fig


@app.callback(
    Output("training-history-chart", "figure"),
    Input("interval", "n_intervals"),
)
def update_history(_):
    # Load from TensorBoard logs if available
    return go.Figure().add_annotation(
        text="After training, view TensorBoard:\ntensorboard --logdir logs/tensorboard",
        showarrow=False,
        font=dict(size=12)
    ).update_layout(template="plotly_dark", height=300)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  IoT-DDoS Dashboard")
    print(f"  Open: http://localhost:{DASHBOARD['port']}")
    print("="*60 + "\n")
    app.run(host=DASHBOARD["host"],
            port=DASHBOARD["port"],
            debug=DASHBOARD["debug"])
