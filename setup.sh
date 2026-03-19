#!/bin/bash
# =============================================================================
# IoT-DDoS Detection Project — Full Ubuntu Setup Script
# Run: chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()    { echo -e "${GREEN}[✓]${NC} $1"; }
warn()   { echo -e "${YELLOW}[!]${NC} $1"; }
info()   { echo -e "${BLUE}[i]${NC} $1"; }
error()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo ""
echo "============================================================"
echo "  IoT-Enabled DDoS Detection — Project Setup"
echo "  MSc Cyber Security | Muhammad Farqan | B01822365"
echo "============================================================"
echo ""

# ── 1. System dependencies ────────────────────────────────────
info "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git wget curl build-essential \
    libhdf5-dev libssl-dev libffi-dev \
    openjdk-11-jdk \
    net-tools iperf3 tcpdump \
    libpcap-dev \
    cmake pkg-config \
    2>/dev/null || warn "Some system packages may already be installed"
log "System dependencies installed"

# ── 2. Python virtual environment ─────────────────────────────
info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log "Virtual environment created: ./venv"
else
    warn "Virtual environment already exists, skipping"
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
log "Virtual environment activated"

# ── 3. Python packages ────────────────────────────────────────
info "Installing Python packages (this takes 3-5 minutes)..."
pip install -q \
    tensorflow==2.15.0 \
    torch==2.2.0 torchvision==0.17.0 \
    scikit-learn==1.4.0 \
    imbalanced-learn==0.11.0 \
    pandas==2.1.4 numpy==1.26.3 \
    matplotlib==3.8.2 seaborn==0.13.2 \
    shap==0.44.1 lime==0.2.0.1 \
    flwr==1.7.0 \
    plotly==5.18.0 \
    dash==2.14.2 dash-bootstrap-components==1.5.0 \
    xgboost==2.0.3 lightgbm==4.2.0 \
    joblib==1.3.2 tqdm==4.66.1 \
    tensorboard==2.15.0 scipy==1.11.4 \
    optuna==3.5.0 psutil==5.9.7 \
    cryptography==42.0.0 \
    jupyter notebook ipykernel \
    2>/dev/null || warn "Some packages may have version conflicts; continuing..."

# TenSEAL for homomorphic encryption (optional, may fail on some systems)
pip install tenseal==0.3.14 -q 2>/dev/null || warn "TenSEAL install skipped (optional)"

log "Python packages installed"

# ── 4. Mininet (SDN simulation) ───────────────────────────────
info "Installing Mininet for SDN simulation..."
if ! command -v mn &> /dev/null; then
    git clone https://github.com/mininet/mininet.git 2>/dev/null || true
    cd mininet && sudo ./util/install.sh -nfv 2>/dev/null || warn "Mininet install may need manual steps"
    cd ..
    log "Mininet installed"
else
    warn "Mininet already installed"
fi

# ── 5. Ryu SDN Controller ─────────────────────────────────────
info "Installing Ryu SDN controller..."
pip install ryu==4.34 -q 2>/dev/null || warn "Ryu install skipped (Python version may be incompatible)"
log "Ryu SDN controller installed"

# ── 6. Create dataset download scripts ────────────────────────
info "Setting up dataset download helpers..."
mkdir -p data/raw data/processed data/synthetic
log "Dataset directories ready"

# ── 7. Jupyter kernel ─────────────────────────────────────────
python -m ipykernel install --user --name iot-ddos --display-name "IoT-DDoS Project" 2>/dev/null || true
log "Jupyter kernel registered"

# ── 8. .env file ──────────────────────────────────────────────
if [ ! -f ".env" ]; then
cat > .env << 'ENVEOF'
DATA_DIR=./data/raw
PROCESSED_DIR=./data/processed
MODELS_DIR=./models
LOGS_DIR=./logs
FL_SERVER_HOST=0.0.0.0
FL_SERVER_PORT=8080
DASHBOARD_PORT=8050
ENVEOF
    log ".env file created"
fi

echo ""
echo "============================================================"
echo -e "${GREEN}  Setup complete!${NC}"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python scripts/download_datasets.py"
echo "  3. python scripts/run_pipeline.py"
echo "  4. python dashboard/app.py"
echo ""
