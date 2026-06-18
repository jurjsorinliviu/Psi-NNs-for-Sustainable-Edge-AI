#!/bin/bash
set -e

echo "🚀 Setting up Hardware Right-Sizing Edge AI development environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install main project dependencies
echo "📦 Installing main project dependencies..."
pip install -r requirements.txt

# Install PSI-HDL implementation dependencies
echo "📦 Installing PSI-HDL implementation dependencies..."
pip install -r PSI-HDL-implementation/requirements.txt

# Install Jupyter kernel
echo "📦 Installing Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name hardware-right-sizing-edge-ai --display-name "Hardware Right-Sizing Edge AI"

# Verify PyTorch installation
echo "🔍 Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Pre-download any model weights or data if needed
echo "📁 Creating results directory structure..."
mkdir -p results

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "   - Run experiment (Burgers PDE, three regimes): python experiments/three_regime_burgers_experiment.py"
echo "   - Generate Figure 2 (decomposition framework): python generate_figure2_decomposition.py"
echo "   - Generate Figure 3 (Burgers κ-sweep curve):   python generate_figure3_kappa_sweep.py"
echo "   - Open Jupyter: jupyter notebook"
echo ""
