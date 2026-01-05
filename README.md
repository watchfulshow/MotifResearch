# Enhancer Repair Research: AI-Driven Regulatory Grammar Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

## Overview

**Decoding Regulatory Grammar Through Neural Network Repair: An AI-Driven Approach to Discovering Essential Gene Control Elements**

This project introduces a novel methodology for discovering gene regulatory grammar by using neural network "repair" as a biological probe. Rather than simply identifying important DNA regions through correlation, we test causality by deliberately damaging sequences and attempting computational repair—failure to repair indicates biological essentiality.

### Key Innovation

Just as natural language has syntax rules (subject-verb-object), DNA has combinatorial grammar rules. By teaching an AI to "fix" broken DNA and observing what it cannot repair, we systematically map which regulatory elements are truly essential versus redundant.

## Research Questions

1. **Cross-Species Conservation:** Which regulatory motifs are conserved across 400 million years of vertebrate evolution?
2. **Combinatorial Grammar:** Which motifs work synergistically versus independently?
3. **Generative Understanding:** Can the model create functional enhancers from scratch, proving causal understanding?

## Methodology

```
Phase 1: Training
├── Train hybrid CNN-Transformer on enhancer sequences
├── Model learns to distinguish functional enhancers from background DNA
└── Achieves 75-85% classification accuracy

Phase 2: Repair-Based Discovery
├── Damage enhancer sequences (delete/mutate critical regions)
├── Attempt computational "repair" using gradient descent
├── Success/failure reveals biological essentiality
└── Map repair success across 4 vertebrate species

Phase 3: Grammar Discovery
├── Test pairwise motif interactions
├── Identify synergistic (both needed) vs redundant (backup) motifs
└── Build hierarchical grammar network

Phase 4: Generative Validation
├── Start with random DNA sequence
├── Evolve into functional enhancer using repair algorithm
└── Validate that generated sequences match real biology
```

## Project Structure

```
MotifResearch/
├── README.md                   # This file
├── CHANGELOG.md                # Version history
├── requirements.txt            # Python dependencies
├── configs/
│   └── config.yaml             # Training configuration
├── src/
│   ├── __init__.py
│   ├── model.py                # CNN-Transformer architecture
│   ├── data.py                 # Data loading and processing
│   ├── training.py             # Training loop with checkpointing
│   ├── repair.py               # Repair-based analysis
│   ├── grammar.py              # Grammar discovery
│   ├── generative.py           # Generative validation
│   └── visualization.py        # Plotting utilities
├── notebooks/
│   └── main_pipeline.ipynb     # Main Google Colab notebook
├── data/
│   ├── raw/                    # Downloaded ENCODE files
│   ├── processed/              # Processed sequences
│   └── splits/                 # Train/val/test indices
├── models/                     # Saved model weights
├── checkpoints/                # Training checkpoints
├── results/
│   ├── cross_species/          # Cross-species analysis
│   ├── grammar/                # Grammar discovery
│   ├── generative/             # Generated sequences
│   └── attention/              # Attention analysis
├── figures/                    # All visualizations
└── logs/                       # Training logs
```

## Quick Start (Google Colab)

### Option 1: Run the Notebook Directly

1. Open `notebooks/main_pipeline.ipynb` in Google Colab
2. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU`
3. Run all cells sequentially

### Option 2: Clone and Run

```python
# In Google Colab
!git clone https://github.com/YOUR_USERNAME/MotifResearch.git
%cd MotifResearch

# Install dependencies
!pip install -r requirements.txt

# Run the notebook
```

## Hardware Requirements

### Minimum (Google Colab Free Tier)
- **GPU:** NVIDIA T4 (16 GB VRAM) ✓
- **RAM:** 12.7 GB (sufficient with optimization)
- **Storage:** Google Drive integration
- **Limitations:** 2-3 hour timeout (addressed with checkpointing)

### Recommended (Full-Scale)
- **GPU:** NVIDIA GPU with 16+ GB VRAM
- **RAM:** 32 GB
- **Storage:** 100 GB free space

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MotifResearch.git
cd MotifResearch

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Sources

### ENCODE Project (Human & Mouse)
- **Human cCREs:** ~1.3 million regulatory regions (GRCh38)
- **Mouse cCREs:** ~900,000 regulatory regions (mm10)
- **Format:** BED files with genomic coordinates

### Ensembl (Zebrafish & Chicken)
- **Zebrafish:** ~200,000 regulatory features (GRCz11)
- **Chicken:** ~150,000 regulatory features (GRCg6a)
- **Access:** REST API

## Model Architecture

**Hybrid CNN-Transformer (27M parameters)**

```
Input: One-hot DNA (batch, 4 channels, 200 positions)
    ↓
CNN Encoder (3 conv layers)
├── Conv1D(kernel=7):  Detects 7bp motifs
├── Conv1D(kernel=11): Detects 11bp motifs
└── Conv1D(kernel=15): Detects 15bp motifs
    ↓
Transformer Encoder (10 layers, 8 heads)
├── Multi-Head Self-Attention
├── Feedforward Network
└── Layer Normalization
    ↓
Classification Head
    ↓
Output: Probability [background, enhancer]
```

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| Test Accuracy | 75-85% |
| ROC AUC | 0.85-0.92 |
| Cross-Species Conservation | 60-80% |
| Grammar Synergy Detection | 5-10% of pairs |
| Generation Success Rate | 40-70% |

## Key Features

- **Checkpoint System:** Resume training after Colab timeouts
- **Mixed Precision Training:** Faster training with FP16
- **Stochastic Sampling:** Memory-efficient training on large datasets
- **Integrated Gradients:** Interpretable motif importance
- **Cross-Species Analysis:** Evolutionary conservation mapping

## Citation

If you use this code in your research, please cite:

```bibtex
@software{enhancer_repair_research,
  title={Enhancer Repair Research: AI-Driven Regulatory Grammar Discovery},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/MotifResearch}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ENCODE Project for regulatory element annotations
- Ensembl for genomic data and APIs
- PyTorch team for deep learning framework
- Google Colab for free GPU access

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Document Version:** 1.0
**Last Updated:** January 2025
