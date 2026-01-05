# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Web-based visualization dashboard
- Additional species support (frog, lamprey)
- Pre-trained model weights release
- Integration with JASPAR motif database

## [1.0.0] - 2025-01-05

### Added
- Initial release of Enhancer Repair Research framework
- Hybrid CNN-Transformer model architecture (27M parameters)
- Data pipeline for ENCODE and Ensembl data sources
- Cross-species evolutionary analysis (Human, Mouse, Zebrafish, Chicken)
- Hierarchical grammar discovery with pairwise motif testing
- Generative validation for synthetic enhancer creation
- Comprehensive checkpointing system for Colab compatibility
- Mixed precision training support (FP16)
- Stochastic epoch sampling for memory efficiency
- Integrated Gradients for model interpretability
- Complete Jupyter notebook for Google Colab execution
- Visualization utilities for all analysis types

### Model Architecture
- CNN encoder with 3 convolutional layers (kernels: 7, 11, 15)
- 10-layer Transformer encoder with 8 attention heads
- 256-dimensional model embeddings
- Global average pooling for sequence-level predictions
- Dropout regularization (0.1)

### Training Features
- AdamW optimizer with weight decay
- ReduceLROnPlateau learning rate scheduling
- Gradient clipping (max norm 0.5)
- Early stopping with patience monitoring
- Gradient accumulation for effective larger batch sizes

### Data Support
- ENCODE cCRE annotations (Human GRCh38, Mouse mm10)
- Ensembl regulatory features (Zebrafish GRCz11, Chicken GRCg6a)
- One-hot encoding for DNA sequences
- Stratified train/validation/test splits
- Background sequence generation with configurable GC content

### Analysis Pipelines
- **Cross-Species:** Damage-repair testing across 4 vertebrate species
- **Grammar Discovery:** Pairwise motif interaction testing (synergy/redundancy)
- **Generative:** Random-to-enhancer evolution with validation

### Documentation
- Comprehensive README with quick start guide
- Detailed technical documentation
- Code comments and docstrings
- Troubleshooting guide

## [0.1.0] - 2024-12-28

### Added
- Initial project structure
- Basic model prototype
- Data loading utilities

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-01-05 | Full release with all innovations |
| 0.1.0 | 2024-12-28 | Initial prototype |
