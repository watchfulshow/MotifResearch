# Main Pipeline Notebook - A100 Optimized Version

## Quick Start

### Option 1: Google Colab (Recommended)
1. Open `notebooks/main_pipeline.ipynb` in Google Colab
2. Select GPU: **Runtime → Change runtime type → A100 GPU** (requires Colab Pro)
3. Run all cells: **Runtime → Run all**

### Option 2: Local/Other GPUs
- The notebook automatically detects your GPU and adjusts configuration
- Supported: A100, V100, T4, or CPU fallback
- Automatically scales model size and batch size

## What's New in This Version

### 🚀 A100 GPU Support (~280M Parameters)
- **24 transformer layers** (vs 10 in T4 version)
- **1024 model dimension** (vs 256 in T4 version)
- **16 attention heads** (vs 8 in T4 version)
- **Batch size 256** (vs 64 in T4 version)
- **500K samples/epoch** (vs 100K in T4 version)

### 📊 Enhanced Statistical Analysis
- Bootstrap confidence intervals (95% CI)
- Kolmogorov-Smirnov tests for distribution comparison
- Chi-square tests for cross-species comparisons
- Mann-Whitney U tests for non-parametric comparisons
- Hypothesis testing summaries

### 📈 Publication-Quality Outputs (300 DPI)
All figures saved at 300 DPI:
- Training curves with error markers
- Cross-species analysis with statistical tests
- Grammar discovery visualizations
- Generative validation with test overlays
- Comprehensive 6-panel summary

### 📋 Publication-Ready Tables
Six CSV tables generated:
1. Model performance metrics
2. Test set detailed metrics
3. Cross-species analysis results
4. Grammar discovery summary
5. Generative validation statistics
6. Model configuration summary

### 🧬 Real Data Support
- Option to download real ENCODE cCRE data
- Ensembl API integration framework
- Synthetic data fallback for demo mode

## GPU Configuration Matrix

| GPU Type | Parameters | Layers | Dimension | Batch Size | Time (50 epochs) |
|----------|------------|---------|-----------|------------|------------------|
| A100     | ~280M      | 24      | 1024      | 256        | 6-10 hours       |
| V100     | ~120M      | 16      | 768       | 128        | 8-12 hours       |
| T4       | ~27M       | 10      | 256       | 64         | 10-15 hours      |

## Expected VRAM Usage

- **A100**: ~40-50GB peak (training), ~20-25GB (inference)
- **V100**: ~25-30GB peak (training), ~12-15GB (inference)
- **T4**: ~12-14GB peak (training), ~6-8GB (inference)

## Notebook Structure (50 Cells)

1. **Setup** (Cells 1-6): Environment, GPU detection, dependencies
2. **Configuration** (Cells 7-8): GPU-specific model configs
3. **Data** (Cells 9-16): Acquisition, encoding, preparation
4. **Model** (Cells 17-20): Architecture definition
5. **Training** (Cells 21-30): Training loop, evaluation
6. **Innovation #1** (Cells 31-34): Cross-species evolutionary analysis
7. **Innovation #2** (Cells 35-38): Hierarchical grammar discovery
8. **Innovation #3** (Cells 39-42): Generative validation
9. **Summary** (Cells 43-44): Comprehensive results figure
10. **Tables** (Cells 45-46): Publication-ready tables
11. **Save** (Cells 47-50): Results compilation and storage

## Three Key Research Innovations

### Innovation #1: Cross-Species Evolutionary Analysis
Tests whether the model has learned ancient vs recent regulatory grammar by comparing repair success across vertebrate species separated by millions of years of evolution.

**Statistical Tests**:
- Chi-square tests for pairwise species comparisons
- Wilson score confidence intervals
- Evolutionary distance correlation

### Innovation #2: Hierarchical Grammar Discovery
Discovers which regulatory motifs must work together (synergistic) vs work as backups (redundant) by systematically testing all pairwise combinations.

**Methods**:
- Integrated Gradients for importance extraction
- Pairwise interaction testing
- Classification into synergy/redundancy/independence

### Innovation #3: Generative Validation
Proves the model understands regulatory grammar causally (not just correlation) by generating functional enhancers from random DNA.

**Statistical Validation**:
- Kolmogorov-Smirnov test for GC distribution
- Mann-Whitney U test for motif content
- T-tests for mean comparisons
- Overall pass/fail determination

## Output Files

### Figures (300 DPI)
- `training_curves.png`: Loss and accuracy progression
- `cross_species_repair.png`: Evolutionary analysis with stats
- `grammar_discovery.png`: Motif interactions
- `generative_validation.png`: Generation statistics
- `comprehensive_summary.png`: All-in-one results figure

### Tables (CSV)
- `model_performance.csv`: Train/val/test metrics
- `test_metrics.csv`: Detailed test set analysis
- `cross_species.csv`: Species-specific repair rates
- `grammar_summary.csv`: Motif interaction types
- `generative_validation.csv`: Generation statistics
- `model_config.csv`: Complete configuration

### Data & Models
- `dataset.npz`: Processed sequences (compressed)
- `checkpoint_best.pt`: Best model weights
- `checkpoint_latest.pt`: Most recent checkpoint
- Various result pickles for detailed analysis

## Resuming After Timeout

If Google Colab disconnects:

1. Re-run cells 1-8 (Setup and Configuration)
2. Skip data generation if files exist
3. Re-create model (cell 18)
4. Run training - it auto-resumes from latest checkpoint

Check for checkpoints:
```python
import os
checkpoint_path = f"{PROJECT_DIR}/checkpoints/checkpoint_latest.pt"
if os.path.exists(checkpoint_path):
    print("Checkpoint found! Training will resume.")
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in CONFIG
- Reduce `samples_per_epoch`
- Enable gradient checkpointing (T4 config)

### Slow Training
- Check GPU is actually being used: `torch.cuda.is_available()`
- Verify mixed precision is enabled
- Reduce validation frequency

### API Rate Limits (Ensembl)
- Set `USE_REAL_DATA = False` for demo mode
- Uses synthetic data instead
- Results are qualitatively similar for testing

## Citation

If you use this notebook in your research:

```bibtex
@software{enhancer_repair_research,
  title={Enhancer Repair Research: AI-Driven Regulatory Grammar Discovery},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/MotifResearch}
}
```

## License

MIT License - See LICENSE file for details

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Check the NOTEBOOK_REGENERATION_SUMMARY.md file
- **Updates**: Star/watch the repository for updates

---

**Last Updated**: January 2025
**Notebook Version**: 2.0 (A100 Optimized)
**Total Cells**: 50
**Estimated Runtime**: 6-15 hours depending on GPU
