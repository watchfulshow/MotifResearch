"""
Enhancer Repair Research
========================

AI-Driven Regulatory Grammar Discovery through Neural Network Repair

This package provides tools for:
- Training CNN-Transformer models on enhancer sequences
- Cross-species evolutionary analysis
- Hierarchical grammar discovery
- Generative validation of regulatory elements

Modules:
    model: Neural network architecture
    data: Data loading and preprocessing
    training: Training loop with checkpointing
    repair: Repair-based analysis
    grammar: Grammar discovery pipelines
    generative: Generative validation
    visualization: Plotting utilities
"""

__version__ = "1.0.0"
__author__ = "Enhancer Repair Research Team"

from .model import HybridEnhancerModel, CNNEncoder, PositionalEncoding
from .data import EnhancerDataset, one_hot_encode, decode_sequence
from .training import train_model, evaluate_model, save_checkpoint, load_checkpoint
from .repair import damage_sequence, attempt_repair, test_cross_species_repair
from .grammar import extract_important_regions, discover_grammar_rules
from .generative import generate_enhancer_from_random, validate_generated_sequences

__all__ = [
    # Model
    "HybridEnhancerModel",
    "CNNEncoder",
    "PositionalEncoding",
    # Data
    "EnhancerDataset",
    "one_hot_encode",
    "decode_sequence",
    # Training
    "train_model",
    "evaluate_model",
    "save_checkpoint",
    "load_checkpoint",
    # Repair
    "damage_sequence",
    "attempt_repair",
    "test_cross_species_repair",
    # Grammar
    "extract_important_regions",
    "discover_grammar_rules",
    # Generative
    "generate_enhancer_from_random",
    "validate_generated_sequences",
]
