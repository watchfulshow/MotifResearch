"""
Visualization Utilities for Enhancer Repair Research

Creates publication-quality figures for:
    - Training curves
    - Cross-species analysis
    - Grammar discovery
    - Generative validation
    - Attention analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_plotting_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with train_loss, val_loss, train_acc, val_acc
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training and Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        class_names: Names for classes
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Background', 'Enhancer']

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_cross_species_results(
    results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cross-species repair analysis.

    Args:
        results: Dictionary from test_cross_species_repair
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to species only (exclude 'summary')
    species = [k for k in results.keys() if k != 'summary']
    repair_rates = [results[s]['repair_rate'] for s in species]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(species)]

    bars = ax.bar(species, repair_rates, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    ax.set_ylabel('Repair Success Rate (%)', fontsize=13)
    ax.set_title('Cross-Species Repair Analysis: Evolutionary Conservation',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add evolutionary timeline annotation
    ax.text(
        0.5, -0.12,
        '<-- 400 million years of evolution -->',
        ha='center', fontsize=11, style='italic',
        transform=ax.transAxes
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_grammar_interactions(
    interactions: List[Dict],
    n_motifs: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot motif interaction network as heatmap.

    Args:
        interactions: List of interaction dictionaries
        n_motifs: Total number of motifs
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Create adjacency matrix
    adj_matrix = np.zeros((n_motifs, n_motifs))

    for interaction in interactions:
        i = interaction['motif1']
        j = interaction['motif2']

        if interaction['interaction'] == 'SYNERGY':
            adj_matrix[i, j] = 2
            adj_matrix[j, i] = 2
        elif interaction['interaction'] == 'REDUNDANT':
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(adj_matrix, cmap='RdYlGn', vmin=0, vmax=2)
    ax.set_title('Motif Interaction Network', fontsize=13, fontweight='bold')
    ax.set_xlabel('Motif Index')
    ax.set_ylabel('Motif Index')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Not tested', 'Redundant', 'Synergistic'])

    # Pie chart
    ax = axes[1]
    interaction_types = ['Synergistic', 'Redundant', 'Independent']
    counts = [
        len([x for x in interactions if x['interaction'] == 'SYNERGY']),
        len([x for x in interactions if x['interaction'] == 'REDUNDANT']),
        len([x for x in interactions if x['interaction'] == 'INDEPENDENT'])
    ]

    colors = ['#e74c3c', '#f39c12', '#95a5a6']
    ax.pie(counts, labels=interaction_types, autopct='%1.1f%%',
           colors=colors, explode=(0.05, 0.05, 0))
    ax.set_title('Grammar Rule Distribution', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_generation_trajectories(
    trajectories: List[List[float]],
    target_prob: float = 0.80,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot enhancer generation trajectories.

    Args:
        trajectories: List of probability trajectories
        target_prob: Target probability threshold
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual trajectories
    for trajectory in trajectories[:50]:
        ax.plot(trajectory, alpha=0.2, color='blue', linewidth=1)

    # Plot average trajectory
    if len(trajectories) > 0:
        max_len = max([len(t) for t in trajectories])
        avg_trajectory = []
        for step in range(max_len):
            step_values = [t[step] for t in trajectories if len(t) > step]
            avg_trajectory.append(np.mean(step_values))

        ax.plot(avg_trajectory, color='red', linewidth=3, label='Average', zorder=10)

    ax.axhline(y=target_prob, color='green', linestyle='--', linewidth=2,
               label=f'Target ({target_prob*100:.0f}%)', zorder=5)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Enhancer Probability', fontsize=12)
    ax.set_title('Enhancer Generation Trajectories', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_generative_validation(
    validation_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot generative validation comparison.

    Args:
        validation_results: Dictionary from validate_generated_sequences
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # GC content comparison
    ax = axes[0]
    ax.hist(validation_results['real_gc'], bins=30, alpha=0.6,
            label='Real Enhancers', color='blue', edgecolor='black')
    ax.hist(validation_results['generated_gc'], bins=30, alpha=0.6,
            label='Generated', color='red', edgecolor='black')

    ax.set_xlabel('GC Content', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f"GC Content Distribution (p={validation_results['gc_pvalue']:.4f})",
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Motif count comparison
    ax = axes[1]
    ax.hist(validation_results['real_motifs'], bins=range(0, max(validation_results['real_motifs'])+2),
            alpha=0.6, label='Real Enhancers', color='blue', edgecolor='black')
    ax.hist(validation_results['generated_motifs'], bins=range(0, max(validation_results['generated_motifs'])+2),
            alpha=0.6, label='Generated', color='red', edgecolor='black')

    ax.set_xlabel('Motif Count', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f"Motif Distribution (p={validation_results['motif_pvalue']:.4f})",
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_importance_profile(
    importance_scores: List[np.ndarray],
    motif_regions: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot position importance profile.

    Args:
        importance_scores: List of importance arrays per sequence
        motif_regions: Optional list of (start, end) tuples
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Calculate average importance
    avg_importance = np.mean(importance_scores, axis=0)
    std_importance = np.std(importance_scores, axis=0)

    positions = range(len(avg_importance))

    # Plot average with confidence band
    ax.plot(positions, avg_importance, linewidth=2, color='darkblue')
    ax.fill_between(
        positions,
        avg_importance - std_importance,
        avg_importance + std_importance,
        alpha=0.3, color='skyblue'
    )

    # Highlight motif regions
    if motif_regions:
        for start, end in motif_regions:
            ax.axvspan(start, end, alpha=0.2, color='red')

    ax.set_xlabel('Sequence Position (bp)', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title('Position Importance Profile (Integrated Gradients)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_comprehensive_figure(
    history: Dict,
    test_results: Dict,
    cross_species_results: Dict,
    grammar_interactions: List[Dict],
    validation_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create 6-panel summary figure.

    Args:
        history: Training history
        test_results: Test evaluation results
        cross_species_results: Cross-species analysis results
        grammar_interactions: Grammar discovery results
        validation_results: Generative validation results
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Training curves (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Performance metrics
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['Train', 'Val', 'Test']
    values = [
        history['train_acc'][-1] if history['train_acc'] else 0,
        history['val_acc'][-1] if history['val_acc'] else 0,
        test_results.get('test_accuracy', 0)
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Cross-species
    ax3 = fig.add_subplot(gs[1, 0])
    species = [k for k in cross_species_results.keys() if k != 'summary']
    repair_rates = [cross_species_results[s]['repair_rate'] for s in species]
    colors_species = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(species)]
    ax3.bar(species, repair_rates, color=colors_species, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Repair Success (%)')
    ax3.set_title('Cross-Species Conservation', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Grammar interactions
    ax4 = fig.add_subplot(gs[1, 1])
    interaction_types = ['Synergistic', 'Redundant', 'Independent']
    counts = [
        len([x for x in grammar_interactions if x['interaction'] == 'SYNERGY']),
        len([x for x in grammar_interactions if x['interaction'] == 'REDUNDANT']),
        len([x for x in grammar_interactions if x['interaction'] == 'INDEPENDENT'])
    ]
    if sum(counts) > 0:
        ax4.pie(counts, labels=interaction_types, autopct='%1.1f%%',
                colors=['#e74c3c', '#f39c12', '#95a5a6'])
    ax4.set_title('Motif Grammar Types', fontweight='bold')

    # Panel 5: Generation success
    ax5 = fig.add_subplot(gs[1, 2])
    gen_stats = ['Success Rate', 'GC Match', 'Motif Match']
    gen_values = [
        validation_results.get('success_rate', 0) if 'success_rate' in validation_results else 50,
        100 if validation_results.get('gc_pvalue', 0) > 0.05 else 50,
        100 if validation_results.get('motif_pvalue', 0) > 0.05 else 50
    ]
    ax5.bar(gen_stats, gen_values, color=['#9b59b6', '#1abc9c', '#e67e22'],
            alpha=0.8, edgecolor='black')
    ax5.set_ylim(0, 100)
    ax5.set_ylabel('Score')
    ax5.set_title('Generative Validation', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: GC content comparison (spans full width)
    ax6 = fig.add_subplot(gs[2, :])
    if 'real_gc' in validation_results and 'generated_gc' in validation_results:
        ax6.hist(validation_results['real_gc'], bins=30, alpha=0.6,
                label='Real Enhancers', color='blue', edgecolor='black')
        ax6.hist(validation_results['generated_gc'], bins=30, alpha=0.6,
                label='Generated', color='red', edgecolor='black')
        ax6.set_xlabel('GC Content')
        ax6.set_ylabel('Count')
        ax6.set_title('Generative Validation: GC Content Distribution', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('ENHANCER REPAIR RESEARCH: COMPREHENSIVE ANALYSIS',
                fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    setup_plotting_style()
    print("Visualization module loaded successfully")
