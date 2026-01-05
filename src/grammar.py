"""
Hierarchical Grammar Discovery for Enhancer Research

Innovation #2: Discovers which motifs work synergistically vs independently.

Method:
    1. Extract important motif regions using Integrated Gradients
    2. Test pairwise combinations
    3. Classify interactions:
        - SYNERGY: Both motifs needed
        - REDUNDANT: Either motif works (backup)
        - INDEPENDENT: No interaction
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from tqdm import tqdm

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum not installed. Integrated Gradients unavailable.")

from .repair import attempt_repair, damage_region


def extract_important_regions(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    n_sequences: int = 100,
    n_ig_steps: int = 30,
    importance_percentile: float = 75,
    min_motif_distance: int = 6,
    motif_window: int = 4
) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """
    Extract high-importance regions using Integrated Gradients.

    Integrated Gradients computes attribution by integrating gradients
    along a path from baseline (all zeros) to input.

    Args:
        model: Trained model
        sequences: Input sequences (N, 4, 200)
        device: Torch device
        n_sequences: Number of sequences to analyze
        n_ig_steps: Integration steps for IG
        importance_percentile: Percentile threshold for peaks
        min_motif_distance: Minimum distance between motifs
        motif_window: Window size around peaks

    Returns:
        motif_regions: List of (start, end) tuples
        importance_scores: List of importance arrays per sequence
    """
    if not CAPTUM_AVAILABLE:
        print("Error: captum required for Integrated Gradients")
        print("Install with: pip install captum")
        return [], []

    model.eval()

    # Wrapper for captum (needs single output)
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            # Return enhancer class probability
            return outputs[:, 1]

    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)

    all_importances = []

    print("Extracting motif regions using Integrated Gradients...")

    n_analyze = min(n_sequences, len(sequences))

    for i in tqdm(range(n_analyze), desc="Analyzing"):
        seq = sequences[i]
        seq_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
        seq_tensor.requires_grad_(True)

        # Baseline: all zeros
        baseline = torch.zeros_like(seq_tensor)

        try:
            # Compute attributions
            attributions = ig.attribute(
                seq_tensor,
                baselines=baseline,
                n_steps=n_ig_steps
            )

            # Sum importance across nucleotide channels
            importance = attributions.abs().sum(dim=1).squeeze().detach().cpu().numpy()
            all_importances.append(importance)
        except Exception as e:
            print(f"Warning: IG failed for sequence {i}: {e}")
            continue

    if len(all_importances) == 0:
        print("No importances computed!")
        return [], []

    # Average importance across sequences
    avg_importance = np.mean(all_importances, axis=0)  # Shape: (200,)

    # Find peaks (candidate motif regions)
    threshold = np.percentile(avg_importance, importance_percentile)
    peaks, properties = find_peaks(
        avg_importance,
        height=threshold,
        distance=min_motif_distance
    )

    # Define motif regions around peaks
    motif_regions = []
    for peak in peaks:
        start = max(0, peak - motif_window)
        end = min(200, peak + motif_window + 1)
        motif_regions.append((start, end))

    print(f"   Found {len(motif_regions)} candidate motif regions")

    return motif_regions, all_importances


def extract_motifs_gradient(
    model: nn.Module,
    sequences: np.ndarray,
    device: torch.device,
    n_sequences: int = 100,
    importance_percentile: float = 75,
    min_motif_distance: int = 6,
    motif_window: int = 4
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Extract motif regions using simple gradient-based importance.

    Fallback method when captum is not available.

    Args:
        model: Trained model
        sequences: Input sequences (N, 4, 200)
        device: Torch device
        n_sequences: Number of sequences to analyze
        importance_percentile: Percentile threshold
        min_motif_distance: Minimum distance between motifs
        motif_window: Window size around peaks

    Returns:
        motif_regions: List of (start, end) tuples
        avg_importance: Average importance array
    """
    model.eval()

    all_importances = []

    print("Extracting motif regions using gradient importance...")

    n_analyze = min(n_sequences, len(sequences))

    for i in tqdm(range(n_analyze), desc="Analyzing"):
        seq = sequences[i]
        seq_tensor = torch.from_numpy(seq).float().unsqueeze(0).to(device)
        seq_tensor.requires_grad_(True)

        # Forward pass
        outputs = model(seq_tensor)
        enhancer_prob = torch.softmax(outputs, dim=1)[0, 1]

        # Backward pass
        enhancer_prob.backward()

        # Get gradient importance
        grad = seq_tensor.grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
        all_importances.append(grad)

    # Average importance
    avg_importance = np.mean(all_importances, axis=0)

    # Find peaks
    threshold = np.percentile(avg_importance, importance_percentile)
    peaks, _ = find_peaks(
        avg_importance,
        height=threshold,
        distance=min_motif_distance
    )

    # Define motif regions
    motif_regions = []
    for peak in peaks:
        start = max(0, peak - motif_window)
        end = min(200, peak + motif_window + 1)
        motif_regions.append((start, end))

    print(f"   Found {len(motif_regions)} candidate motif regions")

    return motif_regions, avg_importance


def test_motif_interaction(
    model: nn.Module,
    sequence: np.ndarray,
    motif1_pos: Tuple[int, int],
    motif2_pos: Tuple[int, int],
    device: torch.device
) -> Tuple[bool, bool, bool]:
    """
    Test if two motifs work synergistically.

    Tests three conditions:
        1. Damage motif 1 alone -> repair success R1
        2. Damage motif 2 alone -> repair success R2
        3. Damage both motifs -> repair success R_both

    Classification:
        If R_both << R1 * R2: SYNERGY
        If R_both >> R1 * R2: REDUNDANT
        Otherwise: INDEPENDENT

    Args:
        model: Trained model
        sequence: Test sequence (4, 200)
        motif1_pos: (start, end) for motif 1
        motif2_pos: (start, end) for motif 2
        device: Torch device

    Returns:
        (repair1, repair2, repair_both): Success booleans
    """
    # Test motif 1 alone
    dam1 = damage_region(sequence, motif1_pos[0], motif1_pos[1])
    repair1, _, _ = attempt_repair(dam1, model, device)

    # Test motif 2 alone
    dam2 = damage_region(sequence, motif2_pos[0], motif2_pos[1])
    repair2, _, _ = attempt_repair(dam2, model, device)

    # Test both motifs
    dam_both = damage_region(dam1, motif2_pos[0], motif2_pos[1])
    repair_both, _, _ = attempt_repair(dam_both, model, device)

    return repair1, repair2, repair_both


def discover_grammar_rules(
    model: nn.Module,
    sequences: np.ndarray,
    motif_regions: List[Tuple[int, int]],
    device: torch.device,
    n_pairs: int = 1000,
    n_sequences_per_pair: int = 3,
    synergy_threshold: float = 0.5,
    redundancy_threshold: float = 1.5
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Test pairwise motif interactions to discover grammar rules.

    Args:
        model: Trained model
        sequences: Test sequences (enhancers only)
        motif_regions: List of (start, end) tuples
        device: Torch device
        n_pairs: Number of motif pairs to test
        n_sequences_per_pair: Sequences to test per pair
        synergy_threshold: R_both < expected * threshold -> SYNERGY
        redundancy_threshold: R_both > expected * threshold -> REDUNDANT

    Returns:
        (all_interactions, synergistic, redundant): Lists of interaction dicts
    """
    print("Discovering hierarchical grammar rules...")
    print(f"   Testing {n_pairs} motif pairs")
    print(f"   {n_sequences_per_pair} sequences per pair")

    interactions = []
    n_motifs = len(motif_regions)

    if n_motifs < 2:
        print("Error: Need at least 2 motifs for interaction testing")
        return [], [], []

    # Generate random pairs
    pairs_tested = set()

    for _ in tqdm(range(n_pairs), desc="Testing pairs"):
        # Random pair (avoid duplicates)
        attempts = 0
        while attempts < 100:
            i, j = np.random.choice(n_motifs, size=2, replace=False)
            pair_key = (min(i, j), max(i, j))
            if pair_key not in pairs_tested:
                pairs_tested.add(pair_key)
                break
            attempts += 1

        if attempts >= 100:
            break

        motif1 = motif_regions[i]
        motif2 = motif_regions[j]

        # Test on multiple sequences
        results = []
        for seq_idx in range(n_sequences_per_pair):
            seq = sequences[seq_idx % len(sequences)]
            r1, r2, r_both = test_motif_interaction(model, seq, motif1, motif2, device)
            results.append((r1, r2, r_both))

        # Average results
        avg_r1 = np.mean([r[0] for r in results])
        avg_r2 = np.mean([r[1] for r in results])
        avg_r_both = np.mean([r[2] for r in results])

        # Classify interaction
        expected_both = avg_r1 * avg_r2  # Independent expectation

        if expected_both > 0:
            ratio = avg_r_both / expected_both
        else:
            ratio = 1.0

        if ratio < synergy_threshold:
            interaction_type = "SYNERGY"
        elif ratio > redundancy_threshold:
            interaction_type = "REDUNDANT"
        else:
            interaction_type = "INDEPENDENT"

        interactions.append({
            'motif1': i,
            'motif2': j,
            'motif1_pos': motif1,
            'motif2_pos': motif2,
            'repair1': avg_r1,
            'repair2': avg_r2,
            'repair_both': avg_r_both,
            'expected_independent': expected_both,
            'ratio': ratio,
            'interaction': interaction_type
        })

    # Categorize
    synergistic = [x for x in interactions if x['interaction'] == 'SYNERGY']
    redundant = [x for x in interactions if x['interaction'] == 'REDUNDANT']
    independent = [x for x in interactions if x['interaction'] == 'INDEPENDENT']

    print(f"\nGrammar Discovery Results:")
    print(f"   Synergistic pairs: {len(synergistic)} ({len(synergistic)/len(interactions)*100:.1f}%)")
    print(f"   Redundant pairs: {len(redundant)} ({len(redundant)/len(interactions)*100:.1f}%)")
    print(f"   Independent pairs: {len(independent)} ({len(independent)/len(interactions)*100:.1f}%)")

    return interactions, synergistic, redundant


def build_interaction_matrix(
    interactions: List[Dict],
    n_motifs: int
) -> np.ndarray:
    """
    Build adjacency matrix from interactions.

    Encoding:
        0: Not tested
        1: Independent
        2: Synergistic
        3: Redundant

    Args:
        interactions: List of interaction dictionaries
        n_motifs: Number of motifs

    Returns:
        Adjacency matrix (n_motifs, n_motifs)
    """
    matrix = np.zeros((n_motifs, n_motifs), dtype=np.int32)

    type_to_value = {
        'INDEPENDENT': 1,
        'SYNERGY': 2,
        'REDUNDANT': 3
    }

    for interaction in interactions:
        i = interaction['motif1']
        j = interaction['motif2']
        value = type_to_value.get(interaction['interaction'], 0)
        matrix[i, j] = value
        matrix[j, i] = value

    return matrix


if __name__ == "__main__":
    print("Grammar discovery module loaded successfully")
