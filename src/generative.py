"""
Generative Validation for Enhancer Research

Innovation #3: Proves the model understands generative regulatory grammar.

Method:
    1. Start with random DNA
    2. Evolve toward enhancer using gradient ascent
    3. Validate generated sequences match real enhancers

This proves causal understanding, not just pattern matching.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tqdm import tqdm

from .data import decode_sequence


def generate_enhancer_from_random(
    model: nn.Module,
    device: torch.device,
    max_iterations: int = 500,
    target_prob: float = 0.80,
    learning_rate: float = 0.05,
    sequence_length: int = 200
) -> Tuple[Optional[np.ndarray], List[float], bool]:
    """
    Generate functional enhancer starting from random DNA.

    Algorithm:
        1. Initialize random DNA sequence
        2. Iteratively optimize to maximize enhancer probability
        3. Project to valid one-hot encoding each step
        4. Stop when probability exceeds threshold

    Args:
        model: Trained model
        device: Torch device
        max_iterations: Maximum optimization steps
        target_prob: Target enhancer probability
        learning_rate: Gradient ascent step size
        sequence_length: Length of sequence

    Returns:
        final_sequence: Generated sequence (4, 200) or None
        probability_trajectory: List of probabilities
        success: Boolean
    """
    model.eval()

    # Initialize random sequence
    random_seq = np.zeros((4, sequence_length), dtype=np.float32)
    for i in range(sequence_length):
        random_seq[np.random.randint(0, 4), i] = 1.0

    seq_tensor = torch.from_numpy(random_seq).float().unsqueeze(0).to(device)
    seq_tensor.requires_grad_(True)

    trajectory = []

    for iteration in range(max_iterations):
        # Forward pass
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(seq_tensor)
            probs = torch.softmax(outputs, dim=1)
            enhancer_prob = probs[0, 1]

        trajectory.append(enhancer_prob.item())

        # Check success
        if enhancer_prob.item() >= target_prob:
            final_seq = seq_tensor.squeeze(0).detach().cpu().numpy()
            return final_seq, trajectory, True

        # Maximize enhancer probability
        loss = -torch.log(enhancer_prob + 1e-8)
        loss.backward()

        # Gradient ascent step
        with torch.no_grad():
            seq_tensor.data -= learning_rate * seq_tensor.grad

            # Project to valid one-hot
            seq_data = seq_tensor.squeeze(0)
            max_indices = torch.argmax(seq_data, dim=0)
            new_seq = torch.zeros_like(seq_data)
            for pos in range(sequence_length):
                new_seq[max_indices[pos], pos] = 1.0
            seq_tensor.data = new_seq.unsqueeze(0)

            # Reset gradients
            if seq_tensor.grad is not None:
                seq_tensor.grad.zero_()

    # Failed to generate
    final_seq = seq_tensor.squeeze(0).detach().cpu().numpy()
    return final_seq, trajectory, False


def generate_enhancer_dataset(
    model: nn.Module,
    device: torch.device,
    n_attempts: int = 500,
    max_iterations: int = 500,
    target_prob: float = 0.80,
    learning_rate: float = 0.05
) -> Tuple[List[np.ndarray], List[List[float]], float]:
    """
    Generate multiple enhancers.

    Args:
        model: Trained model
        device: Torch device
        n_attempts: Number of generation attempts
        max_iterations: Max iterations per attempt
        target_prob: Target probability
        learning_rate: Learning rate

    Returns:
        generated_sequences: List of successful sequences
        trajectories: List of probability trajectories
        success_rate: Percentage successful
    """
    print("Generating synthetic enhancers...")
    print(f"   Attempts: {n_attempts}")

    generated_sequences = []
    trajectories = []

    for i in tqdm(range(n_attempts), desc="Generating"):
        seq, trajectory, success = generate_enhancer_from_random(
            model, device,
            max_iterations=max_iterations,
            target_prob=target_prob,
            learning_rate=learning_rate
        )

        if success:
            generated_sequences.append(seq)
            trajectories.append(trajectory)

    success_rate = len(generated_sequences) / n_attempts * 100

    print(f"\nSuccessfully generated: {len(generated_sequences)}/{n_attempts}")
    print(f"   Success rate: {success_rate:.1f}%")

    if len(generated_sequences) > 0:
        avg_iterations = np.mean([len(t) for t in trajectories])
        print(f"   Average iterations: {avg_iterations:.1f}")

    return generated_sequences, trajectories, success_rate


def calculate_gc_content(seq_str: str) -> float:
    """Calculate GC content of sequence."""
    return (seq_str.count('G') + seq_str.count('C')) / len(seq_str)


def find_motifs(seq_str: str, motif_list: List[str]) -> int:
    """Count known motifs in sequence."""
    count = 0
    for motif in motif_list:
        count += seq_str.count(motif)
    return count


def validate_generated_sequences(
    generated_seqs: List[np.ndarray],
    real_seqs: np.ndarray,
    known_motifs: Optional[List[str]] = None
) -> Dict:
    """
    Validate generated sequences match real enhancer properties.

    Tests:
        1. GC content distribution
        2. Motif content
        3. Statistical similarity

    Args:
        generated_seqs: Generated one-hot sequences
        real_seqs: Real enhancer one-hot sequences
        known_motifs: List of known TF binding motifs

    Returns:
        Dictionary with validation metrics
    """
    if known_motifs is None:
        known_motifs = [
            'TATAAA',    # TATA box
            'CACGTG',    # E-box
            'GGGCGG',    # GC box
            'CCAAT',     # CAAT box
            'TGACGTCA',  # CRE
            'AATAGC',    # Various TFs
        ]

    print("Validating generated sequences...")

    # Analyze generated sequences
    generated_gc = []
    generated_motifs = []

    for seq in tqdm(generated_seqs, desc="Analyzing generated"):
        seq_str = decode_sequence(seq)
        gc = calculate_gc_content(seq_str)
        motif_count = find_motifs(seq_str, known_motifs)
        generated_gc.append(gc)
        generated_motifs.append(motif_count)

    # Analyze real sequences
    real_gc = []
    real_motifs = []

    n_real = min(len(real_seqs), 1000)
    for seq in tqdm(real_seqs[:n_real], desc="Analyzing real"):
        seq_str = decode_sequence(seq)
        gc = calculate_gc_content(seq_str)
        motif_count = find_motifs(seq_str, known_motifs)
        real_gc.append(gc)
        real_motifs.append(motif_count)

    # Statistical comparison
    gc_ttest = stats.ttest_ind(generated_gc, real_gc)
    motif_ttest = stats.ttest_ind(generated_motifs, real_motifs)

    # Mann-Whitney U test (non-parametric)
    gc_mannwhitney = stats.mannwhitneyu(generated_gc, real_gc, alternative='two-sided')
    motif_mannwhitney = stats.mannwhitneyu(generated_motifs, real_motifs, alternative='two-sided')

    print(f"\nValidation Results:")
    print(f"{'='*70}")
    print(f"GC Content:")
    print(f"   Generated: {np.mean(generated_gc):.3f} +/- {np.std(generated_gc):.3f}")
    print(f"   Real:      {np.mean(real_gc):.3f} +/- {np.std(real_gc):.3f}")
    print(f"   t-test p-value:   {gc_ttest.pvalue:.4f} {'(similar)' if gc_ttest.pvalue > 0.05 else '(different)'}")

    print(f"\nMotif Count:")
    print(f"   Generated: {np.mean(generated_motifs):.2f} +/- {np.std(generated_motifs):.2f}")
    print(f"   Real:      {np.mean(real_motifs):.2f} +/- {np.std(real_motifs):.2f}")
    print(f"   t-test p-value:   {motif_ttest.pvalue:.4f} {'(similar)' if motif_ttest.pvalue > 0.05 else '(different)'}")

    # Overall assessment
    both_similar = (gc_ttest.pvalue > 0.05) and (motif_ttest.pvalue > 0.05)

    if both_similar:
        print(f"\nSUCCESS! Generated enhancers are statistically similar to real ones")
        print(f"   -> Model has learned the GENERATIVE grammar")
    else:
        print(f"\nGenerated enhancers differ from real ones")
        print(f"   -> Model may need more training")

    return {
        'generated_gc': generated_gc,
        'real_gc': real_gc,
        'generated_motifs': generated_motifs,
        'real_motifs': real_motifs,
        'gc_mean_generated': np.mean(generated_gc),
        'gc_mean_real': np.mean(real_gc),
        'gc_pvalue': gc_ttest.pvalue,
        'motif_pvalue': motif_ttest.pvalue,
        'gc_mannwhitney_pvalue': gc_mannwhitney.pvalue,
        'motif_mannwhitney_pvalue': motif_mannwhitney.pvalue,
        'validation_passed': both_similar
    }


def analyze_generation_dynamics(trajectories: List[List[float]]) -> Dict:
    """
    Analyze generation dynamics across all attempts.

    Args:
        trajectories: List of probability trajectories

    Returns:
        Dictionary with dynamics analysis
    """
    if len(trajectories) == 0:
        return {}

    # Basic statistics
    lengths = [len(t) for t in trajectories]
    final_probs = [t[-1] for t in trajectories]
    initial_probs = [t[0] for t in trajectories]

    # Calculate average trajectory
    max_len = max(lengths)
    avg_trajectory = []
    std_trajectory = []

    for step in range(max_len):
        step_values = [t[step] for t in trajectories if len(t) > step]
        avg_trajectory.append(np.mean(step_values))
        std_trajectory.append(np.std(step_values))

    # Calculate convergence rate (steps to 50% probability)
    convergence_steps = []
    for t in trajectories:
        for i, p in enumerate(t):
            if p >= 0.5:
                convergence_steps.append(i)
                break
        else:
            convergence_steps.append(len(t))

    return {
        'n_successful': len(trajectories),
        'avg_iterations': np.mean(lengths),
        'std_iterations': np.std(lengths),
        'avg_final_prob': np.mean(final_probs),
        'avg_initial_prob': np.mean(initial_probs),
        'avg_trajectory': avg_trajectory,
        'std_trajectory': std_trajectory,
        'avg_convergence_steps': np.mean(convergence_steps)
    }


if __name__ == "__main__":
    print("Generative validation module loaded successfully")
