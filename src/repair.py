"""
Repair-Based Analysis for Enhancer Research

Core Innovation: Uses neural network "repair" to discover essential regulatory elements.

Method:
    1. Damage enhancer sequences
    2. Attempt computational repair using gradient descent
    3. Success/failure reveals biological essentiality

Cross-species analysis compares repair success across vertebrate evolution.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def damage_sequence(
    sequence: np.ndarray,
    damage_fraction: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly damage DNA sequence.

    Args:
        sequence: One-hot encoded (4, 200)
        damage_fraction: Fraction of positions to damage
        seed: Random seed for reproducibility

    Returns:
        damaged_sequence: One-hot encoded (4, 200)
        damaged_positions: Array of damaged position indices
    """
    if seed is not None:
        np.random.seed(seed)

    sequence = sequence.copy()
    seq_length = sequence.shape[1]

    # Determine positions to damage
    n_damage = int(seq_length * damage_fraction)
    damaged_positions = np.random.choice(seq_length, size=n_damage, replace=False)

    # For each position, randomly select new nucleotide
    for pos in damaged_positions:
        current_nt = np.argmax(sequence[:, pos])
        possible_nts = [0, 1, 2, 3]
        possible_nts.remove(current_nt)
        new_nt = np.random.choice(possible_nts)

        # One-hot encode new nucleotide
        sequence[:, pos] = 0
        sequence[new_nt, pos] = 1

    return sequence, damaged_positions


def damage_region(
    sequence: np.ndarray,
    start: int,
    end: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Damage specific region of sequence.

    Args:
        sequence: One-hot encoded (4, 200)
        start: Start position
        end: End position
        seed: Random seed

    Returns:
        Damaged sequence
    """
    if seed is not None:
        np.random.seed(seed)

    damaged = sequence.copy()

    for pos in range(start, min(end, sequence.shape[1])):
        current_nt = np.argmax(sequence[:, pos])
        possible = [0, 1, 2, 3]
        possible.remove(current_nt)
        new_nt = np.random.choice(possible)
        damaged[:, pos] = 0
        damaged[new_nt, pos] = 1

    return damaged


def attempt_repair(
    damaged_sequence: np.ndarray,
    model: nn.Module,
    device: torch.device,
    max_iterations: int = 30,
    learning_rate: float = 0.01,
    target_prob: float = 0.9
) -> Tuple[bool, float, List[float]]:
    """
    Attempt to repair damaged sequence using gradient descent.

    Algorithm:
        1. Start with damaged sequence
        2. Compute model's prediction (enhancer probability)
        3. If probability > target: SUCCESS
        4. Else: Compute gradient and update sequence
        5. Project back to valid one-hot encoding
        6. Repeat until success or max_iterations

    Args:
        damaged_sequence: One-hot (4, 200)
        model: Trained enhancer model
        device: Torch device
        max_iterations: Maximum repair attempts
        learning_rate: Gradient descent step size
        target_prob: Success threshold

    Returns:
        success: Boolean (True if repaired)
        final_prob: Final enhancer probability
        trajectory: List of probabilities over iterations
    """
    model.eval()

    # Convert to tensor
    seq_tensor = torch.from_numpy(damaged_sequence).float().unsqueeze(0).to(device)
    seq_tensor.requires_grad_(True)

    trajectory = []

    for iteration in range(max_iterations):
        # Forward pass (disable mixed precision for repair)
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(seq_tensor)
            probs = torch.softmax(outputs, dim=1)
            enhancer_prob = probs[0, 1]

        trajectory.append(enhancer_prob.item())

        # Check success
        if enhancer_prob.item() >= target_prob:
            return True, enhancer_prob.item(), trajectory

        # Compute gradient (maximize enhancer probability)
        loss = -torch.log(enhancer_prob + 1e-8)
        loss.backward()

        # Gradient descent step
        with torch.no_grad():
            seq_tensor.data -= learning_rate * seq_tensor.grad

            # Project to valid one-hot encoding
            seq_data = seq_tensor.squeeze(0)  # (4, 200)

            # For each position, select nucleotide with highest value
            max_indices = torch.argmax(seq_data, dim=0)  # (200,)

            # Create new one-hot encoding
            new_seq = torch.zeros_like(seq_data)
            for pos in range(200):
                new_seq[max_indices[pos], pos] = 1.0

            seq_tensor.data = new_seq.unsqueeze(0)

            # Reset gradients
            if seq_tensor.grad is not None:
                seq_tensor.grad.zero_()

    # Failed to repair
    final_prob = trajectory[-1] if trajectory else 0.0
    return False, final_prob, trajectory


def test_cross_species_repair(
    model: nn.Module,
    sequences: np.ndarray,
    species_labels: np.ndarray,
    device: torch.device,
    n_sequences_per_species: int = 200,
    damage_fraction: float = 0.1,
    max_repair_iterations: int = 30,
    repair_learning_rate: float = 0.01,
    target_probability: float = 0.9
) -> Dict:
    """
    Test repair success across multiple species.

    Args:
        model: Trained model
        sequences: Test sequences (N, 4, 200)
        species_labels: Species for each sequence (N,)
        device: Torch device
        n_sequences_per_species: Number to test per species
        damage_fraction: Fraction of sequence to damage
        max_repair_iterations: Max repair iterations
        repair_learning_rate: Repair step size
        target_probability: Success threshold

    Returns:
        Dictionary with repair rates per species
    """
    species_names = ['Human', 'Mouse', 'Zebrafish', 'Chicken']
    results = {}

    print("Running cross-species repair analysis...")
    print(f"   Testing {n_sequences_per_species} sequences per species")
    print(f"   Damage fraction: {damage_fraction*100:.0f}%")

    for species_id, species_name in enumerate(species_names):
        # Get enhancer sequences for this species
        species_mask = (species_labels == species_id)
        species_seqs = sequences[species_mask]

        if len(species_seqs) == 0:
            print(f"\nNo sequences for {species_name}")
            continue

        # Sample sequences
        n_test = min(n_sequences_per_species, len(species_seqs))
        test_indices = np.random.choice(len(species_seqs), size=n_test, replace=False)

        repair_successes = []
        final_probs = []

        print(f"\n{species_name}:")
        for idx in tqdm(test_indices, desc=f"  Testing"):
            seq = species_seqs[idx]

            # Damage sequence
            damaged_seq, _ = damage_sequence(seq, damage_fraction=damage_fraction)

            # Attempt repair
            success, final_prob, trajectory = attempt_repair(
                damaged_seq, model, device,
                max_iterations=max_repair_iterations,
                learning_rate=repair_learning_rate,
                target_prob=target_probability
            )

            repair_successes.append(success)
            final_probs.append(final_prob)

        # Calculate statistics
        repair_rate = np.mean(repair_successes) * 100
        avg_final_prob = np.mean(final_probs)

        results[species_name] = {
            'repair_rate': repair_rate,
            'avg_final_prob': avg_final_prob,
            'n_tested': n_test,
            'successes': sum(repair_successes)
        }

        print(f"  Repair rate: {repair_rate:.1f}%")
        print(f"  Avg final prob: {avg_final_prob:.3f}")

    # Calculate conservation score
    if len(results) >= 4:
        mammal_rate = np.mean([results['Human']['repair_rate'],
                              results['Mouse']['repair_rate']])
        non_mammal_rate = np.mean([results['Zebrafish']['repair_rate'],
                                  results['Chicken']['repair_rate']])
        conservation_score = min(mammal_rate, non_mammal_rate)

        print(f"\nCross-Species Summary:")
        print(f"   Mammalian repair rate: {mammal_rate:.1f}%")
        print(f"   Non-mammalian repair rate: {non_mammal_rate:.1f}%")
        print(f"   Conservation score: {conservation_score:.1f}%")

        results['summary'] = {
            'mammal_rate': mammal_rate,
            'non_mammal_rate': non_mammal_rate,
            'conservation_score': conservation_score
        }

    return results


def test_motif_repair(
    model: nn.Module,
    sequence: np.ndarray,
    motif_positions: List[Tuple[int, int]],
    device: torch.device,
    n_trials: int = 5
) -> Dict:
    """
    Test repair success when specific motifs are damaged.

    Args:
        model: Trained model
        sequence: Test sequence (4, 200)
        motif_positions: List of (start, end) tuples
        device: Torch device
        n_trials: Number of trials per motif

    Returns:
        Dictionary with repair rates per motif
    """
    results = {}

    for motif_idx, (start, end) in enumerate(motif_positions):
        successes = []

        for trial in range(n_trials):
            # Damage only this motif region
            damaged = damage_region(sequence, start, end, seed=trial)

            # Attempt repair
            success, _, _ = attempt_repair(damaged, model, device)
            successes.append(success)

        repair_rate = np.mean(successes) * 100
        results[f'motif_{motif_idx}'] = {
            'position': (start, end),
            'repair_rate': repair_rate,
            'n_trials': n_trials
        }

    return results


if __name__ == "__main__":
    # Test damage function
    test_seq = np.zeros((4, 200), dtype=np.float32)
    for i in range(200):
        test_seq[np.random.randint(0, 4), i] = 1.0

    damaged, positions = damage_sequence(test_seq, damage_fraction=0.1, seed=42)
    print(f"Damaged {len(positions)} positions: {positions[:10]}...")
