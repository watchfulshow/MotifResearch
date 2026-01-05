"""
Data Loading and Processing for Enhancer Repair Research

Handles:
    - DNA sequence encoding (one-hot)
    - ENCODE data download and parsing
    - Ensembl API integration
    - Background sequence generation
    - Dataset creation and splitting
"""

import gzip
import json
import pickle
import random
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


# =============================================================================
# DNA Encoding Functions
# =============================================================================

def one_hot_encode(sequence: str) -> np.ndarray:
    """
    Convert DNA string to one-hot encoded array.

    Encoding:
        A -> [1, 0, 0, 0]
        C -> [0, 1, 0, 0]
        G -> [0, 0, 1, 0]
        T -> [0, 0, 0, 1]
        N -> [0, 0, 0, 0]

    Args:
        sequence: DNA string (A, C, G, T characters)

    Returns:
        One-hot encoded array of shape (4, len(sequence))
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    encoded = np.zeros((4, seq_len), dtype=np.float32)

    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in mapping:
            encoded[mapping[nucleotide], i] = 1.0
        # N and other characters remain as zeros

    return encoded


def decode_sequence(one_hot: np.ndarray) -> str:
    """
    Convert one-hot encoded array back to DNA string.

    Args:
        one_hot: One-hot encoded array of shape (4, seq_len)

    Returns:
        DNA string
    """
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_len = one_hot.shape[1]
    sequence = []

    for i in range(seq_len):
        idx = np.argmax(one_hot[:, i])
        if one_hot[idx, i] > 0:
            sequence.append(mapping[idx])
        else:
            sequence.append('N')

    return ''.join(sequence)


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of a DNA sequence."""
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0.0


# =============================================================================
# Data Download Functions
# =============================================================================

def download_encode_coordinates(
    species: str = 'human',
    output_file: Optional[str] = None,
    max_sequences: int = 500000
) -> List[Dict]:
    """
    Download ENCODE cCRE coordinates.

    Args:
        species: 'human' or 'mouse'
        output_file: Path to save coordinates (pickle)
        max_sequences: Maximum number of sequences to download

    Returns:
        List of coordinate dictionaries {chrom, start, end, length}
    """
    urls = {
        'human': 'https://downloads.wenglab.org/Registry/V3/GRCh38-cCREs.bed.gz',
        'mouse': 'https://downloads.wenglab.org/Registry/V3/mm10-cCREs.bed.gz'
    }

    if species not in urls:
        raise ValueError(f"Species must be 'human' or 'mouse', got {species}")

    print(f"Downloading {species} ENCODE data...")
    print(f"URL: {urls[species]}")

    # Download to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bed.gz')
    urllib.request.urlretrieve(urls[species], temp_file.name)

    print("Parsing BED file...")
    coordinates = []

    with gzip.open(temp_file.name, 'rt') as f:
        for line in tqdm(f, desc="Reading coordinates"):
            # Skip header lines
            if line.startswith('#') or line.startswith('track'):
                continue

            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                length = end - start

                # Filter for appropriate size (150-250bp)
                if 150 <= length <= 250:
                    coordinates.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'length': length
                    })

            # Stop if we have enough
            if len(coordinates) >= max_sequences * 2:
                break

    # Clean up
    Path(temp_file.name).unlink()

    # Random sample to target size
    if len(coordinates) > max_sequences:
        random.seed(42)
        coordinates = random.sample(coordinates, max_sequences)

    print(f"Downloaded {len(coordinates):,} {species} coordinates")

    # Save if output file specified
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(coordinates, f)
        print(f"Saved to {output_file}")

    return coordinates


def fetch_sequence_ensembl(
    chrom: str,
    start: int,
    end: int,
    species: str = 'human',
    max_retries: int = 3
) -> Optional[str]:
    """
    Fetch DNA sequence from Ensembl REST API.

    Args:
        chrom: Chromosome (e.g., '1', 'X')
        start: Start position
        end: End position
        species: Species name
        max_retries: Number of retry attempts

    Returns:
        DNA sequence string or None if failed
    """
    species_map = {
        'human': 'homo_sapiens',
        'mouse': 'mus_musculus',
        'zebrafish': 'danio_rerio',
        'chicken': 'gallus_gallus'
    }

    species_name = species_map.get(species, 'homo_sapiens')

    # Ensembl uses chromosome without 'chr' prefix
    chrom_clean = chrom.replace('chr', '')

    url = (f"https://rest.ensembl.org/sequence/region/"
           f"{species_name}/{chrom_clean}:{start}-{end}"
           f"?content-type=text/plain")

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                sequence = response.read().decode('utf-8').strip().upper()
                return sequence
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None

    return None


def download_sequences_api(
    coordinates: List[Dict],
    species: str,
    output_file: Optional[str] = None,
    batch_size: int = 1000,
    target_length: int = 200
) -> List[str]:
    """
    Download sequences using Ensembl API with rate limiting.

    Args:
        coordinates: List of coordinate dictionaries
        species: Species name
        output_file: Path to save sequences
        batch_size: Checkpoint save frequency
        target_length: Standardize sequences to this length

    Returns:
        List of DNA sequences
    """
    sequences = []
    failed = 0

    print(f"Downloading {len(coordinates):,} sequences via Ensembl API")
    print("Rate limit: ~15 requests/second")
    print(f"Estimated time: {len(coordinates) / 15 / 60:.1f} minutes")

    for i, coord in enumerate(tqdm(coordinates, desc=f"{species} sequences")):
        seq = fetch_sequence_ensembl(
            coord['chrom'],
            coord['start'],
            coord['end'],
            species
        )

        if seq and len(seq) >= 150 and 'N' not in seq:
            # Standardize length
            if len(seq) > target_length:
                seq = seq[:target_length]
            elif len(seq) < target_length:
                seq = seq + 'A' * (target_length - len(seq))

            sequences.append(seq)
        else:
            failed += 1

        # Rate limiting: 15 requests per second
        if (i + 1) % 15 == 0:
            time.sleep(1)

        # Checkpoint saving
        if (i + 1) % batch_size == 0 and output_file:
            checkpoint_file = output_file.replace('.pkl', f'_checkpoint_{i+1}.pkl')
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(sequences, f)
            print(f"\nCheckpoint saved: {len(sequences):,} sequences")

    print(f"\nDownloaded {len(sequences):,} sequences")
    print(f"Failed: {failed:,} sequences")

    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"Saved to {output_file}")

    return sequences


def generate_background_sequences(
    n_sequences: int,
    length: int = 200,
    gc_content: float = 0.42,
    seed: int = 42
) -> List[str]:
    """
    Generate random background DNA sequences.

    Args:
        n_sequences: Number of sequences to generate
        length: Length of each sequence
        gc_content: Target GC content (0.42 is genome average)
        seed: Random seed for reproducibility

    Returns:
        List of random DNA sequences
    """
    random.seed(seed)

    # Calculate nucleotide probabilities
    gc_prob = gc_content / 2
    at_prob = (1 - gc_content) / 2

    nucleotides = ['A', 'C', 'G', 'T']
    probabilities = [at_prob, gc_prob, gc_prob, at_prob]

    sequences = []

    for _ in tqdm(range(n_sequences), desc="Generating background"):
        seq = ''.join(random.choices(nucleotides, weights=probabilities, k=length))
        sequences.append(seq)

    return sequences


# =============================================================================
# Dataset Classes
# =============================================================================

class EnhancerDataset(Dataset):
    """
    PyTorch Dataset for enhancer sequences.

    Args:
        X: One-hot encoded sequences (N, 4, 200)
        y: Labels (N,)
        species: Optional species labels (N,)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        species: Optional[np.ndarray] = None
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.species = torch.from_numpy(species).long() if species is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.species is not None:
            return self.X[idx], self.y[idx], self.species[idx]
        return self.X[idx], self.y[idx]


class StochasticEpochSampler:
    """
    Samples random subset of training data each epoch.

    Benefits:
        - Faster epochs (less data per epoch)
        - Better generalization (different samples each time)
        - Memory efficient

    Args:
        train_indices: Full training indices
        samples_per_epoch: Number to sample each epoch
        seed: Random seed
    """

    def __init__(
        self,
        train_indices: Union[List[int], np.ndarray],
        samples_per_epoch: int = 100000,
        seed: int = 42
    ):
        self.train_indices = np.array(train_indices)
        self.samples_per_epoch = min(samples_per_epoch, len(self.train_indices))
        self.rng = np.random.RandomState(seed)

    def sample_epoch(self) -> np.ndarray:
        """Sample random subset for this epoch."""
        return self.rng.choice(
            self.train_indices,
            size=self.samples_per_epoch,
            replace=False
        )


# =============================================================================
# Dataset Preparation Functions
# =============================================================================

def prepare_dataset(
    human_seqs: List[str],
    mouse_seqs: List[str],
    zebrafish_seqs: List[str],
    chicken_seqs: List[str],
    background_seqs: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine all sequences and create train/val/test splits.

    Args:
        *_seqs: Lists of DNA sequence strings
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        X: One-hot encoded sequences (N, 4, 200)
        y: Labels (N,)
        species: Species labels (N,)
        train_idx, val_idx, test_idx: Split indices
    """
    print("Preparing dataset...")

    all_sequences = []
    all_labels = []
    species_labels = []

    # Add enhancers from each species
    species_data = [
        (human_seqs, 0, 'Human'),
        (mouse_seqs, 1, 'Mouse'),
        (zebrafish_seqs, 2, 'Zebrafish'),
        (chicken_seqs, 3, 'Chicken'),
    ]

    for seqs, species_id, species_name in species_data:
        if seqs:
            all_sequences.extend(seqs)
            all_labels.extend([1] * len(seqs))
            species_labels.extend([species_id] * len(seqs))
            print(f"  {species_name}: {len(seqs):,} enhancers")

    # Add background
    all_sequences.extend(background_seqs)
    all_labels.extend([0] * len(background_seqs))
    species_labels.extend([4] * len(background_seqs))
    print(f"  Background: {len(background_seqs):,} sequences")

    print(f"Total sequences: {len(all_sequences):,}")

    # One-hot encode
    print("\nOne-hot encoding...")
    X = np.array([one_hot_encode(seq) for seq in tqdm(all_sequences, desc="Encoding")])
    y = np.array(all_labels, dtype=np.int64)
    species = np.array(species_labels, dtype=np.int64)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Create stratified splits
    print("\nCreating stratified splits...")
    indices = np.arange(len(y))

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        stratify=y[train_val_idx],
        random_state=random_state
    )

    print(f"Split sizes:")
    print(f"  Train: {len(train_idx):,} ({len(train_idx)/len(y)*100:.1f}%)")
    print(f"  Validation: {len(val_idx):,} ({len(val_idx)/len(y)*100:.1f}%)")
    print(f"  Test: {len(test_idx):,} ({len(test_idx)/len(y)*100:.1f}%)")

    return X, y, species, train_idx, val_idx, test_idx


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    species: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: str
) -> None:
    """
    Save processed dataset to disk.

    Args:
        X, y, species: Dataset arrays
        train_idx, val_idx, test_idx: Split indices
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Saving dataset...")

    # Save full dataset (compressed)
    np.savez_compressed(
        output_dir / 'dataset.npz',
        X=X,
        y=y,
        species=species
    )

    # Save split indices
    np.savez(
        output_dir / 'splits.npz',
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )

    # Save metadata
    metadata = {
        'total_sequences': len(X),
        'sequence_length': X.shape[2],
        'n_channels': X.shape[1],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'class_names': ['background', 'enhancer'],
        'species_names': ['human', 'mouse', 'zebrafish', 'chicken', 'background']
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    file_size = (output_dir / 'dataset.npz').stat().st_size / 1024**3
    print(f"Dataset saved to {output_dir}")
    print(f"  dataset.npz: {file_size:.2f} GB")


def load_dataset(
    data_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load processed dataset from disk.

    Args:
        data_dir: Directory containing dataset files

    Returns:
        X, y, species, train_idx, val_idx, test_idx
    """
    data_dir = Path(data_dir)

    print(f"Loading dataset from {data_dir}...")

    # Load arrays
    data = np.load(data_dir / 'dataset.npz')
    X = data['X']
    y = data['y']
    species = data['species']

    # Load splits
    splits = np.load(data_dir / 'splits.npz')
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    print("Dataset loaded")
    print(f"  Total sequences: {metadata['total_sequences']:,}")
    print(f"  Train: {metadata['n_train']:,}")
    print(f"  Val: {metadata['n_val']:,}")
    print(f"  Test: {metadata['n_test']:,}")

    return X, y, species, train_idx, val_idx, test_idx


if __name__ == "__main__":
    # Test encoding
    test_seq = "ATGCGTACGATCGATCGTAGCTAGCTACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    encoded = one_hot_encode(test_seq)
    decoded = decode_sequence(encoded)
    print(f"Original: {test_seq[:50]}...")
    print(f"Decoded:  {decoded[:50]}...")
    print(f"Match: {test_seq == decoded}")
    print(f"Encoding shape: {encoded.shape}")

    # Test GC content
    gc = calculate_gc_content(test_seq)
    print(f"GC content: {gc:.2%}")
