"""
Training Pipeline for Enhancer Repair Research

Features:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Robust checkpointing for Colab resumption
    - Stochastic epoch sampling
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from .data import EnhancerDataset, StochasticEpochSampler


# =============================================================================
# Checkpoint Functions
# =============================================================================

def save_checkpoint(
    state: Dict,
    checkpoint_dir: str,
    is_best: bool = False
) -> None:
    """
    Save training checkpoint.

    Args:
        state: Dictionary containing:
            - epoch: Current epoch number
            - model_state_dict: Model weights
            - optimizer_state_dict: Optimizer state
            - scheduler_state_dict: Scheduler state
            - scaler_state_dict: Mixed precision scaler state
            - best_val_loss: Best validation loss so far
            - patience_counter: Early stopping counter
            - history: Training history
        checkpoint_dir: Where to save checkpoints
        is_best: If True, also save as best model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Save latest checkpoint
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(state, latest_path)
    print(f"   Checkpoint saved: epoch {state['epoch']}")

    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pt'
        torch.save(state, best_path)
        print(f"   Best model updated: loss={state['best_val_loss']:.4f}")

    # Periodic backup (every 5 epochs)
    if state['epoch'] % 5 == 0:
        backup_path = checkpoint_dir / f"checkpoint_epoch_{state['epoch']}.pt"
        torch.save(state, backup_path)
        print(f"   Backup saved: epoch {state['epoch']}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cpu'
) -> Optional[Dict]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        scaler: Optional mixed precision scaler to restore
        device: Device to load checkpoint to

    Returns:
        Checkpoint dictionary or None if file doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return None

    print(f"Loading checkpoint: {checkpoint_path.name}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scheduler
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore scaler
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"   Resumed from epoch {checkpoint['epoch']}")
    print(f"   Best val_loss: {checkpoint['best_val_loss']:.4f}")

    return checkpoint


# =============================================================================
# Training Function
# =============================================================================

def train_model(
    model: nn.Module,
    train_dataset: EnhancerDataset,
    val_dataset: EnhancerDataset,
    device: torch.device,
    config: Dict,
    checkpoint_dir: str = 'checkpoints'
) -> Dict[str, List[float]]:
    """
    Complete training loop with all optimizations.

    Features:
        - Mixed precision training (FP16)
        - Gradient accumulation
        - Gradient clipping
        - Learning rate scheduling
        - Early stopping
        - Checkpointing

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to train on
        config: Training configuration dictionary
        checkpoint_dir: Directory for checkpoints

    Returns:
        Training history dictionary
    """
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 0.0001),
        weight_decay=config.get('weight_decay', 0.01)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get('scheduler_factor', 0.5),
        patience=config.get('scheduler_patience', 3),
        min_lr=config.get('scheduler_min_lr', 1e-6)
    )

    # Mixed precision scaler
    use_amp = config.get('use_mixed_precision', True)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Stochastic sampler
    sampler = StochasticEpochSampler(
        train_indices=range(len(train_dataset)),
        samples_per_epoch=config.get('samples_per_epoch', 100000)
    )

    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True)
    )

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Try to resume from checkpoint
    checkpoint_state = load_checkpoint(
        Path(checkpoint_dir) / 'checkpoint_latest.pt',
        model, optimizer, scheduler, scaler, device
    )

    if checkpoint_state:
        start_epoch = checkpoint_state['epoch'] + 1
        best_val_loss = checkpoint_state['best_val_loss']
        patience_counter = checkpoint_state['patience_counter']
        history = checkpoint_state['history']
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting new training")

    # Configuration
    num_epochs = config.get('num_epochs', 30)
    batch_size = config.get('batch_size', 64)
    grad_accum_steps = config.get('gradient_accumulation_steps', 4)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    early_stopping_patience = config.get('early_stopping_patience', 7)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*70}")

        # Sample data for this epoch
        epoch_indices = sampler.sample_epoch()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(epoch_indices),
            num_workers=config.get('num_workers', 2),
            pin_memory=config.get('pin_memory', True)
        )

        print(f"Training on {len(epoch_indices):,} randomly sampled sequences")

        # TRAINING PHASE
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Handle both (X, y) and (X, y, species) formats
            if len(batch) == 3:
                sequences, labels, _ = batch
            else:
                sequences, labels = batch

            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps

            # Backward pass with gradient scaling
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Unscale before clipping
                if scaler:
                    scaler.unscale_(optimizer)

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )

                # Check for NaN gradients
                if torch.isnan(grad_norm):
                    print(f"\nNaN gradient detected! Skipping update.")
                    optimizer.zero_grad()
                    continue

                # Optimizer step
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            # Statistics
            train_loss += loss.item() * grad_accum_steps
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * grad_accum_steps:.4f}',
                'acc': f'{100. * train_correct / train_total:.1f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if len(batch) == 3:
                    sequences, labels, _ = batch
                else:
                    sequences, labels = batch

                sequences = sequences.to(device)
                labels = labels.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Check for improvement
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"   New best model!")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{early_stopping_patience})")

        # Save checkpoint
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'history': history,
            'config': config,
        }

        save_checkpoint(checkpoint_state, checkpoint_dir, is_best=is_best)

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    load_checkpoint(
        Path(checkpoint_dir) / 'checkpoint_best.pt',
        model,
        device=device
    )

    print(f"\nTraining complete!")
    print(f"   Best val_loss: {best_val_loss:.4f}")
    print(f"   Total epochs: {epoch+1}")

    return history


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_model(
    model: nn.Module,
    test_dataset: EnhancerDataset,
    device: torch.device,
    batch_size: int = 64
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("Evaluating on test set...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if len(batch) == 3:
                sequences, labels, _ = batch
            else:
                sequences, labels = batch

            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    # Calculate metrics
    test_loss = test_loss / len(test_loader)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    test_acc = 100. * (predictions == labels).sum() / len(labels)

    cm = confusion_matrix(labels, predictions)
    auc = roc_auc_score(labels, all_probabilities)

    # Calculate precision, recall, F1
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nTest Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   ROC AUC: {auc:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")

    print(f"\nConfusion Matrix:")
    print(f"   TN: {tn:,} | FP: {fp:,}")
    print(f"   FN: {fn:,} | TP: {tp:,}")

    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': predictions,
        'labels': labels,
        'probabilities': all_probabilities,
        'confusion_matrix': cm,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    print("Training module loaded successfully")
