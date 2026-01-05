"""
Model Architecture for Enhancer Repair Research

Hybrid CNN-Transformer model for enhancer prediction and repair.

Architecture:
    - CNN Encoder: Detects local motifs (7bp, 11bp, 15bp windows)
    - Transformer Encoder: Models long-range motif interactions
    - Classification Head: Predicts enhancer vs background

Total Parameters: ~27M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.

    Adds position information to sequence embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNEncoder(nn.Module):
    """
    Convolutional encoder for detecting local DNA motifs.

    Uses multiple kernel sizes to capture motifs of different lengths:
        - 7bp: Short motifs (e.g., TATAAA)
        - 11bp: Medium motifs (e.g., CACGTGCACG)
        - 15bp: Long composite motifs

    Args:
        input_channels: Number of input channels (4 for DNA one-hot)
        hidden_channels: List of output channels for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
    """

    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: list = None,
        kernel_sizes: list = None
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [128, 256, 384]
        if kernel_sizes is None:
            kernel_sizes = [7, 11, 15]

        self.hidden_channels = hidden_channels

        # First convolutional layer: detects short motifs
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels[0],
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2  # "same" padding
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])

        # Second convolutional layer: detects medium motifs
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels[0],
            out_channels=hidden_channels[1],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels[1])

        # Third convolutional layer: detects long motifs
        self.conv3 = nn.Conv1d(
            in_channels=hidden_channels[1],
            out_channels=hidden_channels[2],
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2
        )
        self.bn3 = nn.BatchNorm1d(hidden_channels[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.

        Args:
            x: Input tensor of shape (batch, 4, 200)

        Returns:
            Encoded features of shape (batch, hidden_channels[-1], 200)
        """
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        return x


class HybridEnhancerModel(nn.Module):
    """
    Hybrid CNN-Transformer model for enhancer prediction.

    Architecture:
        1. CNN Encoder: Detects local motifs
        2. Linear Projection: Maps CNN features to d_model
        3. Positional Encoding: Adds position information
        4. Transformer Encoder: Models motif interactions
        5. Global Pooling: Aggregates sequence representations
        6. Classification Head: Predicts enhancer probability

    Args:
        n_transformer_layers: Number of transformer encoder layers
        n_attention_heads: Number of attention heads per layer
        d_model: Transformer embedding dimension
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        cnn_channels: List of CNN channel sizes
        cnn_kernels: List of CNN kernel sizes
        sequence_length: Input sequence length
        num_classes: Number of output classes
    """

    def __init__(
        self,
        n_transformer_layers: int = 10,
        n_attention_heads: int = 8,
        d_model: int = 256,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        cnn_channels: list = None,
        cnn_kernels: list = None,
        sequence_length: int = 200,
        num_classes: int = 2
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [128, 256, 384]
        if cnn_kernels is None:
            cnn_kernels = [7, 11, 15]

        # Store configuration
        self.d_model = d_model
        self.n_layers = n_transformer_layers
        self.n_heads = n_attention_heads
        self.sequence_length = sequence_length

        # CNN Encoder
        self.cnn_encoder = CNNEncoder(
            input_channels=4,
            hidden_channels=cnn_channels,
            kernel_sizes=cnn_kernels
        )

        # Project CNN features to d_model
        self.projection = nn.Linear(cnn_channels[-1], d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=sequence_length,
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input sequences of shape (batch, 4, 200)
            return_features: If True, also return intermediate features

        Returns:
            logits: Class logits of shape (batch, num_classes)
            features: (Optional) Intermediate features for analysis
        """
        # CNN encoding
        x = self.cnn_encoder(x)  # (batch, 384, 200)

        # Transpose for transformer: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)  # (batch, 200, 384)

        # Project to d_model
        x = self.projection(x)  # (batch, 200, 256)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, 200, 256)

        # Save features before transformer for potential analysis
        pre_transformer = x

        # Transformer encoding
        x = self.transformer(x)  # (batch, 200, 256)

        # Save post-transformer features
        post_transformer = x

        # Global pooling
        x = x.transpose(1, 2)  # (batch, 256, 200)
        x = self.pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)

        # Classification head
        x = self.fc1(x)  # (batch, 256)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (batch, num_classes)

        if return_features:
            return logits, {
                'pre_transformer': pre_transformer,
                'post_transformer': post_transformer
            }

        return logits

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention weights from transformer.

        Note: This requires hooks or custom forward pass.
        For simplicity, this returns None and should be implemented
        based on specific analysis needs.

        Args:
            x: Input sequences
            layer_idx: Which transformer layer to extract from (-1 for last)

        Returns:
            Attention weights or None
        """
        # TODO: Implement attention extraction using hooks
        # For now, use Integrated Gradients instead
        return None


def create_model(config: dict = None) -> HybridEnhancerModel:
    """
    Factory function to create model from config.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized HybridEnhancerModel
    """
    if config is None:
        config = {}

    model_config = config.get('model', {})

    return HybridEnhancerModel(
        n_transformer_layers=model_config.get('n_transformer_layers', 10),
        n_attention_heads=model_config.get('n_attention_heads', 8),
        d_model=model_config.get('d_model', 256),
        dim_feedforward=model_config.get('dim_feedforward', 1536),
        dropout=model_config.get('dropout', 0.1),
        cnn_channels=model_config.get('cnn_channels', [128, 256, 384]),
        cnn_kernels=model_config.get('cnn_kernels', [7, 11, 15]),
        sequence_length=model_config.get('sequence_length', 200),
        num_classes=model_config.get('num_classes', 2)
    )


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created with {model.get_num_params() / 1e6:.2f}M parameters")

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 4, 200)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with features
    output, features = model(x, return_features=True)
    print(f"Pre-transformer features: {features['pre_transformer'].shape}")
    print(f"Post-transformer features: {features['post_transformer'].shape}")
