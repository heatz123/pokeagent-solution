"""
Neural network architectures for Pokemon IL
Based on ImpalaCNN from cleanrl/qdagger_dqn_atari_jax_impalacnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x + residual
        x = F.relu(x)
        return x


class ConvSequence(nn.Module):
    """Convolutional sequence: Conv -> MaxPool -> ResBlock x2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """
    ImpalaCNN architecture for Pokemon

    Input: (B, 3, 84, 84) - preprocessed RGB images
    Output: (B, action_dim) - action logits or Q-values

    Architecture:
    - 3x ConvSequence blocks (16 -> 32 -> 32 channels)
    - Flatten
    - FC(hidden_dim) -> FC(action_dim)
    """

    def __init__(self, action_dim: int = 10, hidden_dim: int = 256):
        super().__init__()

        # Three conv sequences
        self.conv_seq1 = ConvSequence(3, 16)   # 84x84 -> 42x42
        self.conv_seq2 = ConvSequence(16, 32)  # 42x42 -> 21x21
        self.conv_seq3 = ConvSequence(32, 32)  # 21x21 -> 11x11

        # Calculate flattened size: 32 * 11 * 11 = 3872
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32 * 11 * 11, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (B, 3, 84, 84) or (B, 84, 84, 3) preprocessed observations

        Returns:
            (B, action_dim) logits/Q-values
        """
        # Ensure channel-first format (B, C, H, W)
        if x.shape[-1] == 3:  # If channel-last
            x = x.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Conv blocks
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)

        # FC layers
        x = self.fc(x)

        return x
