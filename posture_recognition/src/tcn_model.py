#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


POSE_CLASSES = ["standing", "sitting", "squatting", "lying", "bending"]
INPUT_DIM = 62
DEFAULT_WINDOW_SIZE = 24


class CausalConv1d(nn.Conv1d):
    """Only pads the past, so output at t never observes frames after t."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation, bias=bias)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, inputs):
        return super().forward(F.pad(inputs, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time step."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs):
        return self.norm(inputs.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation)
        self.norm1 = ChannelLayerNorm(channels)
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        outputs = self.dropout(F.silu(self.norm1(self.conv1(inputs))))
        outputs = self.dropout(F.silu(self.norm2(self.conv2(outputs))))
        return F.silu(outputs + residual)


class PoseTCN(nn.Module):
    """Causal TCN over normalized per-frame 3D pose features."""

    def __init__(self, input_dim=INPUT_DIM, channels=96, kernel_size=3,
                 dilations=(1, 2, 4, 8), dropout=0.2,
                 feature_mean=None, feature_std=None):
        super().__init__()
        if channels % 8 != 0:
            raise ValueError("channels must be divisible by 8")
        mean = (torch.zeros(input_dim) if feature_mean is None
                else torch.as_tensor(feature_mean, dtype=torch.float32))
        std = (torch.ones(input_dim) if feature_std is None
               else torch.as_tensor(feature_std, dtype=torch.float32))
        if mean.numel() != input_dim or std.numel() != input_dim:
            raise ValueError("feature statistics do not match input_dim")
        self.input_dim = int(input_dim)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout_rate = float(dropout)
        self.register_buffer("feature_mean", mean.reshape(1, 1, -1))
        self.register_buffer("feature_std", std.reshape(1, 1, -1))

        # Position and first-order velocity are both used by the temporal model.
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim * 2, channels, kernel_size=1),
            ChannelLayerNorm(channels),
            nn.SiLU(),
        )
        self.temporal_blocks = nn.Sequential(*[
            TemporalResidualBlock(
                channels, kernel_size, dilation, dropout)
            for dilation in self.dilations
        ])
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, len(POSE_CLASSES)),
        )

    def _temporal_features(self, windows):
        if windows.ndim != 3 or windows.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected (batch,time,{self.input_dim}), got "
                f"{tuple(windows.shape)}")
        normalized = (windows - self.feature_mean) / self.feature_std
        normalized = torch.clamp(normalized, -8.0, 8.0)
        velocity = torch.zeros_like(normalized)
        velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        inputs = torch.cat([normalized, velocity], dim=-1).transpose(1, 2)
        return self.temporal_blocks(self.input_projection(inputs))

    def forward_sequence(self, windows):
        """Return logits for every frame, shaped (batch,time,class)."""
        encoded = self._temporal_features(windows).transpose(1, 2)
        return self.classifier(encoded)

    def forward(self, windows):
        return self.forward_sequence(windows)[:, -1]


def model_from_checkpoint(checkpoint):
    model = PoseTCN(
        input_dim=checkpoint.get("input_dim", INPUT_DIM),
        channels=checkpoint.get("channels", 96),
        kernel_size=checkpoint.get("kernel_size", 3),
        dilations=checkpoint.get("dilations", (1, 2, 4, 8)),
        dropout=checkpoint.get("dropout", 0.2),
        feature_mean=checkpoint.get("feature_mean"),
        feature_std=checkpoint.get("feature_std"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
