"""
Model factory for 3D mandible segmentation.

Provides two MONAI architectures:
- UNet:      Standard 3D U-Net. Reliable baseline, fast to train.
- SegResNet: Residual encoder with VAE regularisation. Better with
             limited data due to the regularisation term.

Both are well-established in the medical imaging literature and available
as peer-reviewed MONAI implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, UNet


def build_model(
    model_name: str = "unet",
    in_channels: int = 1,
    num_classes: int = 4,
    channels: tuple[int, ...] = (32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2),
    dropout: float = 0.2,
) -> nn.Module:
    """
    Build a 3D segmentation model.

    Parameters
    ----------
    model_name : str
        ``"unet"`` or ``"segresnet"``.
    in_channels : int
        Number of input channels (1 for grayscale CT).
    num_classes : int
        Number of output classes (4: background + 3 structures).
    channels : tuple[int, ...]
        Feature map sizes at each encoder level (UNet only).
    strides : tuple[int, ...]
        Down-sampling strides between levels (UNet only).
    dropout : float
        Dropout probability.

    Returns
    -------
    nn.Module
    """
    if model_name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=channels,
            strides=strides,
            dropout=dropout,
            num_res_units=2,
        )

    if model_name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=dropout,
        )

    raise ValueError(f"Unknown model: {model_name!r}. Choose 'unet' or 'segresnet'.")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> dict:
    """
    Load model weights and training state from a checkpoint.

    Returns the full checkpoint dict (contains epoch, optimizer state, etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
