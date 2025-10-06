"""Utility functions for Swift-RL modules."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def validate_tensor_shape(
    tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str
) -> None:
    """Validate tensor shape matches expected dimensions.

    Args:
        tensor: Input tensor to validate
        expected_shape: Expected shape tuple
        name: Name of tensor for error message

    Raises:
        ValueError: If shape mismatch
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape {tensor.shape} != expected {expected_shape}")


def ensure_device_compatibility(
    module: nn.Module, tensor: torch.Tensor
) -> torch.Tensor:
    """Move tensor to module's device if needed.

    Args:
        module: PyTorch module
        tensor: Input tensor

    Returns:
        Tensor on same device as module
    """
    module_device = next(module.parameters()).device
    if tensor.device != module_device:
        return tensor.to(module_device)
    return tensor


def masked_reset(buffer: torch.Tensor, indices: Optional[torch.Tensor]) -> None:
    """Zero out buffer at specified batch indices (in-place).

    Args:
        buffer: Tensor buffer to reset
        indices: Batch indices to reset. If None, reset all.
    """
    if indices is None:
        buffer.zero_()
    else:
        buffer[indices] = 0.0
