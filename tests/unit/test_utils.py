"""Unit tests for utility functions."""

import pytest
import torch
import torch.nn as nn

from swift_rl.utils import (
    ensure_device_compatibility,
    masked_reset,
    validate_tensor_shape,
)

# ============================================================================
# T026: Utils Tests
# ============================================================================


def test_validate_tensor_shape_valid():
    """Test validate_tensor_shape accepts correct shape."""
    tensor = torch.randn(10, 5)
    # Should not raise
    validate_tensor_shape(tensor, (10, 5), "test_tensor")


def test_validate_tensor_shape_invalid():
    """Test validate_tensor_shape rejects incorrect shape."""
    tensor = torch.randn(10, 5)

    with pytest.raises(ValueError, match="test_tensor shape"):
        validate_tensor_shape(tensor, (10, 3), "test_tensor")

    with pytest.raises(ValueError, match="test_tensor shape"):
        validate_tensor_shape(tensor, (8, 5), "test_tensor")


def test_validate_tensor_shape_scalar():
    """Test validate_tensor_shape with scalar tensors."""
    tensor = torch.tensor(5.0)
    validate_tensor_shape(tensor, (), "scalar")

    with pytest.raises(ValueError):
        validate_tensor_shape(tensor, (1,), "scalar")


def test_validate_tensor_shape_1d():
    """Test validate_tensor_shape with 1D tensors."""
    tensor = torch.randn(20)
    validate_tensor_shape(tensor, (20,), "vector")

    with pytest.raises(ValueError):
        validate_tensor_shape(tensor, (20, 1), "vector")


def test_ensure_device_compatibility_same_device():
    """Test ensure_device_compatibility with matching devices."""
    module = nn.Linear(5, 3)  # On CPU by default
    tensor = torch.randn(10, 5)  # On CPU

    result = ensure_device_compatibility(module, tensor)
    assert result.device == tensor.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ensure_device_compatibility_cuda():
    """Test ensure_device_compatibility with CUDA module."""
    module = nn.Linear(5, 3).to("cuda")
    tensor_cpu = torch.randn(10, 5)  # On CPU

    result = ensure_device_compatibility(module, tensor_cpu)
    assert result.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ensure_device_compatibility_already_on_device():
    """Test ensure_device_compatibility when tensor already on module device."""
    module = nn.Linear(5, 3).to("cuda")
    tensor_cuda = torch.randn(10, 5, device="cuda")

    result = ensure_device_compatibility(module, tensor_cuda)
    assert result is tensor_cuda  # Should return same tensor


def test_masked_reset_all_reset():
    """Test masked_reset with None indices (reset all)."""
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0])
    masked_reset(buffer, None)

    assert torch.equal(buffer, torch.zeros(4))


def test_masked_reset_partial_reset():
    """Test masked_reset with specific indices."""
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0])
    indices = torch.tensor([0, 2])

    masked_reset(buffer, indices)

    expected = torch.tensor([0.0, 2.0, 0.0, 4.0])
    assert torch.equal(buffer, expected)


def test_masked_reset_no_reset():
    """Test masked_reset with empty indices."""
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0])
    original = buffer.clone()
    indices = torch.tensor([], dtype=torch.long)

    masked_reset(buffer, indices)

    assert torch.equal(buffer, original)


def test_masked_reset_2d_buffer():
    """Test masked_reset with 2D buffer."""
    buffer = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    indices = torch.tensor([0, 2])

    masked_reset(buffer, indices)

    expected = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    assert torch.equal(buffer, expected)


def test_masked_reset_inplace():
    """Test masked_reset modifies buffer in-place."""
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0])
    buffer_id = id(buffer)
    indices = torch.tensor([0, 2])

    masked_reset(buffer, indices)

    # Should be same object (in-place)
    assert id(buffer) == buffer_id
    assert torch.equal(buffer, torch.tensor([0.0, 2.0, 0.0, 4.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masked_reset_cuda():
    """Test masked_reset with CUDA buffers."""
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    indices = torch.tensor([0, 2], device="cuda")

    masked_reset(buffer, indices)

    expected = torch.tensor([0.0, 2.0, 0.0, 4.0], device="cuda")
    assert torch.equal(buffer, expected)
    assert buffer.device.type == "cuda"


def test_masked_reset_empty_buffer():
    """Test masked_reset with empty buffer."""
    buffer = torch.tensor([])
    indices = torch.tensor([], dtype=torch.long)

    masked_reset(buffer, indices)

    assert buffer.shape == (0,)
