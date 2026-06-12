"""Tests for circuitry.patching.mean_ablation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.mean_ablation import compute_mean_activation, mean_ablation

# ---------------------------------------------------------------------------
# Shared toy model
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4, bias=False)
        self.out = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.out(self.layer(x))


@pytest.fixture()
def model():
    m = TinyModel()
    torch.manual_seed(0)
    nn.init.normal_(m.layer.weight)
    nn.init.normal_(m.out.weight)
    return m


# ---------------------------------------------------------------------------
# compute_mean_activation tests
# ---------------------------------------------------------------------------

def test_compute_mean_activation_returns_tensor(model):
    """compute_mean_activation returns a torch.Tensor."""
    inputs = [torch.randn(3, 4) for _ in range(5)]
    result = compute_mean_activation(model, model.layer, inputs)
    assert isinstance(result, Tensor)


def test_compute_mean_activation_shape(model):
    """For (batch=3, d=4) inputs through model.layer (output (3,4)), mean shape is (4,)."""
    inputs = [torch.randn(3, 4) for _ in range(4)]
    result = compute_mean_activation(model, model.layer, inputs)
    assert result.shape == (4,)


def test_compute_mean_activation_correctness_constant_input(model):
    """For constant input ones, mean equals model.layer(ones[0])."""
    ones = torch.ones(3, 4)
    inputs = [ones, ones, ones]
    result = compute_mean_activation(model, model.layer, inputs)
    with torch.no_grad():
        expected = model.layer(ones[0])
    assert torch.allclose(result, expected, atol=1e-5)


def test_compute_mean_activation_averages_multiple_inputs(model):
    """compute_mean_activation correctly averages across multiple dataset inputs."""
    torch.manual_seed(42)
    inp_a = torch.randn(2, 4)
    inp_b = torch.randn(2, 4)
    result = compute_mean_activation(model, model.layer, [inp_a, inp_b])

    with torch.no_grad():
        mean_a = model.layer(inp_a).mean(dim=0)
        mean_b = model.layer(inp_b).mean(dim=0)
    expected = (mean_a + mean_b) / 2.0
    assert torch.allclose(result, expected, atol=1e-5)


def test_compute_mean_activation_raises_on_empty_inputs(model):
    """compute_mean_activation raises ValueError for empty dataset_inputs."""
    with pytest.raises(ValueError):
        compute_mean_activation(model, model.layer, [])


# ---------------------------------------------------------------------------
# mean_ablation tests
# ---------------------------------------------------------------------------

def test_mean_ablation_is_context_manager(model):
    """mean_ablation can be used as a context manager without error."""
    mean_act = torch.zeros(4)
    x = torch.randn(2, 4)
    with mean_ablation(model, model.layer, mean_act):
        _ = model(x)  # should not raise


def test_mean_ablation_replaces_output(model):
    """Output inside mean_ablation context differs from un-ablated output."""
    x = torch.randn(2, 4)
    with torch.no_grad():
        normal_out = model(x).clone()

    # Use a non-trivial mean (all ones) that is unlikely to match normal output
    mean_act = torch.ones(4)
    with mean_ablation(model, model.layer, mean_act):
        with torch.no_grad():
            ablated_out = model(x)

    assert not torch.allclose(normal_out, ablated_out)


def test_mean_ablation_restores_original_behavior(model):
    """After context exits, model returns to normal (un-ablated) behavior."""
    x = torch.randn(2, 4)
    with torch.no_grad():
        before = model(x).clone()

    mean_act = torch.ones(4) * 99.0
    with mean_ablation(model, model.layer, mean_act):
        pass  # enter and immediately exit

    with torch.no_grad():
        after = model(x)

    assert torch.allclose(before, after, atol=1e-6)


def test_mean_ablation_output_is_deterministic(model):
    """Inside context, repeated calls with same input give identical output."""
    x = torch.randn(2, 4)
    mean_act = torch.randn(4)

    with mean_ablation(model, model.layer, mean_act):
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()

    assert torch.allclose(out1, out2)


def test_mean_ablation_broadcasts_correctly(model):
    """(d_model,) mean_act broadcasts correctly to (batch, d_model) layer output."""
    batch = 5
    x = torch.randn(batch, 4)
    mean_act = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Capture what the layer actually produces inside the context
    captured = []

    def _spy(mod, inp, out):
        # This runs after the ablation hook replaces the output
        captured.append(out.detach().clone())

    with mean_ablation(model, model.layer, mean_act):
        # We need to capture layer output after ablation; register a second hook
        handle = model.layer.register_forward_hook(_spy)
        with torch.no_grad():
            model(x)
        handle.remove()

    # Each row of the captured output should equal mean_act
    layer_out = captured[0]  # shape (batch, 4)
    assert layer_out.shape == (batch, 4)
    for i in range(batch):
        assert torch.allclose(layer_out[i], mean_act.float(), atol=1e-5)


def test_mean_ablation_works_with_dict_inputs(model):
    """mean_ablation works when inputs are passed as keyword arguments."""
    # Wrap TinyModel to accept x as a keyword argument
    class WrappedModel(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    wrapped = WrappedModel(model)
    x_tensor = torch.randn(2, 4)
    # Pass dict input to compute_mean_activation — exercises the dict branch
    dict_inputs = [{"x": x_tensor}]
    result = compute_mean_activation(wrapped, wrapped.inner.layer, dict_inputs)
    assert isinstance(result, Tensor)
    assert result.shape == (4,)


def test_mean_ablation_no_hooks_remain_after_exit(model):
    """After context exits, no extra forward hooks remain on the module."""
    original_count = len(model.layer._forward_hooks)
    mean_act = torch.zeros(4)
    with mean_ablation(model, model.layer, mean_act):
        pass
    assert len(model.layer._forward_hooks) == original_count
