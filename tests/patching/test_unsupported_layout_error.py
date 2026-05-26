"""Test that EAPRunner and AtPRunner raise a clear ValueError (pointing to
to_hooked_transformer) when given a model without Llama-family layer layout.
"""
import pytest
import torch.nn as nn

from circuitry.patching import AtPRunner, EAPRunner
from circuitry.patching.sites import HFSiteResolver


class _NoLayers(nn.Module):
    """A non-Llama-layout model: has neither model.layers nor model.model.layers."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8)])
        self.config = type("C", (), {"num_attention_heads": 2, "hidden_size": 8})()

    def forward(self, **kw):
        raise AssertionError("should fail at layer location, before forward")


@pytest.mark.parametrize("Runner", [EAPRunner, AtPRunner])
def test_unsupported_layout_raises_clear_error(Runner):
    resolver = HFSiteResolver(n_heads=2, d_model=8)
    with pytest.raises(ValueError, match="to_hooked_transformer"):
        Runner(_NoLayers(), resolver=resolver)
