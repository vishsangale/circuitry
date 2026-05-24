"""Prompt-pair activation patching runner. Design spec §4."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.intervene import patch_site
from circuitry.patching.sites import ResolvedSite, Site, SiteResolver

# Model inputs may be a single tensor (toy models) or a kwargs dict
# (HF-style models called as model(**inputs)).
ModelInputs = Tensor | dict[str, Tensor]


@dataclass
class PatchResult:
    """Result of a patching run."""

    metric_values: dict[Site, float] = field(default_factory=dict)
    cached_activations: dict[Site, Tensor] = field(default_factory=dict)


def _seq_len(inputs: ModelInputs) -> int | None:
    """Best-effort sequence length (dim 1) for precondition checks.

    Returns None when the length can't be determined (e.g. a dict without an
    obvious id tensor), in which case the caller skips validation.
    """
    if isinstance(inputs, Tensor):
        return inputs.shape[1] if inputs.ndim >= 2 else None
    for key in ("input_ids", "inputs_embeds"):
        t = inputs.get(key)
        if isinstance(t, Tensor) and t.ndim >= 2:
            return t.shape[1]
    return None


class PatchRunner:
    """Orchestrates clean/corrupted prompt-pair activation patching.

    Activation patching requires position-aligned prompts: ``clean_inputs`` and
    ``corrupted_inputs`` must share the same sequence length so a cached
    activation can be substituted position-for-position into the other run.
    """

    def __init__(self, model: nn.Module, resolver: SiteResolver) -> None:
        self.model = model
        self.resolver = resolver

    def _call_model(self, inputs: ModelInputs) -> object:
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    @torch.no_grad()
    def _cache_activations(
        self,
        inputs: ModelInputs,
        sites: list[Site],
    ) -> dict[Site, Tensor]:
        """Run a forward pass and cache activations at all requested sites."""
        cache: dict[Site, Tensor] = {}
        handles = []

        for site in sites:
            resolved: ResolvedSite = self.resolver.resolve(self.model, site)

            if resolved.is_input_hook:
                def make_pre_hook(s: Site, r: ResolvedSite):
                    def hook_fn(module: nn.Module, args: tuple) -> None:
                        cache[s] = r.extract(args[0]).detach().clone()
                    return hook_fn

                h = resolved.module.register_forward_pre_hook(make_pre_hook(site, resolved))
            else:
                def make_post_hook(s: Site, r: ResolvedSite):
                    def hook_fn(module: nn.Module, inputs: tuple, output: object) -> None:
                        out = output[0] if isinstance(output, tuple) else output
                        cache[s] = r.extract(out).detach().clone()
                    return hook_fn

                h = resolved.module.register_forward_hook(make_post_hook(site, resolved))

            handles.append(h)

        was_training = self.model.training
        try:
            self.model.eval()
            self._call_model(inputs)
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()

        return cache

    def run_patching(
        self,
        clean_inputs: ModelInputs,
        corrupted_inputs: ModelInputs,
        sites: list[Site],
        metric: Callable[[Tensor], float],
        direction: Literal["denoise", "noise"] = "denoise",
    ) -> PatchResult:
        """Run activation patching over prompt pairs.

        denoise: cache clean activations, patch each into corrupted run.
        noise: cache corrupted activations, patch each into clean run.

        Raises:
            ValueError: if clean and corrupted inputs have different sequence
                lengths (activation patching requires position alignment).
        """
        clean_len = _seq_len(clean_inputs)
        corrupted_len = _seq_len(corrupted_inputs)
        if clean_len is not None and corrupted_len is not None and clean_len != corrupted_len:
            raise ValueError(
                f"activation patching requires position-aligned prompts, but "
                f"clean seq_len={clean_len} != corrupted seq_len={corrupted_len}. "
                f"Pad or trim the prompt pair to the same length."
            )

        if direction == "denoise":
            source_inputs = clean_inputs
            target_inputs = corrupted_inputs
        else:
            source_inputs = corrupted_inputs
            target_inputs = clean_inputs

        cached = self._cache_activations(source_inputs, sites)
        result = PatchResult(cached_activations=cached)

        for site in sites:
            with patch_site(self.model, site, cached[site], self.resolver):
                patched_out = self._call_model(target_inputs)
            result.metric_values[site] = metric(patched_out)

        return result
