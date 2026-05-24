"""Prompt-pair activation patching runner. Design spec §4."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.intervene import patch_site
from circuitry.patching.sites import ResolvedSite, Site


@dataclass
class PatchResult:
    """Result of a patching run."""

    metric_values: dict[Site, float] = field(default_factory=dict)
    cached_activations: dict[Site, Tensor] = field(default_factory=dict)


class PatchRunner:
    """Orchestrates clean/corrupted prompt-pair activation patching."""

    def __init__(self, model: nn.Module, resolver: object) -> None:
        self.model = model
        self.resolver = resolver

    @torch.no_grad()
    def _cache_activations(
        self,
        inputs: Tensor,
        sites: list[Site],
    ) -> dict[Site, Tensor]:
        """Run a forward pass and cache activations at all requested sites."""
        cache: dict[Site, Tensor] = {}
        handles = []

        for site in sites:
            resolved: ResolvedSite = self.resolver.resolve(self.model, site)  # type: ignore[attr-defined]

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
            self.model(inputs)
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()

        return cache

    def run_patching(
        self,
        clean_inputs: Tensor,
        corrupted_inputs: Tensor,
        sites: list[Site],
        metric: Callable[[Tensor], float],
        direction: Literal["denoise", "noise"] = "denoise",
    ) -> PatchResult:
        """Run activation patching over prompt pairs.

        denoise: cache clean activations, patch each into corrupted run.
        noise: cache corrupted activations, patch each into clean run.
        """
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
                patched_out = self.model(target_inputs)
            result.metric_values[site] = metric(patched_out)

        return result
