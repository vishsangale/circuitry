"""LLaMA-family state_dict classifier: param-name → (role, layer_idx).

Supports the common naming variants in the wild (canonical LLaMA reference
plus the ``blocks.N.`` block-prefix convention). For other families, extend
``_ROLE_PATTERNS``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

# Block-prefix variants: "layers.N." (canonical LLaMA) and "blocks.N." (alt).
_LAYER_RE = re.compile(r"^(?:layers|blocks)\.(\d+)\.")

# Role assignment — applied after stripping the layer prefix.
# Order matters: longer / more specific keys first.
_ROLE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"attention\.wq\.weight$"),    "attn_q"),
    (re.compile(r"attention\.wk\.weight$"),    "attn_k"),
    (re.compile(r"attention\.wv\.weight$"),    "attn_v"),
    (re.compile(r"attention\.wo\.weight$"),    "attn_o"),
    (re.compile(r"attention_norm\.weight$"),   "attn_norm"),
    (re.compile(r"feed_forward\.w1\.weight$"), "ffn_gate"),
    (re.compile(r"feed_forward\.w2\.weight$"), "ffn_out"),
    (re.compile(r"feed_forward\.w3\.weight$"), "ffn_in"),
    (re.compile(r"ffn_norm\.weight$"),         "ffn_norm"),
]

_GLOBAL_ROLES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^tok_embeddings\.weight$"), "embedding"),
    (re.compile(r"^output\.weight$"),         "lm_head"),
    (re.compile(r"^norm\.weight$"),           "final_norm"),
]


@dataclass(frozen=True)
class ParamInfo:
    name: str
    shape: tuple[int, ...]
    role: str | None
    layer: int | None


@dataclass
class Discovery:
    params: list[ParamInfo] = field(default_factory=list)

    def params_by_role(self) -> dict[str, list[ParamInfo]]:
        out: dict[str, list[ParamInfo]] = defaultdict(list)
        for p in self.params:
            if p.role is not None:
                out[p.role].append(p)
        return dict(out)


def discover(state_dict: Mapping[str, torch.Tensor]) -> Discovery:
    """Classify each param into (role, layer) for per-role / per-layer aggregation."""
    params: list[ParamInfo] = []
    for name, tensor in state_dict.items():
        layer = None
        layer_match = _LAYER_RE.match(name)
        if layer_match:
            layer = int(layer_match.group(1))
            suffix = name[layer_match.end():]
            role = None
            for pat, r in _ROLE_PATTERNS:
                if pat.search(suffix):
                    role = r
                    break
        else:
            role = None
            for pat, r in _GLOBAL_ROLES:
                if pat.search(name):
                    role = r
                    break
        params.append(ParamInfo(
            name=name,
            shape=tuple(tensor.shape),
            role=role,
            layer=layer,
        ))
    return Discovery(params=params)
