"""Site dataclass + resolution for activation patching. Design spec §3."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch.nn as nn
from torch import Tensor

VALID_COMPONENTS = frozenset({
    "resid_pre",
    "resid_post",
    "attn_head_out",
    "attn_head_q_out",
    "attn_head_k_out",
    "mlp_out",
    "mlp_neuron",
})


@dataclass(frozen=True)
class Site:
    """A named intervention point in the model's computation graph."""

    component: str
    layer: int
    head: int | None = None
    neuron: int | None = None
    position: int | slice | None = None

    def __post_init__(self) -> None:
        if self.component not in VALID_COMPONENTS:
            raise ValueError(
                f"Unknown component {self.component!r}; "
                f"valid: {sorted(VALID_COMPONENTS)}"
            )
        if self.component in ("attn_head_out", "attn_head_q_out", "attn_head_k_out") and self.head is None:
            raise ValueError(f"{self.component} requires head index")
        if self.component == "mlp_neuron" and self.neuron is None:
            raise ValueError("mlp_neuron requires neuron index")


@dataclass
class ResolvedSite:
    """A Site resolved to a concrete module + hook functions."""

    module: nn.Module
    is_input_hook: bool
    extract: Callable[[Tensor], Tensor]
    inject: Callable[[Tensor, Tensor], Tensor]


class SiteResolver(Protocol):
    """Structural type for site resolvers (HFSiteResolver, TLSiteResolver)."""

    def resolve(self, model: nn.Module, site: Site) -> ResolvedSite: ...


# --------------- position helpers ---------------

def _pos_extract(x: Tensor, pos: int | slice | None) -> Tensor:
    if pos is None:
        return x
    return x[:, pos]


def _pos_inject(full: Tensor, val: Tensor, pos: int | slice | None) -> Tensor:
    if pos is None:
        return val
    out = full.clone()
    out[:, pos] = val
    return out


# --------------- head helpers ---------------

def _extract_head(
    x: Tensor, head: int, n_heads: int, head_dim: int,
    pos: int | slice | None,
) -> Tensor:
    b, s, _ = x.shape
    reshaped = x.reshape(b, s, n_heads, head_dim)
    sliced = reshaped[:, :, head, :]
    return _pos_extract(sliced, pos)


def _inject_head(
    full: Tensor, val: Tensor, head: int, n_heads: int, head_dim: int,
    pos: int | slice | None,
) -> Tensor:
    b, s, d = full.shape
    out = full.clone().reshape(b, s, n_heads, head_dim)
    if pos is None:
        out[:, :, head, :] = val
    else:
        out[:, pos, head, :] = val
    return out.reshape(b, s, d)


# --------------- neuron helpers ---------------

def _extract_neuron(x: Tensor, neuron: int, pos: int | slice | None) -> Tensor:
    sliced = _pos_extract(x, pos)
    return sliced[..., neuron]


def _inject_neuron(
    full: Tensor, val: Tensor, neuron: int, pos: int | slice | None,
) -> Tensor:
    out = full.clone()
    if pos is None:
        out[..., neuron] = val
    else:
        out[:, pos, neuron] = val
    return out


# --------------- module traversal ---------------

def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    parts = path.split(".")
    m: Any = model
    for p in parts:
        m = getattr(m, p)
    return m  # type: ignore[return-value]


# --------------- HF site resolver ---------------

class HFSiteResolver:
    """Resolve Sites to HF model modules using config-declared layout.

    Per-architecture slicing notes:
    - attn_head_out hooks o_proj INPUT, reshaping (batch, seq, d_model) into
      (batch, seq, n_heads, head_dim). Requires eager attention to have a
      meaningful per-head decomposition.
    - mlp_neuron hooks the down_proj INPUT and indexes the last dim. This is
      the SwiGLU/GeGLU intermediate (Llama-family). Other MLP layouts need a
      different intermediate module — supply mlp_intermediate explicitly.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_mlp: int | None = None,
        *,
        layer_pattern: str = "model.layers.{L}",
        attn_module: str = "self_attn.o_proj",
        mlp_module: str = "mlp",
        mlp_intermediate: str = "mlp.down_proj",
    ) -> None:
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.head_dim = d_model // n_heads
        self.layer_pattern = layer_pattern
        self.attn_module = attn_module
        self.mlp_module = mlp_module
        self.mlp_intermediate = mlp_intermediate

    @classmethod
    def from_config(cls, config: Any) -> HFSiteResolver:
        n_heads = getattr(config, "num_attention_heads", None)
        d_model = getattr(config, "hidden_size", None)
        if n_heads is None or d_model is None:
            raise ValueError(
                "Config must have num_attention_heads and hidden_size"
            )
        d_mlp = getattr(config, "intermediate_size", None)
        return cls(n_heads=n_heads, d_model=d_model, d_mlp=d_mlp)

    def _layer_module(self, model: nn.Module, layer: int) -> nn.Module:
        path = self.layer_pattern.replace("{L}", str(layer))
        return _get_submodule(model, path)

    def resolve(self, model: nn.Module, site: Site) -> ResolvedSite:
        # Validate d_mlp early so the error is clear even if layer_pattern
        # doesn't match the model structure.
        if site.component == "mlp_neuron" and self.d_mlp is None:
            raise ValueError(
                "mlp_neuron resolution requires d_mlp (intermediate_size) in config"
            )

        layer_mod = self._layer_module(model, site.layer)
        pos = site.position

        if site.component == "resid_pre":
            return ResolvedSite(
                module=layer_mod,
                is_input_hook=True,
                extract=lambda x, _pos=pos: _pos_extract(x, _pos),
                inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
            )

        if site.component == "resid_post":
            return ResolvedSite(
                module=layer_mod,
                is_input_hook=False,
                extract=lambda x, _pos=pos: _pos_extract(x, _pos),
                inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
            )

        if site.component == "attn_head_out":
            attn_mod = _get_submodule(layer_mod, self.attn_module)
            h, nh, hd = site.head, self.n_heads, self.head_dim
            return ResolvedSite(
                module=attn_mod,
                is_input_hook=True,
                extract=lambda x, _h=h, _nh=nh, _hd=hd, _pos=pos: _extract_head(x, _h, _nh, _hd, _pos),
                inject=lambda full, val, _h=h, _nh=nh, _hd=hd, _pos=pos: _inject_head(full, val, _h, _nh, _hd, _pos),
            )

        if site.component == "attn_head_q_out":
            q_proj = _get_submodule(layer_mod, "self_attn.q_proj")
            h, nh, hd = site.head, self.n_heads, self.head_dim
            return ResolvedSite(
                module=q_proj,
                is_input_hook=False,
                extract=lambda x, _h=h, _nh=nh, _hd=hd, _pos=pos: _extract_head(x, _h, _nh, _hd, _pos),
                inject=lambda full, val, _h=h, _nh=nh, _hd=hd, _pos=pos: _inject_head(full, val, _h, _nh, _hd, _pos),
            )

        if site.component == "attn_head_k_out":
            k_proj = _get_submodule(layer_mod, "self_attn.k_proj")
            h, nh, hd = site.head, self.n_heads, self.head_dim
            return ResolvedSite(
                module=k_proj,
                is_input_hook=False,
                extract=lambda x, _h=h, _nh=nh, _hd=hd, _pos=pos: _extract_head(x, _h, _nh, _hd, _pos),
                inject=lambda full, val, _h=h, _nh=nh, _hd=hd, _pos=pos: _inject_head(full, val, _h, _nh, _hd, _pos),
            )

        if site.component == "mlp_out":
            mlp_mod = _get_submodule(layer_mod, self.mlp_module)
            return ResolvedSite(
                module=mlp_mod,
                is_input_hook=False,
                extract=lambda x, _pos=pos: _pos_extract(x, _pos),
                inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
            )

        if site.component == "mlp_neuron":
            intermediate_mod = _get_submodule(layer_mod, self.mlp_intermediate)
            n = site.neuron
            return ResolvedSite(
                module=intermediate_mod,
                is_input_hook=True,
                extract=lambda x, _n=n, _pos=pos: _extract_neuron(x, _n, _pos),
                inject=lambda full, val, _n=n, _pos=pos: _inject_neuron(full, val, _n, _pos),
            )

        raise ValueError(f"Unresolved component: {site.component}")


# --------------- TL site resolver ---------------

_TL_HOOK_MAP = {
    "resid_pre": "blocks.{L}.hook_resid_pre",
    "resid_post": "blocks.{L}.hook_resid_post",
    "attn_head_out": "blocks.{L}.attn.hook_z",
    "attn_head_q_out": "blocks.{L}.attn.hook_q",
    "attn_head_k_out": "blocks.{L}.attn.hook_k",
    "mlp_out": "blocks.{L}.mlp.hook_post",
    "mlp_neuron": "blocks.{L}.mlp.hook_post",
}


class TLSiteResolver:
    """Resolve Sites to TransformerLens hook names. Lazy transformer_lens import."""

    def hook_name(self, site: Site) -> str:
        template = _TL_HOOK_MAP.get(site.component)
        if template is None:
            raise ValueError(f"No TL hook mapping for {site.component}")
        return template.replace("{L}", str(site.layer))

    def resolve(self, model: nn.Module, site: Site) -> ResolvedSite:
        try:
            import transformer_lens  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformer_lens is required for TLSiteResolver.resolve(). "
                "Install it with: pip install transformer_lens"
            ) from None

        hook_name = self.hook_name(site)
        hook_point = model.hook_dict[hook_name]  # type: ignore[attr-defined]
        pos = site.position

        if site.component == "attn_head_out":
            head = site.head
            return ResolvedSite(
                module=hook_point,
                is_input_hook=False,
                extract=lambda x, _h=head, _pos=pos: (
                    _pos_extract(x[:, :, _h, :], _pos) if x.ndim == 4
                    else _pos_extract(x, _pos)
                ),
                inject=lambda full, val, _h=head, _pos=pos: _inject_tl_head(full, val, _h, _pos),
            )

        if site.component in ("attn_head_q_out", "attn_head_k_out"):
            head = site.head
            return ResolvedSite(
                module=hook_point,
                is_input_hook=False,
                extract=lambda x, _h=head, _pos=pos: (
                    _pos_extract(x[:, :, _h, :], _pos) if x.ndim == 4
                    else _pos_extract(x, _pos)
                ),
                inject=lambda full, val, _h=head, _pos=pos: _inject_tl_head(full, val, _h, _pos),
            )

        if site.component == "mlp_neuron":
            neuron = site.neuron
            return ResolvedSite(
                module=hook_point,
                is_input_hook=False,
                extract=lambda x, _n=neuron, _pos=pos: _extract_neuron(x, _n, _pos),
                inject=lambda full, val, _n=neuron, _pos=pos: _inject_neuron(full, val, _n, _pos),
            )

        return ResolvedSite(
            module=hook_point,
            is_input_hook=False,
            extract=lambda x, _pos=pos: _pos_extract(x, _pos),
            inject=lambda full, val, _pos=pos: _pos_inject(full, val, _pos),
        )


def _inject_tl_head(
    full: Tensor, val: Tensor, head: int, pos: int | slice | None,
) -> Tensor:
    out = full.clone()
    if pos is None:
        out[:, :, head, :] = val
    else:
        out[:, pos, head, :] = val
    return out
