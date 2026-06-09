"""SAE feature attribution runner.

Node-level AtP*-style attribution for SAE features. Mechanism: error-term
substitution (Marks "Sparse Feature Circuits"). Supported sites (v1.7+):
resid_post, mlp_out, attn_out. Supported backends: HFSiteResolver (eager) and
TLSiteResolver (TransformerLens). Supported variants: 'attrib' (AtP*) and 'ig'
(integrated gradients, v1.7+). Optional error-node scoring via include_error_node.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from circuitry.patching.atp import AtPNode, AtPResult, AtPRunner
from circuitry.patching.graph import Node
from circuitry.patching.sites import Site
from circuitry.sae.grad import assert_supported_sae, sae_decompose

# Type alias matching AtPRunner
_Inputs = Tensor | dict[str, Any]


def _freeze_sae(sae: Any) -> dict[str, bool]:
    """Freeze all SAE parameters; return original requires_grad map."""
    orig: dict[str, bool] = {}
    if isinstance(sae, nn.Module):
        for name, p in sae.named_parameters():
            orig[name] = p.requires_grad
            p.requires_grad_(False)
    return orig


def _restore_sae(sae: Any, orig: dict[str, bool]) -> None:
    """Restore SAE parameter requires_grad states."""
    if isinstance(sae, nn.Module):
        for name, p in sae.named_parameters():
            if name in orig:
                p.requires_grad_(orig[name])


def _extract_tensor(output: Any) -> Tensor:
    """Extract a Tensor from a layer output that may be a Tensor or tuple."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Cannot extract Tensor from layer output type {type(output)!r}")


def _inject_tensor(output: Any, new_tensor: Tensor) -> Any:
    """Inject new_tensor back into a layer output (Tensor or tuple)."""
    if isinstance(output, Tensor):
        return new_tensor
    if isinstance(output, (tuple, list)):
        lst = list(output)
        lst[0] = new_tensor
        return type(output)(lst)
    raise TypeError(f"Cannot inject Tensor into layer output type {type(output)!r}")


def _routed_extract(resolved: Any, output: Any) -> Tensor:
    """Unwrap a (possibly tuple) module output then slice to the resolved sub-activation.

    Composition order (spec §1):
      output → _extract_tensor(output) → resolved.extract(full_tensor)

    _extract_tensor must be the OUTER layer because resolved.extract indexes a
    tensor (x[:, pos]) and would crash on a raw tuple.  For resid_post +
    position=None the extract is the identity, so this is a no-op.
    """
    assert not resolved.is_input_hook, (
        "_routed_extract must only be called inside forward hooks "
        "(is_input_hook=False).  All valid SAE sites (resid_post, mlp_out, "
        "attn_out) are output hooks."
    )
    return resolved.extract(_extract_tensor(output))


def _routed_inject(resolved: Any, output: Any, new_sub: Tensor) -> Any:
    """Write the resolved sub-activation back, then rewrap into the (possibly tuple) output.

    Composition order (spec §1):
      new_sub → resolved.inject(full_tensor, new_sub) → _inject_tensor(output, new_full)

    For resid_post + position=None the inject is the identity (returns new_sub),
    so this reduces to _inject_tensor(output, new_sub).
    """
    assert not resolved.is_input_hook, (
        "_routed_inject must only be called inside forward hooks "
        "(is_input_hook=False).  All valid SAE sites (resid_post, mlp_out, "
        "attn_out) are output hooks."
    )
    full = _extract_tensor(output)
    return _inject_tensor(output, resolved.inject(full, new_sub))


class TranscoderWrapper:
    """Wrap a transcoder (module-input → module-output feature decomposition) as an
    SAE-compatible object for SAEFeatureRunner and SAEFeatureEdgeRunner.

    A transcoder encodes from the MODULE INPUT (``inp[0]`` in the forward hook) and
    produces a reconstruction ``x_hat`` in the MODULE OUTPUT space.  The error term
    is ``eps = output − x_hat`` (in output space).  The splice is always lossless:
    ``x_hat + eps = output``.

    ``hook_input = True`` signals the attribution hooks to route encoding through
    ``inp[0]`` instead of ``output``.

    Usage::

        tc = TranscoderWrapper(my_transcoder)  # wraps any obj with encode/decode
        runner = SAEFeatureRunner(model, {Site("mlp_out", layer=0): tc}, resolver)

    The wrapped transcoder must implement ``encode(x_in: Tensor) -> Tensor`` (where
    ``x_in`` is the module input) and ``decode(f: Tensor) -> Tensor`` (where the
    output is in module output space).
    """

    hook_input: bool = True

    def __init__(self, transcoder: Any) -> None:
        self._tc = transcoder

    def encode(self, x: Tensor) -> Tensor:
        return self._tc.encode(x)

    def decode(self, f: Tensor) -> Tensor:
        return self._tc.decode(f)

    @property
    def device(self) -> Any:
        return getattr(self._tc, "device", torch.device("cpu"))

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self._tc, "dtype", torch.float32)


class CrosscoderWrapper:
    """Wraps a crosscoder SAE as a single-site intervention point.

    A crosscoder reads activations from N layers simultaneously and produces
    a shared feature vector.  This wrapper exposes it as a standard SAE
    (encode / decode) by fixing a *primary layer* for single-site attribution.

    hook_input is False (residual-stream output hook, same as standard SAEs).
    Cross-layer analysis (all layers simultaneously) is available via
    encode_all() for users who need it.

    The wrapped crosscoder object must implement one of:
      • crosscoder.encode(x)                   — single-tensor input
      • crosscoder.encode_at_layer(x, layer)   — explicit layer routing
    and one of:
      • crosscoder.decode(f)                   — single output
      • crosscoder.decode_at_layer(f, layer)   — layer-specific decode
    """

    hook_input: bool = False

    def __init__(self, crosscoder: Any, *, primary_layer: int = 0) -> None:
        self._cc = crosscoder
        self.primary_layer = primary_layer

    def encode(self, x: Tensor) -> Tensor:
        if hasattr(self._cc, "encode_at_layer"):
            return self._cc.encode_at_layer(x, self.primary_layer)
        return self._cc.encode(x)

    def decode(self, f: Tensor) -> Tensor:
        if hasattr(self._cc, "decode_at_layer"):
            return self._cc.decode_at_layer(f, self.primary_layer)
        return self._cc.decode(f)

    def encode_all(self, acts: list[Tensor]) -> Tensor:
        """Encode from all layers simultaneously (full cross-layer mode)."""
        return self._cc.encode(acts)

    @property
    def device(self) -> Any:
        return getattr(self._cc, "device", torch.device("cpu"))

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self._cc, "dtype", torch.float32)


class SAEFeatureRunner:
    """Node-level SAE feature attribution via AtP*-style gradient estimation.

    For each configured SAE site, splices the SAE losslessly into the
    clean forward pass and scores each active feature by:

        score(i) = Σ_pos ( Δf[..., i] · gradf[..., i] )

    where Δf = f_corrupt − f_clean and gradf = ∂metric/∂f at the clean
    activation.  This is the same formula as mlp_neuron scoring in atp.py.

    Supported components (v1.7 P2a): resid_post, mlp_out, attn_out.
    Per-head/per-neuron sub-slices (attn_head_out, mlp_neuron) are NOT supported.
    TransformerLens backend (TLSiteResolver) is supported (v1.7 P3).
    Only one SAE site per layer is supported (multi-site-per-layer is P2b).

    Arch note (Llama-family): mlp_out captures the MLP submodule output BEFORE
    the residual add; attn_out captures self_attn output before the residual add.
    For Gemma2 (post-norm before residual), the submodule output is the pre-norm
    tensor — the splice is lossless but the decomposed tensor differs from what
    a Gemma2-trained SAE would expect.  Scope the equivalence claim to Llama-family.
    Parallel-attention architectures (GPT-J-style) make attn_out@L and mlp_out@L
    causally unordered; intra-layer edges are invalid there (v1.7 assumes sequential).
    """

    def __init__(
        self,
        model: nn.Module,
        sae_sites: dict[Site, Any],
        resolver: Any,
    ) -> None:
        """
        Args:
            model:      The HF/toy model to analyse.
            sae_sites:  Mapping from Site → SAE object (or (release, sae_id) tuple
                        which is loaded via circuitry.sae.loader.load_sae).
            resolver:   An HFSiteResolver or TLSiteResolver.
        """
        self.model = model
        self.resolver = resolver

        # Resolve (release, sae_id) tuples; gate valid SAE components + validate architecture
        _VALID_SAE_COMPONENTS = {"resid_post", "mlp_out", "attn_out"}
        resolved_sae_sites: dict[Site, Any] = {}
        for site, sae_or_tuple in sae_sites.items():
            if site.component not in _VALID_SAE_COMPONENTS:
                raise NotImplementedError(
                    f"SAEFeatureRunner supports only {sorted(_VALID_SAE_COMPONENTS)} sites "
                    f"(got {site.component!r}). Per-head/per-neuron sub-slices "
                    "(attn_head_out, mlp_neuron, resid_pre, ...) are not supported."
                )
            if site.position is not None:
                raise NotImplementedError(
                    f"SAEFeatureRunner does not support positional slicing (site.position={site.position!r}). "
                    "The block-hook splice consumed the whole tensor; routing must not silently introduce "
                    "positional slicing. Only position=None is supported."
                )
            if isinstance(sae_or_tuple, tuple) and len(sae_or_tuple) == 2:
                from circuitry.sae.loader import load_sae
                release, sae_id = sae_or_tuple
                sae = load_sae(release, sae_id)
            else:
                sae = sae_or_tuple
            assert_supported_sae(sae)
            resolved_sae_sites[site] = sae

        self._sae_sites = resolved_sae_sites

        # Borrow _freeze_eval / _restore / _locate_layers from AtPRunner by composition
        self._atp = AtPRunner(model, resolver)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_model(self, inputs: _Inputs) -> Any:
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        graddrop: bool = False,
        include_error_node: bool = False,
        max_features: int | None = None,
        variant: str = "attrib",
        n_ig_steps: int = 0,
    ) -> AtPResult:
        """Compute per-feature attribution scores.

        Steps:
          1. Freeze model + all SAE params (try/finally restore).
          2. Corrupted forward (no_grad, no splice): capture f_corrupt per site.
             Also capture eps_corrupt if include_error_node.
          3. Clean forward + backward (enable_grad, SPLICED per §2.1/2.2):
             capture f (retain_grad), f_clean, eps_clean, err_leaf.
          4. Enumerate features where Δf ≠ 0 (union clean-active | corr-active).
             Score each by Σ_pos Δf_i · gradf_i (fp32 accumulation).
          5. If include_error_node: score the error node from err_leaf.grad.
          6. Restore; return AtPResult.

        graddrop:
            Use |per-position contribution| instead of signed sum.
        include_error_node:
            Add a sae_error node for the reconstruction error term.
        max_features:
            If set, keep only the top-|score| features per site.
        variant:
            'attrib' (default): single-point AtP gradient attribution (Δf · grad@clean).
            'ig': integrated gradients along the path f_clean → f_corrupt (Δf = f_corrupt − f_clean).
                  Uses N midpoint samples α_k=(k-0.5)/N for O(1/N²) convergence.
                  Scores complete to metric(decode(f_corrupt)+eps_clean)−metric(decode(f_clean)+eps_clean)
                  (eps-frozen spliced delta) when include_error_node=False.
                  When include_error_node=True, the joint path (features + error interpolated together)
                  completes to the real forward delta metric(corrupt)−metric(clean).
                  Feature scores may differ slightly between include_error_node True/False because the
                  gradient is evaluated along different eps trajectories — this is correct.
            Other values: NotImplementedError.
        n_ig_steps:
            Number of IG integration steps (variant='ig' only). 0 → default of 32.
        """
        was_training, orig_rg = self._atp._freeze_eval()
        sae_orig_rg: dict[Site, dict[str, bool]] = {}
        for site, sae in self._sae_sites.items():
            sae_orig_rg[site] = _freeze_sae(sae)

        if variant not in ("attrib", "ig"):
            raise NotImplementedError(
                f"SAEFeatureRunner variant={variant!r} is not supported. "
                "Supported values: 'attrib', 'ig'."
            )

        try:
            scores: dict[AtPNode, float] = {}

            for site, sae in self._sae_sites.items():
                site_scores = self._run_site(
                    site=site,
                    sae=sae,
                    clean_inputs=clean_inputs,
                    corrupted_inputs=corrupted_inputs,
                    metric=metric,
                    graddrop=graddrop,
                    include_error_node=include_error_node,
                    max_features=max_features,
                    variant=variant,
                    n_ig_steps=n_ig_steps,
                )
                scores.update(site_scores)

            return AtPResult(scores)
        finally:
            self._atp._restore(was_training, orig_rg)
            for site, sae in self._sae_sites.items():
                _restore_sae(sae, sae_orig_rg.get(site, {}))

    def _run_site(
        self,
        site: Site,
        sae: Any,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        *,
        graddrop: bool,
        include_error_node: bool,
        max_features: int | None,
        variant: str = "attrib",
        n_ig_steps: int = 0,
    ) -> dict[AtPNode, float]:
        """Run attribution for a single (site, sae) pair."""
        # Route through ResolvedSite — for resid_post + position=None this is the identity.
        resolved = self.resolver.resolve(self.model, site)
        layer_mod = resolved.module

        # Model dtype/device — used to cast spliced tensor back.
        # On the TL path, HookPoint.parameters() == [], so the params-fallback
        # silently downcasts (e.g. fp64 → fp32).  Read from model.cfg instead.
        from circuitry.patching.sites import TLSiteResolver
        if isinstance(self.resolver, TLSiteResolver):
            model_dtype = self.model.cfg.dtype
            model_device = torch.device(self.model.cfg.device)
        else:
            model_params = list(layer_mod.parameters())
            if model_params:
                model_dtype = model_params[0].dtype
                model_device = model_params[0].device
            else:
                model_dtype = torch.float32
                model_device = torch.device("cpu")

        # ----------------------------------------------------------------
        # Step 1: corrupted forward — capture f_corrupt (and eps_corrupt)
        # ----------------------------------------------------------------
        f_corrupt_store: dict[str, Tensor] = {}
        eps_corrupt_store: dict[str, Tensor] = {}

        def _corr_output_hook(
            module: nn.Module, inp: Any, output: Any,
            _sae: Any = sae,
            _store: dict = f_corrupt_store,
            _eps_store: dict = eps_corrupt_store,
            _inc_err: bool = include_error_node,
            _resolved: Any = resolved,
        ) -> None:
            with torch.no_grad():
                if getattr(_sae, "hook_input", False):
                    # Transcoder: encode from module input, eps in output space
                    a_in = inp[0].detach().to(
                        getattr(_sae, "device", inp[0].device),
                        getattr(_sae, "dtype", inp[0].dtype),
                    )
                    a_out = _extract_tensor(output).detach().to(a_in.device, a_in.dtype)
                    f_c = _sae.encode(a_in)
                    x_hat_c = _sae.decode(f_c)
                    eps_c = (a_out - x_hat_c).detach()
                else:
                    a = _routed_extract(_resolved, output).detach()
                    a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                    f_c, x_hat_c, eps_c = sae_decompose(_sae, a_in)
                _store["f"] = f_c.detach()
                if _inc_err:
                    _eps_store["eps"] = eps_c  # already detached

        corr_hook = layer_mod.register_forward_hook(_corr_output_hook)
        try:
            with torch.no_grad():
                self._call_model(corrupted_inputs)
        finally:
            corr_hook.remove()

        f_corrupt = f_corrupt_store.get("f")
        if f_corrupt is None:
            return {}

        eps_corrupt = eps_corrupt_store.get("eps")  # may be None

        # ----------------------------------------------------------------
        # Step 2: clean forward + backward — spliced
        # ----------------------------------------------------------------
        f_leaf_store: dict[str, Tensor] = {}
        err_leaf_store: dict[str, Tensor] = {}
        eps_clean_store: dict[str, Tensor] = {}

        def _clean_output_hook(
            module: nn.Module, inp: Any, output: Any,
            _sae: Any = sae,
            _f_store: dict = f_leaf_store,
            _err_store: dict = err_leaf_store,
            _eps_clean_s: dict = eps_clean_store,
            _inc_err: bool = include_error_node,
            _mdtype: torch.dtype = model_dtype,
            _mdev: Any = model_device,
            _resolved: Any = resolved,
        ) -> Any:
            if getattr(_sae, "hook_input", False):
                # Transcoder: encode from module input; eps = output − x_hat (output space)
                a_in = inp[0].detach().to(
                    getattr(_sae, "device", inp[0].device),
                    getattr(_sae, "dtype", inp[0].dtype),
                )
                a_out = _routed_extract(_resolved, output)  # live output for inject + eps
                # §2.1: seed grad AT the feature tensor (from module input)
                f = _sae.encode(a_in).detach().requires_grad_(True)
                f.retain_grad()
                x_hat = _sae.decode(f)
                eps = (a_out.detach() - x_hat.detach()).detach()  # frozen transcoder error
            else:
                # Standard SAE: encode from module output
                a = _routed_extract(_resolved, output)
                a_in = a.detach().to(
                    getattr(_sae, "device", a.device),
                    getattr(_sae, "dtype", a.dtype),
                )
                # §2.1: seed grad AT the feature tensor
                f = _sae.encode(a_in).detach().requires_grad_(True)
                f.retain_grad()
                x_hat = _sae.decode(f)
                eps = (a_in - x_hat).detach()  # frozen clean reconstruction error

            _f_store["f"] = f
            _eps_clean_s["eps"] = eps

            if _inc_err:
                # §2.2: independent leaf so both f.grad and err_leaf.grad are non-zero
                err_leaf = eps.clone().requires_grad_(True)
                err_leaf.retain_grad()
                recon = x_hat + err_leaf
                _err_store["err_leaf"] = err_leaf
            else:
                recon = x_hat + eps

            # Cast back to model dtype/device and splice via resolver
            recon_cast = recon.to(_mdev, _mdtype)
            return _routed_inject(_resolved, output, recon_cast)

        clean_hook = layer_mod.register_forward_hook(_clean_output_hook)
        try:
            with torch.enable_grad():
                out = self._call_model(clean_inputs)
                m = metric(out)
                m.backward()
        finally:
            clean_hook.remove()

        f_clean_leaf = f_leaf_store.get("f")
        err_leaf = err_leaf_store.get("err_leaf")
        eps_clean = eps_clean_store.get("eps")

        if f_clean_leaf is None:
            return {}

        # ----------------------------------------------------------------
        # Step 3: check that gradients are populated
        # ----------------------------------------------------------------
        if f_clean_leaf.grad is None:
            raise RuntimeError(
                "f.grad is None after backward — the metric must return a differentiable "
                "Tensor (use logit_diff_t, not the float logit_diff). "
                f"Site: {site}."
            )

        # ----------------------------------------------------------------
        # Step 4: compute scores
        # ----------------------------------------------------------------
        # Everything in fp32 for accumulation
        f_clean = f_clean_leaf.detach().float()
        f_corr = f_corrupt.to(f_clean.device, torch.float32)
        grad_f = f_clean_leaf.grad.float()

        if f_corr.shape != f_clean.shape:
            raise ValueError(
                f"Shape mismatch between clean and corrupted feature tensors at site {site}: "
                f"clean f_clean.shape={f_clean.shape}, corrupted f_corr.shape={f_corr.shape}. "
                "Ensure corrupted_inputs and clean_inputs have the same sequence length."
            )

        delta_f = f_corr - f_clean  # Δf = corrupted − clean  (sign-consistent with attrib)

        # Enumeration: features where Δf ≠ 0 (union clean-active | corr-active).
        # NOTE: For IG we use the SAME Δf≠0 union — NOT gated on grad@clean — so that
        # features dead at clean but live at corrupt still receive a nonzero IG score
        # (this is the saturation fix IG provides over attrib).  max_features caps are
        # applied AFTER scoring so they do not silently re-introduce a grad@clean gate.
        feature_dim = delta_f.shape[-1]
        # delta_f shape: (b, s, d_sae) or (n, d_sae) — feature is last dim
        active_mask = (delta_f != 0).reshape(-1, feature_dim).any(dim=0)  # (d_sae,)
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()

        site_scores: dict[AtPNode, float] = {}
        _comp = site.component if site.component != "resid_post" else None

        if variant == "attrib":
            # Single-point AtP: score_i = Σ_pos Δf_i · grad_i@clean
            for i in active_indices:
                df_i = delta_f[..., i]
                g_i = grad_f[..., i]
                if graddrop:
                    per_pos = df_i * g_i
                    score = float(per_pos.abs().sum().item())
                else:
                    score = float((df_i * g_i).sum().item())
                node = AtPNode(Node("sae_feature", layer=site.layer, neuron=i, component=_comp))
                site_scores[node] = score

            # Error node (opt-in) for attrib path
            if include_error_node and err_leaf is not None and err_leaf.grad is not None:
                if eps_corrupt is not None and eps_clean is not None:
                    eps_c_f = eps_corrupt.to(err_leaf.device, torch.float32)
                    eps_cl_f = eps_clean.float()
                    delta_eps = eps_c_f - eps_cl_f
                    grad_err = err_leaf.grad.float()
                    if graddrop:
                        per_pos = delta_eps * grad_err
                        err_score = float(per_pos.abs().sum().item())
                    else:
                        err_score = float((delta_eps * grad_err).sum().item())
                    _comp_err = site.component if site.component != "resid_post" else None
                    err_node = AtPNode(Node("sae_error", layer=site.layer, component=_comp_err))
                    site_scores[err_node] = err_score

        else:
            # variant == "ig": integrated gradients via N midpoint samples
            # Path: f(α) = f_clean + α·Δf, α: 0→1, α_k = (k-0.5)/N
            # Completeness (feature-only, include_error_node=False):
            #   Σ_i score_i → metric(decode(f_corrupt)+eps_clean) − metric(decode(f_clean)+eps_clean)
            # Completeness (joint path, include_error_node=True):
            #   Σ_i feature_IG_i + error_IG → metric(real corrupt) − metric(real clean)
            _n_ig = n_ig_steps if n_ig_steps > 0 else 32  # default N=32
            f_clean_t = f_clean_leaf.detach()   # original dtype (may be float64)
            f_corr_t = f_corrupt.to(f_clean_t.device, f_clean_t.dtype)
            delta_f_t = f_corr_t - f_clean_t    # Δf in original dtype

            # eps values for joint path (include_error_node=True)
            eps_clean_t = eps_clean  # captured in Step 2; shape same as f→model activation
            eps_corrupt_t = eps_corrupt  # may be None
            if include_error_node and eps_corrupt_t is not None and eps_clean_t is not None:
                eps_corrupt_t_dev = eps_corrupt_t.to(eps_clean_t.device, eps_clean_t.dtype)
                delta_eps_t = eps_corrupt_t_dev - eps_clean_t  # Δeps for joint path
            else:
                delta_eps_t = None

            # Accumulators: sum of gradients over N steps  (same dtype as f)
            grad_sum = torch.zeros_like(f_clean_t)      # (b, s, d_sae)
            grad_err_sum: Tensor | None = None
            if include_error_node and delta_eps_t is not None:
                grad_err_sum = torch.zeros_like(eps_clean_t)

            for k in range(1, _n_ig + 1):
                alpha_k = (k - 0.5) / _n_ig
                # f_k is the interpolated feature vector at this alpha
                f_k_val = (f_clean_t + alpha_k * delta_f_t).detach().requires_grad_(True)

                # eps_k: frozen at clean for feature-only path; interpolated for joint path
                if include_error_node and delta_eps_t is not None:
                    eps_k_val = (eps_clean_t + alpha_k * delta_eps_t).detach().requires_grad_(True)
                else:
                    # Feature-only: eps frozen at clean throughout
                    eps_k_val = None

                # Splice hook: override f with f_k, eps with eps_k (or eps_clean)
                f_k_store: dict[str, Tensor] = {}
                eps_leaf_store: dict[str, Tensor] = {}

                def _ig_clean_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _sae: Any = sae,
                    _f_k: Tensor = f_k_val,
                    _eps_k: Tensor | None = eps_k_val,
                    _eps_frozen: Tensor = eps_clean_t,  # type: ignore[assignment]
                    _inc_err: bool = include_error_node,
                    _mdtype: torch.dtype = model_dtype,
                    _mdev: Any = model_device,
                    _fkst: dict = f_k_store,
                    _epslst: dict = eps_leaf_store,
                    _resolved: Any = resolved,
                ) -> Any:
                    # We don't re-encode; we use the interpolated f_k directly
                    x_hat = _sae.decode(_f_k)
                    if _inc_err and _eps_k is not None:
                        # Joint path: interpolated eps leaf
                        recon = x_hat + _eps_k
                        _fkst["f_k"] = _f_k
                        _epslst["eps_k"] = _eps_k
                    else:
                        # Feature-only path: eps frozen at clean (detached)
                        recon = x_hat + _eps_frozen
                        _fkst["f_k"] = _f_k
                    recon_cast = recon.to(_mdev, _mdtype)
                    return _routed_inject(_resolved, output, recon_cast)

                ig_hook = layer_mod.register_forward_hook(_ig_clean_hook)
                try:
                    with torch.enable_grad():
                        out_k = self._call_model(clean_inputs)
                        m_k = metric(out_k)
                        # Compute gradients w.r.t. f_k and (optionally) eps_k_val
                        grad_inputs: list[Tensor] = [f_k_val]
                        if include_error_node and eps_k_val is not None:
                            grad_inputs.append(eps_k_val)
                        grads = torch.autograd.grad(
                            m_k, grad_inputs, allow_unused=True
                        )
                finally:
                    ig_hook.remove()

                g_f_k = grads[0]
                if g_f_k is not None:
                    grad_sum = grad_sum + g_f_k.to(grad_sum.device, grad_sum.dtype)
                if include_error_node and len(grads) > 1 and grads[1] is not None:
                    g_e_k = grads[1]
                    if grad_err_sum is not None:
                        grad_err_sum = grad_err_sum + g_e_k.to(grad_err_sum.device, grad_err_sum.dtype)

                # Explicitly free the per-step graph
                del out_k, m_k, grads, f_k_val
                if eps_k_val is not None:
                    del eps_k_val

            # Average gradients and compute scores
            avg_grad = grad_sum.float() / _n_ig    # (1/N) Σ_k grad_f@α_k

            for i in active_indices:
                df_i = delta_f[..., i].float()
                avg_g_i = avg_grad[..., i].to(df_i.device)
                if graddrop:
                    score = float((df_i * avg_g_i).abs().sum().item())
                else:
                    score = float((df_i * avg_g_i).sum().item())
                node = AtPNode(Node("sae_feature", layer=site.layer, neuron=i, component=_comp))
                site_scores[node] = score

            # Error node for IG joint path
            if include_error_node and grad_err_sum is not None and delta_eps_t is not None:
                avg_grad_err = grad_err_sum.float() / _n_ig
                delta_eps_f = delta_eps_t.float()
                if graddrop:
                    err_score = float((delta_eps_f * avg_grad_err).abs().sum().item())
                else:
                    err_score = float((delta_eps_f * avg_grad_err).sum().item())
                _comp_err = site.component if site.component != "resid_post" else None
                err_node = AtPNode(Node("sae_error", layer=site.layer, component=_comp_err))
                site_scores[err_node] = err_score

        # Optional max_features cap (applied after scoring to avoid re-introducing grad@clean gate)
        if max_features is not None:
            feat_items = [(k, v) for k, v in site_scores.items() if k.node.kind == "sae_feature"]
            err_items  = [(k, v) for k, v in site_scores.items() if k.node.kind == "sae_error"]
            if len(feat_items) > max_features:
                feat_items = sorted(feat_items, key=lambda kv: abs(kv[1]), reverse=True)[:max_features]
            site_scores = dict(feat_items + err_items)

        return site_scores

    def bruteforce_feature_scores(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        nodes: list[AtPNode],
    ) -> dict[AtPNode, float]:
        """Independent ground truth via real forward intervention.

        For each sae_feature node i: in a spliced clean forward, patch
        f[..., i] = f_corrupt[..., i] BEFORE decode, re-run, score =
        metric(patched) - metric(spliced_clean_baseline).

        For sae_error node: patch eps = eps_corrupt.

        NEVER derived from the analytic score — completely independent path.
        Mirrors AtPRunner.bruteforce_node_scores.
        """
        was_training, orig_rg = self._atp._freeze_eval()
        sae_orig_rg: dict[Site, dict[str, bool]] = {}
        for site, sae in self._sae_sites.items():
            sae_orig_rg[site] = _freeze_sae(sae)

        try:
            with torch.no_grad():
                return self._bruteforce_inner(
                    clean_inputs, corrupted_inputs, metric, nodes
                )
        finally:
            self._atp._restore(was_training, orig_rg)
            for site, sae in self._sae_sites.items():
                _restore_sae(sae, sae_orig_rg.get(site, {}))

    def _bruteforce_inner(
        self,
        clean_inputs: _Inputs,
        corrupted_inputs: _Inputs,
        metric: Callable[[Any], Tensor],
        nodes: list[AtPNode],
    ) -> dict[AtPNode, float]:
        """Inner (no_grad already active) bruteforce computation."""
        # Group nodes by composite (layer, component) key — P2b: multiple sites per layer
        from circuitry.patching.sae_edges import _node_site_key, _site_key
        key_to_nodes: dict[tuple[int, str], list[AtPNode]] = {}
        for n in nodes:
            if n.node.kind in ("sae_feature", "sae_error"):
                if n.node.layer is not None:
                    nk = _node_site_key(n.node)
                    key_to_nodes.setdefault(nk, []).append(n)

        # For each site, build composite-key lookup
        site_by_key: dict[tuple[int, str], tuple[Site, Any]] = {
            _site_key(site): (site, sae) for site, sae in self._sae_sites.items()
        }

        # Pre-resolve all needed sites (route through ResolvedSite)
        resolved_by_key: dict[tuple[int, str], Any] = {}
        for sk in key_to_nodes:
            if sk not in site_by_key:
                continue
            site, _sae = site_by_key[sk]
            resolved_by_key[sk] = self.resolver.resolve(self.model, site)

        corrupt_data: dict[tuple[int, str], dict[str, Any]] = {}
        for sk in key_to_nodes:
            if sk not in site_by_key:
                continue
            site, sae = site_by_key[sk]
            resolved = resolved_by_key[sk]
            layer_mod = resolved.module
            store: dict[str, Any] = {}

            def _corr_hook(
                module: nn.Module, inp: Any, output: Any,
                _sae: Any = sae,
                _st: dict = store,
                _resolved: Any = resolved,
            ) -> None:
                a = _routed_extract(_resolved, output).detach()
                a_in = a.to(getattr(_sae, "device", a.device), getattr(_sae, "dtype", a.dtype))
                f_c, x_hat_c, eps_c = sae_decompose(_sae, a_in)
                _st["f_corrupt"] = f_c.detach()
                _st["eps_corrupt"] = eps_c  # already detached

            h = layer_mod.register_forward_hook(_corr_hook)
            try:
                self._call_model(corrupted_inputs)
            finally:
                h.remove()
            corrupt_data[sk] = store

        # Baseline: spliced clean forward (SAE inserted but no feature patching)
        baseline_store: dict[tuple[int, str], dict[str, Any]] = {}
        for sk in key_to_nodes:
            if sk not in site_by_key:
                continue
            baseline_store[sk] = {}

        def _make_baseline_splice_hook(sk: tuple[int, str], sae: Any, resolved: Any) -> Any:
            _store = baseline_store[sk]

            def _hook(module: nn.Module, inp: Any, output: Any) -> Any:
                a = _routed_extract(resolved, output)
                a_in = a.detach().to(
                    getattr(sae, "device", a.device),
                    getattr(sae, "dtype", a.dtype),
                )
                with torch.no_grad():
                    f_cl, x_hat_cl, eps_cl = sae_decompose(sae, a_in)
                _store["f_clean"] = f_cl.detach()
                _store["eps_clean"] = eps_cl
                recon_cast = (x_hat_cl + eps_cl).to(a.device, a.dtype)
                return _routed_inject(resolved, output, recon_cast)

            return _hook

        baseline_handles: list[Any] = []
        for sk in key_to_nodes:
            if sk not in site_by_key:
                continue
            site, sae = site_by_key[sk]
            resolved = resolved_by_key[sk]
            layer_mod = resolved.module
            baseline_handles.append(
                layer_mod.register_forward_hook(_make_baseline_splice_hook(sk, sae, resolved))
            )

        try:
            m0_out = self._call_model(clean_inputs)
            m0 = float(metric(m0_out).item())
        finally:
            for h in baseline_handles:
                h.remove()

        # Per-node patching
        scores: dict[AtPNode, float] = {}

        for atp_node in nodes:
            inner = atp_node.node
            if inner.kind not in ("sae_feature", "sae_error"):
                scores[atp_node] = 0.0
                continue

            if inner.layer is None:
                scores[atp_node] = 0.0
                continue

            nk = _node_site_key(inner)
            if nk not in site_by_key:
                scores[atp_node] = 0.0
                continue

            site, sae = site_by_key[nk]
            resolved = resolved_by_key[nk]
            layer_mod = resolved.module
            cd = corrupt_data.get(nk, {})
            bd = baseline_store.get(nk, {})

            f_corrupt = cd.get("f_corrupt")
            eps_corrupt = cd.get("eps_corrupt")
            f_clean_base = bd.get("f_clean")
            eps_clean_base = bd.get("eps_clean")

            if inner.kind == "sae_feature":
                feat_idx = inner.neuron
                if feat_idx is None or f_corrupt is None or f_clean_base is None:
                    scores[atp_node] = 0.0
                    continue

                # Patch feature i in clean, measure metric delta
                _fi = feat_idx
                _fc = f_corrupt
                _sae = sae

                def _patch_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _idx: int = _fi,
                    _f_corr: Tensor = _fc,
                    __sae: Any = _sae,
                    _res: Any = resolved,
                ) -> Any:
                    a = _routed_extract(_res, output)
                    a_in = a.detach().to(
                        getattr(__sae, "device", a.device),
                        getattr(__sae, "dtype", a.dtype),
                    )
                    with torch.no_grad():
                        f_cl, x_hat_cl, eps_cl = sae_decompose(__sae, a_in)
                    # Patch feature idx
                    f_patched = f_cl.clone()
                    f_patched[..., _idx] = _f_corr[..., _idx].to(f_patched.device, f_patched.dtype)
                    x_hat_patched = __sae.decode(f_patched)
                    recon_cast = (x_hat_patched + eps_cl).to(a.device, a.dtype)
                    return _routed_inject(_res, output, recon_cast)

            else:  # sae_error
                if eps_corrupt is None or eps_clean_base is None:
                    scores[atp_node] = 0.0
                    continue

                _ec = eps_corrupt
                _sae = sae

                def _patch_hook(
                    module: nn.Module, inp: Any, output: Any,
                    _eps_c: Tensor = _ec,
                    __sae: Any = _sae,
                    _res: Any = resolved,
                ) -> Any:
                    a = _routed_extract(_res, output)
                    a_in = a.detach().to(
                        getattr(__sae, "device", a.device),
                        getattr(__sae, "dtype", a.dtype),
                    )
                    with torch.no_grad():
                        f_cl, x_hat_cl, _ = sae_decompose(__sae, a_in)
                    # Patch error term to corrupted eps
                    recon_cast = (x_hat_cl + _eps_c.to(x_hat_cl.device, x_hat_cl.dtype)).to(a.device, a.dtype)
                    return _routed_inject(_res, output, recon_cast)

            h = layer_mod.register_forward_hook(_patch_hook)
            try:
                patched_out = self._call_model(clean_inputs)
                patched_metric = float(metric(patched_out).item())
            finally:
                h.remove()

            scores[atp_node] = patched_metric - m0

        return scores
