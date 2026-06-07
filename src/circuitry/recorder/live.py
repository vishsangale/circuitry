"""LiveRecorder — attach hooks per recipe, snapshot tensors at emit steps,
run diagnostics, write scalars through a MetricWriter.

See docs/design.md §4.2, §4.4, §10, §11. Single-process — non-zero ranks
no-op in attach().
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch
import torch.nn as nn

from circuitry.core import activation as _act
from circuitry.core import gradient as _grad
from circuitry.core import weight as _w
from circuitry.core.inventory import ModelInventory
from circuitry.core.weight import attention_head_rank as _attention_head_rank
from circuitry.recipes import Recipe, get_recipe
from circuitry.recorder.hooks import StepContext, TensorSource, filtered_matches
from circuitry.writers.base import MetricWriter

logger = logging.getLogger("circuitry")

# All four are SVD-derived: they consume a matrix's singular values. The
# recorder computes the SVD once per matrix per step (see _run_diagnostics)
# and feeds these `_from_sv` variants, so the stock recipe's 4 SVD diagnostics
# + sv_histogram share one SVD instead of recomputing it 5x per matrix.
_WEIGHT_DIAGS = {
    "effective_rank": _w._effective_rank_from_sv,
    "stable_rank": _w._stable_rank_from_sv,
    "condition_number": _w._condition_number_from_sv,
    "heavy_tail_alpha": _w._heavy_tail_alpha_from_sv,
}

_ACT_DIAGS = {
    "dead_fraction": _act.dead_fraction,
    "participation_ratio": _act.participation_ratio,
    "kurtosis": lambda x: float(_act.kurtosis(x).mean().item()),
}

_GRAD_DIAGS = {
    "grad_norm_per_module": _grad.grad_norm_per_module,  # dict in, dict out
}

_WRITERS: dict[str, Any] = {}  # name → factory; populated below

# Fixed seed for the randperm subsample inside singular_values().  Using a
# constant means every emit step draws the *same* column subset for a given
# weight matrix, so cross-step comparisons (rank_trajectory, update_delta,
# direction_cosine) reflect true weight changes rather than subsample churn.
_SUBSAMPLE_SEED: int = 0


class _AttnMeta:
    """Resolved attention head config for attention_head_rank dispatch.

    Built once at attach() time from ``model.config`` if available.
    ``None`` means we couldn't resolve it — attention_head_rank logs WARN
    and emits nothing for this run.
    """
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim


def _first_attr(obj: Any, names: tuple[str, ...]) -> int | None:
    """Return the first present, int-coercible, non-None attribute in *names*."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def _attn_meta_from_config(source: Any) -> _AttnMeta | None:
    """Build ``_AttnMeta`` from an HF-style config object, or ``None``."""
    if source is None:
        return None
    n_heads = getattr(source, "num_attention_heads", None)
    if n_heads is None:
        return None
    n_heads = int(n_heads)
    if n_heads == 0:
        return None
    n_kv_heads = int(getattr(source, "num_key_value_heads", n_heads) or n_heads)
    head_dim = getattr(source, "head_dim", None)
    if head_dim is None:
        hidden = getattr(source, "hidden_size", None)
        if hidden is None:
            return None
        head_dim = int(hidden) // n_heads
    return _AttnMeta(n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=int(head_dim))


def _attn_meta_from_attn_module(mod: nn.Module) -> _AttnMeta | None:
    """Read head metadata directly off an attention submodule (config-less
    custom models that still name their projections q_proj/k_proj/...)."""
    n_heads = _first_attr(
        mod, ("num_attention_heads", "num_heads", "n_heads", "n_head")
    )
    if not n_heads:
        return None
    n_kv_heads = (
        _first_attr(mod, ("num_key_value_heads", "num_kv_heads", "n_kv_heads"))
        or n_heads
    )
    head_dim = _first_attr(mod, ("head_dim", "head_size", "d_head"))
    if head_dim is None:
        embed = _first_attr(
            mod, ("embed_dim", "hidden_size", "d_model", "all_head_size")
        )
        if not embed:
            return None
        head_dim = embed // n_heads
    return _AttnMeta(n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)


class _LensMeta:
    """Resolved unembed + final-LN for logit_lens_kl dispatch.

    Built once at attach() time. ``None`` means we couldn't resolve it.
    """
    def __init__(self, unembed: torch.Tensor,
                 layer_norm: nn.Module | None) -> None:
        self.unembed = unembed
        self.layer_norm = layer_norm


def _scalarize(value: float | torch.Tensor) -> float:
    """Coerce a Python number or 0-d Tensor to a plain Python ``float``.

    Detaches Tensors before ``.item()`` so a ``requires_grad=True`` loss is
    accepted without the PyTorch UserWarning. Non-scalar Tensors are
    rejected explicitly.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"scalar metric expected a 0-d / size-1 Tensor, got shape {tuple(value.shape)}"
            )
        return float(value.detach().item())
    return float(value)


def _resolve_writer(writer: MetricWriter | str, run_dir: pathlib.Path) -> MetricWriter:
    if not isinstance(writer, str):
        return writer
    from circuitry.writers.jsonl import JsonlWriter
    from circuitry.writers.null import NullWriter
    table = {
        "auto": lambda: _make_auto_writer(run_dir),
        "tensorboard": lambda: _make_tensorboard_writer(run_dir),
        "jsonl": lambda: JsonlWriter(run_dir),
        "null": lambda: NullWriter(),
    }
    if writer not in table:
        raise ValueError(f"unknown writer {writer!r}; known: {sorted(table)}")
    return table[writer]()


def _make_tensorboard_writer(run_dir: pathlib.Path):
    from circuitry.writers.tensorboard import TensorBoardWriter
    return TensorBoardWriter(run_dir)


def _make_auto_writer(run_dir: pathlib.Path):
    """Prefer TensorBoard when the optional extra is installed; otherwise fall
    back to the no-dep jsonl writer with a one-time warning. This is the default
    so a lean ``pip install circuitry`` works out of the box. Pass
    ``writer="tensorboard"`` explicitly to get a hard error when it's missing."""
    try:
        return _make_tensorboard_writer(run_dir)
    except ImportError:
        from circuitry.writers.jsonl import JsonlWriter
        logger.warning(
            "circuitry: writer=\"auto\" — the 'tensorboard' extra is not "
            "installed, falling back to the jsonl writer (metrics.jsonl). "
            "Install `pip install \"circuitry[tensorboard]\"` for TensorBoard "
            "output, or pass writer=\"jsonl\" to silence this warning."
        )
        return JsonlWriter(run_dir)


class Recorder:
    """Single-process training-time diagnostics recorder.

    On a multi-rank setup (``torch.distributed.is_initialized()`` true and
    ``get_rank() != 0``), every method is a no-op so existing scripts don't
    crash. Multi-process support is planned for a future release (design §11).
    """

    def __init__(
        self,
        model: nn.Module,
        run_dir: str | pathlib.Path,
        recipe: str | Recipe,
        writer: MetricWriter | str = "auto",
        every_n_steps: int = 200,
        strict: bool = True,
    ) -> None:
        self.model = model
        self.run_dir = pathlib.Path(run_dir)
        self.recipe = recipe if isinstance(recipe, Recipe) else get_recipe(recipe)
        self.every_n_steps = int(every_n_steps)
        self.strict = bool(strict)
        self._writer_arg = writer
        self._writer: MetricWriter | None = None
        self._hook_handles: list[Any] = []
        # name → tensor captured by hook, refreshed each emit step
        self._captured_activations: dict[str, torch.Tensor] = {}
        self._matched: dict[int, list[str]] = {}  # hp index → names
        # Inventory-derived module-name → parameter-name resolution for
        # WEIGHT / GRAD HookPoints. Built once at attach() time. Catches
        # weights hidden inside wrapper Linear classes (e.g.
        # ``Gemma4ClippableLinear`` → resolves to ``<module>.linear.weight``).
        self._inventory: ModelInventory | None = None
        self._param_for_module: dict[str, str] = {}
        self._attn_meta: _AttnMeta | None = None
        self._lens_meta: _LensMeta | None = None
        # v1.10 tuned lens: a fitted TunedLens whose fingerprint matches this
        # model, or None (requested-but-unsupplied / mismatch — warned at attach).
        self._tuned_lens: Any | None = None
        self._induction_probe: torch.Tensor | None = None
        self._main_pass_attn: dict[str, torch.Tensor] = {}
        self._warned_unnormalized_attn = False
        self._output_attentions_restore: list[tuple[Any, Any]] = []
        self._saes: dict[str, Any] = {}  # module_name → loaded SAE
        self._noop = False
        self._current_step: int = -1
        # Modules whose primary weight is a batched 3-D expert tensor (e.g.
        # OlmoeExperts.gate_up_proj / down_proj).  Populated at attach() time.
        # Maps module_name → list of param_names for all batched-expert params
        # in the module's subtree. The recorder iterates the leading expert axis
        # for each param and emits per-expert weight diagnostics instead of
        # passing a 3-D tensor to the rank primitives (which raise on >2-D input).
        self._moe_expert_params: dict[str, list[str]] = {}
        # Cross-step weight snapshots for training-dynamics diagnostics (v1.3).
        # Detached CPU copies of ctx.weights from prior emit steps.
        # Empty at attach(); populated after each emit; cleared in detach().
        self._prev_weights: dict[str, torch.Tensor] = {}
        self._prev_prev_weights: dict[str, torch.Tensor] = {}
        # v1.4 drift-probe: CPU copies of per-layer activations captured during
        # the first probe forward pass (the "anchor").  None = not yet captured.
        # Cleared in detach() and by reset_drift_reference().  copy=True is
        # load-bearing (same lesson as _prev_weights v1.3: .to("cpu") is a no-op
        # on CPU and would alias the live tensor).
        self._ref_probe_activations: dict[str, torch.Tensor] | None = None
        # Warn-once flag for errors during the drift-probe forward pass.
        self._warned_probe_forward = False
        # Set to True by _set_output_attentions_true() when the model's
        # attn_implementation rejects output_attentions=True (e.g. sdpa/flash).
        # When True the induction_score and attention_pattern_entropy blocks
        # skip silently rather than raising.
        self._attn_diags_sdpa_skip = False
        # Per-step cache for the induction probe forward pass.  Both
        # induction_score and copy_suppression_score need the same captured
        # attention patterns; this avoids running two probe passes when both
        # are enabled.  Reset to empty dict at the start of each step's
        # diagnostic loop; -1 sentinel means "not yet run this step".
        self._probe_attn_cache: dict[str, torch.Tensor] = {}
        self._probe_attn_step: int = -1

    # ---- internal helpers ------------------------------------------------

    def _is_inactive_rank(self) -> bool:
        if not torch.distributed.is_available():
            return False
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return True
        except Exception:  # pragma: no cover — defensive
            return False
        return False

    def _should_capture(self, step: int) -> bool:
        return step % self.every_n_steps == 0

    def _resolve_attn_meta(self) -> _AttnMeta | None:
        """Resolve attention-head metadata for attention_head_rank.

        Order: (1) explicit ``recipe.attn_head_meta``; (2) any submodule
        exposing a ``.config`` (or ``.config.text_config``) with
        ``num_attention_heads`` — ``named_modules()`` yields the model itself
        first, then ``model.model`` etc., so this covers plain HF models,
        multimodal ``text_config``, and HF-wrapped ``model.model.config``;
        (3) head attributes read directly off a ``self_attn`` / ``attn`` /
        ``attention`` submodule for config-less custom models. ``None`` means
        nothing resolved (caller warns)."""
        ov = self.recipe.attn_head_meta
        if ov:
            n_heads = ov.get("n_heads")
            if n_heads:
                n_heads = int(n_heads)
                n_kv = int(ov.get("n_kv_heads", n_heads) or n_heads)
                head_dim = ov.get("head_dim")
                if head_dim is None:
                    hidden = ov.get("hidden_size")
                    if hidden and n_heads:
                        head_dim = int(hidden) // n_heads
                if head_dim:
                    return _AttnMeta(n_heads, n_kv, int(head_dim))

        for _name, mod in self.model.named_modules():
            cfg = getattr(mod, "config", None)
            if cfg is None:
                continue
            for source in (getattr(cfg, "text_config", None), cfg):
                meta = _attn_meta_from_config(source)
                if meta is not None:
                    return meta

        for name, mod in self.model.named_modules():
            short = name.rsplit(".", 1)[-1]
            if short in ("self_attn", "attn", "attention"):
                meta = _attn_meta_from_attn_module(mod)
                if meta is not None:
                    return meta
        return None

    # ---- lifecycle -------------------------------------------------------

    def attach(self) -> None:
        if self._is_inactive_rank():
            self._noop = True
            logger.info("circuitry: rank != 0 — Recorder is no-op")
            return

        (self.run_dir / "circuitry").mkdir(parents=True, exist_ok=True)
        self._writer = _resolve_writer(self._writer_arg, self.run_dir)

        # Inventory pass: enumerate every Parameter once. Source of truth for
        # WEIGHT/GRAD HookPoint resolution; replaces the old
        # ``getattr(module, "weight", None)`` heuristic that silently dropped
        # weights hidden behind wrapper Linear classes.
        self._inventory = ModelInventory.build(self.model)
        (self.run_dir / "circuitry" / "inventory.json").write_text(
            self._inventory.to_json()
        )

        import json as _json

        # If the recipe requests attention_head_rank, resolve head metadata
        # once. Resolution order (see _resolve_attn_meta): explicit
        # recipe.attn_head_meta -> any submodule exposing a `.config` with
        # num_attention_heads (covers model.config, text_config, and
        # HF-wrapped model.model.config) -> head attrs read directly off an
        # attention submodule (config-less custom models). WARN here at
        # attach() — not at first emit — naming what was searched, so a
        # config-less model surfaces the gap before any step runs.
        if "attention_head_rank" in self.recipe.weight_diagnostics:
            self._attn_meta = self._resolve_attn_meta()
            if self._attn_meta is None:
                logger.warning(
                    "circuitry: attention_head_rank requested but head metadata "
                    "could not be resolved (searched recipe.attn_head_meta, "
                    "model.config / .text_config, every submodule's .config, and "
                    "num_heads/head_dim attributes on self_attn/attn/attention "
                    "submodules). Pass recipe.attn_head_meta={'n_heads': ..., "
                    "'head_dim': ...} to enable it; emitting no head-rank tags."
                )

        # Both the logit lens and the v1.10 tuned lens share unembed + final-LN
        # resolution, so resolve _lens_meta if either diagnostic is requested.
        _lens_requested = (
            "logit_lens_kl" in self.recipe.activation_diagnostics
            or "tuned_lens_kl" in self.recipe.activation_diagnostics
        )
        if _lens_requested:
            try:
                emb = self.model.get_output_embeddings()
            except (AttributeError, NotImplementedError):
                emb = None
            unembed_w = getattr(emb, "weight", None) if emb is not None else None
            if isinstance(unembed_w, torch.Tensor):
                ln = (
                    getattr(getattr(self.model, "model", None), "norm", None)
                    or getattr(getattr(self.model, "transformer", None), "ln_f", None)
                    or getattr(self.model, "norm", None)
                    or getattr(self.model, "ln_f", None)
                )
                self._lens_meta = _LensMeta(unembed=unembed_w.detach(),
                                            layer_norm=ln)
            else:
                logger.warning(
                    "circuitry: a lens diagnostic (logit_lens_kl / tuned_lens_kl) "
                    "was requested but the model has no resolvable output "
                    "embedding (get_output_embeddings or .lm_head) — skipping"
                )

        # v1.10 tuned lens: requires a fitted Recipe.tuned_lens whose
        # fingerprint matches this model. Resolve once at attach() so the
        # diagnostic loop only ever applies a validated, frozen lens.
        if "tuned_lens_kl" in self.recipe.activation_diagnostics:
            tl = getattr(self.recipe, "tuned_lens", None)
            if tl is None:
                logger.warning(
                    "circuitry: tuned_lens_kl requested but no fitted "
                    "Recipe.tuned_lens supplied — fit one with "
                    "`circuitry fit-tuned-lens` (or circuitry.tuned_lens."
                    "fit_tuned_lens) and set recipe.tuned_lens; skipping."
                )
            else:
                from circuitry.tuned_lens import model_fingerprint as _fp
                got = _fp(self.model)
                if got != tl.model_fingerprint:
                    logger.warning(
                        "circuitry: tuned_lens_kl skipped — the supplied "
                        "Recipe.tuned_lens was fitted on a different model "
                        "(fingerprint %s) than the attached one (%s). Re-fit the "
                        "lens for this model.",
                        tl.model_fingerprint, got,
                    )
                else:
                    self._tuned_lens = tl

        # SDPA / flash-attention silently drop per-head attention weights even
        # when output_attentions=True is injected, so induction_score and
        # attention_pattern_entropy would emit zero tags with no other signal.
        # Warn once at attach time and point at the eager workaround. Only warn
        # when a non-eager implementation is positively detected — stay quiet
        # when the implementation can't be determined (non-HF models).
        _attn_diags = [
            d for d in ("induction_score", "attention_pattern_entropy")
            if d in self.recipe.activation_diagnostics
        ]
        if _attn_diags:
            cfg = getattr(self.model, "config", None)
            text_cfg = getattr(cfg, "text_config", None)
            impl = None
            for source in (text_cfg, cfg):
                if source is None:
                    continue
                impl = getattr(source, "_attn_implementation", None)
                if impl is not None:
                    break
            if impl is not None and impl != "eager":
                logger.warning(
                    "circuitry: %s requested but the model uses "
                    "attn_implementation=%r, which does not return per-head "
                    "attention weights — these diagnostics will emit no tags. "
                    'Reload the model with attn_implementation="eager" to '
                    "capture them.",
                    " / ".join(_attn_diags), impl,
                )

        _needs_probe = (
            "induction_score" in self.recipe.activation_diagnostics
            or "copy_suppression_score" in self.recipe.activation_diagnostics
        )
        if _needs_probe:
            cfg = getattr(self.model, "config", None)
            text_cfg = getattr(cfg, "text_config", None)
            vocab = None
            for source in (text_cfg, cfg):
                if source is None:
                    continue
                vocab = getattr(source, "vocab_size", None)
                if vocab is not None:
                    break
            if vocab is None:
                emb = self.model.get_input_embeddings() if hasattr(
                    self.model, "get_input_embeddings") else None
                if emb is not None:
                    vocab = int(emb.num_embeddings)
            if vocab is None:
                # Last resort: find any Embedding in named_modules (covers
                # models that don't implement get_input_embeddings).
                for _mn, _mod in self.model.named_modules():
                    if isinstance(_mod, nn.Embedding):
                        vocab = int(_mod.num_embeddings)
                        break
            if vocab is None:
                logger.warning(
                    "circuitry: induction_score / copy_suppression_score "
                    "requested but cannot resolve vocab_size — skipping"
                )
            else:
                n = self.recipe.induction_probe_seq_len
                half = torch.randint(0, vocab, (1, n), dtype=torch.long)
                self._induction_probe = torch.cat([half, half], dim=1)

        name_to_mod = dict(self.model.named_modules())
        matched_lines: list[str] = []
        summary_hook_points: list[dict] = []
        totals: dict[str, int] = {"matched": 0, "resolved": 0, "unresolved": 0}

        for idx, hp in enumerate(self.recipe.hook_points):
            names = filtered_matches(self.model, hp, self.recipe)
            self._matched[idx] = names

            if hp.pattern is not None:
                label = hp.pattern
            elif hp.modules is not None:
                label = "<modules>"
            else:
                label = "<selector>"
            matched_lines.append(f"# hook_point[{idx}] source={hp.source.value} target={label}")
            logger.info("circuitry: hook_point[%d] (%s) matched %d modules: %s",
                        idx, label, len(names), names)

            if len(names) == 0:
                msg = f"HookPoint {idx} ({label}) matched 0 modules"
                if self.strict and not hp.optional:
                    raise RuntimeError(
                        msg + " — refusing to attach (pass strict=False to skip "
                        "unmatched HookPoints with a warning)"
                    )
                if hp.optional:
                    # Structurally-absent pattern for this architecture (e.g. MoE
                    # patterns on a dense model) — expected, INFO not WARNING.
                    logger.info(
                        "circuitry: %s — optional HookPoint not present in this "
                        "model, skipping", msg)
                else:
                    logger.warning("circuitry: %s — skipping this HookPoint", msg)
                matched_lines.append("")
                summary_hook_points.append({
                    "idx": idx,
                    "source": hp.source.value,
                    "label": label,
                    "matched": 0,
                    "resolved": 0,
                    "unresolved": 0,
                    "optional": hp.optional,
                })
                continue
            expected = self.recipe.expected_min_matches.get(hp.pattern or "", 0)
            if expected and len(names) < expected:
                msg = (f"HookPoint {idx} ({label}) matched {len(names)} modules but "
                       f"expected at least {expected}")
                if self.strict:
                    raise RuntimeError(msg)
                logger.warning("circuitry: %s", msg)

            # For WEIGHT / GRAD HookPoints, resolve each matched module name
            # to its primary weight Parameter via the inventory. Loud-on-fail:
            # unresolvable matches WARN per module so wrapper-Linear classes
            # don't silently drop diagnostics.
            if hp.source in (TensorSource.WEIGHT, TensorSource.GRAD):
                unresolved = 0
                for mn in names:
                    rec = self._inventory.find_primary_weight(mn)
                    if rec is None:
                        # For MoE batched-expert modules the subtree has multiple
                        # 3-D params (gate_up_proj, down_proj) — find_primary_weight
                        # returns None because there are >1 candidates.  Fall back
                        # to checking for multiple 3-D params directly; if found,
                        # register ALL of them and flag the module as MoE-experts.
                        # We identify batched-expert params as 3-D tensors whose
                        # leaf_attr is NOT "weight" (Conv weights are 3-D but
                        # have leaf_attr=="weight"; expert batches use the param
                        # name itself, e.g. "gate_up_proj").
                        prefix = mn + "." if mn else ""
                        batched = [
                            r for r in self._inventory.parameters
                            if len(r.shape) == 3
                            and r.leaf_attr != "weight"  # not a Conv weight
                            and (
                                r.owning_module_name == mn
                                or (prefix and r.owning_module_name.startswith(prefix))
                            )
                        ]
                        if batched:
                            self._moe_expert_params[mn] = [r.name for r in batched]
                            for r in batched:
                                tail = r.name[len(mn) + 1:] if mn else r.name
                                matched_lines.append(
                                    f"{mn} → {tail} {tuple(r.shape)} [moe_experts]"
                                )
                        else:
                            parent_cls = type(name_to_mod.get(mn, self.model)).__name__
                            logger.warning(
                                "circuitry: hook_point[%d] %s: module %r (%s) has no "
                                "resolvable 2-D+ weight in its subtree — skipping",
                                idx, hp.source.value, mn, parent_cls,
                            )
                            matched_lines.append(f"{mn} → UNRESOLVED ({parent_cls})")
                            unresolved += 1
                    else:
                        # Don't overwrite an earlier WEIGHT mapping with a GRAD
                        # one (they should be identical, but be explicit).
                        self._param_for_module.setdefault(mn, rec.name)
                        tail = rec.name[len(mn) + 1:] if mn else rec.name
                        matched_lines.append(f"{mn} → {tail} {tuple(rec.shape)}")
                if unresolved:
                    logger.warning(
                        "circuitry: hook_point[%d] (%s): %d of %d matched modules "
                        "had no resolvable primary weight",
                        idx, label, unresolved, len(names),
                    )
                hp_resolved = len(names) - unresolved
                hp_unresolved = unresolved
            elif hp.source is TensorSource.NAMED_PARAM:
                # NAMED_PARAM matches parameter names directly; the matched name
                # IS the param name, so the module→param mapping is the identity.
                # Skip non-≥2-D params (the rank diagnostics require 2-D) with a
                # warning, mirroring WEIGHT's "primary 2-D weight" guarantee.
                name_to_param = dict(self.model.named_parameters())
                unresolved = 0
                for pn in names:
                    p = name_to_param.get(pn)
                    if p is None or p.ndim < 2:
                        logger.warning(
                            "circuitry: hook_point[%d] named_param %r is missing "
                            "or not >=2-D — skipping (weight diagnostics need 2-D)",
                            idx, pn,
                        )
                        matched_lines.append(f"{pn} → UNRESOLVED (<2-D or missing)")
                        unresolved += 1
                        continue
                    self._param_for_module[pn] = pn
                    matched_lines.append(f"{pn} {tuple(p.shape)} [named_param]")
                hp_resolved = len(names) - unresolved
                hp_unresolved = unresolved
            else:
                # OUTPUT / INPUT: hooks attach to the module itself; nothing
                # to resolve.
                for n in names:
                    matched_lines.append(n)
                hp_resolved = len(names)
                hp_unresolved = 0

            matched_lines.append("")
            summary_hook_points.append({
                "idx": idx,
                "source": hp.source.value,
                "label": label,
                "matched": len(names),
                "resolved": hp_resolved,
                "unresolved": hp_unresolved,
                "optional": hp.optional,
            })
            totals["matched"] += len(names)
            totals["resolved"] += hp_resolved
            totals["unresolved"] += hp_unresolved

        # F37: aggregate WARNING when weight-source HookPoints collectively match
        # 0 modules.  The per-HookPoint messages above are terse (one line each);
        # this summary gives users one actionable signal at WARNING level that
        # weight diagnostics will be silent for those patterns — e.g. on MoE
        # models where the standard MLP-proj pattern finds nothing.
        # Exclude optional HookPoints (e.g. MoE patterns on a dense model): their
        # absence is expected for the architecture and must not raise warning noise
        # on every common-case attach. Resolves the F37 follow-up.
        _zero_weight_hps = [
            hp_info
            for hp_info in summary_hook_points
            if hp_info["source"] in (TensorSource.WEIGHT.value, TensorSource.GRAD.value)
            and hp_info["matched"] == 0
            and not hp_info.get("optional", False)
        ]
        if _zero_weight_hps:
            _zero_labels = [hp_info["label"] for hp_info in _zero_weight_hps]
            logger.warning(
                "circuitry: %d weight pattern(s) matched 0 modules — weight "
                "diagnostics will not run for those patterns. "
                "Patterns with 0 matches: %s. "
                "If this is a MoE model, ensure the recipe includes patterns for "
                "the expert and router modules.",
                len(_zero_weight_hps),
                ", ".join(_zero_labels),
            )

        (self.run_dir / "circuitry" / "matched_modules.txt").write_text(
            "\n".join(matched_lines)
        )

        attach_summary = {
            "hook_points": summary_hook_points,
            "totals": totals,
        }
        (self.run_dir / "circuitry" / "attach_summary.json").write_text(
            _json.dumps(attach_summary, indent=2)
        )

        # Install hooks for INPUT / OUTPUT sources (WEIGHT/GRAD read directly at step time).
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is TensorSource.OUTPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_hook(self._mk_fwd_hook(n))
                    self._hook_handles.append(handle)
            elif hp.source is TensorSource.INPUT:
                for n in self._matched[idx]:
                    handle = name_to_mod[n].register_forward_pre_hook(self._mk_pre_hook(n))
                    self._hook_handles.append(handle)
            # WEIGHT / GRAD are pulled in step() directly from the inventory.

        # If attention_pattern_entropy is requested, capture attn_weights from
        # matched self_attn modules during the main forward. Per-head weights
        # are enabled via config.output_attentions (set at the END of attach;
        # see _set_output_attentions_true) rather than by injecting an
        # output_attentions=True forward kwarg — the kwarg path raises TypeError
        # on wrapper models whose forward() lacks **kwargs.
        if "attention_pattern_entropy" in self.recipe.activation_diagnostics:
            attn_modules: list[str] = []
            for idx, hp in enumerate(self.recipe.hook_points):
                if hp.source is not TensorSource.OUTPUT:
                    continue
                for mn in self._matched[idx]:
                    short = mn.rsplit(".", 1)[-1]
                    if short in ("self_attn", "attn", "attention"):
                        attn_modules.append(mn)

            def _mk_attn_capture(mn: str, _store=self._main_pass_attn):
                def _h(_mod, _inp, out):
                    if isinstance(out, tuple) and len(out) >= 2 and isinstance(
                        out[1], torch.Tensor
                    ):
                        _store[mn] = out[1].detach()
                return _h

            for mn in attn_modules:
                mod = name_to_mod.get(mn)
                if mod is None:
                    continue
                handle = mod.register_forward_hook(_mk_attn_capture(mn))
                self._hook_handles.append(handle)

        # Load SAEs declared in recipe.sae_checkpoints. Loading is cheap
        # (one HF download per checkpoint, cached afterwards); the per-step
        # encode+decode cost is what's gated by adding "sae_reconstruction"
        # to activation_diagnostics.
        sae_ckpts = self.recipe.sae_checkpoints or {}
        if sae_ckpts:
            import re

            from circuitry.sae.loader import load_sae
            module_names: list[str] = []
            for idx, hp in enumerate(self.recipe.hook_points):
                if hp.source is TensorSource.OUTPUT:
                    module_names.extend(self._matched[idx])
            device = str(next(self.model.parameters()).device)
            for pat, (release, sae_id) in sae_ckpts.items():
                rx = re.compile(pat)
                matched = [mn for mn in module_names if rx.fullmatch(mn)]
                if not matched:
                    logger.warning(
                        "circuitry: sae_checkpoints pattern %r matched 0 "
                        "hooked OUTPUT modules", pat,
                    )
                    continue
                sae = load_sae(release, sae_id, device=device)
                for mn in matched:
                    self._saes[mn] = sae

        # Set last: attach() runs no forward, so nothing above needs the flag;
        # putting it last guarantees a failed attach never mutates user config.
        if "attention_pattern_entropy" in self.recipe.activation_diagnostics:
            self._set_output_attentions_true()

    def _mk_fwd_hook(self, name: str):
        # Hooks always capture (cheap detach); step() decides what to consume.
        # Gating the hook itself on `_current_step` is broken: forward runs
        # before step(), so the hook would see a stale step index.
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self._captured_activations[name] = t.detach()
        return _hook

    def _mk_pre_hook(self, name: str):
        def _hook(_mod, inputs):
            t = inputs[0]
            self._captured_activations[name] = t.detach()
        return _hook

    def _probe_forward(self, probe: Any) -> Any:
        """Run a forward pass for an internal probe (induction / drift).

        Uses ``recipe.forward_fn(model, batch)`` when supplied — the entry point
        for non-HF models (e.g. SASRec.predict_scores) whose ``forward`` is not
        HF-style. Otherwise calls ``model(probe, output_attentions=True)`` with a
        ``TypeError`` fallback to ``model(probe)`` for wrappers whose forward
        lacks ``**kwargs``. The recorder's capture hooks fire either way.
        """
        if self.recipe.forward_fn is not None:
            return self.recipe.forward_fn(self.model, probe)
        try:
            return self.model(probe, output_attentions=True)
        except TypeError:
            return self.model(probe)

    def _get_probe_attn(self, ctx: "StepContext") -> dict[str, "torch.Tensor"]:
        """Return {module_name: attn_tensor} from the induction probe forward.

        Lazily runs the probe pass once per step and caches the result so that
        both ``induction_score`` and ``copy_suppression_score`` share a single
        forward pass when both are enabled.  Returns an empty dict when the
        probe is unavailable (SDPA backend, probe not built, etc.).
        """
        if self._probe_attn_step == ctx.step:
            return self._probe_attn_cache

        self._probe_attn_cache = {}
        self._probe_attn_step = ctx.step

        if self._induction_probe is None or self._attn_diags_sdpa_skip:
            return self._probe_attn_cache

        # Collect self_attn modules matched by OUTPUT hook points.
        self_attn_modules: dict[str, nn.Module] = {}
        name_to_mod = dict(self.model.named_modules())
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source is not TensorSource.OUTPUT:
                continue
            for mn in self._matched[idx]:
                mod = name_to_mod.get(mn)
                if mod is None:
                    continue
                short = mn.rsplit(".", 1)[-1]
                if short in ("self_attn", "attn", "attention"):
                    self_attn_modules[mn] = mod
        if not self_attn_modules:
            return self._probe_attn_cache

        captured: dict[str, torch.Tensor] = {}

        def _mk_capture(mn: str, _cap: dict[str, torch.Tensor] = captured):
            def _h(_mod, _inp, hook_out):
                if (
                    isinstance(hook_out, tuple)
                    and len(hook_out) >= 2
                    and isinstance(hook_out[1], torch.Tensor)
                ):
                    _cap[mn] = hook_out[1].detach()
            return _h

        handles = [
            mod.register_forward_hook(_mk_capture(mn))
            for mn, mod in self_attn_modules.items()
        ]
        # Snapshot training-forward attention so the permanent _main_pass_attn
        # hook (which fires during the probe forward too) doesn't overwrite the
        # training values needed by attention_pattern_entropy later this step.
        _main_pass_attn_snapshot = dict(self._main_pass_attn)
        try:
            probe_dev = next(self.model.parameters()).device
            probe = self._induction_probe.to(probe_dev)
            with torch.inference_mode():
                self._probe_forward(probe)
        finally:
            for h in handles:
                h.remove()
            self._main_pass_attn.update(_main_pass_attn_snapshot)

        self._probe_attn_cache = captured
        return self._probe_attn_cache

    def _set_output_attentions_true(self) -> None:
        """Enable per-head attention output via the HF config (not a forward
        kwarg, which breaks wrappers whose forward lacks **kwargs). Records the
        original value(s) so detach() can restore exactly. Call LAST in attach()
        so a failed attach never leaves the config mutated.

        If the model's attention implementation (e.g. sdpa / flash_attention_2)
        does not support output_attentions, the assignment raises ValueError.
        In that case we degrade gracefully: set ``_attn_diags_sdpa_skip=True``
        so induction_score / attention_pattern_entropy emit no tags, and log a
        single warning instead of propagating the exception.
        """
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        if cfg is None and text_cfg is None:
            # Non-HF model: no config to flip, so attention capture would
            # silently no-op. Warn once and point at the recipe.forward_fn
            # escape hatch (and the eager-attention requirement).
            self._attn_diags_sdpa_skip = True
            logger.warning(
                "circuitry: attention_pattern_entropy / induction_score "
                "requested but the model has no resolvable `config` (non-HF "
                "model). Per-head attention can't be enabled via config, so "
                "these diagnostics will emit no tags. Provide a "
                "recipe.forward_fn(model, batch) that returns per-head "
                "attention (e.g. need_weights=True) to capture them."
            )
            return
        for source in (cfg, text_cfg):
            if source is None:
                continue
            orig = getattr(source, "output_attentions", False)
            try:
                source.output_attentions = True
            except (ValueError, AttributeError) as exc:
                logger.warning(
                    "circuitry: could not enable output_attentions on model "
                    "config (%s). induction_score and attention_pattern_entropy "
                    "will be skipped. To capture attention diagnostics reload "
                    'the model with attn_implementation="eager". Reason: %s',
                    type(source).__name__, exc,
                )
                self._attn_diags_sdpa_skip = True
                return
            self._output_attentions_restore.append((source, orig))

    def _restore_output_attentions(self) -> None:
        for source, original in self._output_attentions_restore:
            source.output_attentions = original
        self._output_attentions_restore.clear()

    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        # Release cross-step snapshots to free RAM (v1.3).
        self._prev_weights.clear()
        self._prev_prev_weights.clear()
        # Release drift-probe reference activations (v1.4).
        self._ref_probe_activations = None
        # Release the fitted tuned lens reference (v1.10).
        self._tuned_lens = None
        # Release per-step probe-attn cache (v1.11).
        self._probe_attn_cache = {}
        self._probe_attn_step = -1
        self._restore_output_attentions()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def reset_drift_reference(self) -> None:
        """Clear the drift-probe reference anchor.

        The next emit step after calling this method will capture a fresh
        anchor (no drift tag emitted on that step), and subsequent steps will
        compute drift relative to the new anchor.

        Use this after resuming from a checkpoint, or after a learning-rate
        phase change, to re-anchor the representational-drift baseline.
        """
        self._ref_probe_activations = None

    def step(self, step: int, loss: float | torch.Tensor | None = None, *,
             loss_components: dict[str, float | torch.Tensor] | None = None,
             **user: Any) -> None:
        if self._noop:
            return
        self._current_step = int(step)
        assert self._writer is not None, "Recorder.step called before attach()"

        if loss is not None:
            self._writer.add_scalar("train/loss", _scalarize(loss), self._current_step)

        if loss_components is not None and self._writer is not None:
            for name, value in loss_components.items():
                self._writer.add_scalar(f"train/{name}", _scalarize(value),
                                        self._current_step)

        if not self._should_capture(self._current_step):
            return

        # Build StepContext from currently-captured tensors + read-on-demand
        # weights/grads. WEIGHT/GRAD HookPoints use the inventory-derived
        # module→param mapping from attach() so wrapper Linear classes
        # (e.g. ``Gemma4ClippableLinear`` → ``<module>.linear.weight``) are
        # handled without silent drops.
        name_to_param = dict(self.model.named_parameters())
        weights: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}
        # MoE batched-expert weights: module_name → list of (param_name, tensor).
        # These 3-D tensors are handled per-expert in _run_diagnostics and are
        # NOT placed in ctx.weights (which the rank primitives expect to be 2-D).
        moe_weights: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source not in (
                TensorSource.WEIGHT, TensorSource.GRAD, TensorSource.NAMED_PARAM
            ):
                continue
            for mod_name in self._matched[idx]:
                # MoE batched-expert module: collect all their 3-D params.
                if mod_name in self._moe_expert_params and hp.source is TensorSource.WEIGHT:
                    entries: list[tuple[str, torch.Tensor]] = []
                    for pname in self._moe_expert_params[mod_name]:
                        p = name_to_param.get(pname)
                        if isinstance(p, torch.Tensor):
                            entries.append((pname, p.detach()))
                    if entries:
                        moe_weights[mod_name] = entries
                    continue
                param_name = self._param_for_module.get(mod_name)
                if param_name is None:
                    continue  # already WARNed at attach()
                p = name_to_param.get(param_name)
                if not isinstance(p, torch.Tensor):
                    continue
                if hp.source in (TensorSource.WEIGHT, TensorSource.NAMED_PARAM):
                    weights[mod_name] = p.detach()
                else:  # GRAD
                    if p.grad is not None:
                        gradients[mod_name] = p.grad.detach()

        ctx = StepContext(
            step=self._current_step,
            model=self.model,
            activations=dict(self._captured_activations),
            gradients=gradients,
            weights=weights,
            loss=loss,
            user=dict(user),
        )

        self._run_diagnostics(ctx, moe_weights=moe_weights)
        # Discard activations now that we've consumed them.
        self._captured_activations.clear()
        # Roll cross-step weight snapshots forward (v1.3 training-dynamics).
        # copy=True is load-bearing: ctx.weights holds detached *views* of the live
        # params, and .cpu() is a no-op on a CPU model, so without a forced copy the
        # snapshot would alias the live storage and silently track in-place optimizer
        # updates — making update_delta/direction_cosine identically zero on CPU.
        self._prev_prev_weights = self._prev_weights
        self._prev_weights = {
            name: t.detach().to("cpu", copy=True) for name, t in ctx.weights.items()
        }

    def _enabled(self, name: str) -> bool:
        return self.recipe.enabled.get(name, True)

    @staticmethod
    def _block_layer_idx(name: str) -> int | None:
        """Layer index N from a residual-block module name ending in ``layers.N``."""
        import re as _re
        m = _re.search(r'(?:^|\.)layers\.(\d+)$', name)
        return int(m.group(1)) if m else None

    def _lens_block_outputs(
        self, ctx: StepContext,
    ) -> tuple[list[tuple[str, torch.Tensor]], torch.Tensor, int | None] | None:
        """Shared setup for the logit / tuned lens diagnostics.

        Returns ``(block_outputs, final_logits, max_tok)`` where ``block_outputs``
        is the per-layer ``[(module_name, residual), ...]`` sorted by numeric
        layer index, and ``final_logits`` is the model's final distribution
        reconstructed as ``LN_f(last_block) @ W_U`` — the same reconstruction the
        tuned lens was fitted against. Returns ``None`` (and disables further
        lens emission this attach) when no residual-stream block outputs match.
        """
        if self._lens_meta is None:
            return None
        _W_raw = self._lens_meta.unembed
        # HF convention is (vocab, d_model) so d_model is the smaller dim; for
        # the unusual square case we fall back to dim 1.
        d_model_unembed = (
            _W_raw.shape[-1]
            if _W_raw.shape[0] >= _W_raw.shape[-1]
            else _W_raw.shape[0]
        )
        # Keep only residual-stream block outputs (module name ends in
        # `.layers.N`); the d_model check is a secondary guard. Sort by numeric
        # layer index so block_outputs[-1] is the true final layer regardless of
        # digit count (layer 9 must not sort after layer 34).
        block_outputs = sorted(
            ((k, v) for k, v in ctx.activations.items()
             if self._block_layer_idx(k) is not None
             and v.shape[-1] == d_model_unembed),
            key=lambda kv: self._block_layer_idx(kv[0]),
        )
        if not block_outputs:
            logger.warning(
                "circuitry: lens diagnostic found no activations with last-dim "
                "matching unembed d_model=%d — skipping.",
                d_model_unembed,
            )
            self._lens_meta = None
            return None
        _last_name, last_x = block_outputs[-1]
        max_tok = self.recipe.lens_max_tokens
        with torch.inference_mode():
            last_f32 = last_x.detach().to(torch.float32)
            if max_tok is not None:
                last_f32 = last_f32[:, :max_tok, :]
            ln = self._lens_meta.layer_norm
            last_normed = ln(last_f32) if ln is not None else last_f32
            W = _W_raw.to(torch.float32)
            # unembed for HF is (vocab, d_model); transpose if needed.
            if W.shape[-1] == last_normed.shape[-1]:
                final_logits = last_normed @ W.t()
            else:
                final_logits = last_normed @ W
        return block_outputs, final_logits, max_tok

    def _run_diagnostics(
        self,
        ctx: StepContext,
        moe_weights: dict[str, list[tuple[str, torch.Tensor]]] | None = None,
    ) -> None:
        assert self._writer is not None
        if moe_weights is None:
            moe_weights = {}

        # Singular values of each weight matrix are computed at most once per
        # step and shared across all SVD-derived diagnostics (effective_rank,
        # stable_rank, condition_number, heavy_tail_alpha, sv_histogram). Keyed
        # by id(); the weight tensors are stable for the duration of this call.
        from circuitry.core.weight import singular_values as _singular_values
        _sv_cache: dict[int, torch.Tensor] = {}

        def _sv(w: torch.Tensor) -> torch.Tensor:
            s = _sv_cache.get(id(w))
            if s is None:
                # Pass a fixed seed so cross-step rank/delta comparisons are
                # not polluted by subsample churn (A1 determinism fix).
                s = _singular_values(w, seed=_SUBSAMPLE_SEED)
                _sv_cache[id(w)] = s
            return s

        for name in self.recipe.weight_diagnostics:
            if not self._enabled(name):
                continue
            if name == "attention_head_rank":
                if self._attn_meta is None:
                    # Already warned once at attach() (with the full search
                    # list); stay quiet per emit step.
                    continue
                meta = self._attn_meta
                for mod_name, w in ctx.weights.items():
                    short = mod_name.rsplit(".", 1)[-1]
                    if short in ("q_proj",):
                        nh, axis = meta.n_heads, 0
                    elif short in ("k_proj", "v_proj"):
                        nh, axis = meta.n_kv_heads, 0
                    elif short in ("o_proj",):
                        nh, axis = meta.n_heads, 1
                    else:
                        continue  # not an attention projection; skip
                    try:
                        ranks = _attention_head_rank(
                            w, n_heads=nh, head_dim=meta.head_dim, axis=axis,
                        )
                    except ValueError as e:
                        logger.warning(
                            "circuitry: attention_head_rank on %s failed: %s",
                            mod_name, e,
                        )
                        continue
                    for i, r in enumerate(ranks):
                        self._writer.add_scalar(
                            f"weight/attention_head_rank/{mod_name}/head_{i}",
                            r, ctx.step,
                        )
                continue
            elif name == "sv_histogram":
                if ctx.weights:
                    for mod_name, w in ctx.weights.items():
                        s = _sv(w)
                        self._writer.add_histogram(
                            f"spectral/per_param/{mod_name}/sv_histogram",
                            s, ctx.step)
                        # Companion summary scalars so the spectrum is visible to
                        # scalar / CSV consumers (a histogram alone is invisible
                        # to tabular exports). Reuses the shared SVD via _sv.
                        if s.numel():
                            self._writer.add_scalar(
                                f"spectral/per_param/{mod_name}/sv_max",
                                float(s[0].item()), ctx.step)
                            self._writer.add_scalar(
                                f"spectral/per_param/{mod_name}/sv_min",
                                float(s[-1].item()), ctx.step)
                            self._writer.add_scalar(
                                f"spectral/per_param/{mod_name}/spectral_entropy",
                                _w._spectral_entropy_from_sv(s), ctx.step)
            elif name == "update_delta":
                if not self._prev_weights:
                    continue  # first emit step — no prior snapshot yet
                deltas = _w.update_delta(ctx.weights, self._prev_weights)
                for mod_name, val in deltas.items():
                    self._writer.add_scalar(
                        f"weight/update_delta/{mod_name}", val, ctx.step
                    )
                # Scale-invariant companion (v1.10): ||ΔW|| / ||W||. The
                # update_delta_vanishing flag keys on this so its threshold means
                # the same thing across parameter sizes.
                rel = _w.relative_update_delta(ctx.weights, self._prev_weights)
                for mod_name, val in rel.items():
                    self._writer.add_scalar(
                        f"weight/update_delta_rel/{mod_name}", val, ctx.step
                    )
            elif name == "direction_cosine":
                if not self._prev_weights or not self._prev_prev_weights:
                    continue  # need two prior snapshots; skip first two emit steps
                cosines = _w.direction_cosine(
                    ctx.weights, self._prev_weights, self._prev_prev_weights
                )
                for mod_name, val in cosines.items():
                    self._writer.add_scalar(
                        f"weight/direction_cosine/{mod_name}", val, ctx.step
                    )
            elif name == "rank_trajectory":
                if not self._prev_weights:
                    continue  # first emit step
                # Reuse the per-step SVD cache (_sv) to avoid a redundant SVD.
                # Equivalent to rank_trajectory([prev, now])[-1] but zero extra SVD cost.
                for mod_name, w in ctx.weights.items():
                    rank_now = _w._effective_rank_from_sv(_sv(w))
                    self._writer.add_scalar(
                        f"weight/rank_trajectory/{mod_name}", rank_now, ctx.step
                    )
            else:
                fn = _WEIGHT_DIAGS.get(name)
                if fn is None:
                    logger.warning("circuitry: unknown weight diagnostic %r — skipping", name)
                    continue
                for mod_name, w in ctx.weights.items():
                    self._writer.add_scalar(f"weight/{name}/{mod_name}", float(fn(_sv(w))), ctx.step)
                # MoE batched-expert weights: iterate leading expert axis so each
                # 2-D slice [d_in, d_out] goes to the rank primitives individually.
                # Tags: weight/<diag>/<mod_name>/expert_<i> where <mod_name> is
                # the experts module name (e.g. "model.layers.0.mlp.experts") and
                # <i> is the expert index within the batch.
                for experts_mod, param_entries in moe_weights.items():
                    for pname, w3d in param_entries:
                        # param name relative to the experts module
                        param_short = pname[len(experts_mod) + 1:] if pname.startswith(experts_mod + ".") else pname
                        n_experts = w3d.shape[0]
                        for ei in range(n_experts):
                            w2d = w3d[ei]  # shape [d_in, d_out] — 2-D, safe for rank primitives
                            tag = f"weight/{name}/{experts_mod}/{param_short}/expert_{ei}"
                            self._writer.add_scalar(tag, float(fn(_sv(w2d))), ctx.step)

        for name in self.recipe.activation_diagnostics:
            if not self._enabled(name):
                continue
            if name == "gate_stats":
                from circuitry.core.activation import gate_stats as _gs
                for mod_name, x in ctx.activations.items():
                    out = _gs(x)
                    for sub, val in out.items():
                        self._writer.add_scalar(
                            f"activation/gate_stats/{mod_name}/{sub}",
                            val, ctx.step,
                        )
                continue
            if name == "logit_lens_kl":
                resolved = self._lens_block_outputs(ctx)
                if resolved is None:
                    continue
                block_outputs, final_logits, max_tok = resolved
                from circuitry.core.lens import logit_lens_kl as _llk
                for mod_name, x in block_outputs:
                    if max_tok is not None:
                        x = x[:, :max_tok, :]
                    try:
                        kl = _llk(
                            x, self._lens_meta.unembed, final_logits,
                            layer_norm=self._lens_meta.layer_norm,
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if (not isinstance(e, torch.cuda.OutOfMemoryError)
                                and "out of memory" not in str(e).lower()):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(
                            "circuitry: logit_lens_kl ran out of memory on %s — "
                            "skipping this layer's emission for this step. Set "
                            "recipe.lens_max_tokens to cap the lens cost. (%s)",
                            mod_name, e,
                        )
                        continue
                    self._writer.add_scalar(
                        f"activation/logit_lens_kl/{mod_name}", kl, ctx.step,
                    )
                continue
            if name == "tuned_lens_kl":
                # Opt-in (v1.10). Needs a fitted, fingerprint-matched lens,
                # resolved at attach() into self._tuned_lens (else None).
                if self._tuned_lens is None:
                    continue
                resolved = self._lens_block_outputs(ctx)
                if resolved is None:
                    continue
                block_outputs, final_logits, max_tok = resolved
                from circuitry.core.lens import tuned_lens_kl as _tlk
                for mod_name, x in block_outputs:
                    layer = self._block_layer_idx(mod_name)
                    translator = self._tuned_lens.translator_for(layer)
                    if translator is None:
                        continue  # this block wasn't fitted (e.g. final frame)
                    if max_tok is not None:
                        x = x[:, :max_tok, :]
                    try:
                        kl = _tlk(
                            x, translator, self._lens_meta.unembed, final_logits,
                            layer_norm=self._lens_meta.layer_norm,
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if (not isinstance(e, torch.cuda.OutOfMemoryError)
                                and "out of memory" not in str(e).lower()):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.warning(
                            "circuitry: tuned_lens_kl ran out of memory on %s — "
                            "skipping this layer's emission for this step. Set "
                            "recipe.lens_max_tokens to cap the lens cost. (%s)",
                            mod_name, e,
                        )
                        continue
                    self._writer.add_scalar(
                        f"activation/tuned_lens_kl/{mod_name}", kl, ctx.step,
                    )
                continue
            if name == "induction_score":
                from circuitry.core.attention import induction_score as _is
                for mn, attn in self._get_probe_attn(ctx).items():
                    try:
                        scores = _is(
                            attn,
                            seq_len_repeat=self.recipe.induction_probe_seq_len,
                        )
                    except ValueError as e:
                        logger.warning(
                            "circuitry: induction_score on %s failed: %s", mn, e,
                        )
                        continue
                    for i, s in enumerate(scores):
                        self._writer.add_scalar(
                            f"activation/induction_score/{mn}/head_{i}",
                            s, ctx.step,
                        )
            if name == "copy_suppression_score":
                from circuitry.core.attention import copy_suppression_score as _css
                for mn, attn in self._get_probe_attn(ctx).items():
                    try:
                        scores = _css(
                            attn,
                            seq_len_repeat=self.recipe.induction_probe_seq_len,
                        )
                    except ValueError as e:
                        logger.warning(
                            "circuitry: copy_suppression_score on %s failed: %s",
                            mn, e,
                        )
                        continue
                    for i, s in enumerate(scores):
                        self._writer.add_scalar(
                            f"activation/copy_suppression_score/{mn}/head_{i}",
                            s, ctx.step,
                        )
                continue
            if name == "attention_pattern_entropy":
                if self._attn_diags_sdpa_skip:
                    continue
                from circuitry.core.attention import (
                    attention_pattern_entropy as _ape,
                )
                for mn, attn in self._main_pass_attn.items():
                    if not self._warned_unnormalized_attn:
                        rs = attn.detach().to(torch.float32).sum(dim=-1)
                        dev = (rs - 1.0).abs().max().item()
                        if dev > 1e-3:
                            logger.warning(
                                "circuitry: attention_pattern_entropy rows do "
                                "not sum to 1 (max deviation %.3g) — entropy is "
                                "computed over the normalized attention shape; "
                                "total attention mass is discarded. Values are "
                                "comparable across attention variants but are "
                                "not raw softmax entropy.", dev,
                            )
                            self._warned_unnormalized_attn = True
                    ents = _ape(attn)
                    for i, e in enumerate(ents):
                        self._writer.add_scalar(
                            f"activation/attention_pattern_entropy/{mn}/head_{i}",
                            e, ctx.step,
                        )
                continue
            if name == "sae_reconstruction":
                if not self._saes:
                    continue
                from circuitry.sae.metrics import sae_reconstruction_error
                for mn, x in ctx.activations.items():
                    sae = self._saes.get(mn)
                    if sae is None:
                        continue
                    out = sae_reconstruction_error(x, sae)
                    for sub, val in out.items():
                        self._writer.add_scalar(
                            f"activation/sae/{mn}/{sub}", val, ctx.step,
                        )
                continue
            if name == "drift_probe":
                if self.recipe.probe_batch is None:
                    continue
                # Run a second forward pass on the fixed probe_batch to capture
                # per-layer activations for representational-drift comparison.
                # Template: induction_score block above (temporary hooks,
                # inference_mode, try/finally removal).
                from circuitry.core.activation import repr_drift as _repr_drift

                # Collect all OUTPUT-source matched module names (same layers
                # that produce main-pass activations) as the candidate set for
                # comparison.  We use modules matched by OUTPUT HookPoints so
                # the same layers that appear in ctx.activations are compared.
                probe_module_names: dict[str, nn.Module] = {}
                _name_to_mod = dict(self.model.named_modules())
                for _idx, _hp in enumerate(self.recipe.hook_points):
                    if _hp.source is not TensorSource.OUTPUT:
                        continue
                    for _mn in self._matched[_idx]:
                        _mod = _name_to_mod.get(_mn)
                        if _mod is not None:
                            probe_module_names[_mn] = _mod

                if not probe_module_names:
                    continue

                # Determine model dtype/device for probe casting.
                _probe_dev = next(self.model.parameters()).device
                _probe_dtype = next(self.model.parameters()).dtype

                # Cast probe_batch: always move to the model's device, but cast
                # dtype only for floating-point probes.  Integer token-ID probes
                # must NOT be cast to the model's float dtype (that would corrupt
                # the indices and crash the embedding lookup).
                probe = self.recipe.probe_batch.to(device=_probe_dev)
                if probe.is_floating_point():
                    probe = probe.to(dtype=_probe_dtype)

                # Apply drift_max_tokens cap (truncate token/seq dim).
                _max_tok = self.recipe.drift_max_tokens
                if _max_tok is not None and probe.dim() >= 2:
                    probe = probe[:, :_max_tok]

                # Register temporary hooks to capture OUTPUT activations.
                _probe_captured: dict[str, torch.Tensor] = {}

                def _mk_probe_hook(
                    _mn: str,
                    _cap: dict[str, torch.Tensor] = _probe_captured,
                ):
                    def _hook(_mod, _inp, _out):
                        t = _out[0] if isinstance(_out, tuple) else _out
                        _cap[_mn] = t.detach()
                    return _hook

                _probe_handles = [
                    mod.register_forward_hook(_mk_probe_hook(mn))
                    for mn, mod in probe_module_names.items()
                ]
                try:
                    with torch.inference_mode():
                        self._probe_forward(probe)
                except Exception as _probe_err:
                    if not self._warned_probe_forward:
                        logger.warning(
                            "circuitry: drift_probe forward pass failed (%s); "
                            "skipping drift emission for this step. "
                            "(Further warnings suppressed.)",
                            _probe_err,
                        )
                        self._warned_probe_forward = True
                    continue
                finally:
                    for _ph in _probe_handles:
                        _ph.remove()

                # First emit: store anchor; emit NO drift tag.
                if self._ref_probe_activations is None:
                    # copy=True is load-bearing: on a CPU model .to("cpu") is a
                    # no-op and would alias the probe-captured tensor, silently
                    # making CKA ~1.0 forever (same lesson as _prev_weights v1.3).
                    self._ref_probe_activations = {
                        mn: t.detach().to("cpu", copy=True)
                        for mn, t in _probe_captured.items()
                    }
                    continue

                # Subsequent emits: compute drift per layer and emit scalars.
                for _mn, _cur_t in _probe_captured.items():
                    _ref_t = self._ref_probe_activations.get(_mn)
                    if _ref_t is None:
                        continue
                    # Bring reference to same device as current for comparison.
                    _ref_on_dev = _ref_t.to(_cur_t.device)
                    try:
                        drift_val = _repr_drift(
                            _ref_on_dev,
                            _cur_t,
                            self.recipe.drift_method,
                            max_samples=256,
                            seed=0,
                        )
                    except Exception as _e:
                        logger.warning(
                            "circuitry: drift_probe on %s failed: %s", _mn, _e
                        )
                        continue
                    self._writer.add_scalar(
                        f"activation/repr_drift/{_mn}", drift_val, ctx.step
                    )
                continue
            fn = _ACT_DIAGS.get(name)
            if fn is None:
                logger.warning("circuitry: unknown activation diagnostic %r — skipping", name)
                continue
            for mod_name, x in ctx.activations.items():
                self._writer.add_scalar(f"activation/{name}/{mod_name}", float(fn(x)), ctx.step)

        for name in self.recipe.gradient_diagnostics:
            if not self._enabled(name):
                continue
            if name == "grad_norm_per_module":
                for mod_name, val in _grad.grad_norm_per_module(ctx.gradients).items():
                    self._writer.add_scalar(f"gradient/grad_norm_per_module/{mod_name}", val, ctx.step)
            elif name == "norms_per_param":
                if ctx.gradients:
                    per = _grad.grad_norm_per_module(ctx.gradients)
                    for mod_name, n in per.items():
                        self._writer.add_scalar(f"grad/per_param/{mod_name}/norm", n, ctx.step)
                    self._writer.add_scalar(
                        "grad/global/total_norm", _grad.total_grad_norm(per), ctx.step,
                    )
            else:
                logger.warning("circuitry: unknown gradient diagnostic %r — skipping", name)

        for fn in self.recipe.custom:
            out = fn(ctx)
            for tag, val in out.items():
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        self._writer.add_scalar(f"custom/{tag}", float(val.item()), ctx.step)
                    else:
                        self._writer.add_histogram(f"custom/{tag}", val, ctx.step)
                else:
                    self._writer.add_scalar(f"custom/{tag}", float(val), ctx.step)
        self._main_pass_attn.clear()
