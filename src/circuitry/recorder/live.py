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
        writer: MetricWriter | str = "tensorboard",
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
        self._induction_probe: torch.Tensor | None = None
        self._main_pass_attn: dict[str, torch.Tensor] = {}
        self._warned_unnormalized_attn = False
        self._output_attentions_restore: list[tuple[Any, Any]] = []
        self._saes: dict[str, Any] = {}  # module_name → loaded SAE
        self._noop = False
        self._current_step: int = -1

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
        # from model.config once. Skip silently if config is missing or
        # incomplete — WARN at step time on first emit so users see why.
        if "attention_head_rank" in self.recipe.weight_diagnostics:
            cfg = getattr(self.model, "config", None)
            text_cfg = getattr(cfg, "text_config", None)  # multimodal HF
            for source in (text_cfg, cfg):
                if source is None:
                    continue
                n_heads = getattr(source, "num_attention_heads", None)
                if n_heads is None:
                    continue
                n_kv_heads = getattr(source, "num_key_value_heads", n_heads)
                head_dim = getattr(source, "head_dim", None)
                if head_dim is None:
                    hidden = getattr(source, "hidden_size", None)
                    if hidden is None or n_heads == 0:
                        continue
                    head_dim = hidden // n_heads
                self._attn_meta = _AttnMeta(
                    n_heads=int(n_heads),
                    n_kv_heads=int(n_kv_heads),
                    head_dim=int(head_dim),
                )
                break

        if "logit_lens_kl" in self.recipe.activation_diagnostics:
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
                    "circuitry: logit_lens_kl requested but model has no "
                    "resolvable output embedding (get_output_embeddings or "
                    ".lm_head) — skipping"
                )

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

        if "induction_score" in self.recipe.activation_diagnostics:
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
                    "circuitry: induction_score requested but cannot resolve "
                    "vocab_size — skipping"
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
                if self.strict:
                    raise RuntimeError(
                        msg + " — refusing to attach (pass strict=False to skip "
                        "unmatched HookPoints with a warning)"
                    )
                logger.warning("circuitry: %s — skipping this HookPoint", msg)
                matched_lines.append("")
                summary_hook_points.append({
                    "idx": idx,
                    "source": hp.source.value,
                    "label": label,
                    "matched": 0,
                    "resolved": 0,
                    "unresolved": 0,
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
            })
            totals["matched"] += len(names)
            totals["resolved"] += hp_resolved
            totals["unresolved"] += hp_unresolved

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

    def _set_output_attentions_true(self) -> None:
        """Enable per-head attention output via the HF config (not a forward
        kwarg, which breaks wrappers whose forward lacks **kwargs). Records the
        original value(s) so detach() can restore exactly. Call LAST in attach()
        so a failed attach never leaves the config mutated."""
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        for source in (cfg, text_cfg):
            if source is None:
                continue
            self._output_attentions_restore.append(
                (source, getattr(source, "output_attentions", False))
            )
            source.output_attentions = True

    def _restore_output_attentions(self) -> None:
        for source, original in self._output_attentions_restore:
            source.output_attentions = original
        self._output_attentions_restore.clear()

    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._restore_output_attentions()
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

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
        for idx, hp in enumerate(self.recipe.hook_points):
            if hp.source not in (TensorSource.WEIGHT, TensorSource.GRAD):
                continue
            for mod_name in self._matched[idx]:
                param_name = self._param_for_module.get(mod_name)
                if param_name is None:
                    continue  # already WARNed at attach()
                p = name_to_param.get(param_name)
                if not isinstance(p, torch.Tensor):
                    continue
                if hp.source is TensorSource.WEIGHT:
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

        self._run_diagnostics(ctx)
        # Discard activations now that we've consumed them.
        self._captured_activations.clear()

    def _enabled(self, name: str) -> bool:
        return self.recipe.enabled.get(name, True)

    def _run_diagnostics(self, ctx: StepContext) -> None:
        assert self._writer is not None

        # Singular values of each weight matrix are computed at most once per
        # step and shared across all SVD-derived diagnostics (effective_rank,
        # stable_rank, condition_number, heavy_tail_alpha, sv_histogram). Keyed
        # by id(); the weight tensors are stable for the duration of this call.
        from circuitry.core.weight import singular_values as _singular_values
        _sv_cache: dict[int, torch.Tensor] = {}

        def _sv(w: torch.Tensor) -> torch.Tensor:
            s = _sv_cache.get(id(w))
            if s is None:
                s = _singular_values(w)
                _sv_cache[id(w)] = s
            return s

        for name in self.recipe.weight_diagnostics:
            if not self._enabled(name):
                continue
            if name == "attention_head_rank":
                if self._attn_meta is None:
                    logger.warning(
                        "circuitry: attention_head_rank requested but model "
                        "has no usable config (num_attention_heads / head_dim) — "
                        "skipping"
                    )
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
                        self._writer.add_histogram(
                            f"spectral/per_param/{mod_name}/sv_histogram",
                            _sv(w), ctx.step)
            else:
                fn = _WEIGHT_DIAGS.get(name)
                if fn is None:
                    logger.warning("circuitry: unknown weight diagnostic %r — skipping", name)
                    continue
                for mod_name, w in ctx.weights.items():
                    self._writer.add_scalar(f"weight/{name}/{mod_name}", float(fn(_sv(w))), ctx.step)

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
                if self._lens_meta is None:
                    continue
                from circuitry.core.lens import logit_lens_kl as _llk
                # Derive d_model from the unembed weight. HF convention is
                # (vocab, d_model) so d_model is the smaller dim; for the
                # unusual square case we fall back to dim 1.
                _W_raw = self._lens_meta.unembed
                d_model_unembed = (
                    _W_raw.shape[-1]
                    if _W_raw.shape[0] >= _W_raw.shape[-1]
                    else _W_raw.shape[0]
                )
                # Keep only residual-stream block outputs: activations whose
                # module name ends in `.layers.N` (or is exactly `layers.N`).
                # Sub-module outputs (self_attn / mlp / layernorm) and the
                # down_proj INPUT capture are excluded even though several share
                # d_model — otherwise the lens runs once per d_model-shaped
                # activation (175 on Gemma 4) instead of once per layer (35).
                # The d_model check is a secondary guard. Sort by numeric layer
                # index so block_outputs[-1] is the true final layer regardless
                # of digit count (layer 9 must not sort after layer 34).
                import re as _re
                def _block_layer_idx(_n: str) -> int | None:
                    _m = _re.search(r'(?:^|\.)layers\.(\d+)$', _n)
                    return int(_m.group(1)) if _m else None
                block_outputs = sorted(
                    ((k, v) for k, v in ctx.activations.items()
                     if _block_layer_idx(k) is not None
                     and v.shape[-1] == d_model_unembed),
                    key=lambda kv: _block_layer_idx(kv[0]),
                )
                if not block_outputs:
                    logger.warning(
                        "circuitry: logit_lens_kl found no activations with "
                        "last-dim matching unembed d_model=%d — skipping.",
                        d_model_unembed,
                    )
                    self._lens_meta = None
                    continue
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
            if name == "induction_score":
                if self._induction_probe is None:
                    continue
                from circuitry.core.attention import induction_score as _is
                # Find matched self_attn modules from the hook points.
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
                    continue
                # Capture attn_weights from each self_attn during probe pass.
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
                try:
                    probe_dev = next(self.model.parameters()).device
                    probe = self._induction_probe.to(probe_dev)
                    with torch.inference_mode():
                        try:
                            self.model(probe, output_attentions=True)
                        except TypeError:
                            self.model(probe)
                finally:
                    for h in handles:
                        h.remove()
                for mn, attn in captured.items():
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
                continue
            if name == "attention_pattern_entropy":
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
