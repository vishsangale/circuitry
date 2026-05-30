"""Recorder drift-probe wiring tests (Workstream B-recorder).

Covers:
- default-OFF: stock llm recipe with probe_batch set but drift_probe disabled
  emits no repr_drift tags.
- enabled + probe_batch on a toy model: first emit captures anchor and emits no
  repr_drift tag; second emit emits one repr_drift tag per matched OUTPUT layer;
  values are finite and >= 0.
- Reference storage invariants: stored references on CPU; byte-unchanged after
  a subsequent probe pass; drift > 0 after weight mutation.
- Temporary hooks removed: no leftover hook handles after step().
- detach() clears _ref_probe_activations; reset_drift_reference() re-anchors
  (next emit captures a fresh anchor, emits no tag).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup():
    _clear_registry_for_tests()


def _teardown():
    _clear_registry_for_tests()


class _TinyMlp(nn.Module):
    """Two-layer MLP with a named OUTPUT-hookable module ``mlp``."""

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _drift_recipe(enabled_drift: bool = True, drift_method: str = "linear_cka",
                  probe_batch: torch.Tensor | None = None,
                  drift_max_tokens: int | None = None) -> Recipe:
    """Build a minimal recipe with drift_probe in activation_diagnostics."""
    return Recipe(
        name="__drift_test__",
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT, pattern=r"^mlp$"),
        ],
        activation_diagnostics=["drift_probe"],
        enabled={"drift_probe": enabled_drift},
        probe_batch=probe_batch,
        drift_method=drift_method,
        drift_max_tokens=drift_max_tokens,
    )


def _make_recorder(model, recipe, tmp_path) -> tuple[Recorder, RecordingWriter]:
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer=writer,
                   every_n_steps=1, strict=False)
    rec.attach()
    return rec, writer


def _drift_tags(writer: RecordingWriter) -> list[str]:
    return [t for t, _, _ in writer.scalars if "repr_drift" in t]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_off_no_repr_drift_tags(tmp_path):
    """Stock llm recipe with probe_batch set but drift_probe disabled emits
    no repr_drift tags."""
    _setup()
    try:
        import dataclasses

        from circuitry.recipes.llm import RECIPE

        model = _TinyMlp()
        probe = torch.randn(1, 8)
        # Patch probe_batch onto the stock recipe; drift_probe is disabled by
        # default in the stock recipe.
        recipe = dataclasses.replace(RECIPE, probe_batch=probe)
        assert recipe.enabled.get("drift_probe") is False, (
            "stock llm recipe should have drift_probe disabled by default"
        )
        writer = RecordingWriter()
        rec = Recorder(model, run_dir=tmp_path, recipe=recipe, writer=writer,
                       every_n_steps=1, strict=False)
        rec.attach()
        model(torch.randn(2, 8))
        rec.step(0)
        model(torch.randn(2, 8))
        rec.step(1)
        rec.detach()

        assert len(_drift_tags(writer)) == 0, (
            "drift_probe disabled → no repr_drift tags should be emitted"
        )
    finally:
        _teardown()


def test_first_emit_captures_anchor_no_drift_tag(tmp_path):
    """With drift_probe enabled: first emit captures the anchor and emits NO
    repr_drift tag."""
    _setup()
    try:
        model = _TinyMlp()
        probe = torch.randn(1, 8)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            model(torch.randn(2, 8))
            rec.step(0)
            # First emit: anchor captured, no drift tag.
            assert _drift_tags(writer) == [], (
                "first emit should not emit any repr_drift tag (anchor step)"
            )
            assert rec._ref_probe_activations is not None, (
                "_ref_probe_activations should be populated after first emit"
            )
        finally:
            rec.detach()
    finally:
        _teardown()


def test_second_emit_produces_repr_drift_tags(tmp_path):
    """With drift_probe enabled: second emit emits one repr_drift tag per
    matched OUTPUT layer, with finite non-negative values."""
    _setup()
    try:
        model = _TinyMlp()
        # Use >= 2 rows so linear_cka (the default drift method) can run.
        probe = torch.randn(2, 8)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            # Step 0: anchor (no tag)
            model(torch.randn(2, 8))
            rec.step(0)
            # Step 1: drift comparison (should emit tags)
            model(torch.randn(2, 8))
            rec.step(1)
            tags = _drift_tags(writer)
            assert len(tags) >= 1, (
                f"second emit should emit at least one repr_drift tag; got {writer.scalars}"
            )
            # Values should be finite and in [0, 1]
            for tag, val, step in writer.scalars:
                if "repr_drift" in tag:
                    assert step == 1, f"drift tag at unexpected step {step}"
                    assert 0.0 <= val <= 1.0, f"drift value {val} out of [0, 1]"
                    assert not (val != val), f"drift value is NaN for {tag}"
        finally:
            rec.detach()
    finally:
        _teardown()


def test_reference_storage_invariants(tmp_path):
    """Reference tensor storage invariants: CPU placement and immutability.

    Three sub-checks:

    1. **CPU placement**: stored reference tensors must be on CPU regardless of
       model device (ensures minimal VRAM usage and cross-device portability).

    2. **Reference immutability**: clone the reference right after the anchor
       step; after a subsequent probe pass the stored reference must be
       byte-identical to the clone.  Captured activations are fresh allocations
       each forward pass so this tests that the stored tensor is genuinely
       independent.

    3. **Drift detects weight change**: after substantially randomising the model
       weights, the emitted drift must be > 0 (CKA < 1.0).
    """
    _setup()
    try:
        torch.manual_seed(42)
        model = _TinyMlp(d=16)
        # Fixed non-trivial probe so activations span a non-degenerate subspace.
        torch.manual_seed(7)
        probe = torch.randn(4, 16)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            # Step 0: anchor.
            model(torch.randn(3, 16))
            rec.step(0)
            assert rec._ref_probe_activations is not None

            # Check 1: stored references must be on CPU.
            for mn, t in rec._ref_probe_activations.items():
                assert t.device.type == "cpu", (
                    f"reference tensor for {mn!r} is on {t.device}, expected CPU"
                )

            # Clone reference values immediately after anchor.
            ref_values_after_anchor = {
                mn: t.clone() for mn, t in rec._ref_probe_activations.items()
            }

            # Substantially randomise model weights so probe activations change.
            torch.manual_seed(999)
            with torch.no_grad():
                for p in model.parameters():
                    p.data = torch.randn_like(p) * 10.0

            # Step 1: compute drift.
            model(torch.randn(3, 16))
            rec.step(1)

            # Check 2: reference values must be byte-identical to the clone.
            for mn in ref_values_after_anchor:
                assert mn in rec._ref_probe_activations, (
                    f"reference key {mn!r} missing after step 1"
                )
                assert torch.equal(
                    ref_values_after_anchor[mn],
                    rec._ref_probe_activations[mn],
                ), (
                    f"reference activations for {mn!r} changed between steps "
                    "(reference must be a stable CPU copy, not a mutable alias)"
                )

            # Check 3: drift must be > 0 after weight randomisation.
            tags_vals = [(t, v) for t, v, _ in writer.scalars if "repr_drift" in t]
            assert len(tags_vals) >= 1, (
                "expected repr_drift tags after weight mutation"
            )
            for tag, val in tags_vals:
                assert val > 0.0, (
                    f"drift must be > 0 after weight mutation; got {val} for {tag}. "
                    "Large weight changes must produce nonzero drift."
                )
        finally:
            rec.detach()
    finally:
        _teardown()


def test_no_leftover_hook_handles_after_step(tmp_path):
    """Temporary hooks installed by the drift probe pass must all be removed
    in the try/finally block before step() returns."""
    _setup()
    try:
        model = _TinyMlp()
        probe = torch.randn(1, 8)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            # Count hook handles before.
            handles_before = len(rec._hook_handles)
            model(torch.randn(2, 8))
            rec.step(0)  # anchor
            model(torch.randn(2, 8))
            rec.step(1)  # drift step
            handles_after = len(rec._hook_handles)
            # The count must not grow — all temporary probe hooks removed.
            assert handles_after == handles_before, (
                f"leftover hook handles: before={handles_before}, after={handles_after}. "
                "Temporary drift-probe hooks were not removed."
            )
        finally:
            rec.detach()
    finally:
        _teardown()


def test_detach_clears_ref_probe_activations(tmp_path):
    """detach() must clear _ref_probe_activations to release RAM."""
    _setup()
    try:
        model = _TinyMlp()
        probe = torch.randn(1, 8)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        model(torch.randn(2, 8))
        rec.step(0)
        assert rec._ref_probe_activations is not None
        rec.detach()
        assert rec._ref_probe_activations is None, (
            "detach() must clear _ref_probe_activations"
        )
    finally:
        _teardown()


def test_reset_drift_reference_reanchors(tmp_path):
    """reset_drift_reference() clears the anchor so the next emit captures a
    fresh anchor (no drift tag on that step); subsequent steps emit drift
    relative to the new anchor."""
    _setup()
    try:
        model = _TinyMlp()
        # Use >= 2 rows so linear_cka (the default drift method) can run.
        probe = torch.randn(2, 8)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            # Step 0: initial anchor.
            model(torch.randn(2, 8))
            rec.step(0)
            assert rec._ref_probe_activations is not None

            # Reset.
            rec.reset_drift_reference()
            assert rec._ref_probe_activations is None

            # Step 1: new anchor (no drift tag).
            drift_tags_before = set(_drift_tags(writer))
            model(torch.randn(2, 8))
            rec.step(1)
            drift_tags_after_re_anchor = set(_drift_tags(writer))
            assert drift_tags_after_re_anchor == drift_tags_before, (
                "after reset_drift_reference, the next emit should capture a "
                "fresh anchor and emit NO drift tag"
            )
            assert rec._ref_probe_activations is not None, (
                "new anchor should be stored after re-anchor step"
            )

            # Step 2: drift relative to new anchor.
            model(torch.randn(2, 8))
            rec.step(2)
            new_tags = _drift_tags(writer)
            assert len(new_tags) >= 1, (
                "after re-anchoring, the second emit should produce drift tags"
            )
            for _, val, step in writer.scalars:
                if "repr_drift" in _:
                    if step == 2:
                        assert 0.0 <= val <= 1.0
        finally:
            rec.detach()
    finally:
        _teardown()


def test_drift_probe_disabled_no_probe_batch_no_tag(tmp_path):
    """With probe_batch=None (even if drift_probe were somehow enabled), the
    drift branch must not emit any tags or error."""
    _setup()
    try:
        model = _TinyMlp()
        recipe = _drift_recipe(enabled_drift=True, probe_batch=None)
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            model(torch.randn(2, 8))
            rec.step(0)
            model(torch.randn(2, 8))
            rec.step(1)
            assert _drift_tags(writer) == [], (
                "probe_batch=None → no repr_drift tags regardless of enabled flag"
            )
        finally:
            rec.detach()
    finally:
        _teardown()


class _TinyEmbedMlp(nn.Module):
    """Tiny model with an Embedding layer, simulating an LLM token pipeline.

    Requires integer token-ID inputs — crashes if fed a float tensor directly
    to the embedding lookup, so it is the canonical test for the int-probe fix.
    """

    def __init__(self, vocab: int = 16, d: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.mlp = nn.Linear(d, d)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq) int64
        return self.mlp(self.embed(token_ids).mean(dim=1))


def test_integer_probe_does_not_crash(tmp_path):
    """Regression: integer token-ID probe_batch must NOT crash step().

    Before fix #1 the recorder unconditionally cast probe_batch to the model's
    float dtype, which corrupted integer indices and raised an error inside
    nn.Embedding.  With the fix, only floating-point probes are dtype-cast;
    integer probes are moved to device only.
    """
    _setup()
    try:
        torch.manual_seed(55)
        vocab, d = 16, 8
        model = _TinyEmbedMlp(vocab=vocab, d=d)
        # Integer token-ID probe: shape (batch=2, seq=4).
        probe = torch.randint(0, vocab, (2, 4))
        assert probe.dtype == torch.int64, "probe must be integer dtype"

        recipe = Recipe(
            name="__embed_drift_test__",
            hook_points=[
                HookPoint(source=TensorSource.OUTPUT, pattern=r"^mlp$"),
            ],
            activation_diagnostics=["drift_probe"],
            enabled={"drift_probe": True},
            probe_batch=probe,
            drift_method="cosine",  # cosine works with 2 rows
        )
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            # Step 0: anchor — must not raise.
            model(torch.randint(0, vocab, (3, 4)))
            rec.step(0)

            # Step 1: drift step — must not raise and must emit finite scalars.
            model(torch.randint(0, vocab, (3, 4)))
            rec.step(1)

            tags_vals = [(_t, v) for _t, v, _ in writer.scalars if "repr_drift" in _t]
            assert len(tags_vals) >= 1, (
                "integer probe drift step should emit repr_drift scalars"
            )
            for _t, v in tags_vals:
                import math
                assert math.isfinite(v), f"repr_drift value is not finite: {v}"
                assert 0.0 <= v <= 1.0, f"repr_drift value out of [0,1]: {v}"
        finally:
            rec.detach()
    finally:
        _teardown()


def test_drift_method_cosine(tmp_path):
    """drift_method='cosine' flows through without error and emits finite tags."""
    _setup()
    try:
        model = _TinyMlp()
        probe = torch.randn(1, 8)
        recipe = _drift_recipe(enabled_drift=True, probe_batch=probe,
                               drift_method="cosine")
        register_recipe(recipe)
        rec, writer = _make_recorder(model, recipe, tmp_path)
        try:
            model(torch.randn(2, 8))
            rec.step(0)
            model(torch.randn(2, 8))
            rec.step(1)
            tags_vals = [(_t, v) for _t, v, _ in writer.scalars if "repr_drift" in _t]
            assert len(tags_vals) >= 1
            for _t, v in tags_vals:
                assert 0.0 <= v <= 1.0, f"cosine drift out of range: {v}"
        finally:
            rec.detach()
    finally:
        _teardown()
