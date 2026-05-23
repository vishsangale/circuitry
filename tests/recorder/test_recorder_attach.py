# tests/recorder/test_recorder_attach.py
from __future__ import annotations

import logging

import pytest
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def _register_demo(pattern: str = r"^\d+$", min_matches: int = 0) -> None:
    register_recipe(Recipe(
        name="demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=pattern)],
        weight_diagnostics=["effective_rank"],
        expected_min_matches={pattern: min_matches},
    ))


def test_attach_writes_matched_modules_file(tmp_path):
    _register_demo()
    model = _toy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    f = tmp_path / "circuitry" / "matched_modules.txt"
    assert f.exists()
    content = f.read_text()
    assert "0" in content and "2" in content
    rec.detach()


def test_matched_modules_label_uses_pattern_for_pattern_hookpoints(tmp_path):
    """Regression: live.py:128 used to mis-parenthesize a ternary so every
    HookPoint ended up labeled `<selector>` in matched_modules.txt, regardless
    of whether it was pattern/modules/selector-based."""
    pattern = r"^\d+$"
    _register_demo(pattern=pattern)
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    content = (tmp_path / "circuitry" / "matched_modules.txt").read_text()
    assert f"target={pattern}" in content, content
    assert "<selector>" not in content, content


def test_attach_writes_inventory_json(tmp_path):
    """v0.6.0: attach() writes <run_dir>/circuitry/inventory.json with one
    entry per Parameter — auditable record of what circuitry can see."""
    import json as _json
    _register_demo()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    inv_path = tmp_path / "circuitry" / "inventory.json"
    assert inv_path.exists()
    parsed = _json.loads(inv_path.read_text())
    names = {r["name"] for r in parsed}
    assert "0.weight" in names and "2.weight" in names


def test_attach_resolves_wrapped_linear_via_inventory(tmp_path, caplog):
    """v0.6.0: WEIGHT HookPoints matching a wrapper module (no direct
    ``.weight``) resolve to the wrapper's inner Linear via the inventory.
    matched_modules.txt shows the resolution tail. No silent drop."""
    import logging

    class _Wrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 8, bias=False)

    class _WrappedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _Wrap()
            self.k_proj = _Wrap()

    register_recipe(Recipe(
        name="wrapped",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"(q|k)_proj$")],
        weight_diagnostics=["effective_rank"],
    ))
    writer = RecordingWriter()
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec = Recorder(_WrappedModel(), run_dir=tmp_path, recipe="wrapped",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()

    # No silent-drop WARN — both wrapper modules resolved.
    drop_warnings = [r for r in caplog.records if "UNRESOLVED" in r.message
                     or "had no resolvable" in r.message]
    assert not drop_warnings, [r.message for r in drop_warnings]

    # matched_modules.txt shows the resolution tail.
    content = (tmp_path / "circuitry" / "matched_modules.txt").read_text()
    assert "q_proj → linear.weight" in content, content
    assert "k_proj → linear.weight" in content, content

    # Both wrapper modules got effective_rank emitted (keyed by module name,
    # not parameter name, to preserve recipe-tag layout).
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank/q_proj" in t for t in tags)
    assert any("effective_rank/k_proj" in t for t in tags)


def test_attach_warns_when_module_has_no_resolvable_weight(tmp_path, caplog):
    """v0.6.0: a matched module with no 2-D+ Parameter in its subtree (or
    with ambiguous multiple candidates) is logged loudly with UNRESOLVED in
    matched_modules.txt — no silent drop."""
    import logging

    class _Ambiguous(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Two Linear children; ambiguous which is the "primary" weight.
            self.a = nn.Linear(4, 8, bias=False)
            self.b = nn.Linear(8, 4, bias=False)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target = _Ambiguous()

    register_recipe(Recipe(
        name="ambig",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"^target$")],
        weight_diagnostics=["effective_rank"],
    ))
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec = Recorder(_M(), run_dir=tmp_path, recipe="ambig",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.detach()

    assert any("UNRESOLVED" in (tmp_path / "circuitry" / "matched_modules.txt").read_text()
               for _ in [0])
    assert any("no resolvable 2-D+ weight" in r.message for r in caplog.records)


def test_attach_logs_matched_modules_at_info(tmp_path, caplog):
    _register_demo()
    caplog.set_level(logging.INFO, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    assert any("matched" in r.message.lower() for r in caplog.records)


def test_attach_raises_on_zero_matches_in_strict_mode(tmp_path):
    register_recipe(Recipe(
        name="bad",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"this-matches-nothing")],
    ))
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="bad",
                   writer=RecordingWriter(), every_n_steps=1, strict=True)
    with pytest.raises(RuntimeError, match="matched 0 modules"):
        rec.attach()


def test_attach_warns_and_skips_zero_matches_in_non_strict_mode(tmp_path, caplog):
    """v0.4.0 contract: strict=False relaxes both 0-match and under-match
    failures to warnings so circuitry can be dropped into an existing
    training script without authoring a perfect recipe first."""
    register_recipe(Recipe(
        name="partial",
        hook_points=[
            HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$"),  # matches
            HookPoint(source=TensorSource.WEIGHT,
                      pattern=r"this-matches-nothing"),  # 0-match, skipped
        ],
        weight_diagnostics=["effective_rank"],
    ))
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="partial",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.detach()
    assert any("matched 0 modules" in r.message for r in caplog.records)


def test_attach_raises_on_min_matches_violation_in_strict_mode(tmp_path):
    _register_demo(pattern=r"^\d+$", min_matches=99)
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1, strict=True)
    with pytest.raises(RuntimeError, match="expected at least 99"):
        rec.attach()


def test_attach_warns_on_min_matches_violation_in_non_strict_mode(tmp_path, caplog):
    _register_demo(pattern=r"^\d+$", min_matches=99)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.detach()
    assert any("expected at least 99" in r.message for r in caplog.records)


def test_detach_removes_all_hooks(tmp_path):
    _register_demo()
    model = _toy_model()
    rec = Recorder(model, run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()
    post = sum(len(m._forward_hooks) + len(m._forward_pre_hooks)
               + len(m._backward_hooks) for m in model.modules())
    assert post == 0
    # We don't assert pre > 0 — pure-weight recipes may not install hooks.


def test_recorder_noop_on_non_zero_rank(monkeypatch, tmp_path):
    _register_demo()
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0, loss=1.0)
    rec.detach()
    assert writer.scalars == []
    assert not (tmp_path / "circuitry" / "matched_modules.txt").exists()


# --- v0.7.0: with_prefix + attach_summary.json ----------------------------

def test_attach_with_prefix_only_keeps_modules_under_prefix(tmp_path):
    """A recipe scoped with with_prefix() should only match modules under that prefix."""
    import torch.nn as nn

    class _Scoped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4, bias=False))
            self.other = nn.Linear(4, 4, bias=False)

    model = _Scoped()
    recipe = Recipe(
        name="scoped_test",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r".*")],
        weight_diagnostics=["effective_rank"],
    ).with_prefix("lm")

    register_recipe(recipe)
    rec = Recorder(model, run_dir=tmp_path, recipe=recipe,
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.step(0)
    rec.detach()

    # Only modules under "lm" prefix should be matched: "lm.0" and "lm.1" (and "lm" itself)
    # "other" must NOT appear.
    matched_txt = (tmp_path / "circuitry" / "matched_modules.txt").read_text()
    assert "other" not in matched_txt


def test_attach_writes_attach_summary_json(tmp_path):
    """attach_summary.json is written by Recorder.attach() with correct schema."""
    import json as _json

    _register_demo()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="demo",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()

    summary_path = tmp_path / "circuitry" / "attach_summary.json"
    assert summary_path.exists()
    data = _json.loads(summary_path.read_text())
    assert "hook_points" in data
    assert "totals" in data
    assert isinstance(data["hook_points"], list)
    totals = data["totals"]
    assert "matched" in totals and "resolved" in totals and "unresolved" in totals
    # _toy_model pattern r"^\d+$" matches modules "0" (Linear), "1" (ReLU), "2" (Linear).
    # ReLU is unresolvable (no 2-D+ weight), both Linears resolve cleanly.
    assert totals["matched"] == 3
    assert totals["resolved"] == 2
    assert totals["unresolved"] == 1


def test_attach_summary_counts_resolved_and_unresolved(tmp_path):
    """attach_summary.json counts resolved=1, unresolved=1 when one module resolves
    and one is ambiguous (v0.7.0 design §3 example)."""
    import json as _json

    class _Wrap(nn.Module):
        """Resolves — single Linear child."""
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 8, bias=False)

    class _Ambiguous(nn.Module):
        """Unresolvable — two Linear children, ambiguous primary."""
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 8, bias=False)
            self.b = nn.Linear(8, 4, bias=False)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wrap = _Wrap()
            self.ambig = _Ambiguous()

    register_recipe(Recipe(
        name="mixed",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^(wrap|ambig)$")],
        weight_diagnostics=["effective_rank"],
    ))
    rec = Recorder(_M(), run_dir=tmp_path, recipe="mixed",
                   writer=RecordingWriter(), every_n_steps=1, strict=False)
    rec.attach()
    rec.detach()

    data = _json.loads((tmp_path / "circuitry" / "attach_summary.json").read_text())
    totals = data["totals"]
    assert totals["matched"] == 2
    assert totals["resolved"] == 1
    assert totals["unresolved"] == 1

    # The single hook_points entry should reflect the same counts.
    hp0 = data["hook_points"][0]
    assert hp0["matched"] == 2
    assert hp0["resolved"] == 1
    assert hp0["unresolved"] == 1


def test_attach_summary_output_hookpoint_has_resolved_equals_matched(tmp_path):
    """For OUTPUT hooks, resolved == matched and unresolved == 0."""
    import json as _json

    register_recipe(Recipe(
        name="output_recipe",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^\d+$")],
        activation_diagnostics=["dead_fraction"],
    ))
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="output_recipe",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    rec.detach()

    data = _json.loads((tmp_path / "circuitry" / "attach_summary.json").read_text())
    hp0 = data["hook_points"][0]
    assert hp0["source"] == "output"
    assert hp0["resolved"] == hp0["matched"]
    assert hp0["unresolved"] == 0


def test_attention_head_rank_emits_per_head_tags(tmp_path):
    """v0.8.0: attention_head_rank emits one scalar per head when the
    recipe requests it and the model carries usable config metadata."""
    class _Cfg:
        num_attention_heads = 4
        num_key_value_heads = 4
        head_dim = 8
        hidden_size = 32

    class _AttentionLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.o_proj = nn.Linear(32, 32, bias=False)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg()
            self.layer = _AttentionLike()

    register_recipe(Recipe(
        name="head_rank_demo",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"(q|o)_proj$")],
        weight_diagnostics=["attention_head_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_M(), run_dir=tmp_path, recipe="head_rank_demo",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()

    tags = {t for t, _, _ in writer.scalars}
    # 4 heads × 2 modules (q_proj + o_proj) = 8 tags expected.
    head_tags = [t for t in tags if "attention_head_rank" in t]
    assert len(head_tags) == 8, head_tags
    # Tag layout: weight/attention_head_rank/<module>/head_<i>
    assert any("layer.q_proj/head_0" in t for t in head_tags)
    assert any("layer.o_proj/head_3" in t for t in head_tags)


def test_attention_head_rank_skips_when_no_config(tmp_path, caplog):
    """v0.8.0: when model has no usable config, attention_head_rank
    logs a WARN and emits nothing (rather than crashing)."""
    import logging

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)

    register_recipe(Recipe(
        name="no_cfg",
        hook_points=[HookPoint(source=TensorSource.WEIGHT,
                               pattern=r"q_proj$")],
        weight_diagnostics=["attention_head_rank"],
    ))
    caplog.set_level(logging.WARNING, logger="circuitry")
    writer = RecordingWriter()
    rec = Recorder(_M(), run_dir=tmp_path, recipe="no_cfg",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()

    assert not any("attention_head_rank" in t for t, _, _ in writer.scalars)
    assert any("attention_head_rank" in r.message and "config" in r.message.lower()
               for r in caplog.records)


def test_gate_stats_emits_three_subscalars_per_module(tmp_path):
    """v0.8.0: gate_stats is an activation diagnostic that emits three
    subscalars per hooked module: frac_active, mean_abs, std."""
    import torch

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = nn.Linear(16, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # The INPUT pre-hook captures whatever we pass to down_proj.
            return self.down_proj(x)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _Mlp()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

    register_recipe(Recipe(
        name="gate_demo",
        hook_points=[HookPoint(source=TensorSource.INPUT,
                               pattern=r"down_proj$")],
        activation_diagnostics=["gate_stats"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_M(), run_dir=tmp_path, recipe="gate_demo",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = _M().forward  # warm-up of registered hook trees not needed; use rec.model
    x = torch.randn(2, 16)
    rec.model(x)
    rec.step(0)
    rec.detach()

    tags = {t for t, _, _ in writer.scalars}
    assert any("activation/gate_stats/mlp.down_proj/frac_active" in t for t in tags)
    assert any("activation/gate_stats/mlp.down_proj/mean_abs" in t for t in tags)
    assert any("activation/gate_stats/mlp.down_proj/std" in t for t in tags)


def test_logit_lens_kl_is_dispatched_per_block(tmp_path):
    """Recorder wires logit_lens_kl: tags like
    activation/logit_lens_kl/<block_name> appear in the JSONL output."""
    import torch
    import torch.nn as nn

    from circuitry import HookPoint, Recipe, Recorder, TensorSource

    d_model, vocab = 8, 16
    n_blocks = 2

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            return x + self.lin(x)

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(n_blocks)])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return self.lm_head(self.ln_f(x))

    model = _Tiny()
    recipe = Recipe(
        name="lens_only",
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT, pattern=r"layers\.\d+$"),
        ],
        activation_diagnostics=["logit_lens_kl"],
    )
    rec = Recorder(model, tmp_path, recipe, writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()
    inp = torch.randn(1, 3, d_model)
    model(inp)  # populate captured activations
    rec.step(0)
    rec.detach()

    out = (tmp_path / "metrics.jsonl").read_text()
    assert "activation/logit_lens_kl/layers.0" in out
    assert "activation/logit_lens_kl/layers.1" in out
