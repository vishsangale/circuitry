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
