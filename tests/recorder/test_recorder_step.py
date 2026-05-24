from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from circuitry.recipes import Recipe, _clear_registry_for_tests, register_recipe
from circuitry.recorder.hooks import HookPoint, StepContext, TensorSource
from circuitry.recorder.live import Recorder
from circuitry.writers.base import RecordingWriter


@pytest.fixture(autouse=True)
def _clean():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def _toy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))


def test_step_writes_weight_diagnostic_scalars(tmp_path):
    register_recipe(Recipe(
        name="w-only",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="w-only",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0, loss=1.0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank" in t for t in tags)
    assert any("stable_rank" in t for t in tags)
    assert ("train/loss", 1.0, 0) in writer.scalars


def test_step_respects_every_n_steps(tmp_path):
    register_recipe(Recipe(
        name="every-3",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="every-3",
                   writer=writer, every_n_steps=3)
    rec.attach()
    for s in range(7):
        rec.step(s, loss=float(s))
    rec.detach()
    steps_with_rank = sorted({s for t, _, s in writer.scalars if "effective_rank" in t})
    # Emit steps: 0, 3, 6
    assert steps_with_rank == [0, 3, 6]
    # Loss is recorded every step.
    assert sorted(s for t, _, s in writer.scalars if t == "train/loss") == list(range(7))


def test_step_runs_activation_diagnostic_after_forward(tmp_path):
    register_recipe(Recipe(
        name="act",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
    ))
    model = _toy_model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="act",
                   writer=writer, every_n_steps=1)
    rec.attach()
    _ = model(torch.randn(2, 4))
    rec.step(0)
    rec.detach()
    assert any("dead_fraction" in t for t, _, _ in writer.scalars)


def test_step_runs_custom_diagnostic(tmp_path):
    def custom(ctx: StepContext) -> dict[str, float]:
        return {"my_metric": float(ctx.step + 1)}

    register_recipe(Recipe(
        name="cust",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        custom=[custom],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="cust",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(2)
    rec.detach()
    assert ("custom/my_metric", 3.0, 2) in writer.scalars


def test_step_skips_disabled_diagnostic(tmp_path):
    register_recipe(Recipe(
        name="dis",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank", "stable_rank"],
        enabled={"stable_rank": False},
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="dis",
                   writer=writer, every_n_steps=1)
    rec.attach()
    rec.step(0)
    rec.detach()
    tags = {t for t, _, _ in writer.scalars}
    assert any("effective_rank" in t for t in tags)
    assert not any("stable_rank" in t for t in tags)


def test_activation_diagnostic_with_every_n_steps_3(tmp_path):
    """Regression: hook capture timing must not depend on stale _current_step.

    With every_n_steps=3, activation tags should appear on steps {0, 3, 6} —
    not on {1, 2, 4, 5} — and must NOT all silently drop because the hook
    gating ran before step() updated _current_step.
    """
    register_recipe(Recipe(
        name="act-every-3",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r"^0$")],
        activation_diagnostics=["dead_fraction"],
    ))
    model = _toy_model()
    writer = RecordingWriter()
    rec = Recorder(model, run_dir=tmp_path, recipe="act-every-3",
                   writer=writer, every_n_steps=3)
    rec.attach()
    for s in range(7):
        _ = model(torch.randn(2, 4))
        rec.step(s)
    rec.detach()
    act_steps = sorted({step for tag, _, step in writer.scalars
                        if "dead_fraction" in tag})
    assert act_steps == [0, 3, 6]


def test_step_accepts_tensor_loss_without_warning(tmp_path):
    """Recorder.step(loss=...) accepts a Tensor (incl. requires_grad=True)
    and detaches internally before logging. Verifies no PyTorch
    'converting tensor with requires_grad to scalar' UserWarning fires."""
    register_recipe(Recipe(
        name="tensor-loss",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
        weight_diagnostics=["effective_rank"],
    ))
    writer = RecordingWriter()
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="tensor-loss",
                   writer=writer, every_n_steps=1)
    rec.attach()
    loss = torch.tensor(2.5, requires_grad=True)
    components = {"aux": torch.tensor(0.5, requires_grad=True)}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote any UserWarning to an error
        rec.step(0, loss=loss, loss_components=components)
    rec.detach()
    assert ("train/loss", 2.5, 0) in writer.scalars
    assert ("train/aux", 0.5, 0) in writer.scalars


def test_step_rejects_multi_element_tensor_loss(tmp_path):
    """Recorder.step(loss=...) requires a scalar; reject multi-element
    tensors with a clear error rather than silently averaging."""
    register_recipe(Recipe(
        name="bad-loss",
        hook_points=[HookPoint(source=TensorSource.WEIGHT, pattern=r"^\d+$")],
    ))
    rec = Recorder(_toy_model(), run_dir=tmp_path, recipe="bad-loss",
                   writer=RecordingWriter(), every_n_steps=1)
    rec.attach()
    with pytest.raises(ValueError, match="0-d / size-1 Tensor"):
        rec.step(0, loss=torch.tensor([1.0, 2.0]))
    rec.detach()


def test_attention_pattern_entropy_uses_main_pass_not_probe(tmp_path):
    """Spec §5: attention_pattern_entropy must source from the user's main
    forward pass (real data), not the induction-score probe.

    We assert this by checking that the entropy value matches what the
    user's forward pass produces — not what a synthetic probe produces."""
    import json

    import pytest
    import torch
    import torch.nn as nn

    from circuitry import HookPoint, Recipe, Recorder, TensorSource

    d_model = 8
    n_heads = 2

    class _Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)
            self.last_attn: torch.Tensor | None = None

        def forward(self, x, output_attentions: bool = False):
            B, T, D = x.shape
            H = n_heads
            HD = D // H
            q = self.q_proj(x).view(B, T, H, HD).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, HD).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, HD).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (HD ** 0.5)
            attn = scores.softmax(dim=-1)
            self.last_attn = attn.detach().clone()
            out = self.o_proj((attn @ v).transpose(1, 2).reshape(B, T, D))
            if output_attentions:
                return out, attn
            return out

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

        def forward(self, x, output_attentions: bool = False):
            return self.self_attn(x, output_attentions=output_attentions)

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            import types
            self.config = types.SimpleNamespace(
                output_attentions=False, _attn_implementation="eager")
            self.tok_embed = nn.Embedding(100, d_model)
            self.layers = nn.ModuleList([_Block()])

        def get_output_embeddings(self):
            return None

        def forward(self, input_ids):
            oa = self.config.output_attentions
            x = self.tok_embed(input_ids)
            for layer in self.layers:
                x = layer(x, output_attentions=oa)
            return x

    model = _Tiny()
    recipe = Recipe(
        name="entropy_only",
        hook_points=[
            HookPoint(source=TensorSource.OUTPUT, pattern=r"layers\.\d+\.self_attn$"),
        ],
        activation_diagnostics=["attention_pattern_entropy"],
    )
    rec = Recorder(model, tmp_path, recipe, writer="jsonl", every_n_steps=1, strict=False)
    rec.attach()

    # Distinctive real-data input so we can recognise its attention pattern.
    torch.manual_seed(42)
    real_input = torch.randint(0, 100, (1, 7), dtype=torch.long)
    model(real_input)
    real_attn_snapshot = model.layers[0].self_attn.last_attn.clone()

    rec.step(0)
    rec.detach()

    # The entropy logged should match the entropy of the captured real-data
    # attention, NOT the entropy of any synthetic probe sequence.
    from circuitry.core.attention import attention_pattern_entropy
    expected_entropies = attention_pattern_entropy(real_attn_snapshot)

    out = (tmp_path / "metrics.jsonl").read_text()
    head_vals: dict[int, float] = {}
    for line in out.splitlines():
        rec_dict = json.loads(line)
        tag = rec_dict.get("tag", "")
        if "attention_pattern_entropy/layers.0.self_attn/head_" in tag:
            idx = int(tag.rsplit("_", 1)[-1])
            head_vals[idx] = rec_dict["value"]

    assert set(head_vals) == {0, 1}
    for i in (0, 1):
        assert head_vals[i] == pytest.approx(expected_entropies[i], abs=1e-4)


def test_attention_entropy_warns_once_on_unnormalized_rows(tmp_path, caplog):
    """When captured attention rows don't sum to 1 (sigmoid-like), the Recorder
    warns once that entropy is over the normalized shape."""
    import logging
    import types
    import torch
    import torch.nn as nn

    from circuitry import HookPoint, Recipe, Recorder, TensorSource

    d_model, n_heads = 8, 2

    class _Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, output_attentions: bool = False):
            B, T, D = x.shape
            H = n_heads
            q = self.q_proj(x).view(B, T, H, D // H).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, D // H).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (D // H) ** 0.5
            attn = torch.sigmoid(scores)  # rows do NOT sum to 1
            out = (attn @ q)  # shape only; value irrelevant
            out = out.transpose(1, 2).reshape(B, T, D)
            if output_attentions:
                return out, attn
            return out

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attn()

        def forward(self, x, output_attentions: bool = False):
            return self.self_attn(x, output_attentions=output_attentions)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(
                output_attentions=False, _attn_implementation="eager")
            self.layers = nn.ModuleList([_Block()])

        def forward(self, x, output_attentions=None):
            # Dual-mode: under the current kwarg injection it receives
            # output_attentions as a kwarg; if absent it reads config.
            if output_attentions is None:
                output_attentions = self.config.output_attentions
            for b in self.layers:
                x = b(x, output_attentions=output_attentions)
            return x

    model = _M()
    recipe = Recipe(
        name="entropy_warn",
        hook_points=[HookPoint(source=TensorSource.OUTPUT,
                               pattern=r"layers\.\d+\.self_attn$")],
        activation_diagnostics=["attention_pattern_entropy"],
    )
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec.attach()
    model(torch.randn(1, 4, d_model))
    rec.step(0)
    rec.detach()

    warns = [r for r in caplog.records if "do not sum to 1" in r.getMessage()]
    assert len(warns) == 1, [r.getMessage() for r in caplog.records]


def _lens_model_and_recipe(lens_max_tokens=None):
    import torch.nn as nn
    from circuitry import HookPoint, Recipe, TensorSource

    d_model, vocab = 8, 16

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            return x + self.lin(x)

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block(), _Block()])
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return self.lm_head(self.ln_f(x))

    recipe = Recipe(
        name=f"lens_{lens_max_tokens}",
        hook_points=[HookPoint(source=TensorSource.OUTPUT,
                               pattern=r"layers\.\d+$")],
        activation_diagnostics=["logit_lens_kl"],
        lens_max_tokens=lens_max_tokens,
    )
    return _Tiny(), recipe, d_model


def test_lens_max_tokens_caps_sequence_dim(tmp_path, monkeypatch):
    import torch
    import circuitry.core.lens as lens_mod
    from circuitry import Recorder

    seen = {}
    real = lens_mod.logit_lens_kl

    def spy(residual, *a, **k):
        seen["seq"] = residual.shape[1]
        return real(residual, *a, **k)

    monkeypatch.setattr(lens_mod, "logit_lens_kl", spy)
    model, recipe, d_model = _lens_model_and_recipe(lens_max_tokens=2)
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 6, d_model))
    rec.step(0)
    rec.detach()
    assert seen["seq"] == 2


def test_logit_lens_kl_oom_is_survived(tmp_path, monkeypatch, caplog):
    import logging
    import torch
    import circuitry.core.lens as lens_mod
    from circuitry import Recorder

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate ...")

    monkeypatch.setattr(lens_mod, "logit_lens_kl", boom)
    model, recipe, d_model = _lens_model_and_recipe()
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    caplog.set_level(logging.WARNING, logger="circuitry")
    rec.attach()
    model(torch.randn(1, 4, d_model))
    rec.step(0)        # must NOT raise
    rec.detach()
    assert any("out of memory" in r.getMessage().lower()
               for r in caplog.records)
    out = (tmp_path / "metrics.jsonl").read_text()
    assert "activation/logit_lens_kl/layers.0" not in out  # skipped


def test_non_oom_runtimeerror_still_propagates(tmp_path, monkeypatch):
    import torch
    import pytest
    import circuitry.core.lens as lens_mod
    from circuitry import Recorder

    def boom(*a, **k):
        raise RuntimeError("some unrelated bug")

    monkeypatch.setattr(lens_mod, "logit_lens_kl", boom)
    model, recipe, d_model = _lens_model_and_recipe()
    rec = Recorder(model, tmp_path, recipe, writer="jsonl",
                   every_n_steps=1, strict=False)
    rec.attach()
    model(torch.randn(1, 4, d_model))
    with pytest.raises(RuntimeError, match="some unrelated bug"):
        rec.step(0)
    rec.detach()
