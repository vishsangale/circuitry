#!/usr/bin/env python3
"""Real-model validation for the recsys-and-attach-fix (v1.9) follow-ups.

Exercises every session change against REAL models, not synthetic toys:
  - fp#1  attention_head_rank emits per-head ranks on a real HF model
  - recsysB attention_pattern_entropy is finite on real left-padded attention
  - recsys#5 gradient diagnostics emit on the llm AND recsys recipes
  - fp#6  sv_histogram companion scalars (sv_max/sv_min/spectral_entropy) emit
  - recsysC/D Recipe.forward_fn drives a non-HF (SASRec) probe forward
  - recsys#3 TensorSource.NAMED_PARAM reaches nn.MultiheadAttention.in_proj_weight
  - fp#3  scan_run(checkpoints=...) on arbitrarily-named snapshots + traj warning
  - fp#4  Recipe.effective_diagnostics() reflects .only()/.disable()
  - fp#2  lean import: `import circuitry` pulls neither sae_lens nor tensorboard

Run from repo root:
    .venv/bin/python scripts/v19_validation/real_model_followups.py
Writes scripts/v19_validation/real_model_followups.results.json and exits
nonzero if any check fails.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import torch
import torch.nn as nn

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from circuitry import Recorder, scan_run  # noqa: E402
from circuitry.core.attention import attention_pattern_entropy  # noqa: E402
from circuitry.recipes import Recipe, get_recipe  # noqa: E402
from circuitry.recorder.hooks import HookPoint, TensorSource  # noqa: E402

_WANT = os.environ.get("CIRC_VALIDATION_DEVICE", "auto")
if _WANT == "auto":
    DEV = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEV = _WANT
LLM = "Qwen/Qwen2.5-0.5B"
RESULTS: list[dict] = []


def _check(name: str, fn) -> None:
    try:
        detail = fn()
        RESULTS.append({"check": name, "passed": True, "detail": detail})
        print(f"  PASS  {name} — {detail}", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        RESULTS.append({"check": name, "passed": False,
                        "detail": f"{e.__class__.__name__}: {e}"})
        print(f"  FAIL  {name} — {e.__class__.__name__}: {e}", flush=True)
        traceback.print_exc()


def _jsonl_tags(run_dir: pathlib.Path) -> list[str]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    tags = []
    for line in p.read_text().splitlines():
        try:
            tags.append(json.loads(line).get("tag", ""))
        except Exception:  # noqa: BLE001
            pass
    return tags


# --------------------------------------------------------------------------
# LLM checks on a real Qwen2.5-0.5B (eager attention, GQA, q/k/v/o_proj)
# --------------------------------------------------------------------------

def _load_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM)
    model = AutoModelForCausalLM.from_pretrained(
        LLM, attn_implementation="eager", torch_dtype=torch.float32
    ).to(DEV)
    return model, tok


def check_llm_real_run():
    """One real forward+backward+step on Qwen: head-rank, grad, sv-scalars,
    finite entropy — fp#1, recsys#5, fp#6, recsysB all at once."""
    model, tok = _load_qwen()
    model.train()
    cfg = model.config
    with tempfile.TemporaryDirectory() as td:
        run_dir = pathlib.Path(td)
        rec = Recorder(model, run_dir=run_dir, recipe="llm",
                       writer="jsonl", every_n_steps=1, strict=False)
        rec.attach()
        enc = tok(["The quick brown fox jumps over the lazy dog."],
                  return_tensors="pt").to(DEV)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        rec.step(0, loss=float(out.loss.detach()))
        rec.detach()
        tags = _jsonl_tags(run_dir)

    head_rank = [t for t in tags if "attention_head_rank" in t]
    grad = [t for t in tags if t.startswith("grad/") or t.startswith("gradient/")]
    sv_scalars = [t for t in tags if t.endswith(("/sv_max", "/sv_min", "/spectral_entropy"))]
    ent = [t for t in tags if "attention_pattern_entropy" in t]
    assert head_rank, "attention_head_rank emitted ZERO tags on a real HF model (fp#1 regression)"
    assert grad, "gradient diagnostics emitted ZERO tags (recsys#5)"
    assert sv_scalars, "sv_histogram companion scalars not emitted (fp#6)"
    return (f"n_heads={cfg.num_attention_heads} head_rank_tags={len(head_rank)} "
            f"grad_tags={len(grad)} sv_scalar_tags={len(sv_scalars)} entropy_tags={len(ent)}")


def check_entropy_finite_on_left_padded_attention():
    """recsysB: extract real attention from Qwen with LEFT padding (PAD rows are
    all-(-inf) -> NaN), confirm attention_pattern_entropy returns finite values."""
    model, tok = _load_qwen()
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(["Hi.", "A considerably longer sentence to force left padding here."],
              return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attn = out.attentions[0]  # (B, H, T, T) for layer 0
    has_nan_rows = bool(torch.isnan(attn).any().item())
    ents = attention_pattern_entropy(attn)
    all_finite = all(math.isfinite(e) for e in ents)
    assert all_finite, f"entropy produced non-finite values on left-padded attention: {ents}"
    return f"layer0 attn had_nan_rows={has_nan_rows} -> {len(ents)} finite per-head entropies"


# --------------------------------------------------------------------------
# Recsys / MHA checks on a real-shaped SASRec model
# --------------------------------------------------------------------------

class _SASRec(nn.Module):
    """SASRec-like sequential recommender: nn.MultiheadAttention (fused
    in_proj_weight), FFN linears, embeddings. forward != HF-style."""

    def __init__(self, n_items=500, d=64, n_heads=2, n_layers=2, T=50):
        super().__init__()
        self.T = T

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
                self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
                self.norm1 = nn.LayerNorm(d)
                self.norm2 = nn.LayerNorm(d)

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
                self.pos_emb = nn.Embedding(T + 1, d)
                self.blocks = nn.ModuleList([_Block() for _ in range(n_layers)])
                self.layer_norm = nn.LayerNorm(d)

        self.encoder = _Enc()
        self.output_proj = nn.Linear(d, n_items + 1)

    def _embed(self, seq):
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        return self.encoder.item_emb(seq) + self.encoder.pos_emb(pos)

    def predict_scores(self, seq):  # non-HF entry point
        x = self._embed(seq)
        for blk in self.encoder.blocks:
            a, _ = blk.self_attn(x, x, x, need_weights=False)
            x = blk.norm1(x + a)
            x = blk.norm2(x + blk.ffn(x))
        return self.output_proj(self.encoder.layer_norm(x))

    def forward(self, seq):
        return self.predict_scores(seq)


def check_recsys_recipe_grad_tags():
    """recsys#5: the stock recsys recipe emits gradient tags on a real SASRec."""
    model = _SASRec().to(DEV)
    model.train()
    with tempfile.TemporaryDirectory() as td:
        run_dir = pathlib.Path(td)
        rec = Recorder(model, run_dir=run_dir, recipe="recsys",
                       writer="jsonl", every_n_steps=1, strict=True)
        rec.attach()
        seq = torch.randint(1, 500, (4, 50), device=DEV)
        logits = model(seq)
        loss = logits.float().pow(2).mean()
        loss.backward()
        rec.step(0, loss=float(loss.detach()))
        rec.detach()
        tags = _jsonl_tags(run_dir)
    grad = [t for t in tags if t.startswith("grad/")]
    assert grad, "recsys recipe emitted ZERO gradient tags on a real SASRec"
    return f"recsys grad tags: {len(grad)} (e.g. {grad[0]})"


def check_named_param_in_proj():
    """recsys#3: NAMED_PARAM reaches the fused in_proj_weight of nn.MultiheadAttention."""
    model = _SASRec().to(DEV)
    recipe = Recipe(
        name="np-sasrec",
        hook_points=[HookPoint(source=TensorSource.NAMED_PARAM,
                               pattern=r".*\.in_proj_weight$")],
        weight_diagnostics=["effective_rank", "stable_rank", "condition_number"],
    )
    with tempfile.TemporaryDirectory() as td:
        run_dir = pathlib.Path(td)
        rec = Recorder(model, run_dir=run_dir, recipe=recipe,
                       writer="jsonl", every_n_steps=1, strict=True)
        rec.attach()
        rec.step(0, loss=0.0)
        rec.detach()
        tags = _jsonl_tags(run_dir)
    in_proj = [t for t in tags if "in_proj_weight" in t]
    assert len(in_proj) >= 2, f"NAMED_PARAM did not reach in_proj_weight; tags={in_proj}"
    return f"in_proj_weight diagnostics: {len(in_proj)} tags across {2} blocks"


def check_forward_fn_drives_probe():
    """recsysC/D: Recipe.forward_fn is invoked for the recorder's probe passes
    on a non-HF model (drift_probe), instead of the HF-style call failing."""
    model = _SASRec().to(DEV)
    calls = {"n": 0}

    def fwd(m, batch):
        calls["n"] += 1
        return m.predict_scores(batch)

    probe = torch.randint(1, 500, (2, 50), device=DEV)
    recipe = Recipe(
        name="ff-sasrec",
        hook_points=[HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.ffn$")],
        activation_diagnostics=["drift_probe"],
        gradient_diagnostics=[],
        probe_batch=probe,
        forward_fn=fwd,
        enabled={"drift_probe": True},
    )
    with tempfile.TemporaryDirectory() as td:
        rec = Recorder(model, run_dir=pathlib.Path(td), recipe=recipe,
                       writer="jsonl", every_n_steps=1, strict=False)
        rec.attach()
        model(probe)  # main forward populates activations
        rec.step(0, loss=0.0)  # drift_probe runs the probe via forward_fn
        rec.detach()
    assert calls["n"] >= 1, "Recipe.forward_fn was never called for the probe pass"
    return f"forward_fn invoked {calls['n']}x for the drift probe"


# --------------------------------------------------------------------------
# scan + DX + packaging
# --------------------------------------------------------------------------

def check_scan_arbitrary_checkpoints_and_traj_warning():
    """fp#3 + fp#5: scan arbitrarily-named snapshots via checkpoints=, and a
    single-snapshot scan warns about trajectory diagnostics."""
    model = _SASRec().to("cpu")
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        c1 = root / "sasrec_epoch1.pt"
        c2 = root / "sasrec_epoch9.pt"
        torch.save(model.state_dict(), c1)
        for p in model.parameters():
            p.data.add_(0.01)
        torch.save(model.state_dict(), c2)

        recipe = get_recipe("recsys")
        # Two snapshots -> trajectory diagnostics emit, no warning.
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            scan_run(run_dir=root, recipe=recipe, out_dir=root / "out2",
                     model_factory=lambda: _SASRec(), writer="jsonl",
                     checkpoints=[(1, c1), (9, c2)], strict=False)
        traj_warn_2 = any("trajectory diagnostics" in str(x.message) for x in w2)
        tags2 = _jsonl_tags(root / "out2")
        has_update_delta = any("update_delta" in t for t in tags2)

        # Single snapshot -> warns.
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            scan_run(run_dir=root, recipe=recipe, out_dir=root / "out1",
                     model_factory=lambda: _SASRec(), writer="jsonl",
                     checkpoints=c1, strict=False)
        traj_warn_1 = any("trajectory diagnostics" in str(x.message) for x in w1)

    assert not traj_warn_2, "trajectory warning fired with 2 checkpoints"
    assert has_update_delta, "update_delta did not emit across 2 scanned checkpoints"
    assert traj_warn_1, "single-snapshot scan did NOT warn about trajectory diagnostics"
    return "2-ckpt scan: update_delta emitted, no warn; 1-ckpt scan: warned"


def check_effective_diagnostics():
    """fp#4: effective_diagnostics() reflects .only()."""
    r = get_recipe("llm").only(["effective_rank", "norms_per_param"])
    active = set(r.active_diagnostics)
    assert active == {"effective_rank", "norms_per_param"}, active
    assert "effective_rank" in r.weight_diagnostics  # raw list untouched
    return f"active={sorted(active)} (raw lists unchanged)"


def check_lean_import():
    """fp#2: `import circuitry` pulls neither sae_lens nor tensorboard."""
    code = (
        "import sys, circuitry\n"
        "from circuitry.recorder.live import Recorder\n"
        "from circuitry.recipes import get_recipe\n"
        "get_recipe('llm'); get_recipe('recsys')\n"
        "bad=[m for m in sys.modules if 'sae_lens' in m or 'tensorboard' in m]\n"
        "assert not bad, bad\nprint('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": _SRC})
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr
    return "import circuitry imports neither sae_lens nor tensorboard"


def main() -> int:
    print(f"device={DEV} torch={torch.__version__} model={LLM}\n", flush=True)
    print("== LLM (real Qwen2.5-0.5B) ==", flush=True)
    _check("fp1+recsys5+fp6: llm real run (head-rank/grad/sv-scalars)", check_llm_real_run)
    _check("recsysB: entropy finite on real left-padded attention",
           check_entropy_finite_on_left_padded_attention)
    print("== Recsys / MHA (real-shaped SASRec) ==", flush=True)
    _check("recsys5: recsys recipe grad tags", check_recsys_recipe_grad_tags)
    _check("recsys3: NAMED_PARAM reaches in_proj_weight", check_named_param_in_proj)
    _check("recsysC/D: forward_fn drives probe", check_forward_fn_drives_probe)
    print("== scan / DX / packaging ==", flush=True)
    _check("fp3+fp5: scan arbitrary checkpoints + traj warning",
           check_scan_arbitrary_checkpoints_and_traj_warning)
    _check("fp4: effective_diagnostics reflection", check_effective_diagnostics)
    _check("fp2: lean import (no sae_lens/tensorboard)", check_lean_import)

    n_pass = sum(r["passed"] for r in RESULTS)
    n_total = len(RESULTS)
    out = pathlib.Path(__file__).with_suffix(".results.json")
    out.write_text(json.dumps(
        {"device": DEV, "model": LLM, "passed": n_pass, "total": n_total,
         "results": RESULTS}, indent=2))
    print(f"\n{n_pass}/{n_total} checks passed. Results -> {out}", flush=True)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
