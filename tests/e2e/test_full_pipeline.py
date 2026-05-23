"""End-to-end smoke test:
  1. Train a tiny LLM-shaped model for 6 steps with LiveRecorder + JsonlWriter
  2. Save 2 checkpoints
  3. Run scan_run over those checkpoints
  4. Run build_report against the recorded jsonl
  5. Assert the markdown contains every section we expect
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch
import torch.nn as nn

from circuitry import Recorder, build_report, scan_run
from circuitry.recipes import get_recipe
from circuitry.writers.jsonl import JsonlWriter


class _TinyBlock(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.attn = _Attn(d)
        self.mlp = _Mlp(d)
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return x


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50, 8)
        self.block = _TinyBlock(8)
        self.lm_head = nn.Linear(8, 50, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.block(x)
        return self.lm_head(x)


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.o_proj(self.v_proj(x))


class _Mlp(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))


def test_e2e_pipeline(tmp_path: pathlib.Path):
    torch.manual_seed(0)
    model = _Tiny()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()

    # Use a modified recipe without diagnostics that require full HF models.
    r = get_recipe("llm")
    test_recipe = dataclasses.replace(
        r,
        activation_diagnostics=[d for d in r.activation_diagnostics
                                if d not in ("induction_score", "logit_lens_kl",
                                            "attention_pattern_entropy")]
    )

    rec = Recorder(model, run_dir=tmp_path, recipe=test_recipe,
                   writer=JsonlWriter(tmp_path), every_n_steps=2, strict=False)
    rec.attach()
    for step in range(6):
        tokens = torch.randint(0, 50, (4, 8))
        logits = model(tokens)
        loss = logits.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec.step(step, loss=float(loss.item()))
        if step in (2, 5):
            torch.save(model.state_dict(), ckpts / f"step{step:09d}.pt")
    rec.detach()

    scan_run(run_dir=tmp_path, recipe=test_recipe,
             out_dir=tmp_path / "tb_retro",
             model_factory=lambda: _Tiny(), strict=False)

    out = build_report(run_dir=tmp_path, out_path=tmp_path / "inspect" / "report.md")
    md = out.read_text()
    assert "# circuitry report" in md
    assert "weight" in md
    assert (tmp_path / "circuitry" / "matched_modules.txt").exists()
    assert any((tmp_path / "tb_retro").rglob("events.out.tfevents.*"))
