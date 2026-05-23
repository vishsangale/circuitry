"""Quick perf sanity check. Uses a much smaller model than scripts/bench_50m.py
so it runs in seconds on CI. The 50M-param run is opt-in via the script."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

from circuitry import Recorder


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, k, nn.Linear(d, d, bias=False))

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


class _Block(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.attn = _Attn(d)
        self.mlp = _Mlp(d)
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.block = _Block(32)
        self.lm_head = nn.Linear(32, 64, bias=False)
        # Initialize for stable numerics
        nn.init.normal_(self.embed.weight, mean=0, std=1.0)
        nn.init.kaiming_normal_(self.lm_head.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t):
        x = self.embed(t)
        x = self.block(x)
        return self.lm_head(x)


def _train(model: nn.Module, steps: int, rec: Recorder | None) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if rec:
        rec.attach()
    t0 = time.perf_counter()
    for s in range(steps):
        tokens = torch.randint(0, 64, (4, 16))
        logits = model(tokens)
        loss = logits.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if rec:
            rec.step(s, loss=float(loss.item()))
    elapsed = time.perf_counter() - t0
    if rec:
        rec.detach()
    return elapsed


@pytest.mark.benchmark(group="overhead")
def test_overhead_under_2x(tmp_path, benchmark, llm_recipe_no_hf_diagnostics):
    """Sanity: tiny model overhead under 2x. Real <10% budget is in
    scripts/bench_50m.py, not enforceable on tiny tests."""
    model = _Small()
    baseline = _train(_Small(), 20, rec=None)

    # Tiny stand-in model: use the llm recipe minus HF-only diagnostics.
    test_recipe = llm_recipe_no_hf_diagnostics

    rec = Recorder(model, run_dir=tmp_path, recipe=test_recipe,
                   writer="null", every_n_steps=5, strict=False)
    instrumented = benchmark(_train, model, 20, rec)
    assert instrumented < baseline * 5.0, (
        f"overhead {instrumented/baseline:.2f}x — investigate"
    )
