"""Evaluate the two_tower / recsys recipe against a DLRM-style model.

Run with: .venv/bin/python scripts/v17_validation/eval_dlrm_recipe.py

Investigates:
  1. Whether the two_tower recipe matches embedding tables and MLPs in DLRM.
  2. Whether spectral/weight primitives handle nn.Embedding.weight (10000×64
     and 10000×768) — effective_rank/condition_number behaviour.
  3. Sparse-grad behaviour with nn.Embedding (sparse=True vs dense).
  4. Whether embedding_alignment custom diagnostic fires.
  5. Whether any embedding-specific breakage occurs.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import tempfile
import traceback

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import circuitry
from circuitry import Recorder, build_report
from circuitry.core.weight import singular_values, effective_rank, stable_rank, condition_number

print(f"circuitry {circuitry.__version__}  |  torch {torch.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# DLRM-style model
# ─────────────────────────────────────────────────────────────────────────────

class DLRM(nn.Module):
    """Mini DLRM:
    - 4 categorical embedding tables (10k × 64 each)
    - bottom MLP for dense features
    - dot-product feature interaction
    - top MLP → binary output
    """
    def __init__(
        self,
        n_tables: int = 4,
        vocab_size: int = 10_000,
        embed_dim: int = 64,
        n_dense: int = 13,
        bottom_hidden: int = 64,
        top_hidden: int = 64,
        sparse: bool = False,
    ):
        super().__init__()
        self.embed_tables = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, sparse=sparse)
            for _ in range(n_tables)
        ])
        self.bottom_mlp = nn.Sequential(
            nn.Linear(n_dense, bottom_hidden),
            nn.ReLU(),
            nn.Linear(bottom_hidden, embed_dim),
            nn.ReLU(),
        )
        # After interaction: n_tables embeddings + 1 bottom_mlp output = n_tables+1 vectors of embed_dim
        # Dot-product interaction (upper triangle) → (n_tables+1)*n_tables/2 scalars + embed_dim
        n_interact = (n_tables + 1) * n_tables // 2  # 10
        self.top_mlp = nn.Sequential(
            nn.Linear(embed_dim + n_interact, top_hidden),
            nn.ReLU(),
            nn.Linear(top_hidden, 1),
        )

    def forward(self, dense: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # cat_ids: (B, n_tables) int
        embs = [self.embed_tables[i](cat_ids[:, i]) for i in range(len(self.embed_tables))]  # each (B, 64)
        bottom = self.bottom_mlp(dense)   # (B, 64)

        all_vecs = embs + [bottom]         # list of (B, 64)
        T = torch.stack(all_vecs, dim=1)  # (B, n_tables+1, 64)

        # Dot-product interaction (upper triangle, no diagonal)
        dots = torch.bmm(T, T.transpose(1, 2))  # (B, n_tables+1, n_tables+1)
        n = T.shape[1]
        idx_i, idx_j = [], []
        for i in range(n):
            for j in range(i):
                idx_i.append(i); idx_j.append(j)
        interact = dots[:, idx_i, idx_j]  # (B, n_interact)

        feat = torch.cat([bottom, interact], dim=1)  # (B, 64 + n_interact)
        return self.top_mlp(feat).squeeze(-1)  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module-name audit for DLRM vs two_tower recipe
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 1 — Module-name audit: two_tower recipe vs DLRM")
print("=" * 70)

model_dlrm = DLRM()
all_names = [n for n, _ in model_dlrm.named_modules() if n]
print(f"Total DLRM named modules: {len(all_names)}")
print(f"Modules: {all_names}")

TWO_TOWER_WEIGHT_PAT = r"(query_tower|item_tower|interaction).*"
TWO_TOWER_OUTPUT_PAT = r"(query_tower|item_tower)$"

matched_w = [n for n in all_names if re.search(TWO_TOWER_WEIGHT_PAT, n)]
matched_o = [n for n in all_names if re.search(TWO_TOWER_OUTPUT_PAT, n)]
print(f"\nTwo-tower WEIGHT pattern matches: {len(matched_w)}: {matched_w}")
print(f"Two-tower OUTPUT pattern matches: {len(matched_o)}: {matched_o}")

if not matched_w and not matched_o:
    print("\n  NOTE: DLRM uses 'embed_tables', 'bottom_mlp', 'top_mlp' names.")
    print("  Two-tower recipe pattern looks for 'query_tower|item_tower|interaction'.")
    print("  ZERO coverage — recipe is name-locked to specific tower naming convention.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Spectral primitives on embedding weight shapes
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2 — Spectral primitives on Embedding weight shapes")
print("=" * 70)

test_cases = {
    "Embedding (10000×64) — narrow, should NOT subsample": (10_000, 64),
    "Embedding (10000×768) — wide, min_dim=768>512, subsample triggers": (10_000, 768),
    "Embedding (10000×128)": (10_000, 128),
    "Linear (64×64) — bottom MLP": (64, 64),
}

torch.manual_seed(0)
for label, shape in test_cases.items():
    W = torch.randn(*shape) * 0.02   # embedding init scale
    min_dim = min(shape)
    max_dim_shape = max(shape)
    # Default max_dim=512: subsample if min_dim > 512
    will_subsample_min = min_dim > 512
    # Actually subsample is on the LARGER axis if it exceeds max_dim
    larger_dim = max_dim_shape
    will_subsample_larger = larger_dim > 512
    try:
        er = effective_rank(W)
        sr = stable_rank(W)
        sv_default = singular_values(W)
        sv_full = singular_values(W, max_dim=None)
        cn_default = condition_number(W)
        cn_full_val = float((sv_full[0] / sv_full[-1]).item()) if sv_full[-1] > 1e-12 else float("inf")
        print(f"  {label}")
        print(f"    shape={shape}, subsampled_sv_shape={tuple(sv_default.shape)}, full_sv_shape={tuple(sv_full.shape)}")
        print(f"    eff_rank={er:.2f}, stable_rank={sr:.2f}")
        print(f"    cond_num(default/subsampled)={cn_default:.2f}, cond_num(full)={cn_full_val:.2f}")
    except Exception as e:
        print(f"  {label}: EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Sparse grad behaviour with nn.Embedding
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 3 — Sparse gradient behaviour with nn.Embedding")
print("=" * 70)

for sparse_flag in [False, True]:
    emb = nn.Embedding(10_000, 64, sparse=sparse_flag)
    ids = torch.randint(0, 10_000, (16,))
    out = emb(ids).sum()
    out.backward()
    grad = emb.weight.grad
    print(f"  sparse={sparse_flag}: grad type={type(grad).__name__ if grad is not None else 'None'}", end="")
    if grad is not None:
        print(f", is_sparse={grad.is_sparse}, shape={tuple(grad.shape)}", end="")
        # Try applying core weight primitives on embedding weight (not grad)
        try:
            er = effective_rank(emb.weight)
            print(f", eff_rank_on_weight={er:.2f}", end="")
        except Exception as e:
            print(f", eff_rank ERROR: {e}", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build a DLRM-compatible model with two_tower naming and run Recorder
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 4 — DLRM-compatible naming for two_tower recipe")
print("=" * 70)

# Rename to match the recipe: query_tower ↔ user/dense side, item_tower ↔ item/cat side
class DLRMTwoTower(nn.Module):
    """DLRM-style with embedding tables renamed to match two_tower recipe.

    query_tower: dense MLP (bottom MLP)
    item_tower: embedding tables + MLP
    """
    def __init__(
        self,
        n_tables: int = 4,
        vocab_size: int = 10_000,
        embed_dim: int = 64,
        n_dense: int = 13,
    ):
        super().__init__()
        # query_tower = bottom MLP for dense features → named to match recipe
        self.query_tower = nn.Sequential(
            nn.Linear(n_dense, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        # item_tower = embedding tables (the BIG 2D weights recsys cares about)
        self.item_tower = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(n_tables)
        ])
        # top MLP — named "interaction" to match two_tower weight pattern
        n_interact = (n_tables + 1) * n_tables // 2
        self.interaction = nn.Sequential(
            nn.Linear(embed_dim + n_interact, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, dense: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        q = self.query_tower(dense)       # (B, 64)
        embs = [self.item_tower[i](cat_ids[:, i]) for i in range(len(self.item_tower))]
        all_vecs = embs + [q]
        T = torch.stack(all_vecs, dim=1)
        dots = torch.bmm(T, T.transpose(1, 2))
        n = T.shape[1]
        idx_i, idx_j = [], []
        for i in range(n):
            for j in range(i):
                idx_i.append(i); idx_j.append(j)
        interact = dots[:, idx_i, idx_j]
        feat = torch.cat([q, interact], dim=1)
        return self.interaction(feat).squeeze(-1)


model_tt = DLRMTwoTower()
all_names_tt = [n for n, _ in model_tt.named_modules() if n]
print(f"DLRMTwoTower modules: {all_names_tt}")
matched_w_tt = [n for n in all_names_tt if re.search(TWO_TOWER_WEIGHT_PAT, n)]
matched_o_tt = [n for n in all_names_tt if re.search(TWO_TOWER_OUTPUT_PAT, n)]
print(f"two_tower WEIGHT pattern matches: {len(matched_w_tt)}: {matched_w_tt}")
print(f"two_tower OUTPUT pattern matches: {len(matched_o_tt)}: {matched_o_tt}")

# Count coverage: embedding tables in item_tower
embed_mods = [n for n in all_names_tt if "item_tower" in n and re.search(r"\d+$", n)]
print(f"\nEmbedding table modules in item_tower: {embed_mods}")
embed_captured_w = [n for n in embed_mods if re.search(TWO_TOWER_WEIGHT_PAT, n)]
embed_captured_o = [n for n in embed_mods if re.search(TWO_TOWER_OUTPUT_PAT, n)]
print(f"  Captured by WEIGHT pattern: {len(embed_captured_w)}/{len(embed_mods)}")
print(f"  Captured by OUTPUT pattern: {len(embed_captured_o)}/{len(embed_mods)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Live Recorder run on DLRMTwoTower (15 steps)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 5 — Recorder + two_tower recipe on DLRMTwoTower (15 steps)")
print("=" * 70)

run_dir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_dlrm_eval_"))
print(f"run_dir: {run_dir}")

model_tt.train()
opt = torch.optim.Adam(model_tt.parameters(), lr=1e-3)

rec = Recorder(
    model_tt,
    run_dir=run_dir,
    recipe="two_tower",
    writer="jsonl",
    every_n_steps=5,
)
rec.attach()

B = 32
n_tables = 4
n_dense = 13

for step in range(15):
    dense = torch.randn(B, n_dense)
    cat_ids = torch.randint(0, 10_000, (B, n_tables))
    labels = torch.randint(0, 2, (B,)).float()
    logits = model_tt(dense, cat_ids)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    rec.step(step, loss=float(loss.item()))
    if step % 5 == 0:
        print(f"  step {step:3d}  loss={loss.item():.4f}")

rec.detach()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Inspect emitted metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 6 — Emitted metrics analysis")
print("=" * 70)

metrics_path = run_dir / "metrics.jsonl"
if not metrics_path.exists():
    print("  WARN: metrics.jsonl not found")
    entries = []
else:
    entries = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]

print(f"Total JSONL entries: {len(entries)}")

if entries:
    tags = sorted(set(e.get("tag", "") for e in entries))
    print(f"Unique tags ({len(tags)}):")
    for t in tags:
        print(f"  {t}")

    # Check embedding_alignment custom diagnostic
    align = [e for e in entries if "embedding_alignment" in str(e.get("tag", ""))]
    print(f"\nembedding_alignment entries: {len(align)}")
    if align:
        vals = [e["value"] for e in align]
        print(f"  values: mean={sum(vals)/len(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}")
    else:
        print("  NOTE: embedding_alignment NOT emitted — custom diagnostic needs query+item tower outputs")
        # Explain why: item_tower is ModuleList, output hook fires on the MODULE not each element
        print("  item_tower is nn.ModuleList — output hook for 'item_tower' fires?")
        item_out = [e for e in entries if "item_tower" in str(e.get("tag", "")) and
                    any(k in str(e.get("tag", "")) for k in ["dead_fraction", "participation_ratio"])]
        print(f"  item_tower activation entries: {len(item_out)}")

    # Check for NaN/Inf
    bad = [(e.get("tag"), e.get("step"), e.get("value"))
           for e in entries
           if isinstance(e.get("value"), float) and (math.isnan(e["value"]) or math.isinf(e["value"]))]
    print(f"\nNaN/Inf entries: {len(bad)}")
    for tag, step, val in bad[:10]:
        print(f"  step={step} tag={tag} val={val}")

    # Weight metrics on embedding tables
    embed_weight_tags = [e["tag"] for e in entries if "item_tower" in str(e.get("tag", ""))
                         and any(k in str(e.get("tag", "")) for k in ["effective_rank", "stable_rank"])]
    print(f"\nEmbedding table weight metric tags: {sorted(set(embed_weight_tags))}")

    # Steps covered
    steps = sorted(set(e.get("step") for e in entries if isinstance(e.get("step"), int)))
    print(f"Steps with data: {steps}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Check: does grad_norm_per_module work with Embedding (sparse=True)?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 7 — Grad norm with sparse Embedding gradients")
print("=" * 70)

class SparseDLRM(nn.Module):
    """Like DLRMTwoTower but with sparse=True embeddings."""
    def __init__(self, n_tables=2, vocab_size=10_000, embed_dim=64, n_dense=8):
        super().__init__()
        self.query_tower = nn.Sequential(nn.Linear(n_dense, embed_dim), nn.ReLU())
        self.item_tower = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, sparse=True) for _ in range(n_tables)
        ])
        self.interaction = nn.Linear(embed_dim, 1)

    def forward(self, dense, cat_ids):
        q = self.query_tower(dense)
        embs = [self.item_tower[i](cat_ids[:, i]) for i in range(len(self.item_tower))]
        it = sum(embs) / len(embs)
        return self.interaction(q + it).squeeze(-1)

sparse_model = SparseDLRM()
run_dir_sparse = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_sparse_"))
rec_sparse = Recorder(sparse_model, run_dir=run_dir_sparse, recipe="two_tower", writer="jsonl", every_n_steps=5)
rec_sparse.attach()

sparse_error = None
try:
    opt_sparse = torch.optim.SparseAdam(
        [p for p in sparse_model.parameters() if p.requires_grad],
        lr=1e-3
    )
    for step in range(10):
        dense = torch.randn(16, 8)
        cat_ids = torch.randint(0, 10_000, (16, 2))
        labels = torch.randint(0, 2, (16,)).float()
        logits = sparse_model(dense, cat_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        opt_sparse.zero_grad()
        loss.backward()
        opt_sparse.step()
        rec_sparse.step(step, loss=float(loss.item()))
    print("  Sparse embedding training + Recorder: OK")
except Exception as e:
    sparse_error = e
    print(f"  Sparse embedding + Recorder ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

rec_sparse.detach()

# ─────────────────────────────────────────────────────────────────────────────
# 8. build_report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 8 — build_report")
print("=" * 70)

try:
    report = build_report(run_dir)
    print(f"  build_report OK: {type(report).__name__}")
except Exception as e:
    print(f"  build_report FAILED: {type(e).__name__}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Bare DLRM (stock names) — two_tower recipe match: {len(matched_w)} weight, {len(matched_o)} output")
print(f"  DLRMTwoTower (renamed) — two_tower recipe match: {len(matched_w_tt)} weight, {len(matched_o_tt)} output")
embed_total = len(embed_mods)
print(f"  Embedding tables covered by WEIGHT pattern: {len(embed_captured_w)}/{embed_total}")
print(f"  Total JSONL entries emitted: {len(entries)}")

print(f"\nScript: {pathlib.Path(__file__).resolve()}")
