"""
Validation script — PART 1-4 (scan / writers / compare / CLI)
=============================================================
PART 1 : scan_run on real multi-checkpoint training
PART 2 : all three writers (JsonlWriter, TensorBoardWriter, NullWriter)
PART 3 : compare_runs / build_compare_report
PART 4 : CLI exercise

Run with:
  /Users/vishsangale/workspace/circuitry/.venv/bin/python \
      scripts/v17_validation/part4_scan_writers_compare_cli.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Tiny model (same shape as examples/tiny_llm.py)
# ---------------------------------------------------------------------------

class TinyAttn(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, k, nn.Linear(d, d, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.v_proj(x))  # type: ignore[return-value]


class TinyMlp(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj   = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))  # type: ignore[return-value]


class TinyBlock(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.attn  = TinyAttn(d)
        self.mlp   = TinyMlp(d)
        self.ln_1  = nn.LayerNorm(d)
        self.ln_2  = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln_1(x))
        x = self.mlp(self.ln_2(x))
        return x


class _TinyBody(nn.Module):
    """Inner model with .layers list — named 'model' in TinyLM so modules are
    model.layers.0, model.layers.1, matching the llm recipe's .*\\.layers\\.\\d+$ hook."""

    def __init__(self, d: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TinyLM(nn.Module):
    """TinyLM with HF-style self.model.layers so .*\\.layers\\.\\d+$ hook matches."""

    def __init__(self, vocab: int = 64, d: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.embed   = nn.Embedding(vocab, d)
        self.model   = _TinyBody(d, n_layers)   # -> model.layers.0, model.layers.1
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.model(x)
        return self.lm_head(x)


VOCAB, D = 64, 32


def make_model() -> TinyLM:
    return TinyLM(vocab=VOCAB, d=D, n_layers=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTHON = str(REPO / ".venv" / "bin" / "python")

FINDINGS: list[str] = []


def record(severity: str, area: str, finding: str, evidence: str, action: str) -> None:
    line = f"{severity} | {area} | {finding} | {evidence} | {action}"
    FINDINGS.append(line)
    print(f"  [{severity}] {area}: {finding}")


def train_run(
    run_dir: pathlib.Path,
    seed: int = 0,
    lr: float = 1e-2,
    n_steps: int = 150,
    ckpt_every: int = 30,
) -> None:
    """Train a TinyLM for n_steps, saving checkpoints every ckpt_every steps."""
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, n_steps + 1):
        tokens = torch.randint(0, VOCAB, (8, 16))
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, VOCAB), tokens.view(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % ckpt_every == 0:
            ckpt_path = ckpt_dir / f"step{step:09d}.pt"
            torch.save(model.state_dict(), ckpt_path)

    print(f"  Trained {n_steps} steps, saved {len(list(ckpt_dir.glob('*.pt')))} checkpoints to {ckpt_dir}")


# ===========================================================================
# PART 1 — scan_run on real multi-checkpoint training
# ===========================================================================

def part1_scan() -> pathlib.Path:
    print("\n=== PART 1: scan_run on real multi-checkpoint training ===")
    from circuitry.recorder.scan import scan_run
    from circuitry.recorder.report import build_report

    run_dir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_scan_"))
    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_scan_out_"))

    train_run(run_dir, seed=42, lr=5e-3, n_steps=150, ckpt_every=30)

    ckpt_files = sorted((run_dir / "checkpoints").glob("step*.pt"))
    print(f"  Checkpoints: {[p.name for p in ckpt_files]}")

    t0 = time.perf_counter()
    scan_run(
        run_dir=run_dir,
        recipe="llm",
        out_dir=out_dir,
        model_factory=make_model,
        writer="jsonl",
    )
    elapsed = time.perf_counter() - t0
    print(f"  scan_run completed in {elapsed:.2f}s")

    # Verify metrics.jsonl was written
    metrics_path = out_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"metrics.jsonl not found at {metrics_path}"

    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    scalars = [r for r in rows if r.get("kind") == "scalar"]
    tags_seen = {r["tag"] for r in scalars}
    steps_seen = sorted({r["step"] for r in scalars})

    print(f"  Scalar tags: {len(tags_seen)}, steps: {steps_seen}")

    # Check cross-step weight primitives
    update_delta_tags = [t for t in tags_seen if "update_delta" in t]
    direction_cosine_tags = [t for t in tags_seen if "direction_cosine" in t]
    rank_trajectory_tags  = [t for t in tags_seen if "rank_trajectory" in t]

    # update_delta: must appear from step 60 onward (not step 30, which has no prior)
    ud_steps_by_tag: dict[str, list] = {}
    for r in scalars:
        if "update_delta" in r["tag"]:
            ud_steps_by_tag.setdefault(r["tag"], []).append((r["step"], r["value"]))

    dc_steps_by_tag: dict[str, list] = {}
    for r in scalars:
        if "direction_cosine" in r["tag"]:
            dc_steps_by_tag.setdefault(r["tag"], []).append((r["step"], r["value"]))

    rt_steps_by_tag: dict[str, list] = {}
    for r in scalars:
        if "rank_trajectory" in r["tag"]:
            rt_steps_by_tag.setdefault(r["tag"], []).append((r["step"], r["value"]))

    print(f"  update_delta tags: {len(update_delta_tags)}, direction_cosine: {len(direction_cosine_tags)}, rank_trajectory: {len(rank_trajectory_tags)}")

    # Verify non-zero update_delta (if weights changed, delta must be > 0)
    if ud_steps_by_tag:
        one_tag = next(iter(ud_steps_by_tag))
        values = [v for _, v in ud_steps_by_tag[one_tag]]
        all_zero = all(abs(v) < 1e-12 for v in values)
        min_step = min(s for s, _ in ud_steps_by_tag[one_tag])
        max_val  = max(values)
        min_val  = min(values)

        print(f"  update_delta ({one_tag}): steps_seen={[s for s,_ in ud_steps_by_tag[one_tag]]}, max={max_val:.6f}, min={min_val:.6f}")

        if all_zero:
            record("BUG", "scan/update_delta",
                   "update_delta is identically zero for all checkpoints — ALIASING suspected",
                   f"tag={one_tag}, all values={values}",
                   "Verify copy=True in _prev_weights snapshot; test on GPU run")
        else:
            record("GOOD", "scan/update_delta",
                   "update_delta is non-zero across checkpoints — no aliasing bug",
                   f"tag={one_tag}, max={max_val:.4g}, min={min_val:.4g}, first_step={min_step}",
                   "No action needed")

        # Check step coverage: update_delta must NOT appear at first checkpoint (step 30)
        first_ckpt_step = steps_seen[0]
        ud_at_first = [v for s, v in ud_steps_by_tag[one_tag] if s == first_ckpt_step]
        if ud_at_first:
            record("GAP", "scan/update_delta",
                   f"update_delta emitted at first checkpoint step={first_ckpt_step} (no prior snapshot exists — value is meaningless)",
                   f"values at first step={ud_at_first}",
                   "Confirm _prev_weights guard: `if not self._prev_weights: continue`")
        else:
            record("GOOD", "scan/update_delta",
                   f"update_delta correctly skipped at first checkpoint (step={first_ckpt_step})",
                   f"first update_delta step={min_step}",
                   "No action needed")
    else:
        record("GAP", "scan/update_delta",
               "No update_delta tags found in scan output",
               f"tags_seen sample={list(tags_seen)[:5]}",
               "Check if llm recipe includes update_delta in weight_diagnostics")

    # Verify direction_cosine: needs two prior snapshots, so NOT at first two checkpoints
    if dc_steps_by_tag:
        one_dc = next(iter(dc_steps_by_tag))
        dc_vals = [v for _, v in dc_steps_by_tag[one_dc]]
        dc_steps = [s for s, _ in dc_steps_by_tag[one_dc]]
        print(f"  direction_cosine ({one_dc}): steps={dc_steps}, vals={[f'{v:.4f}' for v in dc_vals]}")

        if all(abs(v) < 1e-12 for v in dc_vals):
            record("BUG", "scan/direction_cosine",
                   "direction_cosine is identically zero — probable aliasing or same prev snapshot",
                   f"tag={one_dc}, values={dc_vals}",
                   "Check _prev_prev_weights copy=True semantics")
        else:
            record("GOOD", "scan/direction_cosine",
                   "direction_cosine is non-trivial across checkpoints",
                   f"tag={one_dc}, steps={dc_steps}, range=[{min(dc_vals):.4f}, {max(dc_vals):.4f}]",
                   "No action needed")
    else:
        # 5 checkpoints → direction_cosine should have appeared by step 90
        record("GAP", "scan/direction_cosine",
               "No direction_cosine tags found",
               f"total checkpoints={len(ckpt_files)}, steps_seen={steps_seen}",
               "Need at least 3 checkpoints (steps 30, 60, 90); check recipe")

    # rank_trajectory: present from step 60 (needs 1 prior)
    if rt_steps_by_tag:
        one_rt = next(iter(rt_steps_by_tag))
        rt_vals = [v for _, v in rt_steps_by_tag[one_rt]]
        rt_steps_list = [s for s, _ in rt_steps_by_tag[one_rt]]
        print(f"  rank_trajectory ({one_rt}): steps={rt_steps_list}, vals={[f'{v:.2f}' for v in rt_vals]}")
        all_identical = len(set(round(v, 3) for v in rt_vals)) == 1
        record("GOOD" if not all_identical else "GAP",
               "scan/rank_trajectory",
               "rank_trajectory across checkpoints" + (" — all values identical (static rank?)" if all_identical else " — evolving"),
               f"tag={one_rt}, steps={rt_steps_list}, vals={[round(v,3) for v in rt_vals]}",
               "No action needed" if not all_identical else "Verify training produces actual rank changes")
    else:
        record("GAP", "scan/rank_trajectory",
               "No rank_trajectory tags found",
               f"tags_seen count={len(tags_seen)}",
               "Check recipe weight_diagnostics contains rank_trajectory")

    # build_report
    report_path = build_report(out_dir)
    assert report_path.exists(), f"Report not written: {report_path}"
    report_txt = report_path.read_text()
    print(f"  Report: {report_path} ({len(report_txt)} chars)")

    record("GOOD", "scan/build_report",
           "build_report succeeded on real scan output",
           f"path={report_path}, size={len(report_txt)}B, tags={len(tags_seen)}",
           "No action needed")

    return out_dir


# ===========================================================================
# PART 2 — writers
# ===========================================================================

def part2_writers(scan_out_dir: pathlib.Path) -> None:
    print("\n=== PART 2: Writers ===")
    from circuitry.writers.jsonl import JsonlWriter
    from circuitry.writers.null import NullWriter
    from circuitry.writers.tensorboard import TensorBoardWriter

    # ---- JsonlWriter ----
    print("  -- JsonlWriter --")
    jdir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_jsonl_"))
    jw = JsonlWriter(jdir)

    N = 20
    for step in range(N):
        jw.add_scalar("test/loss", float(step) * 0.1, step)
        jw.add_scalar("test/accuracy", 1.0 - float(step) * 0.04, step)
    jw.add_histogram("test/hist", torch.randn(100), step=N - 1)
    jw.add_text("test/note", "hello world", step=0)
    jw.flush()
    jw.close()

    mpath = jdir / "metrics.jsonl"
    assert mpath.exists()
    lines = [l for l in mpath.read_text().splitlines() if l.strip()]
    scalar_rows = [json.loads(l) for l in lines if "scalar" in l]
    hist_rows   = [json.loads(l) for l in lines if "histogram" in l]
    text_rows   = [json.loads(l) for l in lines if "text" in l]

    print(f"    Scalar rows: {len(scalar_rows)}, histogram: {len(hist_rows)}, text: {len(text_rows)}")

    if len(scalar_rows) != N * 2:
        record("BUG", "JsonlWriter",
               f"Expected {N*2} scalar rows, got {len(scalar_rows)}",
               f"path={mpath}",
               "Check add_scalar flush/write")
    else:
        record("GOOD", "JsonlWriter",
               f"Wrote {len(scalar_rows)} scalar + {len(hist_rows)} histogram + {len(text_rows)} text rows cleanly",
               f"path={mpath}, size={mpath.stat().st_size}B",
               "No action needed")

    # Verify histogram artifact file
    if hist_rows:
        art_rel = hist_rows[0]["path"]
        art_abs = jdir / art_rel
        if art_abs.exists():
            record("GOOD", "JsonlWriter/histogram",
                   "Histogram artifact .npy file written",
                   f"path={art_abs}, size={art_abs.stat().st_size}B",
                   "No action needed")
        else:
            record("BUG", "JsonlWriter/histogram",
                   "Histogram artifact path in JSONL points to missing file",
                   f"expected={art_abs}",
                   "Fix artifact path construction in JsonlWriter.add_histogram")

    # ---- TensorBoardWriter (sync, default) ----
    print("  -- TensorBoardWriter (async_writes=False, the default) --")
    import inspect
    from circuitry.writers.tensorboard import TensorBoardWriter
    sig = inspect.signature(TensorBoardWriter.__init__)
    async_default = sig.parameters["async_writes"].default
    print(f"    async_writes default = {async_default!r}")

    if async_default is True:
        record("GAP", "TensorBoardWriter",
               "async_writes defaults to True — design.md §10 claims async-by-default (VERIFIED)",
               "async_writes default=True in __init__ signature",
               "Document clearly; warn users about flush() requirement before process exit")
    else:
        record("GAP", "TensorBoardWriter",
               "async_writes defaults to False (SYNC), contradicting design.md §10 claim of async-by-default",
               f"async_writes default={async_default!r} (tensorboard.py line 23)",
               "Either update design.md §10 to say sync-by-default, or flip the default to True")

    tbdir = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_tb_"))
    tbw = TensorBoardWriter(tbdir, async_writes=False)
    for step in range(10):
        tbw.add_scalar("tb_test/metric", float(step), step)
    tbw.flush()
    tbw.close()

    # Check event files
    event_files = list(tbdir.rglob("events.out.tfevents.*"))
    if not event_files:
        record("BUG", "TensorBoardWriter",
               "No tfevents files found after sync write",
               f"searched in {tbdir}",
               "Check SummaryWriter log_dir construction")
    else:
        ef = event_files[0]
        ef_size = ef.stat().st_size
        print(f"    TFEvents file: {ef.name}, size={ef_size}B")
        if ef_size < 50:
            record("BUG", "TensorBoardWriter",
                   f"TFEvents file suspiciously small ({ef_size}B) — likely empty",
                   f"path={ef}",
                   "Verify SummaryWriter.flush() is called before close()")
        else:
            # Try to parse with EventAccumulator
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                ea = EventAccumulator(str(tbdir))
                ea.Reload()
                scalar_tags = ea.Tags().get("scalars", [])
                print(f"    EventAccumulator scalar tags: {scalar_tags}")
                if scalar_tags:
                    events = ea.Scalars(scalar_tags[0])
                    record("GOOD", "TensorBoardWriter",
                           "TFEvents file readable by EventAccumulator; scalars verified",
                           f"path={ef}, size={ef_size}B, tag={scalar_tags[0]}, n_events={len(events)}",
                           "No action needed")
                else:
                    record("GAP", "TensorBoardWriter",
                           "TFEvents file written but EventAccumulator found no scalar tags",
                           f"path={ef}, size={ef_size}B, tags={ea.Tags()}",
                           "Check if SummaryWriter is flushing fully")
            except ImportError:
                record("GOOD", "TensorBoardWriter",
                       "TFEvents file written (tensorboard EventAccumulator not installed for verification)",
                       f"path={ef}, size={ef_size}B",
                       "Install tensorboard package for full verification; file exists and has content")

    # TensorBoardWriter async mode
    print("  -- TensorBoardWriter (async_writes=True) --")
    tbdir_async = pathlib.Path(tempfile.mkdtemp(prefix="circuitry_tb_async_"))
    tbw_async = TensorBoardWriter(tbdir_async, async_writes=True)
    for step in range(10):
        tbw_async.add_scalar("async_test/metric", float(step), step)
    tbw_async.flush()
    tbw_async.close()

    event_files_async = list(tbdir_async.rglob("events.out.tfevents.*"))
    if event_files_async:
        ef_async = event_files_async[0]
        record("GOOD", "TensorBoardWriter/async",
               "async_writes=True: TFEvents file written and non-empty after flush+close",
               f"path={ef_async}, size={ef_async.stat().st_size}B",
               "No action needed")
    else:
        record("BUG", "TensorBoardWriter/async",
               "async_writes=True: No TFEvents file found after flush+close",
               f"searched in {tbdir_async}",
               "Drain worker thread before calling writer.close()")

    # ---- NullWriter ----
    print("  -- NullWriter --")
    nw = NullWriter()
    try:
        nw.add_scalar("x", 1.0, 0)
        nw.add_histogram("h", torch.randn(10), 0)
        nw.add_image("i", torch.randn(3, 8, 8), 0)
        nw.add_text("t", "hello", 0)
        nw.flush()
        nw.close()
        record("GOOD", "NullWriter",
               "NullWriter no-ops cleanly for all add_* / flush / close",
               "No exception raised, no file written",
               "No action needed")
    except Exception as e:
        record("BUG", "NullWriter",
               f"NullWriter raised exception: {e}",
               str(e),
               "Fix null writer implementation")


# ===========================================================================
# PART 3 — compare_runs / build_compare_report
# ===========================================================================

def part3_compare() -> tuple[pathlib.Path, pathlib.Path]:
    print("\n=== PART 3: compare_runs / build_compare_report ===")
    from circuitry.recorder.compare import build_compare_report, compare_runs
    from circuitry.recorder.scan import scan_run

    run_a_src = pathlib.Path(tempfile.mkdtemp(prefix="cmp_run_a_src_"))
    run_b_src = pathlib.Path(tempfile.mkdtemp(prefix="cmp_run_b_src_"))
    out_a = pathlib.Path(tempfile.mkdtemp(prefix="cmp_run_a_out_"))
    out_b = pathlib.Path(tempfile.mkdtemp(prefix="cmp_run_b_out_"))

    train_run(run_a_src, seed=0,  lr=1e-2, n_steps=120, ckpt_every=30)
    train_run(run_b_src, seed=99, lr=1e-3, n_steps=120, ckpt_every=30)

    scan_run(run_dir=run_a_src, recipe="llm", out_dir=out_a, model_factory=make_model, writer="jsonl")
    scan_run(run_dir=run_b_src, recipe="llm", out_dir=out_b, model_factory=make_model, writer="jsonl")

    # Test compare_runs
    deltas = compare_runs(out_a, out_b)

    families = [fd.section for fd in deltas]
    print(f"  compare_runs: {len(deltas)} FamilyDelta entries")
    print(f"  Families: {families[:8]}{'...' if len(families)>8 else ''}")

    if not deltas:
        record("BUG", "compare_runs",
               "compare_runs returned empty list — no families found",
               f"run_a={out_a}, run_b={out_b}",
               "Verify scan_run produces metrics.jsonl in both out dirs")
    else:
        # Check for NaN deltas where both runs have real values
        both_valid = [fd for fd in deltas if not (fd.last_a != fd.last_a) and not (fd.last_b != fd.last_b)]
        nan_when_both_valid = [fd for fd in both_valid if fd.delta != fd.delta]
        if nan_when_both_valid:
            record("BUG", "compare_runs",
                   "FamilyDelta.delta is NaN despite both last_a and last_b being valid floats",
                   f"affected={[fd.section for fd in nan_when_both_valid[:3]]}",
                   "Check subtraction logic in compare.py")
        else:
            # Spot-check: different LR should produce different last values
            import math
            diffs = [abs(fd.delta) for fd in both_valid if not math.isnan(fd.delta)]
            max_diff = max(diffs) if diffs else 0.0
            agrees_count = sum(1 for fd in deltas if fd.trend_agrees)
            print(f"  Max |delta| across families: {max_diff:.4f}, trend_agrees: {agrees_count}/{len(deltas)}")
            record("GOOD", "compare_runs",
                   "compare_runs produces valid FamilyDelta with non-NaN deltas",
                   f"families={len(deltas)}, max_abs_delta={max_diff:.4g}, trend_agrees={agrees_count}/{len(deltas)}",
                   "No action needed")

    # build_compare_report
    compare_out = pathlib.Path(tempfile.mkdtemp(prefix="cmp_report_")) / "compare.md"
    report_path = build_compare_report(out_a, out_b, out_path=compare_out)
    assert report_path.exists()
    report_txt = report_path.read_text()
    print(f"  Compare report: {report_path} ({len(report_txt)} chars)")

    # Verify markdown table structure
    header_line = "| family/diagnostic | last_a | last_b | Δ (b−a) | trend_a | trend_b | agrees |"
    if header_line in report_txt:
        record("GOOD", "build_compare_report",
               "Markdown report has correct table header",
               f"path={report_path}, size={len(report_txt)}B",
               "No action needed")
    else:
        record("BUG", "build_compare_report",
               "Markdown report missing expected table header",
               f"path={report_path}, first 200 chars: {report_txt[:200]!r}",
               "Fix build_compare_report table construction")

    # Check missing-family sentinel (test with a new empty run)
    empty_dir = pathlib.Path(tempfile.mkdtemp(prefix="cmp_empty_"))
    try:
        compare_runs(out_a, empty_dir)
        record("BUG", "compare_runs",
               "compare_runs did not raise FileNotFoundError for missing metrics.jsonl",
               f"empty_dir={empty_dir}",
               "Check FileNotFoundError guard in compare.py")
    except FileNotFoundError as e:
        record("GOOD", "compare_runs/missing-file",
               "compare_runs raises FileNotFoundError when metrics.jsonl is absent",
               str(e),
               "No action needed")

    return out_a, out_b


# ===========================================================================
# PART 4 — CLI
# ===========================================================================

def part4_cli(scan_out_dir: pathlib.Path, run_a: pathlib.Path, run_b: pathlib.Path) -> None:
    print("\n=== PART 4: CLI ===")
    cli = str(REPO / ".venv" / "bin" / "circuitry")

    def run_cli(*args, expect_ok: bool = True) -> tuple[int, str, str]:
        result = subprocess.run(
            [cli] + list(args),
            capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr

    # --help
    rc, out, err = run_cli("--help")
    print(f"  circuitry --help: rc={rc}")
    if rc == 0 and "usage" in out.lower():
        record("GOOD", "CLI/--help",
               "--help prints usage text and exits 0",
               f"rc={rc}, output={out[:80]!r}",
               "No action needed")
    else:
        record("BUG", "CLI/--help",
               "--help failed or produced no usage",
               f"rc={rc}, stderr={err[:200]!r}",
               "Check CLI entry point")

    # --version
    rc, out, err = run_cli("--version")
    print(f"  circuitry --version: rc={rc}, out={out.strip()!r}")
    if rc == 0 and "circuitry" in out.lower():
        record("GOOD", "CLI/--version",
               "--version prints version string",
               f"output={out.strip()!r}",
               "No action needed")
    else:
        record("BUG", "CLI/--version",
               "--version failed",
               f"rc={rc}, stderr={err[:200]!r}",
               "Check __version__ import in CLI")

    # list-recipes
    rc, out, err = run_cli("list-recipes")
    print(f"  circuitry list-recipes: rc={rc}, out={out.strip()!r}")
    if rc == 0 and out.strip():
        record("GOOD", "CLI/list-recipes",
               "list-recipes prints registered recipe names",
               f"recipes={out.strip()!r}",
               "No action needed")
    else:
        record("BUG", "CLI/list-recipes",
               "list-recipes failed or empty",
               f"rc={rc}, stderr={err[:200]!r}",
               "Check recipes registry")

    # report subcommand
    report_out = pathlib.Path(tempfile.mkdtemp(prefix="cli_report_")) / "cli_report.md"
    rc, out, err = run_cli("report", "--run", str(scan_out_dir), "--out", str(report_out))
    print(f"  circuitry report: rc={rc}")
    if rc == 0 and report_out.exists():
        record("GOOD", "CLI/report",
               "report subcommand writes markdown report to --out path",
               f"rc={rc}, path={report_out}, size={report_out.stat().st_size}B",
               "No action needed")
    else:
        record("BUG", "CLI/report",
               "report subcommand failed or did not produce output",
               f"rc={rc}, stdout={out[:200]!r}, stderr={err[:200]!r}",
               "Investigate CLI report handler")

    # compare subcommand
    compare_out = pathlib.Path(tempfile.mkdtemp(prefix="cli_compare_")) / "cli_compare.md"
    rc, out, err = run_cli("compare", str(run_a), str(run_b), "--out", str(compare_out))
    print(f"  circuitry compare: rc={rc}")
    if rc == 0 and compare_out.exists():
        record("GOOD", "CLI/compare",
               "compare subcommand writes compare report",
               f"rc={rc}, path={compare_out}, size={compare_out.stat().st_size}B",
               "No action needed")
    else:
        record("BUG", "CLI/compare",
               "compare subcommand failed",
               f"rc={rc}, stdout={out[:200]!r}, stderr={err[:200]!r}",
               "Investigate CLI compare handler")

    # scan subcommand — expect rc=2 with helpful error (no model factory)
    rc, out, err = run_cli("scan", "--run", str(scan_out_dir), "--recipe", "llm", expect_ok=False)
    print(f"  circuitry scan: rc={rc}, stderr={err.strip()!r}")
    if rc == 2 and "model factory" in err.lower():
        record("ERGO", "CLI/scan",
               "scan subcommand exits 2 with clear message pointing to programmatic API",
               f"rc={rc}, msg={err.strip()!r}",
               "Expected placeholder behavior — add --model-factory dotted:path in future release to complete the workflow")
    elif rc == 0:
        record("BUG", "CLI/scan",
               "scan subcommand claimed success without a model factory — should not be possible",
               f"rc={rc}, stdout={out[:200]!r}",
               "Investigate scan CLI handler")
    else:
        record("GAP", "CLI/scan",
               f"scan subcommand failed with rc={rc} but different message than expected",
               f"rc={rc}, stderr={err[:200]!r}",
               "Check _cmd_scan error message")

    # compact flag on report
    compact_out = pathlib.Path(tempfile.mkdtemp(prefix="cli_compact_")) / "compact.md"
    rc, out, err = run_cli("report", "--run", str(scan_out_dir), "--out", str(compact_out), "--compact")
    if rc == 0 and compact_out.exists():
        compact_txt = compact_out.read_text()
        full_txt    = report_out.read_text() if report_out.exists() else ""
        if full_txt and len(compact_txt) < len(full_txt):
            record("GOOD", "CLI/report/compact",
                   "--compact produces smaller report (per-tag tables suppressed)",
                   f"compact={len(compact_txt)}B vs full={len(full_txt)}B",
                   "No action needed")
        else:
            record("GAP", "CLI/report/compact",
                   "--compact did not reduce report size",
                   f"compact={len(compact_txt)}B, full={len(full_txt)}B",
                   "Verify compact=True code path in build_report")
    else:
        record("BUG", "CLI/report/compact",
               "--compact flag caused CLI failure",
               f"rc={rc}, stderr={err[:200]!r}",
               "Investigate")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("circuitry v1.7 validation — Parts 1-4: scan / writers / compare / CLI")
    print("=======================================================================")

    scan_out = part1_scan()
    part2_writers(scan_out)
    run_a, run_b = part3_compare()
    part4_cli(scan_out, run_a, run_b)

    print("\n\n=== FINDINGS SUMMARY ===")
    print("severity | area | finding | evidence | suggested action")
    print("-" * 120)
    for f in FINDINGS:
        print(f)

    # Write findings to JSON for programmatic inspection
    out_json = pathlib.Path(__file__).parent / "part4_findings.json"
    out_json.write_text(json.dumps(FINDINGS, indent=2))
    print(f"\nFindings JSON: {out_json.absolute()}")


if __name__ == "__main__":
    main()
