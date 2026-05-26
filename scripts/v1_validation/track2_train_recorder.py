"""Track 2 — Recorder live during REAL training (lr>0).

Every prior validation captured at lr=0 (static, Delta column always "—"). This
trains a tiny from-scratch Llama (eager attn) on TinyStories and verifies the
Recorder's diagnostics actually MOVE over steps, the eager-only diagnostics
(induction_score, attention_pattern_entropy) emit, and the §10 wall-clock budget
(<=10% overhead at the recorder's emit cadence) holds on a real training loop.

Run:  venv/bin/python scripts/v1_validation/track2_train_recorder.py
Saves: scripts/v1_validation/track2_train_recorder.results.json + runs/v1_track2/
"""

from __future__ import annotations

import json
import os
import time

import torch

from circuitry import Recorder, build_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
STEPS = 600
BATCH = 16
SEQLEN = 128
LR = 3e-4
EVERY_N = 100  # emit cadence for the recorded run
RUN_ROOT = os.path.join("runs", "v1_track2")
DATA_CACHE = os.path.join(RUN_ROOT, "tinystories_tokens.pt")


def build_model(vocab: int):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=SEQLEN, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg)


def load_tokens(tokenizer) -> torch.Tensor:
    """Tokenize ~enough TinyStories to fill STEPS*BATCH sequences of SEQLEN; cache."""
    if os.path.exists(DATA_CACHE):
        return torch.load(DATA_CACHE)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    n_needed = STEPS * BATCH * SEQLEN + SEQLEN
    ds = load_dataset("roneneldan/TinyStories", split="train[:20000]")
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for row in ds:
        ids.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        ids.append(eos)
        if len(ids) >= n_needed:
            break
    n_seq = len(ids) // SEQLEN
    toks = torch.tensor(ids[: n_seq * SEQLEN]).view(n_seq, SEQLEN)
    os.makedirs(RUN_ROOT, exist_ok=True)
    torch.save(toks, DATA_CACHE)
    return toks


def train(model, tokens, steps, recorder=None, log_loss=False):
    """One training run; returns (losses, wall_clock_seconds)."""
    torch.manual_seed(SEED)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_seq = tokens.shape[0]
    losses = []
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n_seq, (BATCH,))
        batch = tokens[idx].to(DEVICE)
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        loss.backward()
        if recorder is not None:
            recorder.step(step=step, loss=loss)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.detach().item())
        if log_loss and step % 100 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}", flush=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return losses, time.time() - t0


def parse_jsonl_movement(run_dir):
    """Read metrics.jsonl, return {tag: {first_step,last_step,first,last}} for a few diagnostics."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return {}, 0
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_tag: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        tag = r.get("tag") or r.get("name")
        step = r.get("step")
        val = r.get("value")
        if tag is None or step is None or not isinstance(val, (int, float)):
            continue
        by_tag.setdefault(tag, []).append((step, val))
    return by_tag, len(rows)


def main():
    os.makedirs(RUN_ROOT, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tokens = load_tokens(tok)
    print(f"device={DEVICE} tokens={tuple(tokens.shape)} steps={STEPS} "
          f"batch={BATCH} seqlen={SEQLEN} lr={LR}", flush=True)
    results = {"device": DEVICE, "steps": STEPS, "batch": BATCH, "seqlen": SEQLEN, "lr": LR}

    # ---- Run A: baseline, NO recorder (wall-clock reference) ----
    torch.manual_seed(SEED)
    model = build_model(tok.vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: tiny Llama, {n_params/1e6:.1f}M params (eager)", flush=True)
    print("\n[Run A] baseline training (no Recorder)...", flush=True)
    losses_a, wall_a = train(model, tokens, STEPS, recorder=None, log_loss=True)
    print(f"  loss {losses_a[0]:.3f} -> {losses_a[-1]:.3f}  in {wall_a:.1f}s", flush=True)

    # ---- Run B: same seed/steps WITH Recorder at default-ish cadence ----
    torch.manual_seed(SEED)
    model = build_model(tok.vocab_size).to(DEVICE)
    run_dir = os.path.join(RUN_ROOT, "recorded")
    rec = Recorder(model, run_dir=run_dir, recipe="llm", writer="jsonl",
                   every_n_steps=EVERY_N, strict=False)
    rec.attach()
    print(f"\n[Run B] recorded training (every_n_steps={EVERY_N})...", flush=True)
    losses_b, wall_b = train(model, tokens, STEPS, recorder=rec, log_loss=True)
    rec.detach()
    print(f"  loss {losses_b[0]:.3f} -> {losses_b[-1]:.3f}  in {wall_b:.1f}s", flush=True)

    overhead = (wall_b - wall_a) / wall_a
    print(f"\n=== §10 wall-clock budget: baseline {wall_a:.1f}s vs recorded {wall_b:.1f}s "
          f"=> overhead {overhead:+.1%} (target <=10%) ===", flush=True)
    results.update({"n_params": n_params, "loss_start": losses_b[0], "loss_end": losses_b[-1],
                    "wall_baseline": round(wall_a, 2), "wall_recorded": round(wall_b, 2),
                    "overhead_frac": round(overhead, 4)})

    # ---- Diagnostic movement over steps ----
    by_tag, n_rows = parse_jsonl_movement(run_dir)
    print(f"\n=== Diagnostics emitted: {len(by_tag)} tags, {n_rows} rows over training ===", flush=True)
    movement = {}
    eager_only = [t for t in by_tag if "induction" in t or "entropy" in t]
    interesting = [t for t in by_tag if any(k in t for k in
                   ("logit_lens_kl", "effective_rank", "grad", "induction", "entropy"))]
    for tag in sorted(set(interesting))[:18]:
        pts = sorted(by_tag[tag])
        if len(pts) >= 2 and pts[0][1] != pts[-1][1]:
            movement[tag] = {"first_step": pts[0][0], "first": round(pts[0][1], 4),
                             "last_step": pts[-1][0], "last": round(pts[-1][1], 4)}
            print(f"  {tag:48s} {pts[0][1]:+.4f} (s{pts[0][0]}) -> {pts[-1][1]:+.4f} (s{pts[-1][0]})", flush=True)
    results["eager_only_tags_emitted"] = sorted(eager_only)[:10]
    results["n_diagnostic_tags"] = len(by_tag)
    results["moving_tags_sample"] = movement
    print(f"\n  eager-only diagnostics emitted ({len(eager_only)}): {sorted(eager_only)[:6]}", flush=True)

    # ---- Report ----
    report_path = build_report(run_dir)
    head = "".join(open(report_path).readlines()[:3])
    print(f"\nreport -> {report_path}\n  header: {head.strip()}", flush=True)
    results["report_path"] = str(report_path)
    results["run_dir"] = run_dir

    out = os.path.join(os.path.dirname(__file__), "track2_train_recorder.results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
