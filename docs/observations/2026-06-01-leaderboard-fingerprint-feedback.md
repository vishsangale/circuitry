# Circuitry feedback — static fingerprint of 67 custom 1M-param LMs

**Tested against:** circuitry `1.8.0` @ commit `8f237ee` (single-sourced version).
**Context:** Used circuitry's *retrospective* weight diagnostics to fingerprint ~70 trained
TinyStories architecture variants (`apps/leaderboard-challenge/`) — all custom `nn.Module`
archs (vanilla Llama, MQA/GQA, sigmoid-attention, GLA/gated-linear-attention, RWKV-style
channel-mix, MoE), each ≤1M non-embedding params, one flat `vNN_tinystories.pt` per variant.
Drove a `Recorder` per checkpoint (single snapshot) with the `llm` recipe restricted to static
weight diagnostics, writer=`jsonl`. 67/68 scanned cleanly.

The static weight-diagnostic path was solid and did the job. The items below are the friction
points worth fixing; severity is from the perspective of "applying circuitry to non-HF-standard
models post-hoc."

---

## What worked well (keep)
- `effective_rank` / `stable_rank` / `condition_number` / `heavy_tail_alpha` produced clean,
  sensible per-matrix results across 67 wildly different custom architectures with zero
  per-arch code.
- `writer="jsonl"` + `build_report` were drop-in; the shared per-step SVD cache (`_sv` in
  `_run_diagnostics`) is a nice touch.
- MoE batched-expert weight handling (3-D expert params → per-expert diagnostics) worked
  out of the box on the MoE variants.
- `enabled`-gating is correctly honored at runtime for weight/activation/gradient diagnostics
  (`_enabled()` checked in all three loops) — see #4, the only issue there is discoverability.

---

## 1. [usability, HIGH] `attention_head_rank` silently no-ops unless head metadata is at `model.config`

`Recorder.attach()` resolves head metadata only from `getattr(self.model, "config", None)`
(and `config.text_config`). For two very common shapes this finds nothing:

- **HF-wrapped models** that hold `self.model = LlamaForCausalLM(cfg)` — the config lives at
  `model.model.config`, not `model.config`.
- **Custom architectures** that store `num_heads` / `num_kv_heads` / `head_dim` as attributes
  on the attention *submodules* (no HF-style config object at all).

Result: across **all 68** of our variants, `attention_head_rank` produced **zero** output on the
first pass, with no error and no column — we only noticed because the metric was simply missing
from the aggregate. There *is* a `logger.warning(... "no usable config ... skipping")`, but it
fires at the first **emit step**, not at `attach()`, and is easy to miss under a normal log level.

**Workaround we used:** a resolver that (a) lifts `model.model.config`/any submodule `.config`
exposing `num_attention_heads` to the top level, or (b) synthesizes a `SimpleNamespace` config
from custom submodule attrs, validating `q_rows == n_heads*head_dim` and `k_rows == n_kv_heads*head_dim`
against the state_dict before trusting it. With that shim, head-rank fired on 66/68.

**Suggested fixes (any subset):**
- Walk submodules for a `.config` exposing `num_attention_heads` before giving up.
- Accept **explicit** head metadata — `n_heads` / `n_kv_heads` / `head_dim` — via a `Recorder` /
  `scan_run` kwarg or a recipe field, so config-less custom models are first-class.
- Move the warning to `attach()` (fail-fast when the diagnostic is requested but unresolved),
  and include what was searched so the user knows where to put the metadata.

---

## 2. [packaging, MEDIUM] `sae-lens` is a hard dependency but core diagnostics don't need it

`pyproject.toml` lists `sae-lens>=4.0.0` (and `tensorboard>=2.14`) under top-level
`dependencies`. But `sae_lens` is only **lazy-imported** (`sae/grad.py`, `sae/loader.py`,
`recipes/__init__.py`, `patching/__init__.py`) — we installed with `pip install --no-deps` and
`import circuitry`, `get_recipe("llm")`, `Recorder`, and `scan_run` all worked fine without
sae-lens present. The hard dep drags the heavy `sae-lens` → `transformer-lens` → … tree onto
every user who only wants weight/activation rank diagnostics.

**Suggested fix:** move `sae-lens` to an extra — `circuitry[sae]` — and have the lazy import sites
raise a friendly `"install circuitry[sae] to use SAE features"` on `ImportError`. Consider the
same for `tensorboard` (`circuitry[tensorboard]`) with the `jsonl` writer as the no-dep default,
so a lean core install is `pip install circuitry`. (The `.only()` docstring already frames SAEs as
opt-in functionality — make them opt-in at install time too.)

---

## 3. [API, MEDIUM] `scan_run` only discovers `<run_dir>/checkpoints/step*.pt`

`scan_run` → `_discover_checkpoints` globs `(<run_dir>/"checkpoints").glob("step*.pt")` and
parses `stepNNN` from the name. Retrospective scanning of checkpoints that are **single files** or
use **arbitrary names** (our case: one flat `vNN_tinystories.pt` per model, no step series) isn't
supported, so we bypassed `scan_run` entirely and drove the `Recorder` loop by hand
(`attach()` → `load_state_dict` → `step(0)` → `detach()`).

**Suggested fix:** let `scan_run` accept either an explicit `checkpoints: list[(step, path)]` /
list of paths, or a glob/single-path argument, keeping the `step*.pt` convention as the default.
A first-class "scan one checkpoint" path would also help the single-snapshot use case (#5).

---

## 4. [DX, LOW] `recipe.only(...)` / `.disable(...)` effect is invisible on the dataclass lists

`.only([...])` works correctly — it sets `enabled[name]=False` for the complement, and the run
loops honor `_enabled()`. But it does **not** modify `weight_diagnostics` /
`activation_diagnostics` / `gradient_diagnostics`, so after `r2 = r.only([...])`,
`r2.weight_diagnostics` still shows the *full* list. We inspected exactly that to "verify"
`.only()` and wrongly concluded it was a no-op, then worked around it with `dataclasses.replace`
on the lists (unnecessary). A reasonable user will hit the same footgun.

**Suggested fix:** add an `effective_diagnostics()` / `active_diagnostics` accessor (lists minus
disabled) and/or reflect enabled-state in `Recipe.__repr__`; note in the `.only()`/`.disable()`
docstrings that they toggle `enabled`, not the declared lists.

---

## 5. [docs, LOW] Trajectory weight diagnostics are meaningless on a single-snapshot scan

`update_delta`, `rank_trajectory`, `direction_cosine` (v1.3 training-dynamics) need ≥2 emitted
steps. On a one-checkpoint retrospective scan they have no prior snapshot to diff against and emit
nothing/zero. Not a bug, but a sharp edge for the post-hoc-on-a-checkpoint use case — we had to
know to exclude them and keep only the static set (`effective_rank`, `stable_rank`,
`condition_number`, `heavy_tail_alpha`, `attention_head_rank`, `sv_histogram`).

**Suggested fix:** document a "static vs trajectory" split for retrospective scans, and emit a
one-time warning when a trajectory diagnostic runs with no prior snapshot.

---

## 6. [discoverability, LOW] `sv_histogram` emits artifacts, not scalars

`sv_histogram` writes histogram artifacts rather than scalar tags, so it's invisible to scalar /
CSV consumers (we got no `sv_histogram` columns and had to know to look elsewhere for the spectra).

**Suggested fix:** document where the histogram lands, and/or emit a couple of companion summary
scalars (e.g. spectral entropy, σ_max/σ_min) alongside it so it shows up in tabular exports.

---

## 7. [API, MEDIUM] The default `llm` recipe's MoE-only HookPoints make `strict` attach hard-fail on every dense model

`recipes/llm.py` now carries MoE-only weight HookPoints (e.g. `.*\.mlp\.gate$`, expert
patterns). On any **dense** (non-MoE) model these match 0 modules, and
`Recorder.attach()` with the default `strict=True` **raises** rather than warning:

```
RuntimeError: HookPoint 2 (.*\.mlp\.gate$) matched 0 modules — refusing to attach
(pass strict=False to skip unmatched HookPoints with a warning)
```

So live capture on a plain Llama/GLA model fails out-of-the-box with the stock recipe
unless the caller knows to pass `strict=False`. We hit this attaching to dense 1M-param
leaderboard models and had to set `strict=False` in our harness. The source already has a
`TODO(v1.8 follow-up, F37)` acknowledging the companion warning "fires on EVERY non-MoE
attach" — same root cause.

**Suggested fix (any of):** make the MoE HookPoints an opt-in recipe variant
(`llm_moe`) or gate them behind a recipe flag; OR have `strict` treat "a HookPoint that
*structurally* can't match this model" (MoE patterns on a dense model) as a soft warning
while still erroring on genuinely-misconfigured patterns; OR at minimum document that
dense-model capture with the stock `llm` recipe needs `strict=False`. As-is, the safe
default (`strict=True`) is unusable with the default recipe on the most common model type.

## Already fixed (no action)
- Version mismatch: `pyproject` previously hard-coded `version = "1.4.2"` while
  `__version__ == "1.8.0"`; current HEAD single-sources it via
  `version = { attr = "circuitry.__version__" }` (commit `8f237ee`). Resolved.
