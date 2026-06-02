#!/usr/bin/env python3
"""v1.7 eval — SAE node attribution on REAL GPT-2 + REAL pretrained SAEs.

TransformerLens backend on CPU (the activations the sae_lens SAEs were trained
on; avoids the torch-2.12 MPS silent-incorrectness warning). Validates, on a
real IOI task, for each of the three v1.7 SAE site types:

  1. Splice losslessness  — patch in decode(encode(x))+eps; logits must be
     unchanged to ~machine precision (tests the v1.7 routed splice + that the
     SAE's encode/decode is side-effect-free / not order-dependent).
  2. SAE reconstruction quality (FVU) — informational: is attribution meaningful?
  3. Node attribution (SAEFeatureRunner) — runs, returns sparse non-trivial
     scores, AtPNode.component set correctly per site type.

Site map (sae_lens id -> circuitry Site):
  gpt2-small-res-jb        blocks.8.hook_resid_pre  == TL blocks.7.hook_resid_post  -> Site('resid_post', 7)
  gpt2-small-mlp-out-v5-32k blocks.8.hook_mlp_out                                    -> Site('mlp_out', 8)
  gpt2-small-attn-out-v5-32k blocks.8.hook_attn_out                                  -> Site('attn_out', 8)

Run from repo root:
    .venv/bin/python scripts/v17_validation/sae_nodes_real.py
"""

from __future__ import annotations

import warnings

import torch

warnings.filterwarnings("ignore")
torch.set_grad_enabled(True)

from sae_lens import SAE  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from circuitry.patching.sae_features import SAEFeatureRunner  # noqa: E402
from circuitry.patching.sites import Site, TLSiteResolver  # noqa: E402

DEV = "cpu"


def load_sae(release: str, sae_id: str):
    res = SAE.from_pretrained(release, sae_id, device=DEV)
    return res[0] if isinstance(res, tuple) else res


def main() -> None:
    print("loading TL gpt2 (cpu, default processing)...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2", device=DEV)
    model.eval()

    # ---- IOI task ----
    clean = "When John and Mary went to the store, John gave a drink to"
    corrupt = "When John and Mary went to the store, Mary gave a drink to"
    clean_tok = model.to_tokens(clean)
    corrupt_tok = model.to_tokens(corrupt)
    mary = model.to_single_token(" Mary")
    john = model.to_single_token(" John")

    def metric(logits):
        # logits: [batch, seq, vocab]; IOI logit diff at last position
        return logits[0, -1, mary] - logits[0, -1, john]

    base_logits = model(clean_tok)
    print(f"baseline clean logit-diff (Mary-John) = {metric(base_logits).item():+.4f}")
    print(f"clean tokens={clean_tok.shape} corrupt tokens={corrupt_tok.shape}\n")

    resolver = TLSiteResolver()

    sites = [
        ("resid_post@7 (res-jb)", Site("resid_post", layer=7),
         "gpt2-small-res-jb", "blocks.8.hook_resid_pre",
         "blocks.7.hook_resid_post"),
        ("mlp_out@8 (mlp-v5-32k)", Site("mlp_out", layer=8),
         "gpt2-small-mlp-out-v5-32k", "blocks.8.hook_mlp_out",
         "blocks.8.hook_mlp_out"),
        ("attn_out@8 (attn-v5-32k)", Site("attn_out", layer=8),
         "gpt2-small-attn-out-v5-32k", "blocks.8.hook_attn_out",
         "blocks.8.hook_attn_out"),
    ]

    rows = []
    for label, site, release, sae_id, tl_hook in sites:
        print("=" * 78)
        print(f"SITE: {label}   ->  TL hook {tl_hook}")
        print("=" * 78)
        sae = load_sae(release, sae_id)
        norm_mode = getattr(sae.cfg, "normalize_activations", "?")
        d_sae = getattr(sae.cfg, "d_sae", "?")
        print(f"  SAE d_sae={d_sae} normalize_activations={norm_mode!r}")

        # ---- capture the real activation at this hook on the clean forward ----
        cache = {}

        def grab(t, hook, _c=cache):
            _c["x"] = t.detach().clone()
            return t

        model.run_with_hooks(clean_tok, fwd_hooks=[(tl_hook, grab)])
        x = cache["x"]  # [batch, seq, d_model]

        # ---- SAE reconstruction quality (FVU) on the real activation ----
        with torch.no_grad():
            flat = x.reshape(-1, x.shape[-1]).float()
            f = sae.encode(flat)
            x_hat = sae.decode(f)
            resid = flat - x_hat
            fvu = (resid.pow(2).sum() / (flat - flat.mean(0)).pow(2).sum()).item()
            l0 = (f != 0).float().sum(-1).mean().item()
        print(f"  reconstruction: FVU={fvu:.4f}  mean-L0={l0:.1f}  (d_sae={d_sae})")

        # ---- splice losslessness: replace activation with decode(encode(x))+eps ----
        def splice(t, hook, _sae=sae):
            orig_dtype = t.dtype
            a = t.detach().reshape(-1, t.shape[-1]).float()
            fh = _sae.encode(a)
            xh = _sae.decode(fh)
            eps = (a - xh).detach()
            recon = (xh + eps).reshape(t.shape).to(orig_dtype)
            return recon

        spliced_logits = model.run_with_hooks(clean_tok, fwd_hooks=[(tl_hook, splice)])
        delta = (spliced_logits - base_logits).abs().max().item()
        metric_delta = abs(metric(spliced_logits).item() - metric(base_logits).item())
        lossless = "PASS" if delta < 1e-3 else "FAIL"
        print(f"  splice losslessness: max|Δlogit|={delta:.2e}  Δmetric={metric_delta:.2e}  [{lossless}]")

        # ---- node attribution ----
        runner = SAEFeatureRunner(model=model, sae_sites={site: sae}, resolver=resolver)
        result = runner.run(clean_tok, corrupt_tok, metric, max_features=8)
        scored = sorted(result.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        print(f"  node attribution: {len(result.scores)} features scored; top 5:")
        comps = set()
        for atp, sc in scored[:5]:
            nd = atp.node
            comps.add(nd.component)
            print(f"      layer={nd.layer} component={nd.component!r} feat={nd.neuron:>6} score={sc:+.4f}")
        print(f"  AtPNode.component values seen: {comps}\n")

        rows.append((label, fvu, l0, delta, lossless, len(result.scores), comps))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'site':28} {'FVU':>7} {'L0':>6} {'max|Δlogit|':>12} {'lossless':>9} {'#feat':>6}")
    for label, fvu, l0, delta, lossless, nfeat, comps in rows:
        print(f"{label:28} {fvu:7.4f} {l0:6.1f} {delta:12.2e} {lossless:>9} {nfeat:6d}  comps={comps}")


if __name__ == "__main__":
    main()
