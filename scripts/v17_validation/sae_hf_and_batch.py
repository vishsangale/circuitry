#!/usr/bin/env python3
"""v1.7 eval — HF backend (resolver footgun + losslessness) and batch>1 SAE runs.

Part 1: SAE node attribution via the HuggingFace backend on real GPT-2.
  - Does HFSiteResolver.from_config(gpt2_config) silently build a BROKEN
    (Llama-pathed) resolver, and does resolution then fail loudly or silently?
  - With correct GPT-2 overrides: splice losslessness + node attribution +
    reconstruction quality on raw HF activations (vs TL-processed).
Part 2: batch>1 (TL backend). We only ever tested batch=1. Does the
  whole-tensor splice + attribution handle a real [batch, seq, d] input?

Run from repo root:
    .venv/bin/python scripts/v17_validation/sae_hf_and_batch.py
"""

from __future__ import annotations

import traceback
import warnings

import torch

warnings.filterwarnings("ignore")

from sae_lens import SAE  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402

from circuitry.patching.sae_features import SAEFeatureRunner  # noqa: E402
from circuitry.patching.sites import HFSiteResolver, Site  # noqa: E402

DEV = "cpu"


def load_sae(release, sae_id):
    res = SAE.from_pretrained(release, sae_id, device=DEV)
    return res[0] if isinstance(res, tuple) else res


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def main():
    print("loading HF gpt2...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    sae = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre")  # resid_post@7

    clean = tok("When John and Mary went to the store, John gave a drink to", return_tensors="pt")
    corrupt = tok("When John and Mary went to the store, Mary gave a drink to", return_tensors="pt")
    mary = tok.encode(" Mary")[0]
    john = tok.encode(" John")[0]

    def metric(out):
        logits = out.logits if hasattr(out, "logits") else out
        return logits[0, -1, mary] - logits[0, -1, john]

    site = Site("resid_post", layer=7)

    # ---------------- Part 1a: from_config footgun ----------------
    try:
        hr("1a. HFSiteResolver.from_config(gpt2_config) — does it build a broken resolver?")
        bad = HFSiteResolver.from_config(model.config)
        print(f"  from_config -> layer_pattern={bad.layer_pattern!r}  "
              f"attn_module_full={bad.attn_module_full!r}  mlp_module={bad.mlp_module!r}")
        try:
            runner = SAEFeatureRunner(model=model, sae_sites={site: sae}, resolver=bad)
            res = runner.run(clean, corrupt, metric, max_features=3)
            print(f"  !! resolution SUCCEEDED unexpectedly ({len(res.scores)} feats) — "
                  "did it resolve the WRONG module silently?")
        except Exception as e:
            msg = str(e)
            clear = any(s in msg.lower() for s in
                        ("model.layers", "transformer.h", "no module", "not found",
                         "has no", "attribute", "getattr", "submodule"))
            print(f"  resolution FAILED with {type(e).__name__}: {msg[:160]}")
            print(f"  -> error is {'CLEAR (names a module path)' if clear else 'OPAQUE'} "
                  "for a user who passed a GPT-2 config")
    except Exception:
        print("  SECTION 1a FAILED:"); traceback.print_exc()

    # ---------------- Part 1b: correct GPT-2 overrides ----------------
    try:
        hr("1b. HF backend with correct GPT-2 overrides")
        resolver = HFSiteResolver(
            n_heads=model.config.num_attention_heads,
            d_model=model.config.hidden_size,
            layer_pattern="transformer.h.{L}",
            attn_module="attn.c_proj",
            attn_module_full="attn",
            mlp_module="mlp",
        )
        # splice losslessness on raw HF activation at transformer.h.7 output
        cache = {}
        h = model.transformer.h[7].register_forward_hook(
            lambda m, i, o, _c=cache: _c.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        base = model(**clean).logits
        h.remove()
        x = cache["x"]
        with torch.no_grad():
            flat = x.reshape(-1, x.shape[-1]).float()
            fvu = ((flat - sae.decode(sae.encode(flat))).pow(2).sum()
                   / (flat - flat.mean(0)).pow(2).sum()).item()
        print(f"  raw-HF-activation reconstruction FVU={fvu:.4f} "
              "(res-jb was trained on TL-PROCESSED acts; higher FVU expected)")

        runner = SAEFeatureRunner(model=model, sae_sites={site: sae}, resolver=resolver)
        res = runner.run(clean, corrupt, metric, max_features=5)
        top = sorted(res.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        print(f"  node attribution OK: {len(res.scores)} feats; component={ {a.node.component for a in res.scores} }")
        for a, s in top:
            print(f"      feat={a.node.neuron:>6} score={s:+.4f}")
    except Exception:
        print("  SECTION 1b FAILED:"); traceback.print_exc()

    # ---------------- Part 2: batch>1 ----------------
    try:
        hr("2. batch>1 — node attribution on a real [batch, seq, d] input (TL backend)")
        from transformer_lens import HookedTransformer
        tl = HookedTransformer.from_pretrained("gpt2", device=DEV)
        tl.eval()
        sae_tl = sae  # same res-jb
        site7 = Site("resid_post", 7)
        clean_prompts = [
            "When John and Mary went to the store, John gave a drink to",
            "When Sarah and Paul went to the park, Paul handed a ball to",
            "When Anna and Mark left the school, Mark told a joke to",
        ]
        corrupt_prompts = [p.replace("John gave", "Mary gave").replace("Paul handed", "Sarah handed")
                            .replace("Mark told", "Anna told") for p in clean_prompts]
        ct = tl.to_tokens(clean_prompts)
        rt = tl.to_tokens(corrupt_prompts)
        print(f"  batched clean tokens shape={tuple(ct.shape)} (batch={ct.shape[0]})")
        m_a, m_b = tl.to_single_token(" Mary"), tl.to_single_token(" John")

        def bmetric(logits):
            # batch-aware: mean over batch of last-position logit diff
            return (logits[:, -1, m_a] - logits[:, -1, m_b]).mean()

        from circuitry.patching.sae_edges import SAEFeatureEdgeRunner
        from circuitry.patching.sites import TLSiteResolver
        rsv = TLSiteResolver()
        runner = SAEFeatureRunner(model=tl, sae_sites={site7: sae_tl}, resolver=rsv)
        res = runner.run(ct, rt, bmetric, max_features=6)
        finite = all(torch.isfinite(torch.tensor(s)) for s in res.scores.values())
        print(f"  batch>1 node attribution: {len(res.scores)} feats; all-finite={finite}")
        for a, s in sorted(res.scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]:
            print(f"      feat={a.node.neuron:>6} score={s:+.4f}")

        er = SAEFeatureEdgeRunner(model=tl,
                                  sae_sites={Site("resid_post", 6): load_sae('gpt2-small-res-jb','blocks.7.hook_resid_pre'),
                                             site7: sae_tl},
                                  resolver=rsv)
        circ = er.run(ct, rt, bmetric, layer_pairs="adjacent", top_k_survivors=8)
        ef = all(torch.isfinite(torch.tensor(v)) for v in dict(circ.edges).values())
        print(f"  batch>1 edges: {len(circ.edges)}; all-finite={ef}")
        print("  >>> batch>1 handled without crash." if finite and ef
              else "  >>> batch>1 produced non-finite values — investigate.")
    except Exception:
        print("  SECTION 2 FAILED:"); traceback.print_exc()


if __name__ == "__main__":
    main()
