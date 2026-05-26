from circuitry.patching.sites import HFSiteResolver


class _Cfg:
    """Minimal config stub: head_dim independent of hidden_size/num_attention_heads."""
    num_attention_heads = 8
    hidden_size = 64          # d_model/n_heads == 8
    head_dim = 16             # explicit, != 8  -> the Gemma-2 condition
    intermediate_size = 128
    num_key_value_heads = 8


def test_resolver_honors_explicit_head_dim():
    r = HFSiteResolver.from_config(_Cfg())
    assert r.head_dim == 16            # not 64 // 8 == 8


def test_resolver_falls_back_when_no_head_dim():
    class C:
        num_attention_heads = 4
        hidden_size = 64               # 64 // 4 == 16
    r = HFSiteResolver.from_config(C())
    assert r.head_dim == 16            # fallback d_model // n_heads


def test_explicit_head_dim_kwarg():
    r = HFSiteResolver(n_heads=8, d_model=64, head_dim=16)
    assert r.head_dim == 16


def _explicit_head_dim_model():
    """Tiny real Llama whose head_dim (16) != hidden_size/n_heads (64/8=8)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
        head_dim=16,
    )
    # Guard: the transformers version must actually honor explicit head_dim.
    assert cfg.head_dim == 16
    model = LlamaForCausalLM(cfg).eval()
    assert model.model.layers[0].self_attn.q_proj.out_features == 8 * 16  # 128, not 64
    return model, cfg


def test_eap_runner_uses_config_head_dim_and_runs():
    import torch

    from circuitry.core.patching import logit_diff_t
    from circuitry.patching import EAPRunner

    model, cfg = _explicit_head_dim_model()
    resolver = HFSiteResolver.from_config(cfg)
    runner = EAPRunner(model, resolver=resolver)
    assert runner.head_dim == 16   # was 8 before the fix

    torch.manual_seed(0)
    clean = {"input_ids": torch.randint(0, 64, (1, 6))}
    corrupt = {"input_ids": torch.randint(0, 64, (1, 6))}

    def metric(out):
        return logit_diff_t(out.logits if hasattr(out, "logits") else out, 1, 2)

    res = runner.run(clean, corrupt, metric)   # must not raise the reshape error
    assert len(res.scores) > 0


def test_atp_runner_uses_config_head_dim_and_runs():
    import torch

    from circuitry.core.patching import logit_diff_t
    from circuitry.patching import AtPRunner

    model, cfg = _explicit_head_dim_model()
    resolver = HFSiteResolver.from_config(cfg)
    runner = AtPRunner(model, resolver=resolver)
    assert runner.head_dim == 16

    torch.manual_seed(0)
    clean = {"input_ids": torch.randint(0, 64, (1, 6))}
    corrupt = {"input_ids": torch.randint(0, 64, (1, 6))}

    def metric(out):
        return logit_diff_t(out.logits if hasattr(out, "logits") else out, 1, 2)

    res = runner.run(clean, corrupt, metric, qk_fix=True)
    assert len(res.scores) > 0


def test_acdc_anchors_hold_with_explicit_head_dim():
    """ACDC inherits head_dim from EAP; empty anchor KL==0, full anchor==corrupted run."""
    import torch

    from circuitry.patching import ACDCRunner
    from circuitry.patching.graph import edge_sort_key  # noqa: F401  (ensure import ok)

    model, cfg = _explicit_head_dim_model()
    resolver = HFSiteResolver.from_config(cfg)
    acdc = ACDCRunner(model, resolver=resolver)
    assert acdc.head_dim == 16

    torch.manual_seed(0)
    clean = {"input_ids": torch.randint(0, 64, (1, 6))}
    corrupt = {"input_ids": torch.randint(0, 64, (1, 6))}
    corr_act = acdc._cache_corrupted_acts(corrupt)
    with torch.no_grad():
        clean_logits = acdc._eap._call_model(clean)
        clean_logits = clean_logits.logits if hasattr(clean_logits, "logits") else clean_logits
        corr_logits = acdc._eap._call_model(corrupt)
        corr_logits = corr_logits.logits if hasattr(corr_logits, "logits") else corr_logits
    # empty anchor: nothing removed -> circuit logits == clean run
    empty = acdc._forward(clean, set(), corr_act)
    assert torch.allclose(empty[:, -1], clean_logits[:, -1], atol=1e-4)
    # full anchor: all edges removed -> circuit logits == corrupted run
    full = acdc._forward(clean, set(acdc.graph.edges), corr_act)
    assert torch.allclose(full[:, -1], corr_logits[:, -1], atol=1e-4)
