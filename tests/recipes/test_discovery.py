import torch

from circuitry.recipes._discovery import discover


def _make_llama_sd():
    """Synthetic state_dict matching LLaMA-family naming."""
    return {
        "tok_embeddings.weight": torch.empty(100, 64),
        "norm.weight": torch.empty(64),
        "output.weight": torch.empty(100, 64),
        "layers.0.attention.wq.weight": torch.empty(64, 64),
        "layers.0.attention.wk.weight": torch.empty(64, 64),
        "layers.0.attention.wv.weight": torch.empty(64, 64),
        "layers.0.attention.wo.weight": torch.empty(64, 64),
        "layers.0.attention_norm.weight": torch.empty(64),
        "layers.0.feed_forward.w1.weight": torch.empty(128, 64),
        "layers.0.feed_forward.w2.weight": torch.empty(64, 128),
        "layers.0.feed_forward.w3.weight": torch.empty(128, 64),
        "layers.0.ffn_norm.weight": torch.empty(64),
        "layers.1.attention.wq.weight": torch.empty(64, 64),
    }


def test_discover_assigns_layers():
    sd = _make_llama_sd()
    out = discover(sd)
    names = {p.name: p for p in out.params}
    assert names["layers.0.attention.wq.weight"].layer == 0
    assert names["layers.1.attention.wq.weight"].layer == 1
    assert names["tok_embeddings.weight"].layer is None


def test_discover_assigns_roles():
    sd = _make_llama_sd()
    out = discover(sd)
    names = {p.name: p for p in out.params}
    assert names["layers.0.attention.wq.weight"].role == "attn_q"
    assert names["layers.0.attention.wk.weight"].role == "attn_k"
    assert names["layers.0.attention.wv.weight"].role == "attn_v"
    assert names["layers.0.attention.wo.weight"].role == "attn_o"
    assert names["layers.0.feed_forward.w1.weight"].role in ("ffn_in", "ffn_gate")
    assert names["layers.0.feed_forward.w2.weight"].role == "ffn_out"
    assert names["tok_embeddings.weight"].role == "embedding"
    assert names["output.weight"].role == "lm_head"


def test_discover_params_by_role():
    sd = _make_llama_sd()
    out = discover(sd)
    by_role = out.params_by_role()
    assert "attn_q" in by_role
    assert len(by_role["attn_q"]) == 2  # layers 0 and 1
