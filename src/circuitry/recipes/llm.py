"""Stock LLM recipe. See docs/design.md §5."""

from __future__ import annotations

from circuitry.recipes import Recipe, register_recipe
from circuitry.recorder.hooks import HookPoint, TensorSource

RECIPE = Recipe(
    name="llm",
    hook_points=[
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.(q|k|v|o)_proj$"),
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.(w1|w2|w3|gate_proj|up_proj|down_proj)$"),
        # MoE router weight (e.g. OlmoeTopKRouter — named `gate`, not `gate_proj`).
        # Shape is 2-D [n_experts, hidden_size]; normal rank diagnostics apply.
        # optional=True: absent on dense models, so a 0-match must not fail strict
        # attach (the common case — see tests/recorder/test_optional_hookpoints.py).
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.mlp\.gate$", optional=True),
        # MoE batched expert weights (e.g. OlmoeExperts).  These modules store
        # all experts as a single 3-D tensor [n_experts, d_in, d_out] rather
        # than as separate leaf Linears.  The recorder iterates the leading
        # expert axis and emits per-expert weight diagnostics.
        HookPoint(source=TensorSource.WEIGHT,
                  pattern=r".*\.mlp\.experts$", optional=True),
        # MoE router OUTPUT (router logits, shape (..., n_experts)) for the
        # opt-in ``moe_routing`` activation diagnostic (v1.44). Absent on
        # dense models, hence optional.
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r".*\.mlp\.gate$", optional=True),
        # Attention submodule output. Covers HF (`self_attn`), HF-GPT-2 (`attn`),
        # and canonical LLaMA reference (`attention`).
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r".*\.(self_attn|attn|attention)$"),
        HookPoint(source=TensorSource.OUTPUT, pattern=r".*\.mlp$"),
        # Down-proj input hook for gate_stats on Llama/Gemma-style MLPs.
        HookPoint(source=TensorSource.INPUT, pattern=r".*\.down_proj$"),
        # Per-block layernorms. Covers HF (`input_layernorm`,
        # `post_attention_layernorm`), GPT-2 (`ln_1`, `ln_2`), and canonical
        # LLaMA reference (`attention_norm`, `ffn_norm`).
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r".*\.(input_layernorm|post_attention_layernorm"
                          r"|attention_norm|ffn_norm|ln_[12])$"),
        HookPoint(source=TensorSource.WEIGHT, pattern=r"embed.*"),
        HookPoint(source=TensorSource.WEIGHT, pattern=r"lm_head$"),
        HookPoint(source=TensorSource.GRAD,
                  pattern=r".*\.(q|k|v|o)_proj$"),
        # v0.9: block-output hook for logit_lens_kl per layer.
        HookPoint(source=TensorSource.OUTPUT,
                  pattern=r".*\.layers\.\d+$"),
    ],
    weight_diagnostics=["effective_rank", "attention_head_rank", "stable_rank",
                        "condition_number", "heavy_tail_alpha", "sv_histogram",
                        # v1.3 training-dynamics:
                        "update_delta", "rank_trajectory", "direction_cosine"],
    activation_diagnostics=["gate_stats", "dead_fraction", "kurtosis",
                            "participation_ratio",
                            # v0.9 additions:
                            "logit_lens_kl", "induction_score",
                            "copy_suppression_score",
                            "attention_pattern_entropy",
                            "attention_sink_score",
                            # v1.4 drift probe: default OFF; opt in via
                            # recipe.only(["drift_probe"]) or by clearing the
                            # disable and supplying a probe_batch.
                            "drift_probe",
                            # v1.44 MoE routing: default OFF (dense models
                            # have no router); opt in on MoE models.
                            "moe_routing"],
    gradient_diagnostics=["norms_per_param"],
    # drift_probe is expensive (second forward pass per emit step) so it is
    # gated OFF by default.  Users opt in via recipe.only(["drift_probe"]) or
    # dataclasses.replace(recipe, enabled={**recipe.enabled, "drift_probe": True},
    #                     probe_batch=my_tensor).
    enabled={"drift_probe": False, "moe_routing": False},
)


def register() -> None:
    """Register the LLM recipe. Idempotent under test fixtures via
    ``_clear_registry_for_tests``."""
    register_recipe(RECIPE)
