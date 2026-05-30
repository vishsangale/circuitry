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
                        "heavy_tail_alpha", "sv_histogram",
                        # v1.3 training-dynamics:
                        "update_delta", "rank_trajectory", "direction_cosine"],
    activation_diagnostics=["gate_stats", "dead_fraction", "kurtosis",
                            "participation_ratio",
                            # v0.9 additions:
                            "logit_lens_kl", "induction_score",
                            "attention_pattern_entropy"],
    gradient_diagnostics=["norms_per_param"],
)


def register() -> None:
    """Register the LLM recipe. Idempotent under test fixtures via
    ``_clear_registry_for_tests``."""
    register_recipe(RECIPE)
