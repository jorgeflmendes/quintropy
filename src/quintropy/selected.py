"""Configuration of the model selected for publication and interactive use."""

from .policy import PolicyConfig
from .priors import EditorialRegimeConfig, LinguisticPriorConfig

SELECTED_EDITORIAL_CONFIG = EditorialRegimeConfig(
    frequency_temperature=7.0,
    regime_feature_weight=0.5,
    positional_weight=0.5,
    bigram_weight=0.0,
    structural_weight=0.25,
    recency_half_life_days=90.0,
)

SELECTED_LINGUISTIC_CONFIG = LinguisticPriorConfig(
    regularization=0.01,
    negative_ratio=8,
    retrain_every_games=28,
)

SELECTED_POLICY_CONFIG = PolicyConfig(
    starter="soare",
    direct_hit_weight=3.0,
    exploit_threshold=0.5,
    exact_endgame_limit=3,
    tail_wordfreq_weight=1.0,
    tail_wordfreq_gap=0.1,
    tail_wordfreq_start_turn=3,
    expanded_direct_hit_factor=1.5,
    expanded_language_override=True,
)

__all__ = [
    "SELECTED_EDITORIAL_CONFIG",
    "SELECTED_LINGUISTIC_CONFIG",
    "SELECTED_POLICY_CONFIG",
]
