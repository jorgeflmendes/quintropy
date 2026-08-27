import numpy as np
import pytest

from quintropy.evaluation import (
    bootstrap_mean_interval,
    moving_block_mean_interval,
    run_game,
    summarise,
    wilson_interval,
)
from quintropy.feedback import build_feedback_table
from quintropy.policy import EntropyPolicy, PolicyConfig, entropy_bits


def test_entropy_is_measured_in_bits():
    assert entropy_bits(np.array([0, 1], dtype=np.uint8), np.array([0.5, 0.5])) == 1.0


def test_run_game_retains_path_and_prior_diagnostics():
    words = ("cigar", "rebut", "sissy")
    table = build_feedback_table(words)
    policy = EntropyPolicy(
        words, table, PolicyConfig(starter="cigar", exploit_threshold=0.0)
    )
    result = run_game(
        words, table, policy, "rebut", np.array([0.1, 0.8, 0.1]), 1, "2026-01-01"
    )
    assert result.solved
    assert result.path == ("cigar", "rebut")
    assert result.true_rank == 1


def test_exact_endgame_returns_a_legal_action():
    words = ("cigar", "rebut", "sissy")
    table = build_feedback_table(words)
    policy = EntropyPolicy(
        words,
        table,
        PolicyConfig(starter="cigar", exploit_threshold=1.0, exact_endgame_limit=3),
    )
    action = policy.choose(
        np.array([0, 1, 2], dtype=np.int32), np.array([1 / 3, 1 / 3, 1 / 3]), turn=2
    )
    assert action in range(len(words))


def test_exact_three_candidate_endgame_is_enabled_by_default():
    assert PolicyConfig().exact_endgame_limit == 3


def test_adaptive_starter_returns_a_legal_action():
    words = ("cigar", "rebut", "sissy")
    policy = EntropyPolicy(
        words,
        build_feedback_table(words),
        PolicyConfig(starter="cigar", adaptive_starter=True),
    )
    action = policy.choose(
        np.arange(len(words), dtype=np.int32), np.array([0.6, 0.3, 0.1]), turn=1
    )
    assert action in range(len(words))


def test_tail_wordfreq_policy_returns_a_legal_action():
    words = ("cigar", "rebut", "sissy", "humph")
    table = build_feedback_table(words)
    policy = EntropyPolicy(
        words,
        table,
        PolicyConfig(starter="cigar", tail_wordfreq_weight=1.0, tail_wordfreq_gap=0.0),
    )
    action = policy.choose(
        np.array([1, 2, 3], dtype=np.int32), np.full(len(words), 0.25), turn=3
    )
    assert action in range(len(words))


def test_tail_wordfreq_policy_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="weights"):
        PolicyConfig(tail_wordfreq_weight=-0.1)
    with pytest.raises(ValueError, match="weights"):
        PolicyConfig(expanded_direct_hit_factor=-0.1)


def test_expanded_hit_factor_uses_canonical_vocabulary_regime():
    words = ("cigar", "geode")
    policy = EntropyPolicy(
        words,
        build_feedback_table(words),
        PolicyConfig(starter="cigar", expanded_direct_hit_factor=1.5),
    )
    assert not policy._expanded_mask[policy.index["cigar"]]
    assert policy._expanded_mask[policy.index["geode"]]


def test_expanded_language_override_is_narrow_and_causal():
    words = ("cigar", "rebut", "sissy", "humph", "geode")
    policy = EntropyPolicy(
        words,
        build_feedback_table(words),
        PolicyConfig(
            starter="cigar",
            expanded_language_override=True,
            expanded_language_min_probability=0.2,
            expanded_language_editorial_min=0.15,
            expanded_language_editorial_max=0.2,
        ),
    )
    action = policy.choose(
        np.arange(5, dtype=np.int32),
        np.array([0.25, 0.25, 0.18, 0.17, 0.15]),
        turn=3,
        auxiliary_prior=np.array([0.05, 0.05, 0.05, 0.05, 0.80]),
    )
    assert words[action] == "geode"


def test_expanded_language_override_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="weights"):
        PolicyConfig(
            expanded_language_editorial_min=0.3, expanded_language_editorial_max=0.2
        )
    with pytest.raises(ValueError, match="bounds"):
        PolicyConfig(expanded_language_min_candidates=1)


def test_summary_includes_uncertainty_and_proper_scoring_fields():
    words = ("cigar", "rebut", "sissy")
    table = build_feedback_table(words)
    policy = EntropyPolicy(
        words, table, PolicyConfig(starter="cigar", exploit_threshold=0.0)
    )
    result = run_game(
        words, table, policy, "rebut", np.array([0.1, 0.8, 0.1]), 1, "2026-01-01"
    )
    report = summarise([result], len(words))
    assert report["prior_log_loss_bits"] > 0
    assert report["accuracy_at_6_wilson_95_iid_sensitivity"][0] < 1
    assert (
        report["uncertainty"]["method"] == "circular_moving_block_percentile_bootstrap"
    )
    assert "243" in report["top_k_coverage"]
    assert wilson_interval(1, 1)[1] == 1.0


def test_temporal_interval_preserves_local_dependence():
    values = np.repeat([0.0, 1.0], 20)
    iid = bootstrap_mean_interval(values)
    blocked, block_length = moving_block_mean_interval(values)
    assert block_length > 1
    assert blocked[0] < iid[0]
    assert blocked[1] > iid[1]


def test_game_rejects_secret_outside_declared_answer_universe():
    words = ("cigar", "rebut", "sissy")
    table = build_feedback_table(words)
    policy = EntropyPolicy(
        words, table, PolicyConfig(starter="cigar"), answer_indices=np.array([0, 1])
    )
    with pytest.raises(ValueError, match="declared answer universe"):
        run_game(
            words, table, policy, "sissy", np.array([0.5, 0.5, 0.0]), 1, "2026-01-01"
        )
