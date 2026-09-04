from datetime import date

import numpy as np
import pandas as pd

from quintropy.data import (
    answer_indices,
    load_allowed_words,
    load_answer_words,
    load_history,
)
from quintropy.priors import (
    CausalLinguisticPrior,
    EditorialRegimeConfig,
    EditorialRegimePrior,
    FrequencyPriorConfig,
    HybridPriorConfig,
    LinguisticPriorConfig,
    generate_priors,
    load_prior_artifact,
    save_prior_artifact,
)
from quintropy.selection import (
    purged_temporal_folds,
    select_editorial_config,
    select_policy_config,
)


def test_generate_priors_excludes_target_and_future_answers(monkeypatch):
    history = pd.DataFrame(
        {
            "game": [0, 1, 2],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "answer": ["apple", "baker", "cider"],
        }
    )
    captured_past = []

    class RecordingPrior:
        def __init__(self, words, config):
            self.words = words

        def predict(self, past):
            captured_past.append(tuple(past["answer"]))
            return __import__("numpy").full(len(self.words), 1 / len(self.words))

    monkeypatch.setattr("quintropy.priors.FrequencyPrior", RecordingPrior)
    monkeypatch.setattr("quintropy.priors.data_provenance", lambda: "test-provenance")
    targets, priors, manifest = generate_priors(
        history,
        ("apple", "baker", "cider"),
        date(2026, 1, 2),
        date(2026, 1, 3),
        FrequencyPriorConfig(),
    )
    assert targets["answer"].tolist() == ["baker", "cider"]
    assert captured_past == [("apple",), ("apple", "baker")]
    assert priors.shape == (2, 3)
    assert manifest.information_cutoff == "strictly before each target date"


def test_prior_artifact_round_trip(tmp_path, monkeypatch):
    history = pd.DataFrame(
        {
            "game": [0, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "answer": ["apple", "baker"],
        }
    )
    monkeypatch.setattr("quintropy.priors.data_provenance", lambda: "test-provenance")
    targets, priors, manifest = generate_priors(
        history, ("apple", "baker"), date(2026, 1, 2), date(2026, 1, 2)
    )
    path = tmp_path / "priors.npz"
    save_prior_artifact(path, ("apple", "baker"), targets, priors, manifest)
    (
        words,
        restored_answer_indices,
        restored_targets,
        restored_priors,
        restored_manifest,
    ) = load_prior_artifact(path)
    assert words == ("apple", "baker")
    assert restored_answer_indices.tolist() == [0, 1]
    assert restored_targets.equals(targets)
    assert (restored_priors == priors).all()
    assert restored_manifest["information_cutoff"] == "strictly before each target date"


def test_editorial_regime_prior_rejects_future_rows():
    history = pd.DataFrame(
        {"game": [0], "date": pd.to_datetime(["2026-01-02"]), "answer": ["apple"]}
    )
    model = EditorialRegimePrior(("apple", "baker"), EditorialRegimeConfig())
    with __import__("pytest").raises(ValueError, match="target or future"):
        model.predict(history, pd.Timestamp("2026-01-02"))


def test_editorial_regime_prior_is_normalized_and_causal():
    history = pd.DataFrame(
        {
            "game": [0, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "answer": ["apple", "apple"],
        }
    )
    model = EditorialRegimePrior(("apple", "baker", "cider"), EditorialRegimeConfig())
    probability = model.predict(history, pd.Timestamp("2026-01-03"))
    assert np.isclose(probability.sum(), 1)
    assert (probability > 0).all()


def test_editorial_expanded_temperature_is_explicit_and_normalized():
    history = pd.DataFrame(
        {"game": [0], "date": pd.to_datetime(["2026-01-01"]), "answer": ["apple"]}
    )
    model = EditorialRegimePrior(
        ("apple", "baker", "cider"),
        EditorialRegimeConfig(frequency_temperature_expanded=50),
    )
    probability = model.predict(history, pd.Timestamp("2026-01-02"))
    assert np.isclose(probability.sum(), 1)


def test_regime_feature_weight_is_bounded():
    with __import__("pytest").raises(ValueError, match="regime_feature_weight"):
        EditorialRegimeConfig(regime_feature_weight=1.1)


def test_regime_frequency_profile_weight_is_non_negative():
    with __import__("pytest").raises(
        ValueError, match="regime_frequency_profile_weight"
    ):
        EditorialRegimeConfig(regime_frequency_profile_weight=-0.1)


def test_linguistic_prior_is_normalized_and_rejects_future_rows():
    words = ("apple", "baker", "cider", "dandy")
    history = pd.DataFrame(
        {
            "game": range(24),
            "date": pd.date_range("2026-01-01", periods=24, freq="D"),
            "answer": [words[index % len(words)] for index in range(24)],
        }
    )
    model = CausalLinguisticPrior(
        words, LinguisticPriorConfig(minimum_history=20, retrain_every_games=1)
    )
    probability = model.predict(history, pd.Timestamp("2026-01-25"))
    assert np.isclose(probability.sum(), 1)
    assert (probability > 0).all()
    with __import__("pytest").raises(ValueError, match="target or future"):
        model.predict(history, pd.Timestamp("2026-01-24"))


def test_linguistic_prior_refits_for_a_different_same_length_history():
    words = ("apple", "baker", "cider", "dandy")
    history = pd.DataFrame(
        {
            "game": range(24),
            "date": pd.date_range("2026-01-01", periods=24, freq="D"),
            "answer": [words[index % len(words)] for index in range(24)],
        }
    )
    alternate = history.copy()
    alternate.loc[0, "answer"] = "baker"
    model = CausalLinguisticPrior(
        words, LinguisticPriorConfig(minimum_history=20, retrain_every_games=28)
    )
    model.predict(history, pd.Timestamp("2026-01-25"))
    first_signature = model._last_fit_signature
    model.predict(alternate, pd.Timestamp("2026-01-25"))
    assert first_signature != model._last_fit_signature


def test_hybrid_prior_manifest_is_explicitly_composed():
    history = pd.DataFrame(
        {
            "game": [0, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "answer": ["apple", "baker"],
        }
    )
    config = HybridPriorConfig(
        editorial=EditorialRegimeConfig(),
        linguistic=LinguisticPriorConfig(minimum_history=20),
        linguistic_weight=0.15,
    )
    _, priors, manifest = generate_priors(
        history, ("apple", "baker", "cider"), date(2026, 1, 2), date(2026, 1, 2), config
    )
    assert manifest.model == "hybrid_editorial_linguistic_prior"
    assert manifest.config["linguistic_weight"] == 0.15
    assert np.isclose(priors.sum(), 1)


def test_selection_uses_only_its_chronological_window(monkeypatch):
    history = pd.DataFrame(
        {
            "game": [0, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "answer": ["apple", "baker"],
        }
    )
    configs = (
        EditorialRegimeConfig(frequency_temperature=2),
        EditorialRegimeConfig(frequency_temperature=4),
    )
    chosen, leaderboard = select_editorial_config(
        history, ("apple", "baker"), date(2026, 1, 2), date(2026, 1, 2), configs
    )
    assert len(leaderboard) == 2
    assert chosen in configs


def test_policy_selection_prefers_lower_expected_tries_with_equal_coverage():
    from quintropy.feedback import build_feedback_table
    from quintropy.policy import PolicyConfig

    words = ("cigar", "rebut", "sissy")
    history = pd.DataFrame(
        {
            "game": [0, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "answer": ["cigar", "rebut"],
        }
    )
    priors = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    selected, leaderboard = select_policy_config(
        words,
        build_feedback_table(words),
        history,
        priors,
        (
            PolicyConfig(starter="cigar", direct_hit_weight=0),
            PolicyConfig(starter="cigar", direct_hit_weight=1),
        ),
    )
    assert selected in (
        PolicyConfig(starter="cigar", direct_hit_weight=0),
        PolicyConfig(starter="cigar", direct_hit_weight=1),
    )
    assert len(leaderboard) == 2


def test_answer_universe_is_static_subset_and_covers_history():
    actions = load_allowed_words()
    answers = load_answer_words()
    assert len(answers) < len(actions)
    assert set(actions[:2_315]) <= set(answers) <= set(actions)
    assert set(load_history()["answer"]) <= set(answers)


def test_history_is_a_complete_daily_sequence():
    history = load_history()
    assert np.all(np.diff(history["game"]) == 1)
    assert np.all(history["date"].diff().dropna().dt.days == 1)
    assert history.iloc[-1]["game"] == 1903
    assert history.iloc[-1]["date"] == pd.Timestamp("2026-09-04")
    assert history.iloc[-1]["answer"] == "wager"


def test_linguistic_default_uses_temporally_selected_regularization():
    assert LinguisticPriorConfig().regularization == 0.01


def test_purged_temporal_folds_are_ordered_and_disjoint():
    history = pd.DataFrame(
        {
            "game": range(240),
            "date": pd.date_range("2026-01-01", periods=240, freq="D"),
            "answer": ["cigar"] * 240,
        }
    )
    folds = purged_temporal_folds(
        history,
        history["date"].iloc[160].date(),
        history["date"].iloc[-1].date(),
        n_splits=4,
        train_games=84,
        embargo_games=7,
    )
    assert len(folds) == 4
    for fold in folds:
        assert len(fold.train_games) == 84
        assert len(fold.embargo_games) == 7
        assert set(fold.train_games).isdisjoint(fold.embargo_games)
        assert set(fold.train_games).isdisjoint(fold.validation_games)
        assert max(fold.train_games) < min(fold.embargo_games)
        assert max(fold.embargo_games) < min(fold.validation_games)


def test_frequency_prior_assigns_zero_mass_outside_answer_universe():
    words = load_allowed_words()
    model = __import__("quintropy.priors", fromlist=["FrequencyPrior"]).FrequencyPrior(
        words
    )
    probability = model.predict(pd.DataFrame(columns=["game", "date", "answer"]))
    supported = answer_indices(words)
    outside = np.ones(len(words), dtype=bool)
    outside[supported] = False
    assert np.isclose(probability.sum(), 1)
    assert np.all(probability[outside] == 0)


def test_numeric_configuration_rejects_non_finite_values():
    from quintropy.policy import PolicyConfig

    with __import__("pytest").raises(ValueError, match="finite"):
        PolicyConfig(direct_hit_weight=float("nan"))
    with __import__("pytest").raises(ValueError, match="finite"):
        EditorialRegimeConfig(frequency_temperature=float("inf"))
    with __import__("pytest").raises(ValueError, match="invalid"):
        LinguisticPriorConfig(regularization=float("nan"))
