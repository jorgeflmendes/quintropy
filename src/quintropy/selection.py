"""Chronological hyperparameter selection for causal prior families."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from .evaluation import evaluate_split
from .policy import EntropyPolicy, PolicyConfig
from .priors import (
    EditorialRegimeConfig,
    HybridPriorConfig,
    LinguisticPriorConfig,
    generate_priors,
)


@dataclass(frozen=True)
class PurgedTemporalFold:
    """One forward validation fold with fixed tuning and embargo windows."""

    train_games: tuple[int, ...]
    embargo_games: tuple[int, ...]
    validation_games: tuple[int, ...]


def purged_temporal_folds(
    history: pd.DataFrame,
    start: date,
    end: date,
    *,
    n_splits: int = 4,
    train_games: int = 84,
    embargo_games: int = 7,
) -> tuple[PurgedTemporalFold, ...]:
    """Create contiguous future folds with fixed training and embargo windows."""
    if n_splits < 2 or train_games < 1 or embargo_games < 0:
        raise ValueError("Fold, training, and embargo sizes are invalid.")
    required = ("game", "date", "answer")
    if set(history) != set(required) or history[list(required)].isna().any().any():
        raise ValueError("History must contain complete game, date, and answer rows.")
    if (
        history["game"].duplicated().any()
        or history["date"].duplicated().any()
        or not history["game"].is_monotonic_increasing
        or not history["date"].is_monotonic_increasing
    ):
        raise ValueError("History rows must have unique, increasing games and dates.")
    validation = history[
        (history["date"].dt.date >= start) & (history["date"].dt.date <= end)
    ]
    if len(validation) < n_splits:
        raise ValueError("The validation window is too short for the requested folds.")

    folds: list[PurgedTemporalFold] = []
    for indices in np.array_split(np.arange(len(validation)), n_splits):
        block = validation.iloc[indices]
        eligible = history[history["date"] < block["date"].iloc[0]]
        required_history = train_games + embargo_games
        if len(eligible) < required_history:
            raise ValueError("Insufficient history before a validation fold.")
        train = eligible.iloc[-required_history : -embargo_games or None]
        embargo = (
            eligible.iloc[-embargo_games:] if embargo_games else eligible.iloc[0:0]
        )
        fold = PurgedTemporalFold(
            tuple(int(game) for game in train["game"]),
            tuple(int(game) for game in embargo["game"]),
            tuple(int(game) for game in block["game"]),
        )
        if set(fold.train_games) & set(fold.validation_games):
            raise AssertionError("Training and validation games overlap.")
        if fold.embargo_games and max(fold.train_games) >= min(fold.embargo_games):
            raise AssertionError("Embargo does not follow the training window.")
        if max((*fold.train_games, *fold.embargo_games)) >= min(fold.validation_games):
            raise AssertionError(
                "Validation does not follow training and embargo windows."
            )
        folds.append(fold)
    return tuple(folds)


def editorial_config_grid() -> tuple[EditorialRegimeConfig, ...]:
    """Small predeclared grid; extend it only before opening validation data."""
    return tuple(
        EditorialRegimeConfig(
            frequency_temperature=temperature,
            positional_weight=position_weight,
            bigram_weight=bigram_weight,
            structural_weight=structural_weight,
            positional_smoothing=5.0,
            structural_smoothing=2.0,
            recency_half_life_days=half_life,
        )
        for temperature, position_weight, bigram_weight, structural_weight, half_life in itertools.product(
            (2.0, 4.0, 6.0, 7.0),
            (0.5, 1.0, 1.5),
            (0.0, 0.25),
            (0.25, 0.5),
            (90.0, 365.0),
        )
    )


def select_editorial_config(
    history: pd.DataFrame,
    words: tuple[str, ...],
    start: date,
    end: date,
    configs: tuple[EditorialRegimeConfig, ...] | None = None,
) -> tuple[EditorialRegimeConfig, list[dict[str, object]]]:
    """Select by log-loss on an earlier chronological validation window."""
    if start > end:
        raise ValueError("Selection start must not be after selection end.")
    index = {word: position for position, word in enumerate(words)}
    leaderboard: list[dict[str, object]] = []
    for config in configs or editorial_config_grid():
        targets, priors, _ = generate_priors(history, words, start, end, config)
        truth = np.asarray(
            [index[answer] for answer in targets["answer"]], dtype=np.int32
        )
        probability = priors[np.arange(len(truth)), truth]
        leaderboard.append(
            {
                "config": asdict(config),
                "n_games": len(targets),
                "mean_log_loss_bits": float(np.mean(-np.log2(probability))),
                "median_true_rank": float(
                    np.median((priors > probability[:, None]).sum(axis=1) + 1)
                ),
            }
        )
    leaderboard.sort(key=lambda row: (row["mean_log_loss_bits"], str(row["config"])))
    return EditorialRegimeConfig(**leaderboard[0]["config"]), leaderboard


def linguistic_config_grid() -> tuple[LinguisticPriorConfig, ...]:
    """Small predeclared regularisation/cadence grid for temporal selection."""
    return tuple(
        LinguisticPriorConfig(
            regularization=regularization,
            negative_ratio=negatives,
            retrain_every_games=cadence,
        )
        for regularization, negatives, cadence in itertools.product(
            (0.01, 0.03, 0.1), (4, 8), (14, 28)
        )
    )


def select_linguistic_config(
    history: pd.DataFrame,
    words: tuple[str, ...],
    start: date,
    end: date,
    configs: tuple[LinguisticPriorConfig, ...] | None = None,
) -> tuple[LinguisticPriorConfig, list[dict[str, object]]]:
    """Choose a causal linguistic model by log-loss on a closed past window."""
    if start > end:
        raise ValueError("Selection start must not be after selection end.")
    index = {word: position for position, word in enumerate(words)}
    leaderboard: list[dict[str, object]] = []
    for config in configs or linguistic_config_grid():
        targets, priors, _ = generate_priors(history, words, start, end, config)
        truth = np.asarray(
            [index[answer] for answer in targets["answer"]], dtype=np.int32
        )
        probability = priors[np.arange(len(truth)), truth]
        leaderboard.append(
            {
                "config": asdict(config),
                "n_games": len(targets),
                "mean_log_loss_bits": float(np.mean(-np.log2(probability))),
                "median_true_rank": float(
                    np.median((priors > probability[:, None]).sum(axis=1) + 1)
                ),
            }
        )
    leaderboard.sort(key=lambda row: (row["mean_log_loss_bits"], str(row["config"])))
    return LinguisticPriorConfig(**leaderboard[0]["config"]), leaderboard


def nested_walk_forward_linguistic(
    history: pd.DataFrame,
    words: tuple[str, ...],
    start: date,
    end: date,
    *,
    inner_games: int = 84,
    n_splits: int = 4,
    embargo_games: int = 7,
    configs: tuple[LinguisticPriorConfig, ...] | None = None,
) -> list[dict[str, object]]:
    """Select on purged training folds and score their subsequent blocks."""
    index = {word: position for position, word in enumerate(words)}
    report: list[dict[str, object]] = []
    folds = purged_temporal_folds(
        history,
        start,
        end,
        n_splits=n_splits,
        train_games=inner_games,
        embargo_games=embargo_games,
    )
    for fold_index, fold in enumerate(folds):
        inner = history[history["game"].isin(fold.train_games)]
        outer = history[history["game"].isin(fold.validation_games)]
        config, _ = select_linguistic_config(
            history,
            words,
            inner["date"].iloc[0].date(),
            inner["date"].iloc[-1].date(),
            configs,
        )
        _, priors, _ = generate_priors(
            history,
            words,
            outer["date"].iloc[0].date(),
            outer["date"].iloc[-1].date(),
            config,
        )
        truth = np.asarray(
            [index[answer] for answer in outer["answer"]], dtype=np.int32
        )
        probability = priors[np.arange(len(truth)), truth]
        report.append(
            {
                "fold": fold_index,
                "inner_start": inner["date"].iloc[0].date().isoformat(),
                "inner_end": inner["date"].iloc[-1].date().isoformat(),
                "embargo_games": list(fold.embargo_games),
                "outer_start": outer["date"].iloc[0].date().isoformat(),
                "outer_end": outer["date"].iloc[-1].date().isoformat(),
                "n_games": len(outer),
                "selected_config": asdict(config),
                "mean_log_loss_bits": float(np.mean(-np.log2(probability))),
            }
        )
    return report


def select_hybrid_weight(
    history: pd.DataFrame,
    words: tuple[str, ...],
    start: date,
    end: date,
    editorial: EditorialRegimeConfig,
    linguistic: LinguisticPriorConfig,
    weights: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30),
) -> tuple[HybridPriorConfig, list[dict[str, object]]]:
    """Select an ensemble weight on an earlier closed temporal window."""
    index = {word: position for position, word in enumerate(words)}
    leaderboard: list[dict[str, object]] = []
    for weight in weights:
        config = HybridPriorConfig(
            editorial=editorial, linguistic=linguistic, linguistic_weight=weight
        )
        targets, priors, _ = generate_priors(history, words, start, end, config)
        truth = np.asarray(
            [index[answer] for answer in targets["answer"]], dtype=np.int32
        )
        probability = priors[np.arange(len(truth)), truth]
        leaderboard.append(
            {
                "linguistic_weight": weight,
                "n_games": len(targets),
                "mean_log_loss_bits": float(np.mean(-np.log2(probability))),
            }
        )
    leaderboard.sort(
        key=lambda row: (row["mean_log_loss_bits"], row["linguistic_weight"])
    )
    return HybridPriorConfig(
        editorial=editorial,
        linguistic=linguistic,
        linguistic_weight=float(leaderboard[0]["linguistic_weight"]),
    ), leaderboard


def policy_config_grid() -> tuple[PolicyConfig, ...]:
    """Predeclared small policy grid; its criterion is expected tries, not NLL."""
    return tuple(
        PolicyConfig(
            direct_hit_weight=weight,
            exploit_threshold=threshold,
            tail_wordfreq_weight=tail_weight,
            tail_wordfreq_gap=0.1,
            tail_wordfreq_start_turn=3,
            expanded_direct_hit_factor=expanded_factor,
        )
        for weight, threshold, tail_weight, expanded_factor in itertools.product(
            (0.0, 1.0, 2.0, 3.0),
            (0.15, 0.25, 0.4, 0.5),
            (0.0, 1.0),
            (1.0, 1.5),
        )
    )


def select_policy_config(
    words: tuple[str, ...],
    table: np.ndarray,
    targets: pd.DataFrame,
    priors: np.ndarray,
    configs: tuple[PolicyConfig, ...] | None = None,
) -> tuple[PolicyConfig, list[dict[str, object]]]:
    """Choose a policy on an already-closed earlier validation window."""
    leaderboard: list[dict[str, object]] = []
    for config in configs or policy_config_grid():
        _, report = evaluate_split(
            words, table, EntropyPolicy(words, table, config), targets, priors
        )
        leaderboard.append(
            {
                "config": asdict(config),
                "n_games": len(targets),
                "mean_tries": report["mean_tries"],
                "accuracy_at_6": report["accuracy_at_6"],
                "worst_case": report["worst_case"],
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -row["accuracy_at_6"],
            row["mean_tries"],
            row["worst_case"],
            str(row["config"]),
        )
    )
    return PolicyConfig(**leaderboard[0]["config"]), leaderboard
