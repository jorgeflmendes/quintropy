"""End-to-end causal evaluation, including prior and solver diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .policy import EntropyPolicy, ExactEndgamePlanner


@dataclass(frozen=True)
class GameResult:
    game: int
    date: str
    answer: str
    tries: int
    solved: bool
    path: tuple[str, ...]
    true_probability: float
    true_rank: int


def run_game(
    words: tuple[str, ...],
    table: np.ndarray,
    policy: EntropyPolicy,
    answer: str,
    prior: np.ndarray,
    game: int,
    day: str,
    auxiliary_prior: np.ndarray | None = None,
) -> GameResult:
    """Run one game and retain the full decision path for independent review."""
    index = policy.index
    if answer not in index:
        raise ValueError(f"Observed answer {answer!r} is outside the action universe.")
    secret = index[answer]
    prior = np.asarray(prior, dtype=float)
    if (
        prior.shape != (len(words),)
        or not np.isfinite(prior).all()
        or np.any(prior < 0)
        or prior.sum() <= 0
    ):
        raise ValueError(
            "Prior must be a finite, non-negative vector over the action universe."
        )
    prior = prior / prior.sum()
    if auxiliary_prior is not None:
        auxiliary_prior = np.asarray(auxiliary_prior, dtype=float)
        if (
            auxiliary_prior.shape != (len(words),)
            or not np.isfinite(auxiliary_prior).all()
            or np.any(auxiliary_prior < 0)
            or auxiliary_prior.sum() <= 0
        ):
            raise ValueError(
                "Auxiliary prior must be a finite, non-negative vector over the action universe."
            )
        auxiliary_prior = auxiliary_prior / auxiliary_prior.sum()
    true_probability = float(prior[secret])
    true_rank = int(np.sum(prior > true_probability) + 1)
    candidates = policy.answer_indices.copy()
    if not np.any(candidates == secret):
        raise ValueError(
            f"Observed answer {answer!r} is outside the declared answer universe."
        )
    planner = ExactEndgamePlanner(table, prior, policy.config.exact_endgame_limit)
    path: list[str] = []
    for turn in range(1, 7):
        action = policy.choose(candidates, prior, turn, planner, auxiliary_prior)
        path.append(words[action])
        if action == secret:
            return GameResult(
                game, day, answer, turn, True, tuple(path), true_probability, true_rank
            )
        pattern = table[action, secret]
        candidates = candidates[table[action, candidates] == pattern]
        if not len(candidates):
            break
    return GameResult(
        game, day, answer, 7, False, tuple(path), true_probability, true_rank
    )


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        raise ValueError("Cannot create an interval for zero observations.")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
        / denominator
    )
    return float(centre - radius), float(centre + radius)


def bootstrap_mean_interval(
    values: np.ndarray, seed: int = 0, repetitions: int = 10_000
) -> tuple[float, float]:
    """Legacy IID percentile interval retained for schema-v1 audits only."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot bootstrap zero observations.")
    generator = np.random.default_rng(seed)
    samples = generator.choice(
        values, size=(repetitions, len(values)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def moving_block_mean_interval(
    values: np.ndarray,
    *,
    block_length: int | None = None,
    seed: int = 0,
    repetitions: int = 10_000,
) -> tuple[tuple[float, float], int]:
    """Circular moving-block bootstrap interval for a chronological mean."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot bootstrap zero observations.")
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive.")
    resolved_block = min(
        len(values), block_length or max(1, round(len(values) ** (1 / 3)))
    )
    if resolved_block < 1:
        raise ValueError("Block length must be positive.")
    block_count = int(np.ceil(len(values) / resolved_block))
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, len(values), size=(repetitions, block_count))
    offsets = np.arange(resolved_block)
    indices = (starts[..., None] + offsets) % len(values)
    sample_means = values[indices.reshape(repetitions, -1)[:, : len(values)]].mean(
        axis=1
    )
    lower, upper = np.quantile(sample_means, [0.025, 0.975])
    return (float(lower), float(upper)), resolved_block


def summarise(results: list[GameResult], universe_size: int) -> dict[str, object]:
    """Return solver performance and proper scoring rules for one fixed split."""
    if not results:
        raise ValueError("No results to summarise.")
    tries = np.asarray([result.tries for result in results], dtype=float)
    solved = np.asarray([result.solved for result in results], dtype=bool)
    probability = np.asarray(
        [result.true_probability for result in results], dtype=float
    )
    ranks = np.asarray([result.true_rank for result in results], dtype=int)
    mean_interval, block_length = moving_block_mean_interval(tries)
    accuracy_interval, _ = moving_block_mean_interval(
        solved.astype(float), block_length=block_length
    )
    return {
        "n_games": len(results),
        "mean_tries": float(tries.mean()),
        "mean_tries_moving_block_bootstrap_95": list(mean_interval),
        "median_tries": float(np.median(tries)),
        "worst_case": int(tries.max()),
        "accuracy_at_6": float(solved.mean()),
        "accuracy_at_6_moving_block_bootstrap_95": list(accuracy_interval),
        "accuracy_at_6_wilson_95_iid_sensitivity": list(
            wilson_interval(int(solved.sum()), len(solved))
        ),
        "uncertainty": {
            "method": "circular_moving_block_percentile_bootstrap",
            "block_length_games": block_length,
            "repetitions": 10_000,
            "note": "Wilson is reported only as an IID sensitivity analysis.",
        },
        "tries_distribution": {
            str(key): int(value)
            for key, value in zip(*np.unique(tries.astype(int), return_counts=True))
        },
        "prior_log_loss_bits": float(np.mean(-np.log2(probability))),
        "uniform_log_loss_bits": float(np.log2(universe_size)),
        "true_rank": {"mean": float(ranks.mean()), "median": float(np.median(ranks))},
        "top_k_coverage": {
            str(k): float(np.mean(ranks <= k)) for k in (1, 10, 100, 243, 1000)
        },
    }


def evaluate_split(
    words: tuple[str, ...],
    table: np.ndarray,
    policy: EntropyPolicy,
    targets: pd.DataFrame,
    priors: np.ndarray,
    auxiliary_priors: np.ndarray | None = None,
) -> tuple[list[GameResult], dict[str, object]]:
    if len(targets) != len(priors):
        raise ValueError("Each evaluation target needs exactly one prior.")
    if auxiliary_priors is not None:
        auxiliary_priors = np.asarray(auxiliary_priors, dtype=float)
        if auxiliary_priors.shape != priors.shape:
            raise ValueError(
                "Auxiliary priors must have the same shape as primary priors."
            )
    results = [
        run_game(
            words,
            table,
            policy,
            row.answer,
            prior,
            int(row.game),
            row.date.strftime("%Y-%m-%d"),
            None if auxiliary_priors is None else auxiliary,
        )
        for row, prior, auxiliary in zip(
            targets.itertuples(index=False),
            priors,
            np.zeros_like(priors) if auxiliary_priors is None else auxiliary_priors,
        )
    ]
    report = summarise(results, len(policy.answer_indices))
    normalized = priors / priors.sum(axis=1, keepdims=True)
    index = {word: i for i, word in enumerate(words)}
    truth = np.array([index[result.answer] for result in results])
    report["prior_brier_score"] = float(
        np.mean(
            np.sum(normalized * normalized, axis=1)
            - 2 * normalized[np.arange(len(truth)), truth]
            + 1
        )
    )
    return results, report


def save_evaluation(
    output_dir: Path,
    results: list[GameResult],
    report: dict[str, object],
    manifest: dict[str, object],
) -> None:
    """Write inspectable per-game paths and machine-readable experiment metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(result), "path": " ".join(result.path)} for result in results]
    pd.DataFrame(rows).to_csv(
        output_dir / "games.csv", index=False, lineterminator="\n"
    )
    (output_dir / "metrics.json").write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    (output_dir / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
